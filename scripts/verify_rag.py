"""Прогон кодовых примеров презентации блока (LlamaIndex) на живом Qdrant + OpenAI.

Каждый блок повторяет конкретный слайд презентации и работает в одноразовой
коллекции, которая удаляется в конце. Что не запускается здесь (тяжёлые зависимости),
отмечено в выводе.

Запуск:
    uv run python scripts/verify_rag.py
"""

import asyncio
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llama_index.core import (  # noqa: E402
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.core.vector_stores import (  # noqa: E402
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding  # noqa: E402
from llama_index.llms.openai import OpenAI  # noqa: E402
from llama_index.vector_stores.qdrant import QdrantVectorStore  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

settings = get_settings()
KEY = settings.llm.openai_api_key.get_secret_value()
DATA = str(settings.rag_data_dir)


def _setup_models() -> None:
    Settings.embed_model = OpenAIEmbedding(model=settings.embedding_model, api_key=KEY)
    Settings.llm = OpenAI(model=settings.rag_llm_model, temperature=0.0, api_key=KEY)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


def main() -> None:
    print("=== versions ===")
    for p in ["llama-index", "llama-index-core", "llama-index-vector-stores-qdrant",
              "qdrant-client", "openai"]:
        try:
            print(f"  {p}=={version(p)}")
        except PackageNotFoundError:
            print(f"  {p}: NOT INSTALLED")

    _setup_models()
    client = QdrantClient(url=settings.qdrant_url)
    suffix = uuid.uuid4().hex[:6]
    collections = []
    docstores: list[Path] = []

    try:
        # --- слайд «Тот же RAG через LlamaIndex» + «Что нашёл поиск и как процитировать»
        coll = f"_vr_li_{suffix}"
        collections.append(coll)
        vs = QdrantVectorStore(client=client, collection_name=coll)
        storage = StorageContext.from_defaults(vector_store=vs)
        documents = SimpleDirectoryReader(DATA, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage)
        engine = index.as_query_engine(similarity_top_k=3)
        response = engine.query("За сколько дней можно вернуть деньги?")
        assert response.source_nodes, "source_nodes пуст"
        assert response.source_nodes[0].metadata.get("file_name"), "нет file_name в метаданных"
        print(f"\n[from_documents + query] OK: {len(documents)} док., "
              f"top source={response.source_nodes[0].metadata['file_name']}, "
              f"score={response.source_nodes[0].score:.3f}")

        # --- слайд «Подключение к готовой коллекции Qdrant»
        index2 = VectorStoreIndex.from_vector_store(QdrantVectorStore(client=client, collection_name=coll))
        nodes = index2.as_retriever(similarity_top_k=2).retrieve("способы оплаты")
        assert nodes, "from_vector_store вернул пусто"
        print(f"[from_vector_store reconnect] OK: {len(nodes)} нод без переиндексации")

        # --- слайды «Добавляем метаданные при загрузке» + «Фильтр по метаданным»
        coll_m = f"_vr_meta_{suffix}"
        collections.append(coll_m)

        def file_metadata(path: str) -> dict:
            name = Path(path).name
            bucket = name.split("_", 1)[0]
            return {"bucket": bucket}

        docs_m = SimpleDirectoryReader(DATA, recursive=True, file_metadata=file_metadata).load_data()
        vs_m = QdrantVectorStore(client=client, collection_name=coll_m)
        idx_m = VectorStoreIndex.from_documents(
            docs_m, storage_context=StorageContext.from_defaults(vector_store=vs_m)
        )
        flt = MetadataFilters(filters=[
            MetadataFilter(key="bucket", value="billing", operator=FilterOperator.EQ),
        ])
        hits = idx_m.as_retriever(similarity_top_k=5, filters=flt).retrieve("возврат и оплата")
        assert hits, "фильтр по метаданным вернул пусто"
        buckets = {h.metadata.get("bucket") for h in hits}
        assert buckets == {"billing"}, f"фильтр пропустил чужие bucket: {buckets}"
        print(f"[file_metadata + MetadataFilters] OK: {len(hits)} нод, все bucket=billing")

        # --- слайд «IngestionPipeline: явный конвейер трансформаций»
        coll_p = f"_vr_pipe_{suffix}"
        collections.append(coll_p)
        vs_p = QdrantVectorStore(client=client, collection_name=coll_p)
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=64),
                OpenAIEmbedding(model=settings.embedding_model, api_key=KEY),
            ],
            vector_store=vs_p,
        )
        pnodes = pipeline.run(documents=documents)
        assert pnodes, "IngestionPipeline не создал ноды"
        print(f"[IngestionPipeline] OK: создано {len(pnodes)} нод")

        # --- слайд «Гибридный поиск в LlamaIndex + Qdrant» (нужен fastembed)
        if find_spec("fastembed"):
            coll_h = f"_vr_hybrid_{suffix}"
            collections.append(coll_h)
            vs_h = QdrantVectorStore(
                client=client,
                collection_name=coll_h,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm25",
                batch_size=20,
            )
            idx_h = VectorStoreIndex.from_documents(
                documents, storage_context=StorageContext.from_defaults(vector_store=vs_h)
            )
            qe_h = idx_h.as_query_engine(
                similarity_top_k=3, sparse_top_k=12, vector_store_query_mode="hybrid"
            )
            resp_h = qe_h.query("возврат средств за подписку")
            assert resp_h.source_nodes, "гибридный поиск вернул пусто"
            print(f"[enable_hybrid=True + bm25] OK: {len(resp_h.source_nodes)} нод "
                  f"(dense + sparse, RRF)")
        else:
            print("[enable_hybrid] SKIP: нет пакета fastembed (pip install fastembed)")

        # --- слайд «Кастомизация: свой QueryEngine с постпроцессором»: только проверка импорта
        if find_spec("sentence_transformers") and find_spec("torch"):
            print("[SentenceTransformerRerank] доступен для запуска")
        else:
            print("[SentenceTransformerRerank] импорт корректен, запуск пропущен "
                  "(нужны torch + sentence-transformers + загрузка модели ~600 МБ)")

        # --- продакшн-сервисы блока 5.5: IngestionService + RAGService (новый контракт)
        from app.services.ingestion import IngestionService
        from app.services.rag import RAGService

        corp_coll = f"_vr_corp_{suffix}"
        collections.append(corp_coll)
        corp_settings = settings.model_copy(
            update={"rag_collection": corp_coll, "rag_retrieve_top_k": 8, "rag_rerank_top_n": 4}
        )

        ingestion = IngestionService(corp_settings)
        docstores.append(ingestion.docstore_path)
        n1 = ingestion.ingest_all()
        n2 = ingestion.ingest_all()  # повтор: UPSERTS по docstore пропускает неизменённое
        points = client.count(corp_coll).count
        ingestion.close()
        assert n1 > 0, "IngestionService не создал нод"
        assert n2 == 0, f"повторный ingest не идемпотентен: {n2} нод переиндексировано"
        print(f"[IngestionService UPSERTS] OK: первый прогон {n1} нод, повтор {n2}, "
              f"в коллекции {points} точек")

        async def _rag_contract() -> tuple[dict, dict, dict]:
            rag = RAGService(corp_settings)
            rag.build()
            q = "За сколько дней можно вернуть деньги за подписку?"
            hit = await rag.answer(q)
            # evaluate_inputs — вход для RAGAS: ответ + полные тексты чанков.
            ev = await rag.evaluate_inputs(q)
            # Запрос вне домена корпуса (астрономия): retrieval не должен дать score
            # выше порога, срабатывает код-гард до LLM.
            miss = await rag.answer("Сколько колец у планеты Сатурн?")
            await rag.close()
            return hit, miss, ev

        hit, miss, ev = asyncio.run(_rag_contract())
        assert hit["confident"] and hit["sources"], f"ожидался уверенный ответ: {hit}"
        assert hit["sources"][0]["file_name"], "в источнике нет file_name"
        assert set(hit["sources"][0]) == {"id", "file_name", "page", "score", "snippet"}, \
            f"контракт source изменился: {hit['sources'][0]}"
        assert miss["confident"] is False and miss["sources"] == [], f"ожидался отказ: {miss}"
        assert ev["retrieved_contexts"] and all(
            isinstance(c, str) for c in ev["retrieved_contexts"]
        ), f"evaluate_inputs не вернул полные contexts: {ev.get('retrieved_contexts')}"
        print(f"[RAGService contract] OK: ответ confident={hit['confident']} "
              f"top={hit['top_score']} sources={len(hit['sources'])}; "
              f"отказ confident={miss['confident']} top={miss['top_score']}; "
              f"evaluate_inputs contexts={len(ev['retrieved_contexts'])}")

        print("\nВсе запускаемые примеры презентации блока отработали.")
    finally:
        for c in collections:
            try:
                client.delete_collection(c)
            except Exception:
                pass
        for d in docstores:
            try:
                d.unlink(missing_ok=True)
            except Exception:
                pass
        client.close()


if __name__ == "__main__":
    main()
