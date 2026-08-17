"""RAG на LlamaIndex: широкий retrieval, опциональный реранкинг, ответ с цитатами.

Онлайн-контур построен по принципу «retrieve → guard → synthesize»:

1. Ретривер достаёт `rag_retrieve_top_k` кандидатов из Qdrant.
2. Опциональный реранкер (config-флаг, тяжёлая зависимость) пересортировывает
   и оставляет `rag_rerank_top_n`; без него — обрезка dense-топа до того же N.
3. Код-гард: если лучших score ниже порога — отдаём честный отказ, не дёргая
   LLM (быстрее и дешевле галлюцинации).
4. Иначе синтез ответа по пронумерованному контексту с цитатами [1], [2].

Запуск отдельно:
    uv run python -m app.services.rag
"""

import logging
import re

from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.core.config import Settings as AppSettings
from app.core.config import get_settings

logger = logging.getLogger(__name__)

REFUSAL_TEXT = "В базе знаний я не нашёл ответа на этот вопрос."

CITATION_QA_PROMPT = PromptTemplate(
    "Ниже — пронумерованные источники из базы знаний.\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Ответь на вопрос, опираясь ТОЛЬКО на источники. Каждый факт сопровождай "
    "номером источника в квадратных скобках, например [1] или [2]. Если ответа "
    "в источниках нет — честно напиши, что не нашёл его в базе знаний, и ничего "
    "не выдумывай. Отвечай по-русски, коротко и по делу.\n"
    "Вопрос: {query_str}\n"
    "Ответ: "
)


def build_sources(source_nodes: list[NodeWithScore]) -> list[dict]:
    """Нумерованные цитаты [1..N]: id, file_name, page, score, snippet."""
    sources = []
    for i, sn in enumerate(source_nodes, start=1):
        meta = sn.metadata or {}
        sources.append(
            {
                "id": i,
                "file_name": meta.get("file_name") or meta.get("source") or "unknown",
                "page": meta.get("page"),
                "score": round(sn.score or 0.0, 3),
                "snippet": sn.get_content()[:200].strip(),
            }
        )
    return sources


def parse_citations(text: str, sources: list[dict]) -> str:
    """Разворачивает [1] в [1 — file.pdf], чтобы источник был виден в тексте."""
    by_id = {s["id"]: s for s in sources}

    def replace(match: re.Match) -> str:
        source = by_id.get(int(match.group(1)))
        return f"[{match.group(1)} — {source['file_name']}]" if source else match.group(0)

    return re.sub(r"\[(\d+)\]", replace, text)


def build_filters(
    *, visibility: str | None = "internal", departments: list[str] | None = None
) -> MetadataFilters | None:
    """Фильтр доступа до поиска: документы вне видимости даже не достаются.

    Применяется на уровне векторного хранилища, а не после ретрива — иначе
    кусок недоступного документа может попасть в контекст LLM.
    """
    filters = []
    if visibility:
        filters.append(
            MetadataFilter(key="visibility", value=visibility, operator=FilterOperator.EQ)
        )
    if departments:
        filters.append(
            MetadataFilter(key="department", value=departments, operator=FilterOperator.IN)
        )
    return MetadataFilters(filters=filters) if filters else None


def _numbered_context(nodes: list[NodeWithScore]) -> str:
    return "\n\n".join(f"[{i}] {sn.get_content()}" for i, sn in enumerate(nodes, start=1))


def _contexts_from_nodes(nodes: list[NodeWithScore]) -> list[str]:
    """Полные тексты найденных чанков — вход retrieved_contexts для RAGAS."""
    return [sn.get_content() for sn in nodes]


class RAGService:
    """Один экземпляр на процесс: ретривер и движок собираются один раз на старте."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        api_key = settings.llm.openai_api_key.get_secret_value()
        qdrant_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )

        Settings.llm = OpenAI(model=settings.rag_llm_model, temperature=0.0, api_key=api_key)
        Settings.embed_model = OpenAIEmbedding(model=settings.embedding_model, api_key=api_key)
        Settings.node_parser = SentenceSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

        self._client = QdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._aclient = AsyncQdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._index: VectorStoreIndex | None = None
        self._retriever = None
        self._postprocessors: list = []

    def _vector_store(self) -> QdrantVectorStore:
        kwargs: dict = {
            "collection_name": self._settings.rag_collection,
            "client": self._client,
            "aclient": self._aclient,
        }
        if self._settings.rag_use_hybrid:
            kwargs["enable_hybrid"] = True
            kwargs["fastembed_sparse_model"] = self._settings.rag_sparse_model
        return QdrantVectorStore(**kwargs)

    def _collection_ready(self) -> bool:
        """Коллекция существует и непуста — индексировать заново не нужно."""
        if not self._client.collection_exists(self._settings.rag_collection):
            return False
        return self._client.count(self._settings.rag_collection).count > 0

    def _build_reranker(self):
        """Реранкер — опциональная тяжёлая зависимость (sentence-transformers + torch).
        Импорт и загрузка модели только когда rag_use_reranker=True."""
        from llama_index.core.postprocessor import SentenceTransformerRerank

        return SentenceTransformerRerank(
            model=self._settings.rag_reranker_model,
            top_n=self._settings.rag_rerank_top_n,
        )

    def build(self) -> None:
        """Подключается к готовой коллекции либо индексирует корпус из файлов."""
        vector_store = self._vector_store()
        if self._collection_ready():
            self._index = VectorStoreIndex.from_vector_store(vector_store)
            logger.info(
                "RAG: подключён к коллекции %s (%d точек)",
                self._settings.rag_collection,
                self._client.count(self._settings.rag_collection).count,
            )
        else:
            documents = SimpleDirectoryReader(
                input_dir=str(self._settings.rag_data_dir),
                recursive=True,
            ).load_data()
            storage = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(documents, storage_context=storage)
            logger.info(
                "RAG: проиндексировано %d документов в коллекцию %s",
                len(documents),
                self._settings.rag_collection,
            )

        filters = (
            build_filters(visibility="internal")
            if self._settings.rag_restrict_to_internal
            else None
        )
        retriever_kwargs: dict = {"similarity_top_k": self._settings.rag_retrieve_top_k}
        if filters is not None:
            retriever_kwargs["filters"] = filters
        if self._settings.rag_use_hybrid:
            retriever_kwargs["vector_store_query_mode"] = "hybrid"
            retriever_kwargs["sparse_top_k"] = self._settings.rag_retrieve_top_k
        self._retriever = self._index.as_retriever(**retriever_kwargs)

        if self._settings.rag_use_reranker:
            self._postprocessors = [self._build_reranker()]

    async def _retrieve(self, question: str) -> list[NodeWithScore]:
        nodes = await self._retriever.aretrieve(question)
        for postprocessor in self._postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_str=question)
        return nodes[: self._settings.rag_rerank_top_n]

    async def answer(self, question: str) -> dict:
        """Контракт: {answer, top_score, sources[id,file_name,page,score,snippet], confident}."""
        nodes = await self._retrieve_checked(question)
        return await self._synthesize(question, nodes)

    async def evaluate_inputs(self, question: str) -> dict:
        """answer() + полные retrieved_contexts (тексты чанков) — вход для RAGAS.

        Роут /rag/query отдаёт усечённые snippet'ы, а метрикам нужен полный текст
        найденных чанков; этот метод собирает оба представления за один ретрив.
        """
        nodes = await self._retrieve_checked(question)
        result = await self._synthesize(question, nodes)
        result["retrieved_contexts"] = _contexts_from_nodes(nodes)
        return result

    async def _retrieve_checked(self, question: str) -> list[NodeWithScore]:
        if self._retriever is None:
            raise RuntimeError("RAG-индекс не инициализирован: сначала вызвать build().")
        return await self._retrieve(question)

    async def _synthesize(self, question: str, nodes: list[NodeWithScore]) -> dict:
        top_score = max((sn.score or 0.0 for sn in nodes), default=0.0)
        if not nodes or top_score < self._settings.rag_score_threshold:
            return {
                "answer": REFUSAL_TEXT,
                "top_score": round(top_score, 3),
                "sources": [],
                "confident": False,
            }

        response = await Settings.llm.acomplete(
            CITATION_QA_PROMPT.format(context_str=_numbered_context(nodes), query_str=question)
        )
        sources = build_sources(nodes)
        return {
            "answer": parse_citations(str(response), sources),
            "top_score": round(top_score, 3),
            "sources": sources,
            "confident": True,
        }

    async def close(self) -> None:
        try:
            await self._aclient.close()
        except Exception:
            logger.debug("ошибка при закрытии async Qdrant-клиента", exc_info=True)
        try:
            self._client.close()
        except Exception:
            logger.debug("ошибка при закрытии Qdrant-клиента", exc_info=True)


def _demo() -> None:
    import asyncio
    import json

    async def run() -> None:
        service = RAGService(get_settings())
        service.build()
        for question in (
            "За сколько дней можно вернуть деньги за подписку?",
            "Как приготовить плов?",
        ):
            result = await service.answer(question)
            print(f"\nВопрос: {question}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        await service.close()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    _demo()
