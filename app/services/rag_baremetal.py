"""Тот же RAG, но руками — без LlamaIndex.

Полный путь на чистых openai + qdrant-client: чтение файлов, наивный чанкинг
(один документ = один чанк), эмбеддинги, upsert с плоским payload, поиск и
генерация. Нужен для сравнения с app/services/rag.py (см. docs/rag.md).

Запуск отдельно:
    uv run python -m app.services.rag_baremetal
"""

import logging
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import Settings as AppSettings
from app.core.config import get_settings
from app.services.loader_utils import stable_id

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Отвечай строго по предоставленному контексту. Если ответа в контексте нет — "
    "честно скажи, что не нашёл, и ничего не выдумывай. Отвечай по-русски, коротко."
)


class BareMetalRAG:
    """RAG руками: то же поведение, что у RAGService, но без фреймворка."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._openai = OpenAI(api_key=settings.llm.openai_api_key.get_secret_value())
        qdrant_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )
        self._client = QdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._collection = settings.rag_collection_bare

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._openai.embeddings.create(
            model=self._settings.embedding_model, input=texts
        )
        return [item.embedding for item in resp.data]

    def ensure_indexed(self) -> None:
        """Идемпотентный ingestion: один чанк на файл, плоский payload {text, source}."""
        exists = self._client.collection_exists(self._collection)
        if exists and self._client.count(self._collection).count > 0:
            return
        if not exists:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._settings.embedding_dim, distance=Distance.COSINE
                ),
            )

        files = sorted(Path(self._settings.rag_data_dir).glob("*.md"))
        texts = [f.read_text(encoding="utf-8") for f in files]
        vectors = self._embed(texts)
        points = [
            PointStruct(
                id=stable_id(f.name, 0),
                vector=vec,
                payload={"text": text, "source": f.name},
            )
            for f, text, vec in zip(files, texts, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.info(
            "bare-metal: проиндексировано %d документов в коллекцию %s",
            len(points),
            self._collection,
        )

    def answer(self, question: str) -> dict:
        q_vec = self._embed([question])[0]
        hits = self._client.query_points(
            collection_name=self._collection,
            query=q_vec,
            limit=self._settings.rag_top_k,
            with_payload=True,
        ).points

        context = "\n\n".join(f"[{h.payload['source']}] {h.payload['text']}" for h in hits)
        completion = self._openai.chat.completions.create(
            model=self._settings.rag_llm_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
            ],
        )

        top_score = hits[0].score if hits else 0.0
        answer_text = completion.choices[0].message.content
        if top_score < self._settings.rag_score_threshold:
            answer_text = "В базе знаний нет ответа на этот вопрос."

        return {
            "answer": answer_text,
            "top_score": round(top_score, 3),
            "sources": [
                {
                    "text": h.payload["text"][:300],
                    "source": h.payload["source"],
                    "score": round(h.score, 3),
                }
                for h in hits
            ],
        }

    def close(self) -> None:
        self._client.close()


def _demo() -> None:
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    service = BareMetalRAG(get_settings())
    service.ensure_indexed()
    for question in (
        "За сколько дней можно вернуть деньги за подписку?",
        "Как приготовить плов?",
    ):
        result = service.answer(question)
        print(f"\nВопрос: {question}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    service.close()


if __name__ == "__main__":
    _demo()
