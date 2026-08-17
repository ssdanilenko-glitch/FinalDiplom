"""Сравнение ранжирования cosine vs dot на пяти запросах.

Создаёт временную пару коллекций `{base}_cosine` и `{base}_dot` на одних
и тех же векторах, прогоняет запросы, сохраняет результат в
`docs/metric_comparison.json` и удаляет временные коллекции.

Запуск:
    python scripts/compare_metrics.py
"""


import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI  # noqa: E402
from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import EmbeddingsClient  # noqa: E402
from app.services.loader_utils import read_jsonl, stable_id  # noqa: E402

logger = logging.getLogger("compare")
logging.basicConfig(level=logging.INFO, format="%(message)s")


QUERIES: list[str] = [
    "Как вернуть деньги за подписку",
    "Сбросить пароль если не приходит письмо",
    "Какой rate limit на API для бесплатного плана",
    "Где найти аудит-логи действий пользователей",
    "Как подключить интеграцию со Slack",
]


async def upsert_clone(
    client: AsyncQdrantClient,
    base_name: str,
    suffix: str,
    distance: Distance,
    points: list[PointStruct],
    dim: int,
) -> str:
    """Создаёт коллекцию `{base}_{suffix}` с заданной метрикой и заливает точки."""
    name = f"{base_name}_{suffix}"
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        await client.delete_collection(name)
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=distance),
    )
    await client.upsert(collection_name=name, points=points, wait=True)
    return name


async def top5(
    client: AsyncQdrantClient, collection: str, query_vec: list[float]
) -> list[str]:
    """top-5 id из коллекции для одного вектора."""
    result = await client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=5,
        with_payload=False,
    )
    return [str(p.id) for p in result.points]


async def main() -> None:
    settings = get_settings()
    data_path = Path("data/sample_kb.jsonl")
    if not data_path.exists():
        raise SystemExit("Сначала запустите `python data/generate_sample.py`.")

    docs = read_jsonl(data_path)
    openai_client = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        timeout=settings.llm.request_timeout,
    )
    embeddings = EmbeddingsClient(openai_client, model=settings.embedding_model)

    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )

    try:
        logger.info("Считаю embeddings для %d документов и %d запросов", len(docs), len(QUERIES))
        doc_vectors = await embeddings.embed([d["text"] for d in docs])
        query_vectors = await embeddings.embed(QUERIES)

        points = [
            PointStruct(
                id=stable_id(d["source"], d["chunk_index"]),
                vector=v,
                payload={"source": d["source"], "chunk_index": d["chunk_index"]},
            )
            for d, v in zip(docs, doc_vectors, strict=True)
        ]

        base = settings.qdrant_collection
        cos_name = await upsert_clone(qdrant, base, "cosine", Distance.COSINE, points, settings.embedding_dim)
        dot_name = await upsert_clone(qdrant, base, "dot", Distance.DOT, points, settings.embedding_dim)

        logger.info("\n%-50s | %-30s | %-30s | match", "Запрос", "top-5 cosine", "top-5 dot")
        logger.info("-" * 130)
        rows = []
        for q, qv in zip(QUERIES, query_vectors, strict=True):
            cos_top = await top5(qdrant, cos_name, qv)
            dot_top = await top5(qdrant, dot_name, qv)
            match = "✓" if cos_top == dot_top else "✗"
            rows.append({"query": q, "cosine": cos_top, "dot": dot_top, "match": cos_top == dot_top})
            logger.info(
                "%-50s | %-30s | %-30s | %s",
                q[:48],
                ",".join(c[:6] for c in cos_top),
                ",".join(d[:6] for d in dot_top),
                match,
            )

        out = Path("docs/metric_comparison.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("\nДетали сохранены в %s", out)

        await qdrant.delete_collection(cos_name)
        await qdrant.delete_collection(dot_name)
        logger.info("Временные коллекции %s, %s удалены", cos_name, dot_name)
    finally:
        await qdrant.close()
        await openai_client.close()


if __name__ == "__main__":
    asyncio.run(main())
