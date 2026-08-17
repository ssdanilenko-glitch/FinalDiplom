"""Идемпотентная загрузка JSONL-корпуса в Qdrant.

Запуск:
    python scripts/load_to_qdrant.py
    python scripts/load_to_qdrant.py --data data/sample_kb.jsonl --batch 256
"""


import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI  # noqa: E402
from qdrant_client.models import PointStruct  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import EmbeddingsClient  # noqa: E402
from app.services.loader_utils import read_jsonl, stable_id  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

logger = logging.getLogger("loader")
logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sample_kb.jsonl"),
        help="Путь к JSONL с документами",
    )
    parser.add_argument("--batch", type=int, default=256, help="Размер upsert-батча")
    args = parser.parse_args()

    settings = get_settings()

    if not args.data.exists():
        raise SystemExit(
            f"Файл {args.data} не найден. Запустите `python data/generate_sample.py` "
            f"для генерации учебного корпуса."
        )

    docs = read_jsonl(args.data)
    logger.info("Загружаю %d документов из %s", len(docs), args.data)

    openai_client = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    embeddings = EmbeddingsClient(openai_client, model=settings.embedding_model)

    store = VectorStore(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )

    try:
        await store.ensure_collection()
        logger.info(
            "Коллекция %s готова (dim=%d, distance=COSINE)",
            settings.qdrant_collection,
            settings.embedding_dim,
        )

        texts = [d["text"] for d in docs]
        vectors: list[list[float]] = []
        batch_for_embed = 64
        for i in tqdm(range(0, len(texts), batch_for_embed), desc="embeddings"):
            chunk = texts[i : i + batch_for_embed]
            vectors.extend(await embeddings.embed(chunk))

        if len(vectors) != len(docs):
            raise RuntimeError(
                f"Получено {len(vectors)} embeddings на {len(docs)} документов"
            )

        if vectors and len(vectors[0]) != settings.embedding_dim:
            raise RuntimeError(
                f"Embedding dim={len(vectors[0])} != EMBEDDING_DIM={settings.embedding_dim}. "
                f"Сверьте имя модели и значение EMBEDDING_DIM в .env."
            )

        points = [
            PointStruct(
                id=stable_id(doc["source"], doc["chunk_index"]),
                vector=vec,
                payload={
                    "source": doc["source"],
                    "chunk_index": doc["chunk_index"],
                    "text": doc["text"],
                    "category": doc["category"],
                    "created_at": doc["created_at"],
                },
            )
            for doc, vec in zip(docs, vectors, strict=True)
        ]

        logger.info("Заливаю %d точек батчами по %d", len(points), args.batch)
        await store.upsert(points, batch_size=args.batch)

        total = await store.count()
        logger.info("Готово. В коллекции %d точек", total)
    finally:
        await store.close()
        await openai_client.close()


if __name__ == "__main__":
    asyncio.run(main())
