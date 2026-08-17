"""Async-обёртка над Qdrant."""


import logging
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

logger = logging.getLogger(__name__)


class VectorStoreDimensionMismatch(RuntimeError):
    """Размерность существующей коллекции не совпадает с EMBEDDING_DIM."""


class VectorStore:
    _DEFAULT_PAYLOAD_INDEXES: tuple[tuple[str, PayloadSchemaType], ...] = (
        ("source", PayloadSchemaType.KEYWORD),
        ("created_at", PayloadSchemaType.DATETIME),
        ("category", PayloadSchemaType.KEYWORD),
    )

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection: str,
        dim: int,
        *,
        distance: Distance = Distance.COSINE,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        timeout: float = 10.0,
        payload_indexes: Sequence[tuple[str, PayloadSchemaType]] | None = None,
    ) -> None:
        self.client = AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout)
        self.collection = collection
        self.dim = dim
        self.distance = distance
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.payload_indexes: tuple[tuple[str, PayloadSchemaType], ...] = tuple(
            payload_indexes if payload_indexes is not None else self._DEFAULT_PAYLOAD_INDEXES
        )

    async def ensure_collection(self) -> None:
        """Создаёт коллекцию при отсутствии, проверяет размерность при наличии."""
        existing = await self._existing_collections()
        if self.collection not in existing:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=self.distance),
                hnsw_config=HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
            )
            logger.info(
                "Создана коллекция %s (dim=%d, distance=%s)",
                self.collection,
                self.dim,
                self.distance.name,
            )
        else:
            info = await self.client.get_collection(self.collection)
            actual_dim = info.config.params.vectors.size  # type: ignore[union-attr]
            if actual_dim != self.dim:
                raise VectorStoreDimensionMismatch(
                    f"Коллекция {self.collection!r} имеет dim={actual_dim}, "
                    f"настройки требуют dim={self.dim}. Либо обновите EMBEDDING_DIM, "
                    f"либо пересоздайте коллекцию (rm volume `qdrant_storage`)."
                )

        for field, schema in self.payload_indexes:
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except UnexpectedResponse as e:
                if 400 <= e.status_code < 500:
                    continue
                raise

    async def upsert(
        self,
        points: list[PointStruct],
        batch_size: int = 256,
    ) -> None:
        """Заливает точки батчами, ждёт подтверждения только на последнем батче."""
        if not points:
            return
        total = len(points)
        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            is_last = (i + batch_size) >= total
            await self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=is_last,
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Возвращает top-K точек, отсортированных по похожести."""
        if len(query_vector) != self.dim:
            raise ValueError(
                f"query_vector dim={len(query_vector)} != collection dim={self.dim}"
            )
        result = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return result.points

    async def count(self) -> int:
        """Число точек в коллекции."""
        info = await self.client.get_collection(self.collection)
        return info.points_count or 0

    async def close(self) -> None:
        """Закрывает HTTP/gRPC-соединение."""
        await self.client.close()

    async def _existing_collections(self) -> set[str]:
        resp = await self.client.get_collections()
        return {c.name for c in resp.collections}
