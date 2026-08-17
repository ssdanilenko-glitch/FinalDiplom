"""Прогон всех кодовых сниппетов презентации блока на живом Qdrant.

Запуск:
    QDRANT_TEST_URL=http://localhost:6333 python scripts/verify_presentation.py
"""


import asyncio
import os
import sys
import uuid
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    GeoBoundingBox,
    GeoPoint,
    HnswConfigDiff,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    QuantizationSearchParams,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


async def main() -> None:
    url = os.environ.get("QDRANT_TEST_URL", "http://localhost:6333")
    client = AsyncQdrantClient(url=url, timeout=10.0)

    coll = f"_verify_{uuid.uuid4().hex[:6]}"
    hybrid = f"_verify_hybrid_{uuid.uuid4().hex[:6]}"

    try:
        # --- 1) create_collection + HNSW config (из слайдов «HNSW параметры»)
        await client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        print("✓ create_collection с HNSW config")

        # --- 2) payload_index (из слайда «Payload-индекс»)
        await client.create_payload_index(
            collection_name=coll,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=coll,
            field_name="created_at",
            field_schema=PayloadSchemaType.DATETIME,
        )
        print("✓ create_payload_index (KEYWORD, DATETIME)")

        # --- 3) upsert + PointStruct + wait
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[1.0, 0.0, 0.0, 0.0],
                payload={"category": "billing", "created_at": "2026-04-01T00:00:00Z"},
            ),
            PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0, 1.0, 0.0, 0.0],
                payload={"category": "support", "created_at": "2025-01-01T00:00:00Z"},
            ),
        ]
        await client.upsert(collection_name=coll, points=points, wait=True)
        print("✓ upsert(points, wait=True)")

        # --- 4) query_points + Filter(must) + MatchValue (из слайда «Поиск через query_points»)
        flt_match = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="billing"))]
        )
        res = await client.query_points(
            collection_name=coll,
            query=[0.9, 0.1, 0.0, 0.0],
            query_filter=flt_match,
            limit=3,
            with_payload=True,
        )
        assert res.points, "Ожидался хотя бы один хит"
        print("✓ query_points + must + MatchValue")

        # --- 5) DatetimeRange (из слайда «Range, geo, datetime»)
        flt_dt = Filter(
            must=[FieldCondition(key="created_at", range=DatetimeRange(gte="2026-01-01T00:00:00Z"))]
        )
        res = await client.query_points(
            collection_name=coll, query=[1, 0, 0, 0], query_filter=flt_dt, limit=3
        )
        assert len(res.points) == 1, "DatetimeRange должен отфильтровать старый"
        print("✓ DatetimeRange(gte=...)")

        # --- 6) Range числовой
        flt_range = Filter(must=[FieldCondition(key="chunk_index", range=Range(gte=3))])
        await client.query_points(collection_name=coll, query=[1, 0, 0, 0], query_filter=flt_range, limit=3)
        print("✓ Range(gte=...) для числовых полей")

        # --- 7) GeoBoundingBox (тип конструируется, проверять полноценно негде без geo-данных)
        GeoBoundingBox(
            top_left=GeoPoint(lat=60.0, lon=29.0),
            bottom_right=GeoPoint(lat=59.0, lon=31.0),
        )
        print("✓ GeoBoundingBox / GeoPoint импорты")

        # --- 8) update_collection с quantization (из слайда «как включить и обновить»)
        await client.update_collection(
            collection_name=coll,
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        )
        print("✓ update_collection(quantization_config=ScalarQuantization(...))")

        # --- 9) search с QuantizationSearchParams (из слайда «поиск с rescoring»)
        await client.query_points(
            collection_name=coll,
            query=[1, 0, 0, 0],
            limit=3,
            search_params=SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                ),
            ),
        )
        print("✓ SearchParams(quantization=QuantizationSearchParams(...))")

        # --- 10) Hybrid: dense + sparse + Prefetch + FusionQuery (из слайда «Hybrid в Qdrant: код»)
        await client.create_collection(
            collection_name=hybrid,
            vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        await client.upsert(
            collection_name=hybrid,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": [1.0, 0.0, 0.0, 0.0],
                        "sparse": SparseVector(indices=[1, 5], values=[0.7, 0.3]),
                    },
                    payload={"source": "a"},
                )
            ],
            wait=True,
        )
        hits = await client.query_points(
            collection_name=hybrid,
            prefetch=[
                Prefetch(query=[1.0, 0.0, 0.0, 0.0], using="dense", limit=5),
                Prefetch(
                    query=SparseVector(indices=[1, 5], values=[0.7, 0.3]),
                    using="sparse",
                    limit=5,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=3,
        )
        assert hits.points, "Hybrid вернул пусто"
        print("✓ Prefetch + FusionQuery(fusion=Fusion.RRF)")

        # --- 11) get_collection — для health-check (из слайда «мониторинг»)
        info = await client.get_collection(coll)
        assert info.status is not None
        assert info.points_count is not None
        print(f"✓ get_collection → status={info.status}, points={info.points_count}")

        print(f"\nВсе примеры из презентации работают на qdrant-client {version('qdrant-client')}.")
    finally:
        for c in (coll, hybrid):
            try:
                await client.delete_collection(c)
            except Exception:
                pass
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
