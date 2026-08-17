"""Smoke-тесты для VectorStore.

По умолчанию пропускаются, если на хосте не поднят Qdrant (нет переменной
QDRANT_TEST_URL). На локальной разработке достаточно:

    docker compose up -d qdrant
    QDRANT_TEST_URL=http://localhost:6333 uv run pytest tests/test_vector_store.py -v

В CI без Qdrant — фикстура пропустит весь файл.
"""


import os
import uuid

import pytest
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
)

from app.services.vector_store import VectorStore, VectorStoreDimensionMismatch


pytestmark = pytest.mark.skipif(
    not os.getenv("QDRANT_TEST_URL"),
    reason="Нужен живой Qdrant: задайте QDRANT_TEST_URL",
)


@pytest.fixture
async def store() -> VectorStore:
    """Чистая коллекция с уникальным именем — изолирует тест от соседей."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    s = VectorStore(
        url=os.environ["QDRANT_TEST_URL"],
        api_key=os.getenv("QDRANT_TEST_API_KEY") or None,
        collection=name,
        dim=4,
    )
    await s.ensure_collection()
    try:
        yield s
    finally:
        await s.client.delete_collection(name)
        await s.close()


def _point(id_: str, vec: list[float], **payload) -> PointStruct:
    return PointStruct(id=id_, vector=vec, payload=payload)


async def test_ensure_collection_idempotent(store: VectorStore) -> None:
    """Повторный вызов ensure_collection не должен падать."""
    await store.ensure_collection()
    await store.ensure_collection()
    assert await store.count() == 0


async def test_ensure_collection_rejects_dim_mismatch(store: VectorStore) -> None:
    """При другой размерности EMBEDDING_DIM — осмысленная ошибка."""
    bad = VectorStore(
        url=os.environ["QDRANT_TEST_URL"],
        api_key=os.getenv("QDRANT_TEST_API_KEY") or None,
        collection=store.collection,
        dim=128,
    )
    try:
        with pytest.raises(VectorStoreDimensionMismatch):
            await bad.ensure_collection()
    finally:
        await bad.close()


async def test_upsert_and_search(store: VectorStore) -> None:
    """upsert + search возвращает ближайший вектор."""
    a = _point(str(uuid.uuid4()), [1.0, 0.0, 0.0, 0.0], source="a.md")
    b = _point(str(uuid.uuid4()), [0.0, 1.0, 0.0, 0.0], source="b.md")
    await store.upsert([a, b])

    hits = await store.search(query_vector=[0.9, 0.1, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    assert str(hits[0].id) == str(a.id)
    assert hits[0].payload["source"] == "a.md"


async def test_search_with_match_filter(store: VectorStore) -> None:
    """Filter по строковому полю — должен пропустить только нужные точки."""
    p1 = _point(str(uuid.uuid4()), [1.0, 0.0, 0.0, 0.0], source="a.md", category="billing")
    p2 = _point(str(uuid.uuid4()), [1.0, 0.0, 0.0, 0.0], source="b.md", category="support")
    await store.upsert([p1, p2])

    flt = Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="billing"))]
    )
    hits = await store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10, query_filter=flt)
    assert len(hits) == 1
    assert hits[0].payload["category"] == "billing"


async def test_search_rejects_wrong_dim(store: VectorStore) -> None:
    """Подача вектора чужой размерности — ValueError на клиенте, без удара по серверу."""
    with pytest.raises(ValueError, match="dim="):
        await store.search(query_vector=[1.0, 2.0], top_k=5)


async def test_upsert_is_idempotent(store: VectorStore) -> None:
    """Повторный upsert с теми же id не дублирует точки."""
    pid = str(uuid.uuid4())
    p = _point(pid, [1.0, 0.0, 0.0, 0.0], source="x.md")
    await store.upsert([p])
    await store.upsert([p])
    assert await store.count() == 1


async def test_datetime_range_filter(store: VectorStore) -> None:
    """DatetimeRange — фильтрация по полю created_at в payload."""
    old = _point(str(uuid.uuid4()), [1.0, 0.0, 0.0, 0.0], source="old.md", created_at="2024-01-01T00:00:00Z")
    new = _point(str(uuid.uuid4()), [1.0, 0.0, 0.0, 0.0], source="new.md", created_at="2026-05-01T00:00:00Z")
    await store.upsert([old, new])

    flt = Filter(
        must=[
            FieldCondition(
                key="created_at",
                range=DatetimeRange(gte="2026-01-01T00:00:00Z"),
            )
        ]
    )
    hits = await store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10, query_filter=flt)
    assert len(hits) == 1
    assert hits[0].payload["source"] == "new.md"


async def test_range_filter_numeric(store: VectorStore) -> None:
    """Числовой Range на скалярном поле."""
    a = _point(str(uuid.uuid4()), [1.0, 0.0, 0.0, 0.0], source="a", chunk_index=0)
    b = _point(str(uuid.uuid4()), [1.0, 0.0, 0.0, 0.0], source="b", chunk_index=5)
    await store.upsert([a, b])

    flt = Filter(must=[FieldCondition(key="chunk_index", range=Range(gte=3))])
    hits = await store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10, query_filter=flt)
    assert len(hits) == 1
    assert hits[0].payload["chunk_index"] == 5
