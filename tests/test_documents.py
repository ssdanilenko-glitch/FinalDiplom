"""Тесты ручек /documents — без Qdrant/OpenAI (ASGITransport + фейковый индексатор)."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.deps.providers import get_ingestion_service
from app.main import app


class _FakeIngestion:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def run_for_file(self, path: Path) -> int:
        self.calls.append(("run_for_file", str(path)))
        return 1

    def ingest_all(self) -> int:
        self.calls.append(("ingest_all",))
        return 1

    def ingest_files(self, files: list[str]) -> int:
        self.calls.append(("ingest_files", tuple(files)))
        return 1

    def reindex_all(self) -> int:
        self.calls.append(("reindex_all",))
        return 1


def _settings_with_dir(tmp: Path):
    return get_settings().model_copy(update={"rag_data_dir": tmp})


@pytest.fixture
def fake_ingestion():
    fake = _FakeIngestion()
    app.dependency_overrides[get_ingestion_service] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_upload_saves_file_and_queues_indexing(fake_ingestion, tmp_path) -> None:
    app.dependency_overrides[get_settings] = lambda: _settings_with_dir(tmp_path)
    async with await _client() as ac:
        resp = await ac.post(
            "/documents/upload",
            files={"file": ("policy.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"
    assert (tmp_path / "policy.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert fake_ingestion.calls == [("run_for_file", str(tmp_path / "policy.pdf"))]


async def test_upload_strips_path_traversal(fake_ingestion, tmp_path) -> None:
    app.dependency_overrides[get_settings] = lambda: _settings_with_dir(tmp_path)
    async with await _client() as ac:
        resp = await ac.post(
            "/documents/upload",
            files={"file": ("../../evil.pdf", b"x", "application/pdf")},
        )
    assert resp.status_code == 202
    assert (tmp_path / "evil.pdf").exists()
    assert not (tmp_path.parent / "evil.pdf").exists()


async def test_upload_503_without_ingestion() -> None:
    app.dependency_overrides[get_ingestion_service] = lambda: None
    async with await _client() as ac:
        resp = await ac.post(
            "/documents/upload",
            files={"file": ("a.pdf", b"x", "application/pdf")},
        )
    assert resp.status_code == 503
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "body,expected_call",
    [
        ({"mode": "incremental"}, ("ingest_all",)),
        ({"mode": "full"}, ("reindex_all",)),
        ({"mode": "files", "files": ["a.pdf", "b.pdf"]}, ("ingest_files", ("a.pdf", "b.pdf"))),
    ],
)
async def test_reindex_dispatches_by_mode(fake_ingestion, body, expected_call) -> None:
    async with await _client() as ac:
        resp = await ac.post("/documents/reindex", json=body)
    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"
    assert fake_ingestion.calls == [expected_call]


async def test_reindex_files_requires_nonempty_list(fake_ingestion) -> None:
    async with await _client() as ac:
        resp = await ac.post("/documents/reindex", json={"mode": "files", "files": []})
    assert resp.status_code == 422
    assert fake_ingestion.calls == []


async def test_reindex_rejects_unknown_mode(fake_ingestion) -> None:
    async with await _client() as ac:
        resp = await ac.post("/documents/reindex", json={"mode": "bogus"})
    assert resp.status_code == 422
