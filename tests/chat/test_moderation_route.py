"""Тест: модерация блокирует POST /chats/{id}/messages с 403 ДО старта SSE.

Критично: моя интеграция ставит check_input ПЕРЕД StreamingResponse — если
случайно вернуть проверку в стрим-генератор, status уже будет 200 и клиент
получит truncated SSE вместо 403. Этот тест ловит регрессию.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.chat.deps import get_chat_service, get_repository
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.service import ChatService
from app.deps.providers import get_cache, get_llm
from app.main import app
from app.moderation.domain import ModerationResult
from app.moderation.service import ModerationService


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def blocking_client(tmp_path):
    repo = JsonChatRepository(tmp_path / "chats")
    llm = AsyncMock()

    moderation = MagicMock(spec=ModerationService)
    moderation.check_input = AsyncMock(
        return_value=ModerationResult(
            allowed=False, categories=["custom_blocklist"], layer="regex",
        )
    )

    service = ChatService(
        repository=repo,
        llm_client=llm,
        context_window=10,
        default_model="gpt-5.4-mini",
        moderation=moderation,
        prompt_repo=None,
    )

    app.state.llm = llm
    app.state.redis = None
    app.state.session_factory = None
    app.state.async_engine = None

    async def _override_repo():
        yield repo

    app.dependency_overrides[get_repository] = _override_repo
    app.dependency_overrides[get_chat_service] = lambda: service
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_cache] = lambda: None

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_moderation_blocked_returns_403(blocking_client):
    # Создать чат
    create = await blocking_client.post(
        "/chats", json={"owner_external_id": "u-mod", "interface": "cli"}
    )
    chat_id = create.json()["chat_id"]

    # Послать сообщение — должно вернуть 403 (не 200 с truncated stream)
    r = await blocking_client.post(
        f"/chats/{chat_id}/messages",
        data={"content": "что-нибудь-плохое"},
    )
    assert r.status_code == 403
    body = r.json()
    detail = body["detail"]
    assert detail["code"] == "moderation_blocked"
    assert detail["layer"] == "regex"
    assert "custom_blocklist" in detail["categories"]


async def test_moderation_passes_through_normal_text(tmp_path):
    """Контроль: при allowed=True всё проходит как обычно (200 SSE)."""
    repo = JsonChatRepository(tmp_path / "chats")

    llm = AsyncMock()

    async def fake_create(**kwargs):
        async def gen():
            yield MagicMock(
                choices=[MagicMock(delta=MagicMock(content="ok"))], usage=None
            )

        return gen()

    llm.chat.completions.create = AsyncMock(side_effect=fake_create)

    moderation = MagicMock(spec=ModerationService)
    moderation.check_input = AsyncMock(
        return_value=ModerationResult(allowed=True, layer="passed")
    )

    service = ChatService(
        repository=repo,
        llm_client=llm,
        context_window=10,
        default_model="gpt-5.4-mini",
        moderation=moderation,
        prompt_repo=None,
    )

    app.state.llm = llm
    app.state.redis = None
    app.state.session_factory = None
    app.state.async_engine = None

    async def _override_repo():
        yield repo

    app.dependency_overrides[get_repository] = _override_repo
    app.dependency_overrides[get_chat_service] = lambda: service
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_cache] = lambda: None

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            create = await ac.post(
                "/chats",
                json={"owner_external_id": "u-ok", "interface": "cli"},
            )
            chat_id = create.json()["chat_id"]
            r = await ac.post(
                f"/chats/{chat_id}/messages", data={"content": "привет"}
            )
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
    finally:
        app.dependency_overrides.clear()
