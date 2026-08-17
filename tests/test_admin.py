"""Тесты admin-API: /stats требует X-Admin-Token, 403 без, 200 с правильным."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
async def admin_client():
    """HTTP-клиент без PG — admin endpoint'ы должны работать (graceful empty)."""
    prev_sf = getattr(app.state, "session_factory", None)
    prev_engine = getattr(app.state, "async_engine", None)
    app.state.session_factory = None
    app.state.async_engine = None
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac
    finally:
        app.state.session_factory = prev_sf
        app.state.async_engine = prev_engine


@pytest.mark.asyncio
async def test_stats_requires_token(admin_client):
    r = await admin_client.get("/chats/admin/stats")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_stats_rejects_wrong_token(admin_client):
    r = await admin_client.get(
        "/chats/admin/stats", headers={"X-Admin-Token": "wrong"}
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_stats_returns_200_with_valid_token(admin_client):
    settings = get_settings()
    token = settings.admin_token.get_secret_value()
    r = await admin_client.get(
        "/chats/admin/stats", headers={"X-Admin-Token": token}
    )
    assert r.status_code == 200
    body = r.json()
    # Без PG возвращает нули, но структура должна быть валидной
    assert body == {
        "total_messages": 0,
        "active_users": 0,
        "feedback_ratio": 0.0,
        "refusal_rate": 0.0,
        "negative_feedback_rate": 0.0,
        "knowledge_gaps": [],
    }


@pytest.mark.asyncio
async def test_export_requires_token(admin_client):
    r = await admin_client.get("/chats/admin/export")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_export_returns_empty_without_pg(admin_client):
    settings = get_settings()
    token = settings.admin_token.get_secret_value()
    r = await admin_client.get(
        "/chats/admin/export", headers={"X-Admin-Token": token}
    )
    assert r.status_code == 200
    assert r.json() == {"items": [], "next_after": None}


@pytest.mark.asyncio
async def test_alerts_list_requires_token(admin_client):
    r = await admin_client.get("/chats/admin/alerts")
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_alerts_empty_without_pg(admin_client):
    settings = get_settings()
    token = settings.admin_token.get_secret_value()
    r = await admin_client.get(
        "/chats/admin/alerts", headers={"X-Admin-Token": token}
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_handoff_requires_token(admin_client):
    r = await admin_client.post(
        "/chats/admin/handoff",
        json={
            "owner_external_id": "tg-1",
            "interface": "telegram",
            "status": "paused_for_human",
        },
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_handoff_with_token_returns_ok(admin_client):
    settings = get_settings()
    token = settings.admin_token.get_secret_value()
    r = await admin_client.post(
        "/chats/admin/handoff",
        headers={"X-Admin-Token": token},
        json={
            "owner_external_id": "tg-1",
            "interface": "telegram",
            "status": "paused_for_human",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # без PG updated=0
    assert body["updated"] == 0


@pytest.mark.asyncio
async def test_broadcast_400_without_targets(admin_client):
    """Без owner_ids И без interface backend должен вернуть 400, а не
    молча отправлять в никуда."""
    settings = get_settings()
    token = settings.admin_token.get_secret_value()
    r = await admin_client.post(
        "/chats/admin/broadcast",
        headers={"X-Admin-Token": token},
        json={"text": "hello"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_broadcast_with_explicit_owner_ids_calls_bot(
    admin_client, monkeypatch
):
    """Если owner_ids явно заданы — backend сразу идёт в bot /notify,
    БД не трогает."""
    seen: dict = {}

    async def fake_broadcast(text, owner_ids, bot_url, internal_token):
        seen["text"] = text
        seen["owner_ids"] = owner_ids
        seen["bot_url"] = bot_url
        return {"sent": len(owner_ids), "failed": 0}

    monkeypatch.setattr(
        "app.admin.routes.do_broadcast", fake_broadcast
    )

    settings = get_settings()
    token = settings.admin_token.get_secret_value()
    r = await admin_client.post(
        "/chats/admin/broadcast",
        headers={"X-Admin-Token": token},
        json={"text": "ping", "owner_ids": [1, 2, 3]},
    )
    assert r.status_code == 200
    assert r.json() == {"sent": 3, "failed": 0}
    assert seen["text"] == "ping"
    assert seen["owner_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_broadcast_with_interface_resolves_owner_ids(
    admin_client, monkeypatch
):
    """interface без owner_ids — backend сам тянет список из БД.
    Без PG repo вернёт [] → должен быть 400."""
    settings = get_settings()
    token = settings.admin_token.get_secret_value()
    r = await admin_client.post(
        "/chats/admin/broadcast",
        headers={"X-Admin-Token": token},
        json={"text": "ping", "interface": "telegram"},
    )
    # Без PG список пустой → 400, а не «успешная» рассылка в никого.
    assert r.status_code == 400
