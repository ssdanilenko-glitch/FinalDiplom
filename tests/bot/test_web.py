"""Тесты /notify-endpoint бота (обратный канал backend → bot)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from bot.web import build_api


@pytest.fixture
def bot():
    b = MagicMock()
    b.send_message = AsyncMock(return_value=None)
    return b


@pytest.fixture
def client(bot):
    api = build_api(bot, internal_token="secret-token-123")
    return TestClient(api)


def test_notify_requires_token(client):
    r = client.post("/notify", json={"chat_id": 1, "text": "test"})
    assert r.status_code == 422  # X-Internal-Token Header отсутствует


def test_notify_rejects_wrong_token(client):
    r = client.post(
        "/notify",
        json={"chat_id": 1, "text": "test"},
        headers={"X-Internal-Token": "wrong"},
    )
    assert r.status_code == 401


def test_notify_success(client, bot):
    r = client.post(
        "/notify",
        json={"chat_id": 12345, "text": "Hello"},
        headers={"X-Internal-Token": "secret-token-123"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    bot.send_message.assert_awaited_once_with(chat_id=12345, text="Hello")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_notify_validates_body(client):
    """Без обязательных полей — 422."""
    r = client.post(
        "/notify",
        json={},
        headers={"X-Internal-Token": "secret-token-123"},
    )
    assert r.status_code == 422
