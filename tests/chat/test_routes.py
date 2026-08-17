"""Тесты HTTP-роутов /chats."""

import json
from uuid import UUID

import pytest


pytestmark = pytest.mark.asyncio


async def test_create_chat_returns_uuid(chat_client):
    client, _ = chat_client
    resp = await client.post(
        "/chats",
        json={"owner_external_id": "u-1", "interface": "cli"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "chat_id" in body
    UUID(body["chat_id"])  # валидный UUID


async def test_create_chat_is_idempotent_per_owner_interface(chat_client):
    """Повторный POST /chats с тем же (owner, interface) возвращает тот же id.

    Иначе бот при каждом сообщении создаёт нового пользователя и история не
    подтягивается.
    """
    client, _ = chat_client
    body = {"owner_external_id": "u-idem", "interface": "telegram"}
    a = await client.post("/chats", json=body)
    b = await client.post("/chats", json=body)
    assert a.status_code == 200
    assert b.status_code == 200
    assert a.json()["chat_id"] == b.json()["chat_id"]

    # Другой interface — другой чат
    c = await client.post(
        "/chats",
        json={"owner_external_id": "u-idem", "interface": "cli"},
    )
    assert c.json()["chat_id"] != a.json()["chat_id"]


async def test_get_chat_404_for_unknown(chat_client):
    client, _ = chat_client
    resp = await client.get(f"/chats/{UUID(int=0)}")
    assert resp.status_code == 404


async def test_post_message_streams_sse(chat_client):
    client, _ = chat_client
    create = await client.post(
        "/chats",
        json={"owner_external_id": "u-2", "interface": "cli"},
    )
    chat_id = create.json()["chat_id"]

    async with client.stream(
        "POST",
        f"/chats/{chat_id}/messages",
        data={"content": "hi"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        data_lines = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data_lines.append(line[len("data: ") :])

    # Среди data-строк должны быть token-кадры, message_saved и финальный done.
    parsed = [json.loads(d) for d in data_lines]
    assert any(p.get("type") == "token" for p in parsed)
    assert any("delta" in p for p in parsed if p.get("type") == "token")
    saved = [p for p in parsed if p.get("type") == "message_saved"]
    assert len(saved) == 1
    UUID(saved[0]["message_id"])  # message_id — валидный UUID
    assert parsed[-1] == {"type": "done"}


async def test_list_messages_returns_history(chat_client):
    client, service = chat_client
    create = await client.post(
        "/chats",
        json={"owner_external_id": "u-3", "interface": "cli"},
    )
    chat_id = create.json()["chat_id"]

    # Прогоним сообщение, чтобы появились user+assistant
    async with client.stream(
        "POST",
        f"/chats/{chat_id}/messages",
        data={"content": "hello"},
    ) as resp:
        async for _ in resp.aiter_lines():
            pass

    resp = await client.get(f"/chats/{chat_id}/messages")
    assert resp.status_code == 200
    items = resp.json()
    roles = [m["role"] for m in items]
    assert roles == ["user", "assistant"]


async def test_post_message_accepts_media_multipart(chat_client):
    """Multipart с media-файлом — backend должен принять и стримить ответ."""
    client, _ = chat_client
    create = await client.post(
        "/chats",
        json={"owner_external_id": "u-media", "interface": "cli"},
    )
    chat_id = create.json()["chat_id"]

    files = {"media": ("photo.jpg", b"\xff\xd8\xff\xe0fake-jpeg", "image/jpeg")}
    async with client.stream(
        "POST",
        f"/chats/{chat_id}/messages",
        data={"content": "describe it"},
        files=files,
    ) as resp:
        assert resp.status_code == 200
        data_lines = []
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                data_lines.append(line[len("data: ") :])

    parsed = [json.loads(d) for d in data_lines]
    assert any(p.get("type") == "token" for p in parsed)
    assert parsed[-1] == {"type": "done"}

    # Сообщение пользователя должно сохраниться с media_refs (включая part)
    resp = await client.get(f"/chats/{chat_id}/messages")
    items = resp.json()
    user_msg = next(m for m in items if m["role"] == "user")
    assert user_msg["media_refs"] is not None
    assert user_msg["media_refs"]["mime"] == "image/jpeg"
    assert user_msg["media_refs"]["part"]["type"] == "image_url"


async def test_delete_messages_soft_clears(chat_client):
    client, _ = chat_client
    create = await client.post(
        "/chats",
        json={"owner_external_id": "u-4", "interface": "cli"},
    )
    chat_id = create.json()["chat_id"]

    # Записать пару сообщений
    async with client.stream(
        "POST",
        f"/chats/{chat_id}/messages",
        data={"content": "x"},
    ) as resp:
        async for _ in resp.aiter_lines():
            pass

    # Очистка
    resp = await client.delete(f"/chats/{chat_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    # После очистки — пусто
    resp2 = await client.get(f"/chats/{chat_id}/messages")
    assert resp2.json() == []
