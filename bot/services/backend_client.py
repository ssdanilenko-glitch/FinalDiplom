"""Тонкий async-клиент к chat-сервису.

Бот не хранит истории/контекста — всё это есть на стороне backend.
Здесь только операции: получить chat_id, отправить сообщение (SSE с
опциональным media через multipart/form-data), очистить историю,
оставить feedback, admin-команды (stats/handoff/alerts).

Заголовок `X-Owner-External-Id` передаётся в каждом POST/DELETE-вызове,
где есть владелец, — backend использует его для rate-limit.
"""

import json
from collections.abc import AsyncIterator
from uuid import UUID

import httpx


class BackendClient:
    def __init__(
        self, http: httpx.AsyncClient, admin_token: str = ""
    ) -> None:
        self.http = http
        self._admin_token = admin_token

    # --- chat operations -------------------------------------------------
    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> UUID:
        """POST /chats; идемпотентно по (owner, interface)."""
        r = await self.http.post(
            "/chats",
            json={
                "owner_external_id": owner_external_id,
                "interface": interface,
            },
            headers={"X-Owner-External-Id": owner_external_id},
        )
        r.raise_for_status()
        return UUID(r.json()["chat_id"])

    async def send_message(
        self,
        chat_id: UUID,
        content: str,
        media: bytes | None = None,
        mime: str | None = None,
        filename: str = "file.bin",
        owner_external_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """POST /chats/{id}/messages multipart; парсит SSE, yields dict-события.

        Backend отдаёт поток вида:
            data: {"type":"token","delta":"..."}\\n\\n
            data: {"type":"message_saved","message_id":"..."}\\n\\n
            data: {"type":"done"}\\n\\n

        Этот метод yield-ит ровно то, что пришло (без преобразования), пока
        не встретит `done` — он завершает итерацию (не yield-ится).
        Неизвестные `type` молча игнорируются — для forward compat.
        """
        data = {"content": content}
        files = {"media": (filename, media, mime)} if media is not None else None
        headers = (
            {"X-Owner-External-Id": owner_external_id}
            if owner_external_id
            else {}
        )
        async with self.http.stream(
            "POST",
            f"/chats/{chat_id}/messages",
            data=data,
            files=files,
            headers=headers,
            timeout=120.0,
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                ptype = payload.get("type")
                if ptype == "done":
                    return
                if ptype in ("token", "message_saved"):
                    yield payload

    async def clear_messages(
        self,
        chat_id: UUID,
        owner_external_id: str | None = None,
    ) -> None:
        headers = (
            {"X-Owner-External-Id": owner_external_id}
            if owner_external_id
            else {}
        )
        r = await self.http.delete(
            f"/chats/{chat_id}/messages", headers=headers
        )
        r.raise_for_status()

    # --- feedback --------------------------------------------------------
    async def post_feedback(
        self,
        chat_id: UUID,
        message_id: str,
        owner_external_id: str,
        value: str,
    ) -> None:
        r = await self.http.post(
            f"/chats/{chat_id}/messages/{message_id}/feedback",
            json={"owner_external_id": owner_external_id, "value": value},
            headers={"X-Owner-External-Id": owner_external_id},
        )
        r.raise_for_status()

    # --- admin -----------------------------------------------------------
    def _admin_headers(self) -> dict[str, str]:
        return {"X-Admin-Token": self._admin_token}

    async def get_admin_stats(self, window_hours: int = 24) -> dict:
        r = await self.http.get(
            "/chats/admin/stats",
            params={"window_hours": window_hours},
            headers=self._admin_headers(),
        )
        r.raise_for_status()
        return r.json()

    async def broadcast(
        self, text: str, interface: str = "telegram"
    ) -> dict:
        """POST /chats/admin/broadcast. Backend сам подтянет owner_ids по
        interface и серийно отправит каждому через bot:9000/notify.
        """
        r = await self.http.post(
            "/chats/admin/broadcast",
            json={"text": text, "interface": interface},
            headers=self._admin_headers(),
        )
        r.raise_for_status()
        return r.json()

    async def set_handoff_status(
        self,
        owner_external_id: str,
        status: str,
        interface: str = "telegram",
    ) -> dict:
        r = await self.http.post(
            "/chats/admin/handoff",
            json={
                "owner_external_id": owner_external_id,
                "interface": interface,
                "status": status,
            },
            headers=self._admin_headers(),
        )
        r.raise_for_status()
        return r.json()

    async def fetch_pending_alerts(self) -> list[dict]:
        r = await self.http.get(
            "/chats/admin/alerts", headers=self._admin_headers()
        )
        r.raise_for_status()
        return r.json()

    async def ack_alert(self, alert_id: int) -> None:
        r = await self.http.post(
            f"/chats/admin/alerts/{alert_id}/ack",
            headers=self._admin_headers(),
        )
        r.raise_for_status()

    # --- lifecycle -------------------------------------------------------
    async def aclose(self) -> None:
        await self.http.aclose()
