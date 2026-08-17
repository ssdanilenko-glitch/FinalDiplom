"""JsonChatRepository — JSONL append-only хранилище.

Структура на диске:
    <base_dir>/<chat_id>/chat.json        — метаданные Chat
    <base_dir>/<chat_id>/messages.jsonl   — одна ChatMessage на строку

Soft delete — маркерная запись `{"type": "soft_delete", "at": "..."}` в jsonl.
list_messages пропускает всё, что было ДО последнего такого маркера.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import aiofiles

from app.chat.domain import Chat, ChatMessage

logger = logging.getLogger("llm-service.chat.json_repo")


class JsonChatRepository:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    # -- внутренние пути -------------------------------------------------
    def _chat_dir(self, chat_id: UUID) -> Path:
        return self.base_dir / str(chat_id)

    def _chat_meta_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _messages_path(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "messages.jsonl"

    # -- API -------------------------------------------------------------
    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )
        path = self._chat_meta_path(chat.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w") as f:
            await f.write(chat.model_dump_json())
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        path = self._chat_meta_path(chat_id)
        if not path.exists():
            return None
        async with aiofiles.open(path) as f:
            raw = await f.read()
        return Chat.model_validate_json(raw)

    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> Chat:
        # O(N) скан — для учебного варианта норм.
        if self.base_dir.exists():
            for chat_dir in self.base_dir.iterdir():
                if not chat_dir.is_dir():
                    continue
                meta = chat_dir / "chat.json"
                if not meta.exists():
                    continue
                try:
                    async with aiofiles.open(meta) as f:
                        raw = await f.read()
                    chat = Chat.model_validate_json(raw)
                except (OSError, ValueError) as exc:
                    logger.warning("skip unreadable chat meta %s: %s", meta, exc)
                    continue
                if (
                    chat.owner_external_id == owner_external_id
                    and chat.interface == interface
                ):
                    return chat
        return await self.create_chat(owner_external_id, interface)

    async def append_message(
        self, chat_id: UUID, message: ChatMessage
    ) -> ChatMessage:
        path = self._messages_path(chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a") as f:
            await f.write(message.model_dump_json() + "\n")
        return message

    async def list_messages(
        self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]:
        path = self._messages_path(chat_id)
        if not path.exists():
            return []
        async with aiofiles.open(path) as f:
            lines = await f.readlines()

        # Находим индекс ПОСЛЕДНЕГО soft_delete-маркера, берём всё ПОСЛЕ него.
        last_marker = -1
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "soft_delete":
                last_marker = i

        effective = lines[last_marker + 1 :] if last_marker >= 0 else lines

        messages: list[ChatMessage] = []
        for line in effective:
            line = line.strip()
            if not line:
                continue
            # Пропускаем маркер-строки (хотя их после среза не должно быть).
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("type") == "soft_delete":
                continue
            try:
                messages.append(ChatMessage.model_validate_json(line))
            except ValueError as exc:
                logger.warning("skip malformed message line in %s: %s", path, exc)
                continue

        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        path = self._messages_path(chat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = {"type": "soft_delete", "at": datetime.now(UTC).isoformat()}
        async with aiofiles.open(path, "a") as f:
            await f.write(json.dumps(marker) + "\n")
