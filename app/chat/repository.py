"""Контракты репозиториев чата через typing.Protocol.

Реализации не обязаны наследоваться явно: структурная типизация.
"""
from typing import Protocol
from uuid import UUID

from app.chat.domain import Chat, ChatMessage, SystemPrompt


class ChatRepository(Protocol):
    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat: ...

    async def get_chat(self, chat_id: UUID) -> Chat | None: ...

    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> Chat: ...

    async def append_message(
        self, chat_id: UUID, message: ChatMessage
    ) -> ChatMessage: ...

    async def list_messages(
        self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]: ...

    async def soft_delete_messages(self, chat_id: UUID) -> None: ...


class SystemPromptRepository(Protocol):
    async def list_active(self) -> list[SystemPrompt]:
        """Активные кандидаты A/B-сплита (active=TRUE и traffic_pct>0)."""
        ...
