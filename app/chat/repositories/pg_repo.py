"""Postgres-реализации репозиториев чата поверх async SQLAlchemy 2.x.

`PostgresChatRepository` принимает `AsyncSession` — он живёт в рамках
одного HTTP-запроса (yield-dependency в `deps.py`) и пишет вместе с
основной транзакцией.

`PostgresSystemPromptRepository` принимает `session_factory` и открывает
короткоживущую сессию под единичный SELECT внутри `_pick_prompt`. Это
позволяет не держать дополнительное соединение на тех путях, где
A/B-сплит не используется (например, фон-задачи без LLM-вызова).
"""

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.chat.domain import Chat, ChatMessage, SystemPrompt
from app.chat.repositories.pg_models import (
    ChatMessageRow,
    ChatRow,
    SystemPromptRow,
)


class PostgresChatRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

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
        row = ChatRow(
            id=chat.id,
            owner_external_id=chat.owner_external_id,
            interface=chat.interface,
            system_prompt=chat.system_prompt,
            created_at=chat.created_at,
        )
        self.session.add(row)
        await self.session.commit()
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        stmt = select(ChatRow).where(ChatRow.id == chat_id)
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        return Chat.model_validate(row, from_attributes=True)

    async def get_or_create_chat(
        self,
        owner_external_id: str,
        interface: str,
    ) -> Chat:
        stmt = (
            select(ChatRow)
            .where(
                ChatRow.owner_external_id == owner_external_id,
                ChatRow.interface == interface,
            )
            .order_by(ChatRow.created_at.asc())
            .limit(1)
        )
        row = (await self.session.execute(stmt)).scalar_one_or_none()
        if row is not None:
            return Chat.model_validate(row, from_attributes=True)
        return await self.create_chat(owner_external_id, interface)

    async def append_message(
        self, chat_id: UUID, message: ChatMessage
    ) -> ChatMessage:
        row = ChatMessageRow(
            id=message.id,
            chat_id=chat_id,
            role=message.role,
            content=message.content,
            media_refs=message.media_refs,
            sources=message.sources,
            tokens=message.tokens,
            prompt_id=message.prompt_id,
            created_at=message.created_at,
        )
        self.session.add(row)
        await self.session.commit()
        return ChatMessage.model_validate(row, from_attributes=True)

    async def list_messages(
        self, chat_id: UUID, limit: int = 50
    ) -> list[ChatMessage]:
        stmt = (
            select(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
            .order_by(ChatMessageRow.created_at.desc())
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [
            ChatMessage.model_validate(r, from_attributes=True)
            for r in reversed(rows)
        ]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        stmt = (
            update(ChatMessageRow)
            .where(
                ChatMessageRow.chat_id == chat_id,
                ChatMessageRow.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.now(UTC))
        )
        await self.session.execute(stmt)
        await self.session.commit()


class PostgresSystemPromptRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None):
        self.session_factory = session_factory

    async def list_active(self) -> list[SystemPrompt]:
        """Активные кандидаты A/B-сплита, новые сначала."""
        if self.session_factory is None:
            return []
        stmt = (
            select(SystemPromptRow)
            .where(
                SystemPromptRow.active.is_(True),
                SystemPromptRow.traffic_pct > 0,
            )
            .order_by(SystemPromptRow.created_at.desc())
        )
        async with self.session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [
            SystemPrompt.model_validate(r, from_attributes=True) for r in rows
        ]
