"""ORM-модели для Postgres-хранения чата и аналитики RAG.

Строки чата (`ChatRow`, `ChatMessageRow`, `SystemPromptRow`) — внутренние для
модуля app.chat; граница с доменом — `ChatMessage.model_validate(row, ...)`.
`RagQueryRow` — общая аналитическая строка лога RAG-запросов: её пишет
RAG-роут и читает `AdminRepository` (refusal_rate, пробелы в знаниях).
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

TimestampTZ = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class ChatRow(Base):
    __tablename__ = "chats"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_external_id: Mapped[str]
    interface: Mapped[str]
    system_prompt: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE")
    )
    role: Mapped[str]
    content: Mapped[str]
    media_refs: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # Цитаты RAG-ответа, показанные пользователю: по ним отрицательная оценка
    # указывает, что чинить (плохие источники → поиск, хорошие но плохой ответ → генерация).
    sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    tokens: Mapped[int | None]
    prompt_id: Mapped[UUID | None]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        TimestampTZ, nullable=True
    )


class RagQueryRow(Base):
    """Лог RAG-запросов для аналитики: refusal_rate и пробелы в знаниях."""

    __tablename__ = "rag_queries"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    question_normalized: Mapped[str]
    confident: Mapped[bool]
    top_score: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )


class SystemPromptRow(Base):
    __tablename__ = "system_prompts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    version: Mapped[str]
    body: Mapped[str]
    active: Mapped[bool] = mapped_column(default=False)
    traffic_pct: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(
        TimestampTZ, default=lambda: datetime.now(UTC)
    )
