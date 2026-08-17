"""Pydantic-схемы admin-API."""

from datetime import datetime

from pydantic import BaseModel


class StatsOut(BaseModel):
    """Агрегаты за окно времени, которые backend реально считает.

    M4-метрики (сообщения, активные пользователи, доля положительных оценок) +
    RAG-дельта M5: доля отказов, доля отрицательных оценок и топ вопросов без
    ответа (по таблице rag_queries).
    """

    total_messages: int
    active_users: int
    feedback_ratio: float = 0.0
    # RAG-дельта M5: считаются по rag_queries и message_feedback.
    refusal_rate: float = 0.0
    negative_feedback_rate: float = 0.0
    knowledge_gaps: list[str] = []


class BroadcastIn(BaseModel):
    """Адресаты задаются ровно одним способом:

    - явный `owner_ids: list[int]` — рассылка по списку Telegram chat_id;
    - `interface: "telegram"` — backend сам подтянет всех owner_external_id
      из таблицы `chats` по этому интерфейсу.

    Хотя бы одно из полей должно быть задано, иначе route вернёт 400.
    """

    text: str
    owner_ids: list[int] | None = None
    interface: str | None = None


class BroadcastResult(BaseModel):
    sent: int
    failed: int


class ExportItem(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    created_at: str


class ExportResult(BaseModel):
    items: list[ExportItem]
    next_after: datetime | None = None


class HandoffIn(BaseModel):
    owner_external_id: str
    interface: str = "telegram"
    status: str  # 'active' | 'paused_for_human' | 'resolved'


class AlertOut(BaseModel):
    id: int
    kind: str
    payload: dict
