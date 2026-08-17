"""Inline-клавиатуры для сценариев бота."""

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Префикс callback_data для feedback. Лимит Telegram — 64 байта на
# callback_data. msg_id (UUID, 36 байт) + префикс "fb:up:" / "fb:down:" даёт
# 6+36 = 42 байта, влезает. chat_id в callback НЕ кладём — handler возьмёт
# его через `backend.get_or_create_chat(owner_external_id=str(chat.id))`,
# который идемпотентен и стоит ~5 ms на запрос.
FEEDBACK_CB_PREFIX = "fb"

# Wire-значения feedback. Те же строки на backend (см. app/chat/routes.py::
# FeedbackIn.value — Literal["up","down"]). Совпадение поддерживается ручным
# контрактом: общего Python-модуля у backend и bot нет.
FEEDBACK_UP = "up"
FEEDBACK_DOWN = "down"
FEEDBACK_VALUES = (FEEDBACK_UP, FEEDBACK_DOWN)

# Темы — примеры из диплом-доменов (техподдержка SaaS).
# В реальном дипломе студент подменит на свои.
DEFAULT_TOPICS: list[tuple[str, str]] = [
    ("Биллинг", "billing"),
    ("Установка", "install"),
    ("API", "api"),
]


def topics_kb(
    topics: list[tuple[str, str]] | None = None,
) -> InlineKeyboardMarkup:
    """Inline-кнопки выбора темы + Отмена."""
    topics = topics or DEFAULT_TOPICS
    kb = InlineKeyboardBuilder()
    for label, slug in topics:
        kb.button(text=label, callback_data=f"topic:{slug}")
    kb.button(text="Отмена", callback_data="topic:cancel")
    kb.adjust(1)
    return kb.as_markup()


def feedback_kb(message_id: str) -> InlineKeyboardMarkup:
    """Кнопки оценки ответа. callback_data: fb:<vote>:<msg_id> (42 байта)."""
    kb = InlineKeyboardBuilder()
    kb.button(
        text="👍",
        callback_data=f"{FEEDBACK_CB_PREFIX}:{FEEDBACK_UP}:{message_id}",
    )
    kb.button(
        text="👎",
        callback_data=f"{FEEDBACK_CB_PREFIX}:{FEEDBACK_DOWN}:{message_id}",
    )
    kb.adjust(2)
    return kb.as_markup()
