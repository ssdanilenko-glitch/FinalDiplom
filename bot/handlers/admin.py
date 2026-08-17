"""Админ-команды бота: /stats, /broadcast.

Доступны только пользователям из BOT_ADMIN_IDS.
- /stats — агрегаты за 24ч из backend /chats/admin/stats
- /broadcast <текст> — массовая рассылка всем telegram-юзерам через
  backend /chats/admin/broadcast (backend сам подтягивает список).

Команда /operator — это запрос пользователя «переключите меня на оператора»,
а не админ-действие. Она вынесена в отдельный router (bot/handlers/handoff.py)
без IsAdmin-фильтра.

Backend закрыт X-Admin-Token; BackendClient получает его через конструктор
из BotSettings.admin_token. В compose оба сервиса (app и bot) получают
одинаковый ADMIN_TOKEN.
"""

import logging

from aiogram import Router
from aiogram.filters import Command, CommandObject, Filter
from aiogram.types import Message

from bot.config import get_bot_settings
from bot.services.backend_client import BackendClient

log = logging.getLogger(__name__)


class IsAdmin(Filter):
    """Пропускает только Telegram-юзеров из BotSettings.bot_admin_ids.

    Список парсится через pydantic-settings (валидатор `_parse_ids` в
    BotSettings); `get_bot_settings()` закеширован через `@lru_cache`,
    так что на сообщение — только set-lookup.
    """

    async def __call__(self, message: Message) -> bool:
        if message.from_user is None:
            return False
        return message.from_user.id in set(get_bot_settings().bot_admin_ids)


router = Router(name="admin")
router.message.filter(IsAdmin())


@router.message(Command("stats"))
async def cmd_stats(message: Message, backend: BackendClient) -> None:
    try:
        s = await backend.get_admin_stats()
        text = (
            "<b>Статистика 24ч</b>\n"
            f"Сообщений: <code>{s.get('total_messages', 0)}</code>\n"
            f"DAU: <code>{s.get('active_users', 0)}</code>\n"
            f"Feedback ratio: <code>{s.get('feedback_ratio', 0):.1%}</code>"
        )
        await message.answer(text)
    except Exception as e:
        log.exception("stats failed")
        await message.answer(f"Ошибка: {e}")


@router.message(Command("broadcast"))
async def cmd_broadcast(
    message: Message, command: CommandObject, backend: BackendClient
) -> None:
    text = (command.args or "").strip()
    if not text:
        await message.answer("Использование: /broadcast &lt;текст&gt;")
        return
    try:
        result = await backend.broadcast(text, interface="telegram")
        await message.answer(
            "<b>Рассылка завершена</b>\n"
            f"Отправлено: <code>{result.get('sent', 0)}</code>\n"
            f"Ошибок: <code>{result.get('failed', 0)}</code>"
        )
    except Exception as e:
        log.exception("broadcast failed")
        await message.answer(f"Ошибка: {e}")
