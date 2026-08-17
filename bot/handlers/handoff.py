"""Команда /operator — handoff на оператора.

Доступна любому пользователю: он же просит перевести себя на оператора, а не
админ это делает за него. Вынесено в отдельный router без IsAdmin-фильтра.
"""

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.services.backend_client import BackendClient

log = logging.getLogger(__name__)

router = Router(name="handoff")


@router.message(Command("operator"))
async def cmd_operator(message: Message, backend: BackendClient) -> None:
    """Меняем статус на backend; подтверждение юзеру пришлёт сам backend
    через POST /notify внутри handoff-route. Бот отвечает только при ошибке —
    иначе получаем два почти одинаковых сообщения в чате."""
    try:
        await backend.set_handoff_status(
            owner_external_id=str(message.chat.id),
            interface="telegram",
            status="paused_for_human",
        )
    except Exception as exc:
        log.warning("handoff failed: %s", exc)
        await message.answer("Не удалось переключить — попробуйте позже.")
