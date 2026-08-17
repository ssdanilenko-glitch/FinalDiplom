"""Обработчик свободного текста (catch-all)."""

import asyncio
import logging

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.services.backend_client import BackendClient
from bot.services.error_handling import handle_backend_error
from bot.services.streaming import stream_to_chat
from bot.services.typing import typing_until

router = Router(name="text")
log = logging.getLogger(__name__)


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(
    message: Message, backend: BackendClient, state: FSMContext
) -> None:
    # FSM имеет приоритет — если внутри сценария, отдаём обработку fsm-роутеру.
    # Здесь страховочная проверка: fsm-роутер регистрируется выше text-роутера,
    # так что в норме это не сработает.
    if await state.get_state() is not None:
        return

    chat_id = await backend.get_or_create_chat(
        owner_external_id=str(message.chat.id),
        interface="telegram",
    )
    stop = asyncio.Event()
    typing_task = asyncio.create_task(
        typing_until(message.bot, message.chat.id, stop)
    )
    try:
        events = backend.send_message(
            chat_id,
            message.text,
            owner_external_id=str(message.chat.id),
        )
        await stream_to_chat(message, events, chat_id=chat_id)
    except Exception as exc:
        await handle_backend_error(message, exc)
    finally:
        stop.set()
        await typing_task
