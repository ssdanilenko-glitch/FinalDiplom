"""Обработчик feedback-кнопок под ответами ассистента.

callback_data формат: `fb:<vote>:<message_id>` (42 байта при UUID, влезает
в лимит Telegram 64 байт).

`chat_id` (UUID backend-чата) в callback не передаётся — получаем его
через идемпотентный `backend.get_or_create_chat(owner_external_id=...)`
по Telegram chat.id юзера, нажавшего кнопку. Один лишний HTTP-вызов на
клик, но feedback — редкое событие, и так мы держимся в 64-байтовом
лимите callback_data.
"""

import logging

import httpx
from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.keyboards.inline import (
    FEEDBACK_CB_PREFIX,
    FEEDBACK_UP,
    FEEDBACK_VALUES,
)
from bot.services.backend_client import BackendClient

router = Router(name="feedback")
log = logging.getLogger(__name__)


@router.callback_query(F.data.startswith(f"{FEEDBACK_CB_PREFIX}:"))
async def on_feedback(cb: CallbackQuery, backend: BackendClient) -> None:
    try:
        _, vote, msg_id = cb.data.split(":", 2)
    except ValueError:
        await cb.answer()
        return

    if vote not in FEEDBACK_VALUES:
        await cb.answer()
        return

    try:
        chat_uuid = await backend.get_or_create_chat(
            owner_external_id=str(cb.message.chat.id),
            interface="telegram",
        )
        await backend.post_feedback(
            chat_id=chat_uuid,
            message_id=msg_id,
            owner_external_id=str(cb.message.chat.id),
            value=vote,
        )
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("feedback post failed: %s", exc)
    except Exception:
        log.exception("feedback handler crashed")

    try:
        if cb.message is not None:
            await cb.message.edit_reply_markup(reply_markup=None)
    except Exception as exc:
        # Если Telegram ругается (message_not_modified,
        # сообщение старое) — это не критично, голос уже зафиксирован выше.
        log.debug("clear keyboard failed: %s", exc)
    await cb.answer("Спасибо!" if vote == FEEDBACK_UP else "Учли.")
