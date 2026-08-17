"""FSM-сценарий /ask: выбор темы → текст вопроса → отправка с topic-префиксом."""

import asyncio
import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import topics_kb
from bot.services.backend_client import BackendClient
from bot.services.error_handling import handle_backend_error
from bot.services.streaming import stream_to_chat
from bot.services.typing import typing_until
from bot.states import AskFlow

router = Router(name="fsm")
log = logging.getLogger(__name__)


@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext) -> None:
    await state.set_state(AskFlow.waiting_for_topic)
    await message.answer("Выберите тему:", reply_markup=topics_kb())


@router.callback_query(F.data.startswith("topic:"), AskFlow.waiting_for_topic)
async def on_topic_selected(cb: CallbackQuery, state: FSMContext) -> None:
    _, slug = cb.data.split(":", 1)
    if slug == "cancel":
        await state.clear()
        if cb.message is not None:
            await cb.message.edit_text("Отменено.")
        await cb.answer()
        return
    await state.update_data(topic=slug)
    await state.set_state(AskFlow.waiting_for_question)
    if cb.message is not None:
        await cb.message.edit_text(
            f"Тема: {slug}\nЗадайте ваш вопрос текстом."
        )
    await cb.answer()


@router.message(AskFlow.waiting_for_question, F.text)
async def on_question_received(
    message: Message,
    backend: BackendClient,
    state: FSMContext,
) -> None:
    data = await state.get_data()
    topic = data.get("topic", "general")
    prompt = f"Тема: {topic}. Вопрос: {message.text}"

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
            chat_id, prompt, owner_external_id=str(message.chat.id)
        )
        await stream_to_chat(message, events, chat_id=chat_id)
    except Exception as exc:
        await handle_backend_error(message, exc)
    finally:
        stop.set()
        await typing_task
        await state.clear()
