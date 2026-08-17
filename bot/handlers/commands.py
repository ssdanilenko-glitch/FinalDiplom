"""Команды бота: /start, /help, /clear, /cancel."""

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.services.backend_client import BackendClient

router = Router(name="commands")


@router.message(CommandStart())
async def cmd_start(
    message: Message, backend: BackendClient, state: FSMContext
) -> None:
    await state.clear()
    # Прогреваем chat_id в backend, чтобы /clear / on_text потом не падали.
    await backend.get_or_create_chat(
        owner_external_id=str(message.chat.id),
        interface="telegram",
    )
    await message.answer(
        "Привет! Я подключён к chat-сервису. Пиши сообщения — я отвечу.\n"
        "Команды: /help, /ask, /clear, /cancel"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Доступные команды:\n"
        "/start — начать заново\n"
        "/ask — задать вопрос с выбором темы\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить текущий сценарий\n"
        "/operator — передать диалог живому оператору\n"
        "\n"
        "Для админов: /stats, /broadcast &lt;текст&gt;"
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    current = await state.get_state()
    if current is None:
        await message.answer("Нечего отменять.")
        return
    await state.clear()
    await message.answer("Сценарий отменён.")


@router.message(Command("clear"))
async def cmd_clear(
    message: Message, backend: BackendClient, state: FSMContext
) -> None:
    await state.clear()
    chat_id = await backend.get_or_create_chat(
        owner_external_id=str(message.chat.id),
        interface="telegram",
    )
    await backend.clear_messages(
        chat_id, owner_external_id=str(message.chat.id)
    )
    await message.answer("История очищена.")
