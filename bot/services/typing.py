"""Индикатор «бот печатает…».

Telegram сбрасывает индикатор каждые ~5 сек или после следующего сообщения.
Шлём sendChatAction раз в 4 сек до тех пор, пока stop-event не выставлен.
"""

import asyncio

from aiogram import Bot
from aiogram.enums import ChatAction


async def typing_until(bot: Bot, chat_id: int, stop: asyncio.Event) -> None:
    """Шлёт TYPING каждые 4 сек до stop.set()."""
    while not stop.is_set():
        try:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            # Сетевая ошибка — не валим основной поток; следующая итерация попробует снова.
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            continue
