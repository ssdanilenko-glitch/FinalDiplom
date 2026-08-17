"""Точка входа: long polling + uvicorn-сервер /notify + alert-drain в одном loop.

Polling шлёт Telegram-апдейты в Dispatcher, параллельно крутится небольшой
FastAPI-сервер на 9000 для обратного канала backend → bot (см. bot/web.py),
плюс фон-таска drain_alerts — push алертов из БД в админ-чат.
"""

import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_bot_settings
from bot.handlers import register_routers
from bot.services.alert_drain import drain_alerts
from bot.services.backend_client import BackendClient
from bot.services.http import build_http_client
from bot.web import build_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bot")


async def main() -> None:
    settings = get_bot_settings()

    bot = Bot(
        token=settings.bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())

    http = build_http_client(settings)
    backend = BackendClient(
        http, admin_token=settings.admin_token.get_secret_value()
    )
    dp["backend"] = backend

    register_routers(dp)

    api = build_api(bot, settings.internal_token.get_secret_value())
    config = uvicorn.Config(
        api,
        host="0.0.0.0",
        port=settings.bot_api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    log.info(
        "Bot starting (backend=%s, notify-port=%s, admin_chat_id=%s)",
        settings.backend_url,
        settings.bot_api_port,
        settings.admin_chat_id,
    )
    try:
        await asyncio.gather(
            dp.start_polling(bot),
            server.serve(),
            drain_alerts(bot, backend, settings.admin_chat_id),
        )
    finally:
        await backend.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
