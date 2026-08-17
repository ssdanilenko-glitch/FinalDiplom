# app/routers/express.py
"""
Модуль интеграции с корпоративным мессенджером eXpress.
Аналог Telegram-модуля, построен на pybotx + FastAPI.
"""

import json
import logging
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pybotx import (
    Bot,
    BotAccountWithSecret,
    HandlerCollector,
    IncomingMessage,
    bot,
    lifespan_wrapper,
)

from app.core.config import settings
from app.services.agent_persistent import get_agent, process_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/express", tags=["express"])

# ============================================================================
# 1. НАСТРОЙКА ОБРАБОТЧИКОВ КОМАНД (аналог telegram handlers)
# ============================================================================

collector = HandlerCollector()

@collector.command("/start", description="Начать работу с ассистентом")
async def start_handler(message: IncomingMessage, bot: Bot) -> None:
    """Обработчик команды /start."""
    await bot.answer_message(
        "👋 Привет! Я ИИ-ассистент техподдержки УИТ.\n"
        "Задай свой вопрос по 1С:ERP или любому из 82 сервисов — я помогу найти ответ или создам обращение в ITILIUM."
    )

@collector.command("/help", description="Помощь по командам")
async def help_handler(message: IncomingMessage, bot: Bot) -> None:
    """Обработчик команды /help."""
    await bot.answer_message(
        "📋 Доступные команды:\n"
        "/start — начать работу\n"
        "/help — эта справка\n"
        "/status {UID} — проверить статус обращения в ITILIUM\n\n"
        "Просто напиши свой вопрос — я обработаю его через RAG-поиск или создам обращение."
    )

@collector.command("/status", description="Проверить статус обращения в ITILIUM")
async def status_handler(message: IncomingMessage, bot: Bot) -> None:
    """
    Обработчик команды /status.
    Ожидает аргумент: UID обращения.
    """
    args = message.body.strip().split()
    if len(args) < 2:
        await bot.answer_message("❌ Укажите UID обращения: `/status {UID}`")
        return

    ticket_uid = args[1]
    # Здесь вызывается ваш сервис для проверки статуса в ITILIUM
    # result = await itilium_client.get_ticket_status(ticket_uid)
    # await bot.answer_message(f"Статус обращения {ticket_uid}: {result.status}")

    # Заглушка для демонстрации
    await bot.answer_message(
        f"🔍 Проверяю статус обращения {ticket_uid}...\n"
        "(Интеграция с ITILIUM будет добавлена на следующем этапе)"
    )

# ----------------------------------------------------------------------------
# ОСНОВНОЙ ОБРАБОТЧИК ТЕКСТОВЫХ СООБЩЕНИЙ (вызов ИИ-агента)
# ----------------------------------------------------------------------------

@collector.default_handler
async def default_handler(message: IncomingMessage, bot: Bot) -> None:
    """
    Обработчик всех остальных текстовых сообщений.
    Здесь происходит вызов основного ИИ-агента (RAG + HIL + эскалация).
    """
    user_id = str(message.user.id)          # идентификатор пользователя в eXpress
    chat_id = str(message.chat.id)          # идентификатор чата
    text = message.body

    logger.info(f"Express message from {user_id} in {chat_id}: {text[:100]}...")

    try:
        # 1. Вызов основного агента (ваша существующая логика из DZ6_4)
        #    Функция process_message обрабатывает запрос через RAG, определяет сервис,
        #    создаёт обращение в ITILIUM при необходимости и возвращает ответ.
        response = await process_message(
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            # можно передать дополнительные контекстные данные из eXpress
            # например, имя пользователя, должность и т.д.
        )

        # 2. Отправка ответа обратно в чат
        await bot.answer_message(response.answer)

        # 3. Если в ответе есть вложения (например, ссылка на обращение) — можно отправить их отдельно
        if response.attachments:
            for attachment in response.attachments:
                await bot.answer_message(attachment)

    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        await bot.answer_message(
            "⚠️ Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже или обратитесь в поддержку."
        )

# ============================================================================
# 2. ИНИЦИАЛИЗАЦИЯ БОТА (аналог bot = Bot(...) в Telegram)
# ============================================================================

def create_express_bot() -> Bot:
    """
    Создаёт и настраивает экземпляр Bot для eXpress.
    Данные для подключения берутся из переменных окружения.
    """
    return Bot(
        collectors=[collector],
        bot_accounts=[
            BotAccountWithSecret(
                id=UUID(settings.EXPRESS_BOT_ID),          # Bot ID из панели администратора
                host=settings.EXPRESS_CTS_HOST,            # например, "cts.example.com"
                secret_key=settings.EXPRESS_SECRET_KEY,    # секретный ключ бота
            ),
        ],
    )

# Глобальный экземпляр бота (будет инициализирован в lifespan)
express_bot: Optional[Bot] = None

# ============================================================================
# 3. FASTAPI ЭНДПОИНТЫ (вебхук + статус)
# ============================================================================

@router.post("/webhook")
async def webhook_handler(request: Request) -> JSONResponse:
    """
    Эндпоинт для приёма вебхуков от eXpress.
    eXpress отправляет сюда все сообщения и системные события.
    """
    global express_bot

    if express_bot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Express bot not initialized"
        )

    try:
        raw_body = await request.json()
        logger.debug(f"Webhook payload: {json.dumps(raw_body, ensure_ascii=False)[:200]}...")

        # pybotx самостоятельно разберёт входящий запрос и вызовет нужный обработчик
        await express_bot.async_execute_raw_bot_command(raw_body)

        # eXpress ожидает ответ 200 OK, даже если обработка была асинхронной
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok"}
        )

    except Exception as e:
        logger.exception(f"Webhook processing error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "detail": str(e)}
        )

@router.get("/status")
async def status_check() -> Dict[str, Any]:
    """Эндпоинт для проверки состояния бота."""
    return {
        "status": "running" if express_bot else "not_initialized",
        "bot_id": settings.EXPRESS_BOT_ID,
    }

# ============================================================================
# 4. LIFESPAN ИНТЕГРАЦИЯ (подключение к main.py)
# ============================================================================

async def init_express_bot() -> None:
    """Инициализация бота при старте приложения."""
    global express_bot
    express_bot = create_express_bot()
    # Внутренняя инициализация pybotx (подготовка подключения к CTS)
    await express_bot.startup()
    logger.info("Express bot initialized successfully")

async def shutdown_express_bot() -> None:
    """Корректное завершение работы бота."""
    global express_bot
    if express_bot:
        await express_bot.shutdown()
        express_bot = None
        logger.info("Express bot shutdown successfully")