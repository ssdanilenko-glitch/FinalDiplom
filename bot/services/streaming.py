"""Рендеринг стрима событий в Telegram-сообщение.

Native streaming через `sendMessageDraft`: бот шлёт серию «драфтов» с
общим draft_id (Telegram анимирует приращение текста), а в конце
фиксирует ответ полноценным `send_message`. Драфт — ephemeral preview
на ~30 секунд, поэтому финальный send_message обязателен.

Backend стримит уже dict-события `{"type":"token","delta":"..."}` и
однократно `{"type":"message_saved","message_id":"<uuid>"}`. Если id
известен — финальный send_message получает inline-клавиатуру feedback,
привязанную к этому message_id. Если backend старый или message_id
не пришёл — кнопок нет.

sendMessageDraft — private-chat only. Если бот когда-нибудь окажется в группе,
вызов упадёт; на этот случай оставлен узкий AttributeError-fallback (на
старых версиях aiogram без метода) — он переключает рендер на edit_text-тротлинг.
"""

import logging
import uuid
from collections.abc import AsyncIterable
from time import monotonic
from uuid import UUID

import telegramify_markdown
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from bot.keyboards.inline import feedback_kb

log = logging.getLogger(__name__)


def _to_tg_markdown(text: str) -> str:
    """GitHub-Markdown от LLM → Telegram MarkdownV2 с эскейпом спецсимволов.

    LLM возвращает обычный Markdown (`**bold**`, `# header`, `- list`), а
    Telegram парсит свой MarkdownV2 (требует эскейпа `.`, `-`, `(`, `)`, ...).
    `telegramify-markdown` делает конвертацию и эскейп.
    """
    try:
        return telegramify_markdown.markdownify(text)
    except Exception:
        # На любую ошибку конвертации — отдаём текст как есть; парсер Telegram
        # на это вернёт ошибку, и мы упадём в fallback без parse_mode.
        return text

# Минимальный интервал между sendMessageDraft вызовами на один draft.
# Telegram flood-control режет ~30 вызовов/сек суммарно; на длинном LLM-стриме
# (десятки токенов в секунду) без тротлинга мгновенно ловим TelegramRetryAfter.
# 0.7 сек даёт плавную анимацию и оставляет запас под другие сообщения бота.
DRAFT_MIN_INTERVAL_SEC = 0.7


async def stream_to_chat(
    message: Message,
    events: AsyncIterable[dict],
    chat_id: UUID | None = None,
) -> str:
    """Стримит через sendMessageDraft с общим draft_id. Финальный send_message
    фиксирует ответ в чате и крепит feedback-кнопки, если backend отдал
    message_id."""
    draft_id = uuid.uuid4().int & 0xFFFFFFFF or 1  # ensure non-zero
    buffer = ""
    assistant_message_id: str | None = None
    last_draft_at = 0.0

    # Первый кадр — пустой draft-плейсхолдер. Если метод недоступен (старая
    # aiogram) — graceful fallback на edit_text.
    try:
        await message.bot.send_message_draft(
            chat_id=message.chat.id, draft_id=draft_id, text="",
        )
        last_draft_at = monotonic()
    except AttributeError:
        return await _stream_via_edit_text(message, events, chat_id)
    except TelegramRetryAfter as e:
        log.warning("draft flood on init, falling back to edit_text: retry_after=%s", e.retry_after)
        return await _stream_via_edit_text(message, events, chat_id)

    async for event in events:
        etype = event.get("type")
        if etype == "token":
            buffer += event.get("delta", "")
            if not buffer.strip():
                continue
            now = monotonic()
            if now - last_draft_at < DRAFT_MIN_INTERVAL_SEC:
                continue   # тротлим — финальный send_message покажет полный текст
            try:
                await message.bot.send_message_draft(
                    chat_id=message.chat.id,
                    draft_id=draft_id,
                    text=buffer,
                )
                last_draft_at = now
            except TelegramRetryAfter as e:
                # Telegram сам сказал «подожди N сек» — пропускаем draft'ы
                # на это окно. Финальный send_message всё равно отрисует ответ.
                last_draft_at = now + e.retry_after
            except TelegramBadRequest:
                # draft expired / message_not_modified — игнорируем,
                # доберём финальным send_message.
                pass
        elif etype == "message_saved":
            assistant_message_id = event.get("message_id")

    if buffer:
        reply_markup = (
            feedback_kb(assistant_message_id) if assistant_message_id else None
        )
        await _send_final(message, buffer, reply_markup)
    return buffer


async def _send_final(message: Message, text: str, reply_markup) -> None:
    """Шлёт финальный send_message с Telegram MarkdownV2.

    Если MarkdownV2-парсер Telegram'а спотыкается на конкретном тексте
    (бывает на нестандартных конструкциях LLM) — graceful fallback на plain.
    """
    md = _to_tg_markdown(text)
    try:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=md,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except TelegramBadRequest as e:
        log.warning("MarkdownV2 parse failed, fallback to plain: %s", e)
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=text,
            reply_markup=reply_markup,
        )


async def _stream_via_edit_text(
    message: Message,
    events: AsyncIterable[dict],
    chat_id: UUID | None = None,
) -> str:
    """Fallback: edit_text-тротлинг 1 сек/кадр + finalize с feedback-кнопками."""
    sent = await message.answer("…")
    buffer = ""
    assistant_message_id: str | None = None
    last_edit = monotonic()

    async for event in events:
        etype = event.get("type")
        if etype == "token":
            buffer += event.get("delta", "")
            if monotonic() - last_edit >= 1.0:
                try:
                    await sent.edit_text(buffer)
                    last_edit = monotonic()
                except TelegramRetryAfter as e:
                    last_edit = monotonic() + e.retry_after
                except TelegramBadRequest:
                    last_edit = monotonic()
        elif etype == "message_saved":
            assistant_message_id = event.get("message_id")

    if buffer:
        reply_markup = (
            feedback_kb(assistant_message_id) if assistant_message_id else None
        )
        md = _to_tg_markdown(buffer)
        try:
            await sent.edit_text(
                md,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
        except TelegramBadRequest:
            try:
                await sent.edit_text(buffer, reply_markup=reply_markup)
            except (TelegramBadRequest, TelegramRetryAfter):
                pass
        except TelegramRetryAfter:
            pass
    return buffer
