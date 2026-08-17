"""HTTP-роуты chat-модуля.

`POST /chats/{chat_id}/messages` принимает multipart/form-data с
опциональным media. Перед запуском стрима выполняется модерация (чтобы
вернуть 403 до старта SSE) и rate-limit через Depends; идентификатор
владельца берётся из заголовка X-Owner-External-Id.

SSE-контракт `POST /chats/{chat_id}/messages` (стабильный — клиент
пишется один раз):

    data: {"type":"token","delta":"<chunk-of-text>"}\\n\\n
    ...
    data: {"type":"message_saved","message_id":"<uuid>"}\\n\\n
    data: {"type":"done"}\\n\\n

`message_saved` — кадр после сохранения assistant-сообщения в БД, нужен
боту чтобы прицепить feedback-клавиатуру к финальному send_message.
`done` всегда последний (и при ошибке — в finally).
"""

import json
from typing import Annotated, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage
from app.core.config import get_settings
from app.deps.providers import SessionFactoryDep
from app.ratelimit.dependencies import enforce_rate_limit

# Wire-значения feedback. Те же строки на стороне бота (см.
# bot/keyboards/inline.py::FEEDBACK_UP/FEEDBACK_DOWN). Общего модуля у
# backend и bot нет — это контракт сети, поддерживается ручным синхроном.
FeedbackValue = Literal["up", "down"]

router = APIRouter(prefix="/chats", tags=["chats"])

_settings = get_settings()
_rate_limit_message = enforce_rate_limit(
    "message", _settings.rate_limit_messages_per_min
)


class CreateChatIn(BaseModel):
    owner_external_id: str
    interface: str
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


class FeedbackIn(BaseModel):
    owner_external_id: str
    value: FeedbackValue   # Pydantic сам провалидирует Literal["up","down"]


@router.post("", response_model=CreateChatOut, summary="Создать чат")
async def create_chat(
    body: CreateChatIn, chat_service: ChatServiceDep
) -> CreateChatOut:
    """Идемпотентно по (owner_external_id, interface): повторный POST с теми
    же параметрами возвращает chat_id уже существующего чата. Это нужно ботам,
    которые при каждом сообщении вызывают /chats и не помнят локально id.

    system_prompt применяется только при первом создании; если чат уже
    существует — игнорируется молча (override через PATCH вне scope).
    """
    chat = await chat_service.get_or_create_chat(
        owner_external_id=body.owner_external_id,
        interface=body.interface,
        system_prompt=body.system_prompt,
    )
    return CreateChatOut(chat_id=chat.id)


@router.get("/{chat_id}", response_model=Chat, summary="Метаданные чата")
async def get_chat(chat_id: UUID, chat_service: ChatServiceDep) -> Chat:
    chat = await chat_service.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return chat


@router.post(
    "/{chat_id}/messages",
    summary="Послать сообщение (multipart, SSE streaming)",
    dependencies=[Depends(_rate_limit_message)],
)
async def post_message(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    content: str = Form(""),
    media: UploadFile | None = File(None),
    owner_external_id: Annotated[
        str | None, Header(alias="X-Owner-External-Id")
    ] = None,
) -> StreamingResponse:
    """Принимает текст + опциональное медиа, стримит ответ LLM как SSE.

    Порядок проверок:
    1. rate-limit (Depends) — 429 если превышен.
    2. moderation на input — 403 если заблокирован (вызывается ЗДЕСЬ, не
       внутри генератора, иначе после старта streaming HTTPException
       не сменит уже отправленный 200-status).
    3. StreamingResponse(send_message(...)).

    owner_external_id берётся из того же заголовка, что и rate-limit;
    нужен модерации только для контекста в alert payload.
    """
    mod_result = await chat_service.check_input(
        content, owner_external_id=owner_external_id
    )
    if not mod_result.allowed:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "moderation_blocked",
                "categories": mod_result.categories,
                "layer": mod_result.layer,
            },
        )

    async def event_source():
        try:
            async for event in chat_service.send_message(
                chat_id=chat_id, user_content=content, media=media,
            ):
                # service yield-ит уже структурированные dict-события:
                # {"type":"token","delta":"..."} и
                # {"type":"message_saved","message_id":"..."}.
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.get(
    "/{chat_id}/messages",
    response_model=list[ChatMessage],
    summary="История сообщений (хронологически)",
)
async def list_messages(
    chat_id: UUID,
    chat_service: ChatServiceDep,
    limit: int = Query(50, ge=1, le=500),
) -> list[ChatMessage]:
    return await chat_service.list_messages(chat_id, limit=limit)


@router.delete("/{chat_id}/messages", summary="Очистить историю (soft delete)")
async def delete_messages(
    chat_id: UUID, chat_service: ChatServiceDep
) -> dict:
    await chat_service.clear_history(chat_id)
    return {"status": "ok"}


@router.post(
    "/{chat_id}/messages/{msg_id}/feedback",
    summary="Оценка ответа: up / down",
)
async def post_feedback(
    chat_id: UUID,
    msg_id: UUID,
    body: FeedbackIn,
    session_factory: SessionFactoryDep,
) -> dict:
    """value: 'up' | 'down' (валидируется Pydantic Literal в FeedbackIn).
    UNIQUE по (owner_external_id, message_id) — повторный POST молча
    игнорируется (idempotent)."""
    if session_factory is None:
        raise HTTPException(
            status_code=503, detail="feedback requires postgres"
        )
    async with session_factory() as s:
        await s.execute(
            text(
                """
                INSERT INTO message_feedback
                    (message_id, owner_external_id, value, created_at)
                VALUES (:m, :o, :v, NOW())
                ON CONFLICT (owner_external_id, message_id) DO NOTHING
                """
            ),
            {"m": msg_id, "o": body.owner_external_id, "v": body.value},
        )
        await s.commit()
    return {"status": "ok"}
