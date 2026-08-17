"""DI для chat-модуля."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repositories.pg_repo import (
    PostgresChatRepository,
    PostgresSystemPromptRepository,
)
from app.chat.repository import ChatRepository
from app.chat.service import ChatService
from app.core.config import get_settings
from app.deps.providers import LLMDep, SessionFactoryDep
from app.moderation.service import ModerationService


async def get_repository(
    session_factory: SessionFactoryDep,
) -> AsyncIterator[ChatRepository]:
    """Фабрика репозитория. Postgres-сессия живёт ровно один запрос.

    Реализована как yield-dependency, чтобы session корректно закрывалась.
    """
    settings = get_settings()
    if settings.chat_repository == "json":
        yield JsonChatRepository(settings.chat_storage_dir)
        return

    if settings.chat_repository == "postgres":
        if session_factory is None:
            raise RuntimeError(
                "session_factory not initialised — postgres repository unavailable"
            )
        async with session_factory() as session:
            yield PostgresChatRepository(session)
        return

    raise ValueError(f"unknown chat_repository: {settings.chat_repository}")


RepositoryDep = Annotated[ChatRepository, Depends(get_repository)]


def get_chat_service(
    repo: RepositoryDep,
    llm: LLMDep,
    session_factory: SessionFactoryDep,
) -> ChatService:
    settings = get_settings()

    # Moderation: используем тот же AsyncOpenAI-клиент (он стоит в app.state.llm),
    # — `llm` сюда уже приходит через get_llm, который читает app.state.llm.
    # session_factory нужен сервису, чтобы писать alert при блокировке;
    # если PG нет — алерты молча не пишутся.
    moderation = ModerationService(
        llm_client=llm,
        use_openai_moderation=settings.moderation_use_openai,
        session_factory=session_factory,
    )

    # Prompt repository: только если есть PG-сессия. Иначе choose_by_split
    # вернёт None и в send_message сработает chat.system_prompt.
    prompt_repo = (
        PostgresSystemPromptRepository(session_factory)
        if session_factory is not None
        else None
    )

    return ChatService(
        repository=repo,
        llm_client=llm,
        context_window=settings.chat_context_window,
        default_model=settings.llm.default_model,
        moderation=moderation,
        prompt_repo=prompt_repo,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
