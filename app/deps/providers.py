"""Глобальные FastAPI-зависимости.

Всё, что лежит в `app.state` (LLM-клиент, Redis, session_factory), достаём
через `Depends`-провайдеры — это даёт тип-чек и убирает getattr из роутов.
Lifespan гарантирует, что атрибуты на app.state выставлены: для боевых
клиентов — реальный объект, для опциональных (Redis/Postgres) — `None`.
"""

from typing import Annotated, Any

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.llm import LLMService
from app.services.vector_store import VectorStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_llm(request: Request) -> Any:
    return request.app.state.llm


def get_cache(request: Request) -> Any:
    return request.app.state.redis


def get_session_factory(request: Request) -> Any:
    """Возвращает async_sessionmaker, выставленный в lifespan, либо None,
    если Postgres недоступен. Роуты, которым PG обязателен, должны явно
    проверять на None и отдавать 503/собственный fallback."""
    return request.app.state.session_factory


def get_vector_store(request: Request) -> VectorStore | None:
    """Vector-store, инициализированный в lifespan. None — если Qdrant
    был недоступен на старте: роут должен решить, что делать (503 или
    fallback на чистый LLM-ответ без retrieval)."""
    return request.app.state.vector_store


def get_rag_service(request: Request) -> Any:
    """RAG-сервис на LlamaIndex, собранный один раз в lifespan. None — если
    Qdrant/индекс был недоступен на старте: роут отдаёт 503."""
    return request.app.state.rag_service


def get_ingestion_service(request: Request) -> Any:
    """Индексатор корпуса (офлайн-контур), собранный в lifespan. None — если
    Qdrant был недоступен на старте: ручки /documents отдают 503."""
    return request.app.state.ingestion_service


def get_agent_graph(request: Request) -> Any:
    """Скомпилированный ReAct-граф агента, собранный в lifespan. None — если
    сборка не удалась (нет ключа/модели): /agent/chat отдаёт 503."""
    return request.app.state.agent_graph


LLMDep = Annotated[Any, Depends(get_llm)]
CacheDep = Annotated[Any, Depends(get_cache)]
SessionFactoryDep = Annotated[Any, Depends(get_session_factory)]
VectorStoreDep = Annotated[VectorStore | None, Depends(get_vector_store)]
RAGServiceDep = Annotated[Any, Depends(get_rag_service)]
IngestionServiceDep = Annotated[Any, Depends(get_ingestion_service)]
AgentGraphDep = Annotated[Any, Depends(get_agent_graph)]


def get_llm_service(
    llm: LLMDep,
    cache: CacheDep,
    settings: SettingsDep,
) -> LLMService:
    return LLMService(llm=llm, cache=cache, ttl=settings.cache_ttl_seconds)


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
