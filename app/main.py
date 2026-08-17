import asyncio
import logging
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.admin.routes import router as admin_router
from app.chat.routes import router as chat_router
from app.core.config import get_settings
from app.core.exceptions import (
    LLMAuthError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.observability import setup_tracing
from app.routers import agent, chat, documents, health, models, rag
from app.services.vector_store import VectorStore

# Импорт модуля eXpress
from app.routers import express
from app.routers.express import init_express_bot, shutdown_express_bot

logger = logging.getLogger("llm-service")
logging.basicConfig(level=logging.INFO)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== Phoenix-трейсинг =====
    app.state.tracing_enabled = setup_tracing(settings)

    # ===== LLM-клиент (OpenAI-совместимый) =====
    app.state.llm = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        base_url=settings.llm.base_url,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )

    # ===== Redis (опционально) =====
    app.state.redis = None
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        app.state.redis = redis_client
    except Exception as e:
        logger.warning("Redis недоступен (%s) — продолжаем без кеша", e)

    # ===== PostgreSQL (опционально) =====
    app.state.async_engine = None
    app.state.session_factory = None
    try:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        app.state.async_engine = engine
        app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)
    except Exception as e:
        logger.warning(
            "Postgres engine не создан (%s) — postgres-репозиторий недоступен",
            e,
        )

    # ===== Qdrant (векторная БД) =====
    app.state.vector_store = None
    try:
        vector_store = VectorStore(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key is not None
                else None
            ),
            collection=settings.qdrant_collection,
            dim=settings.embedding_dim,
        )
        await vector_store.ensure_collection()
        app.state.vector_store = vector_store
        logger.info(
            "Qdrant подключён: %s, коллекция %s (dim=%d)",
            settings.qdrant_url,
            settings.qdrant_collection,
            settings.embedding_dim,
        )
    except Exception as e:
        logger.warning("Qdrant недоступен (%s) — vector-роуты вернут 503", e)

    # ===== RAG / Индексация (LlamaIndex) =====
    app.state.ingestion_service = None
    app.state.rag_service = None
    try:
        from app.services.ingestion import IngestionService
        from app.services.rag import RAGService

        ingestion = IngestionService(settings)
        app.state.ingestion_service = ingestion
        if ingestion.is_collection_empty():
            await asyncio.to_thread(ingestion.ingest_all)

        rag_service = RAGService(settings)
        await asyncio.to_thread(rag_service.build)
        app.state.rag_service = rag_service
        logger.info("RAG-сервис готов (коллекция %s)", settings.rag_collection)
    except Exception as e:
        logger.warning(
            "RAG/индексация не инициализированы (%s) — /rag/query и /documents вернут 503",
            e,
        )

    # ===== Агентный слой (LangGraph) =====
    app.state.agent_graph = None
    agent_stack = AsyncExitStack()
    try:
        from langchain_openai import ChatOpenAI
        from app.agents.tools import build_search_knowledge_base, multiply
        from app.services.agent_persistent import agent_lifespan

        agent_model = ChatOpenAI(
            model=settings.llm.default_model,
            base_url=settings.llm.base_url,
            temperature=0,
            api_key=settings.llm.openai_api_key.get_secret_value(),
            timeout=settings.llm.request_timeout,
        )

        async def _search_kb(query: str) -> dict:
            if app.state.rag_service is None:
                return {"answer": "База знаний недоступна.", "sources": [], "confident": False}
            return await app.state.rag_service.answer(query)

        async def _send_email(draft: dict) -> None:
            # Учебный side-effect: реальную отправку (SMTP/API) студент подключает сам.
            logger.info("send_email → %s: %s", draft.get("to"), draft.get("subject"))

        agent_tools = [multiply, build_search_knowledge_base(_search_kb)]
        app.state.agent_graph = await agent_stack.enter_async_context(
            agent_lifespan(
                settings.agent_checkpointer,
                agent_model,
                agent_tools,
                _send_email,
                sqlite_path=settings.agent_sqlite_path,
                postgres_url=settings.database_url,
            )
        )
        logger.info(
            "Персистентный агент собран (backend=%s)", settings.agent_checkpointer
        )
    except Exception as e:
        app.state.agent_graph = None
        logger.warning("Агентный граф не собран (%s) — /agent/* вернут 503", e)

    # ===== ИНИЦИАЛИЗАЦИЯ eXpress-БОТА =====
    try:
        await init_express_bot()
        logger.info("eXpress бот инициализирован")
    except Exception as e:
        logger.warning("eXpress бот не инициализирован (%s) — /express/webhook вернёт 503", e)

    # ===== ПРИЛОЖЕНИЕ ЗАПУЩЕНО =====
    yield

    # ===== ЗАКРЫТИЕ eXpress-БОТА =====
    try:
        await shutdown_express_bot()
        logger.info("eXpress бот остановлен")
    except Exception as e:
        logger.warning("Ошибка при остановке eXpress бота: %s", e)

    # ===== ЗАКРЫТИЕ ОСТАЛЬНЫХ РЕСУРСОВ =====
    await agent_stack.aclose()

    try:
        await app.state.llm.close()
    except Exception:
        logger.exception("ошибка при закрытии LLM-клиента")

    if app.state.redis is not None:
        try:
            await app.state.redis.close()
        except Exception:
            logger.exception("ошибка при закрытии Redis")

    if app.state.async_engine is not None:
        try:
            await app.state.async_engine.dispose()
        except Exception:
            logger.exception("ошибка при остановке engine Postgres")

    if app.state.vector_store is not None:
        try:
            await app.state.vector_store.close()
        except Exception:
            logger.exception("ошибка при закрытии Qdrant-клиента")

    if app.state.rag_service is not None:
        try:
            await app.state.rag_service.close()
        except Exception:
            logger.exception("ошибка при закрытии RAG-сервиса")

    if app.state.ingestion_service is not None:
        try:
            app.state.ingestion_service.close()
        except Exception:
            logger.exception("ошибка при закрытии индексатора")


# ===== СОЗДАНИЕ ПРИЛОЖЕНИЯ FASTAPI =====
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="FastAPI-сервис для LLM с кешированием, стримингом и модерацией",
    lifespan=lifespan,
)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-LLM-Cost-USD"],
)

# ===== OBSERVABILITY MIDDLEWARE =====
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.llm_cost = 0.0
    request.state.llm_tokens = 0

    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled", extra={"request_id": request.state.request_id})
        raise

    duration_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-LLM-Cost-USD"] = f"{request.state.llm_cost:.6f}"
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.state.request_id,
    )
    return response


# ===== ГЛОБАЛЬНЫЕ ОБРАБОТЧИКИ ИСКЛЮЧЕНИЙ =====
_STATUS_MAP: list[tuple[type[LLMError], int, str]] = [
    (LLMRateLimitError, 429, "llm_rate_limit"),
    (LLMAuthError, 502, "llm_auth_error"),
    (LLMTimeoutError, 504, "llm_timeout"),
    (LLMContentFilterError, 400, "content_filter"),
    (LLMError, 502, "llm_error"),
]


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError):
    for cls, status, code in _STATUS_MAP:
        if isinstance(exc, cls):
            return JSONResponse(
                status_code=status,
                content={"error": {"code": code, "message": str(exc)}},
                headers={"X-Request-ID": getattr(request.state, "request_id", "")},
            )
    return JSONResponse(
        status_code=502,
        content={"error": {"code": "llm_error", "message": str(exc)}},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "fields": errors}},
        headers={"X-Request-ID": getattr(request.state, "request_id", "")},
    )


# ===== ПОДКЛЮЧЕНИЕ РОУТЕРОВ =====
app.include_router(chat.router)
app.include_router(admin_router)
app.include_router(models.router)
app.include_router(health.router)
app.include_router(rag.router)
app.include_router(documents.router)
app.include_router(agent.router)
app.include_router(express.router)       # добавлен роутер eXpress