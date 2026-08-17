from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    openai_api_key: SecretStr = SecretStr("sk-test-placeholder")
    base_url: str = "https://api.proxyapi.ru/openai/v1"
    default_model: str = "gpt-5.4-mini"
    request_timeout: float = 30.0
    max_retries: int = 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "llm-service"
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # Chat ---------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://chat:chat@localhost:5432/chat"
    chat_repository: Literal["json", "postgres"] = "json"
    chat_storage_dir: Path = Path("./var/chats")
    chat_context_window: int = 10

    # Production ---------------------------------------------------------
    # X-Admin-Token для /chats/admin/*. Сменить на 32+ hex-байт через
    # `openssl rand -hex 32` в проде.
    admin_token: SecretStr = SecretStr("change-me-admin")
    # Service-to-service: backend ↔ bot (общий с bot /notify).
    internal_token: SecretStr = SecretStr("change-me-internal")
    # Базовый URL bot-сервиса (для broadcast и notify-вызовов из backend).
    bot_url: str = "http://bot:9000"
    # Telegram chat_id админ-группы для alert drain и handoff-уведомлений.
    admin_chat_id: int | None = None
    # Включить OpenAI Moderation API (layer 2 каскада). Если False —
    # только regex-блоклист.
    moderation_use_openai: bool = True
    # Rate limit: сколько сообщений на одного owner_external_id в минуту.
    rate_limit_messages_per_min: int = 15

    # Qdrant ------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"
    embedding_dim: int = 1536
    embedding_model: str = "text-embedding-3-small"

    # RAG ---------------------------------------------------------------
    # Корпус для индексации и отдельные коллекции под LlamaIndex и bare-metal:
    # один корпус и одна embed-модель, но раскладка payload разная.
    rag_data_dir: Path = Path("data/rag-block-03")
    rag_collection: str = "rag_block_03"
    rag_collection_bare: str = "rag_block_03_bare"
    rag_llm_model: str = "gpt-5.4-mini"
    rag_top_k: int = 3
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    # Если top-1 score ниже порога — ответа в корпусе нет, отдаём честный fallback.
    rag_score_threshold: float = 0.3
    # Корпоративный RAG: достаём широко, оставляем top_n лучших.
    rag_retrieve_top_k: int = 10
    rag_rerank_top_n: int = 5
    # Реранкер и гибридный поиск — опциональные тяжёлые зависимости, в репо не
    # держим хард-депендой. Включаются флагом, тогда нужны extras:
    #   reranker -> pip install sentence-transformers torch  (модель ~600 МБ)
    #   hybrid   -> pip install fastembed
    # По умолчанию выключены; dense-поиск с обрезкой до rag_rerank_top_n работает и так.
    rag_use_reranker: bool = False
    rag_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rag_use_hybrid: bool = False
    rag_sparse_model: str = "Qdrant/bm25"
    # Контроль доступа на уровне поиска: фильтр visibility="internal" до ретрива.
    # Включать только когда корпус проиндексирован через IngestionService
    # (он проставляет visibility); на «голой» коллекции фильтр вернёт пусто.
    rag_restrict_to_internal: bool = False

    # Агент и чекпоинтер -------------------------------------------------
    # Бэкенд персистентности графа: memory (unit-тесты) | sqlite (локальная
    # разработка) | postgres (docker-compose). Схему checkpoint-таблиц ведёт
    # setup() чекпоинтера, доменную — Alembic (checkpoint* исключены в env.py).
    agent_checkpointer: Literal["memory", "sqlite", "postgres"] = "sqlite"
    # Файл SQLite-чекпоинтера при agent_checkpointer="sqlite".
    agent_sqlite_path: str = "var/agent_checkpoints.sqlite"

    # Phoenix-трейсинг --------------------------------------------------
    # Инструментирование LlamaIndex в Phoenix — опциональный runtime-путь,
    # группа зависимостей `tracing` (uv sync --extra tracing). По умолчанию
    # выключено; в дипломном стеке включается PHOENIX_ENABLED=true, тогда нужен
    # сервис phoenix (см. compose.yaml).
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"

    # Оценка качества (RAGAS) -------------------------------------------
    # Судья и эмбеддинги для офлайн-оценки (scripts/run_eval.py,
    # generate_testset.py) — группа зависимостей `eval`. Судья отделён от
    # production-LLM в /rag/query (rag_llm_model): роли разные, путать нельзя.
    anthropic_api_key: SecretStr | None = None
    eval_judge_provider: Literal["anthropic", "openai"] = "openai"
    eval_judge_model: str = "gpt-5.4-mini"

    # === Telegram ===
    TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")   # <-- исправлено
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://app:8000")   # <-- исправлено
    TELEGRAM_EDIT_DEBOUNCE_MS: int = int(os.getenv("TELEGRAM_EDIT_DEBOUNCE_MS", "800"))

    EXPRESS_BOT_ID: str = ""          # UUID
    EXPRESS_CTS_HOST: str = ""        # например, "cts.example.com"
    EXPRESS_SECRET_KEY: str = ""      # секретный ключ

@lru_cache
def get_settings() -> Settings:
    return Settings()
