"""Трейсинг в Phoenix через OpenInference (опциональный runtime-путь).

Включается флагом `PHOENIX_ENABLED=true` и группой зависимостей `tracing`
(`uv sync --extra tracing`): openinference-instrumentation-llama-index +
openinference-instrumentation-langchain + opentelemetry-sdk + exporter-otlp.
По умолчанию выключено — сервис поднимается без трейсинга, спаны не пишутся.

Инструменторы подключаются один раз при старте (lifespan):
- LlamaIndex — вызовы RAG (retrieve, embed, LLM) попадают в спаны автоматически;
- LangChain/LangGraph — прогон агентного графа: каждый узел, вызовы инструментов
  с input/output, вызовы LLM и `__interrupt__` становятся дочерними спанами.
"""

import logging
from importlib.util import find_spec

from app.core.config import Settings

logger = logging.getLogger(__name__)


def setup_tracing(settings: Settings) -> bool:
    """Регистрирует инструменторы RAG и агентного графа → Phoenix.

    Возвращает True, если трейсинг включён и хотя бы один инструментор поднят.
    """
    if not settings.phoenix_enabled:
        return False

    has_llama = find_spec("openinference.instrumentation.llama_index") is not None
    has_langchain = find_spec("openinference.instrumentation.langchain") is not None
    if not (has_llama or has_langchain):
        logger.warning(
            "phoenix_enabled=true, но пакеты трейсинга не установлены — "
            "uv sync --extra tracing"
        )
        return False

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint))
    )
    trace.set_tracer_provider(provider)

    instrumented: list[str] = []
    if has_llama:
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        LlamaIndexInstrumentor().instrument(tracer_provider=provider)
        instrumented.append("LlamaIndex")
    if has_langchain:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # LangGraph построен на LangChain-runnable'ах, поэтому этот же инструментор
        # покрывает узлы графа, вызовы инструментов и LLM.
        LangChainInstrumentor().instrument(tracer_provider=provider)
        instrumented.append("LangChain/LangGraph")

    logger.info(
        "Phoenix-трейсинг включён (%s): %s",
        ", ".join(instrumented),
        settings.phoenix_collector_endpoint,
    )
    return True
