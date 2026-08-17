"""Проверка eval/tracing-путей: версии, импорты, опциональный live-прогон.

Тяжёлые зависимости (ragas, arize-phoenix) — опциональны: ставятся группами
`eval`/`tracing`. Скрипт проверяет, что импорты собираются и контракт метрик
на месте; при наличии ключей гоняет одну метрику вживую (по правилу для
опциональных путей — основная проверка это импорт, а не полный конвейер).

Запуск:
    uv run --extra eval --extra tracing python scripts/verify_eval.py
"""

import asyncio
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402


def _print_versions() -> None:
    print("=== versions ===")
    for pkg in [
        "ragas",
        "anthropic",
        "pandas",
        "llama-index-llms-anthropic",
        "openinference-instrumentation-llama-index",
        "opentelemetry-sdk",
        "arize-phoenix",
    ]:
        try:
            print(f"  {pkg}=={version(pkg)}")
        except PackageNotFoundError:
            print(f"  {pkg}: NOT INSTALLED")


async def _live_ascore() -> None:
    """Один прогон Faithfulness + has_citation вживую — нужны ключи OpenAI+судьи."""
    from app.eval.metrics import build_judge, build_metrics, make_has_citation

    settings = get_settings()
    judge, embeddings = build_judge(settings)
    metrics = build_metrics(judge, embeddings)
    has_citation = make_has_citation(judge)

    question = "Сколько дней на возврат?"
    answer = "Возврат возможен в течение 14 дней с момента покупки [1]."
    contexts = ["Возврат товара возможен в течение 14 дней с момента покупки."]
    faith = await metrics.faithfulness.ascore(
        user_input=question, response=answer, retrieved_contexts=contexts
    )
    cite = await has_citation.ascore(response=answer)
    print(f"[live ascore] OK: faithfulness={faith.value:.2f} has_citation={cite.value}")


def main() -> None:
    _print_versions()

    # --- import-check метрик и судьи (группа eval) ---
    from ragas.embeddings import OpenAIEmbeddings  # noqa: F401
    from ragas.llms import llm_factory  # noqa: F401
    from ragas.metrics import discrete_metric  # noqa: F401
    from ragas.metrics.collections import (  # noqa: F401
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )
    from ragas.testset import TestsetGenerator  # noqa: F401

    from app.eval.metrics import (  # noqa: F401
        build_judge,
        build_metrics,
        eval_row,
        make_has_citation,
    )

    print("[ragas imports] OK (collections + llm_factory + discrete_metric + TestsetGenerator)")

    # --- import-check трейсинга (группа tracing) ---
    if find_spec("openinference.instrumentation.llama_index"):
        from openinference.instrumentation.llama_index import (  # noqa: F401
            LlamaIndexInstrumentor,
        )

        print("[tracing imports] OK (LlamaIndexInstrumentor)")
    else:
        print("[tracing imports] SKIP: uv sync --extra tracing")

    # --- phoenix.evals — опциональный продвинутый путь (HallucinationEvaluator) ---
    if find_spec("phoenix"):
        print("[phoenix.evals] доступен (arize-phoenix установлен)")
    else:
        print("[phoenix.evals] SKIP: arize-phoenix не установлен")

    # --- опциональный live-прогон одной метрики ---
    settings = get_settings()
    has_openai = settings.llm.openai_api_key.get_secret_value().startswith("sk-") and \
        "placeholder" not in settings.llm.openai_api_key.get_secret_value()
    has_judge = (
        settings.eval_judge_provider == "anthropic"
        and (os.getenv("ANTHROPIC_API_KEY") or settings.anthropic_api_key is not None)
    ) or settings.eval_judge_provider == "openai"
    if has_openai and has_judge:
        asyncio.run(_live_ascore())
    else:
        print("[live ascore] SKIP: нет ключей OPENAI_API_KEY / судьи (ANTHROPIC_API_KEY)")

    print("\nОбязательные импорты eval-пути отработали.")


if __name__ == "__main__":
    main()
