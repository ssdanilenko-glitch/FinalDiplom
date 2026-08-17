"""Метрики RAGAS 0.4 и сборка строки оценки.

RAGAS 0.4 — collections-API: каждая метрика это объект с корутиной `ascore(...)`
(аргументы только именованные, у разных метрик разный набор полей), результат —
`MetricResult` со `.value`. Судья создаётся через `llm_factory`, эмбеддинги для
`AnswerRelevancy` — через `ragas.embeddings.OpenAIEmbeddings`. Кастомная
категориальная метрика — декоратором `@discrete_metric` (в 0.4 `AspectCritic` убран).

Модуль импортирует ragas на верхнем уровне — тяжёлая группа `eval`
(`uv sync --extra eval`). В проде не используется.
"""

from dataclasses import dataclass
from typing import Any, Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import discrete_metric
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from app.core.config import Settings


@dataclass
class RagasMetrics:
    """Пять collections-метрик RAGAS, собранных на одном судье."""

    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    context_precision: ContextPrecision
    context_recall: ContextRecall
    factual_correctness: FactualCorrectness


def build_judge(settings: Settings) -> tuple[Any, OpenAIEmbeddings]:
    """Судья (LLM) и эмбеддинги для метрик. Судья отделён от production-LLM."""
    openai_client = AsyncOpenAI(api_key=settings.llm.openai_api_key.get_secret_value())
    embeddings = OpenAIEmbeddings(client=openai_client, model=settings.embedding_model)

    if settings.eval_judge_provider == "anthropic":
        api_key = (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key is not None
            else None
        )
        judge = llm_factory(
            settings.eval_judge_model,
            provider="anthropic",
            client=AsyncAnthropic(api_key=api_key),
        )
    else:
        judge = llm_factory(
            settings.eval_judge_model, provider="openai", client=openai_client
        )
    return judge, embeddings


def build_metrics(judge: Any, embeddings: OpenAIEmbeddings) -> RagasMetrics:
    """Пять метрик генерации/контекста на общем судье."""
    return RagasMetrics(
        faithfulness=Faithfulness(llm=judge),
        answer_relevancy=AnswerRelevancy(llm=judge, embeddings=embeddings),
        context_precision=ContextPrecision(llm=judge),
        context_recall=ContextRecall(llm=judge),
        factual_correctness=FactualCorrectness(llm=judge),
    )


class CitationVerdict(BaseModel):
    has_citation: Literal["yes", "no"]


CITATION_PROMPT = (
    "Содержит ли ответ ссылку на источник: маркер вида '[1]'/'[doc_id]', имя "
    "файла, или фразу 'согласно ...', 'в источнике X указано'?\n\n"
    "Ответ: {response}"
)


def make_has_citation(judge: Any):
    """Кастомная метрика «ответ содержит цитату» через @discrete_metric.

    Возвращает метрику, замкнутую на переданном судье, — так её удобно
    тестировать с фейковым судьёй и переиспользовать в run_eval.
    """

    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(response: str) -> str:
        verdict = await judge.agenerate(
            CITATION_PROMPT.format(response=response), response_model=CitationVerdict
        )
        return verdict.has_citation

    return has_citation


async def eval_row(rag: Any, row: dict, metrics: RagasMetrics, has_citation: Any) -> dict:
    """Шесть метрик по одной строке golden dataset.

    `rag` — RAGService (или совместимый), у которого есть `evaluate_inputs`.
    Аргументы метрик именованные: у каждой свой набор полей.
    """
    result = await rag.evaluate_inputs(row["user_input"])
    answer, contexts = result["answer"], result["retrieved_contexts"]
    q, ref = row["user_input"], row["reference"]
    return {
        "user_input": q,
        "faithfulness": (
            await metrics.faithfulness.ascore(
                user_input=q, response=answer, retrieved_contexts=contexts
            )
        ).value,
        "answer_relevancy": (
            await metrics.answer_relevancy.ascore(user_input=q, response=answer)
        ).value,
        "context_precision": (
            await metrics.context_precision.ascore(
                user_input=q, reference=ref, retrieved_contexts=contexts
            )
        ).value,
        "context_recall": (
            await metrics.context_recall.ascore(
                user_input=q, retrieved_contexts=contexts, reference=ref
            )
        ).value,
        "factual_correctness": (
            await metrics.factual_correctness.ascore(response=answer, reference=ref)
        ).value,
        "has_citation": (await has_citation.ascore(response=answer)).value,
    }
