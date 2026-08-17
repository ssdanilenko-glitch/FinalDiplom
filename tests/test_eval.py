"""Тесты RAGAS-оценки: сборка строки и кастомная метрика — без сети.

ragas — опциональная группа (`uv sync --extra eval`); если не установлена,
модуль пропускается. Судья и метрики подменяются фейками, LLM-вызовов нет.
"""

import pytest

pytest.importorskip("ragas")

from app.eval.metrics import (  # noqa: E402
    CitationVerdict,
    RagasMetrics,
    eval_row,
    make_has_citation,
)


class _Result:
    def __init__(self, value) -> None:
        self.value = value


class _FakeMetric:
    """Возвращает фиксированное значение на любой ascore(...)."""

    def __init__(self, value) -> None:
        self._value = value

    async def ascore(self, **kwargs) -> _Result:
        return _Result(self._value)


class _FakeRAG:
    async def evaluate_inputs(self, question: str) -> dict:
        return {
            "answer": "Возврат возможен в течение 14 дней [1].",
            "top_score": 0.81,
            "sources": [{"id": 1, "file_name": "billing_refunds.md"}],
            "confident": True,
            "retrieved_contexts": ["Возврат товара возможен в течение 14 дней."],
        }


class _FakeJudge:
    """Имитирует InstructorLLM из llm_factory: agenerate → response_model."""

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    async def agenerate(self, prompt: str, response_model):
        return response_model(has_citation=self._verdict)


def _metrics() -> RagasMetrics:
    return RagasMetrics(
        faithfulness=_FakeMetric(1.0),
        answer_relevancy=_FakeMetric(0.94),
        context_precision=_FakeMetric(0.8),
        context_recall=_FakeMetric(0.7),
        factual_correctness=_FakeMetric(0.66),
    )


async def test_eval_row_collects_six_metrics() -> None:
    has_citation = make_has_citation(_FakeJudge("yes"))
    row = {"user_input": "Сколько дней на возврат?", "reference": "14 дней."}
    out = await eval_row(_FakeRAG(), row, _metrics(), has_citation)

    assert out["user_input"] == "Сколько дней на возврат?"
    assert out["faithfulness"] == 1.0
    assert out["answer_relevancy"] == 0.94
    assert out["context_precision"] == 0.8
    assert out["context_recall"] == 0.7
    assert out["factual_correctness"] == 0.66
    assert out["has_citation"] == "yes"


async def test_has_citation_uses_judge_verdict() -> None:
    yes_metric = make_has_citation(_FakeJudge("yes"))
    no_metric = make_has_citation(_FakeJudge("no"))

    assert (await yes_metric.ascore(response="Согласно [1].")).value == "yes"
    assert (await no_metric.ascore(response="Просто текст без ссылок.")).value == "no"


def test_citation_verdict_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        CitationVerdict(has_citation="maybe")
