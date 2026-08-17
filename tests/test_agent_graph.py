"""Тесты агентного графа на фейковой модели — без сети и без ключей."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import MAX_ITERATIONS, build_custom_graph
from app.agents.tools import build_search_knowledge_base, multiply


class FakeChat:
    """Заглушка ChatModel: отдаёт заранее заготовленную последовательность ответов.

    Последний ответ повторяется, если вызовов больше, чем заготовлено.
    """

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._i = 0

    def bind_tools(self, tools):  # noqa: ANN001
        return self

    async def ainvoke(self, messages):  # noqa: ANN001
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


async def _run(graph, text: str) -> dict:
    return await graph.ainvoke(
        {"messages": [HumanMessage(text)], "iteration_count": 0, "tool_results": []}
    )


@pytest.mark.asyncio
async def test_custom_graph_runs_tool_and_finishes():
    model = FakeChat(
        [
            _tool_call("multiply", {"a": 17, "b": 23}, "c1"),
            AIMessage(content="Ответ: 391"),
        ]
    )
    graph = build_custom_graph(model, [multiply])

    result = await _run(graph, "Сколько будет 17 * 23?")

    assert result["messages"][-1].content == "Ответ: 391"
    assert result["tool_results"] == [
        {"name": "multiply", "args": {"a": 17, "b": 23}, "result": "391"}
    ]


@pytest.mark.asyncio
async def test_force_finish_stops_runaway_loop():
    # Модель всегда просит инструмент — без лимита это бесконечный цикл.
    # Каждый ответ — новое сообщение с уникальным id, иначе add_messages
    # склеит их по id и цикл не наберёт итераций.
    class LoopingChat:
        def __init__(self) -> None:
            self._n = 0

        def bind_tools(self, tools):  # noqa: ANN001
            return self

        async def ainvoke(self, messages):  # noqa: ANN001
            self._n += 1
            return AIMessage(
                content="",
                id=f"ai-{self._n}",
                tool_calls=[
                    {
                        "name": "multiply",
                        "args": {"a": 2, "b": 2},
                        "id": f"c{self._n}",
                        "type": "tool_call",
                    }
                ],
            )

    graph = build_custom_graph(LoopingChat(), [multiply])

    result = await _run(graph, "зациклись")

    assert result["iteration_count"] == MAX_ITERATIONS


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_not_crash():
    model = FakeChat(
        [
            _tool_call("delete_everything", {}, "c1"),
            AIMessage(content="Не могу это сделать."),
        ]
    )
    graph = build_custom_graph(model, [multiply])

    result = await _run(graph, "удали всё")

    assert result["tool_results"][0]["result"].startswith("error: unknown tool")
    assert result["messages"][-1].content == "Не могу это сделать."


@pytest.mark.asyncio
async def test_search_knowledge_base_tool_formats_sources():
    async def fake_search(query: str) -> dict:
        return {
            "answer": "Срок возврата — 14 дней [1].",
            "sources": [{"id": 1, "file_name": "returns.md"}],
            "confident": True,
        }

    search_tool = build_search_knowledge_base(fake_search)
    out = await search_tool.ainvoke({"query": "срок возврата"})

    assert "Срок возврата — 14 дней" in out
    assert "[1] returns.md" in out
