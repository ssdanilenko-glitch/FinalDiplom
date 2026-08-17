"""Тесты персистентного агента с HIL на фейковой модели и sqlite in-memory.

Без сети, без Postgres: `AsyncSqliteSaver.from_conn_string(":memory:")` достаточно.
"""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.agents.tools import multiply
from app.services.agent_persistent import build_agent


class FakeChat:
    """Заглушка ChatModel: отдаёт заготовленную последовательность ответов."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = responses
        self._i = 0

    def bind_tools(self, tools):  # noqa: ANN001
        return self

    async def ainvoke(self, messages):  # noqa: ANN001
        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return response


def _send_email_call() -> AIMessage:
    return AIMessage(
        content="",
        id="ai-1",
        tool_calls=[
            {
                "name": "send_email",
                "args": {"to": "client@example.com", "subject": "Ответ", "body": "Текст"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


def _model() -> FakeChat:
    # 1) модель просит send_email → HIL-гейт; 2) после отправки — финальный ответ.
    return FakeChat([_send_email_call(), AIMessage(content="Готово.", id="ai-2")])


def _initial() -> dict:
    return {
        "messages": [HumanMessage("отправь письмо клиенту")],
        "iteration_count": 0,
        "tool_results": [],
        "draft": None,
        "sent": False,
    }


@pytest.mark.asyncio
async def test_graph_reaches_interrupt_before_send():
    send_fn = AsyncMock()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, _model(), [multiply], send_fn)
        config = {"configurable": {"thread_id": "t-interrupt"}}

        result = await graph.ainvoke(_initial(), config)

        assert "__interrupt__" in result  # граф встал на interrupt
        snapshot = await graph.aget_state(config)
        assert "confirm_and_send" in snapshot.next  # ждём узел подтверждения
        send_fn.assert_not_called()  # side-effect ещё не выполнялся


@pytest.mark.asyncio
async def test_resume_true_sends_email():
    send_fn = AsyncMock()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, _model(), [multiply], send_fn)
        config = {"configurable": {"thread_id": "t-approve"}}

        await graph.ainvoke(_initial(), config)
        final = await graph.ainvoke(Command(resume=True), config)

        assert final["sent"] is True
        send_fn.assert_awaited_once()
        assert send_fn.await_args.args[0]["subject"] == "Ответ"


@pytest.mark.asyncio
async def test_resume_false_does_not_send():
    send_fn = AsyncMock()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, _model(), [multiply], send_fn)
        config = {"configurable": {"thread_id": "t-reject"}}

        await graph.ainvoke(_initial(), config)
        final = await graph.ainvoke(Command(resume=False), config)

        assert final["sent"] is False
        send_fn.assert_not_called()  # отказ — реального вызова API нет


@pytest.mark.asyncio
async def test_role_full_skips_interrupt():
    # Роль full: узел подтверждения не поднимает interrupt, письмо уходит сразу.
    send_fn = AsyncMock()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, _model(), [multiply], send_fn)
        config = {"configurable": {"thread_id": "t-full", "user_role": "full"}}

        result = await graph.ainvoke(_initial(), config)

        assert "__interrupt__" not in result
        assert result["sent"] is True
        send_fn.assert_awaited_once()
