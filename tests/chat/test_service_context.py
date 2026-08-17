"""Тесты ChatService — стратегия sliding window и подсчёт токенов."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.chat.context import count_tokens, fit_to_budget
from app.chat.domain import ChatMessage
from app.chat.service import ChatService


pytestmark = pytest.mark.asyncio


def _make_streaming_llm(captured: dict):
    """LLM-мок, который фиксирует переданные messages и стримит ответ."""
    llm = AsyncMock()

    async def fake_create(**kwargs):
        captured["messages"] = kwargs.get("messages")
        captured["model"] = kwargs.get("model")
        captured["stream"] = kwargs.get("stream")

        async def gen():
            for piece in ["ok", "!"]:
                yield MagicMock(
                    choices=[MagicMock(delta=MagicMock(content=piece))],
                    usage=None,
                )

        return gen()

    llm.chat.completions.create = AsyncMock(side_effect=fake_create)
    return llm


async def test_sliding_window_limits_history(json_repo):
    captured: dict = {}
    llm = _make_streaming_llm(captured)
    service = ChatService(
        repository=json_repo,
        llm_client=llm,
        context_window=5,
        default_model="gpt-5.4-mini",
    )

    chat = await service.create_chat(
        owner_external_id="u", interface="cli", system_prompt="SYS"
    )

    # Запишем 20 «прошлых» user-сообщений ВРУЧНУЮ в репозиторий
    for i in range(20):
        await json_repo.append_message(
            chat.id,
            ChatMessage(chat_id=chat.id, role="user", content=f"old-{i}"),
        )

    # Теперь отправим новое сообщение через сервис
    events = []
    async for event in service.send_message(chat.id, "new"):
        events.append(event)

    deltas = [e["delta"] for e in events if e.get("type") == "token"]
    assert "".join(deltas) == "ok!"
    # И обязательно был ровно один message_saved-кадр с UUID
    saved = [e for e in events if e.get("type") == "message_saved"]
    assert len(saved) == 1
    assert saved[0]["message_id"]

    messages = captured["messages"]
    # system + 5 последних сообщений из истории
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert len(messages) == 1 + 5

    # Последнее в контексте — только что записанное "new"
    assert messages[-1]["content"] == "new"
    # Перед ним — old-19, old-18, old-17, old-16 (всего 5: 4 старых + 1 новое)
    contents = [m["content"] for m in messages[1:]]
    assert contents == ["old-16", "old-17", "old-18", "old-19", "new"]


async def test_send_message_saves_assistant_after_stream(json_repo):
    captured: dict = {}
    llm = _make_streaming_llm(captured)
    service = ChatService(
        repository=json_repo,
        llm_client=llm,
        context_window=10,
        default_model="gpt-5.4-mini",
    )
    chat = await service.create_chat("u", "cli")

    events = []
    async for event in service.send_message(chat.id, "hi"):
        events.append(event)

    msgs = await json_repo.list_messages(chat.id)
    # 1 user + 1 assistant
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].content == "hi"
    assert msgs[1].content == "ok!"
    # message_saved.message_id должен совпасть с id ассистент-сообщения
    saved = [e for e in events if e.get("type") == "message_saved"]
    assert saved and saved[0]["message_id"] == str(msgs[1].id)


async def test_count_tokens_returns_positive_int():
    n = count_tokens(
        [
            {"role": "system", "content": "Hello"},
            {"role": "user", "content": "Привет!"},
        ]
    )
    assert isinstance(n, int)
    assert n > 0


async def test_fit_to_budget_preserves_system():
    msgs = [
        ChatMessage(
            chat_id=uuid4(),
            role="system",
            content="SYS-PROMPT",
        ),
    ]
    # Добавим 50 длинных user-сообщений
    chat_id = msgs[0].chat_id
    for i in range(50):
        msgs.append(
            ChatMessage(
                chat_id=chat_id,
                role="user",
                content=f"long content number {i} " * 20,
            )
        )

    trimmed = fit_to_budget(msgs, budget=200)
    # system обязан остаться
    assert trimmed[0].role == "system"
    # Должны были что-то срезать
    assert len(trimmed) < len(msgs)


async def test_clear_history_calls_soft_delete(json_repo):
    service = ChatService(
        repository=json_repo,
        llm_client=AsyncMock(),
        context_window=10,
        default_model="gpt-5.4-mini",
    )
    chat = await service.create_chat("u", "cli")
    await json_repo.append_message(
        chat.id, ChatMessage(chat_id=chat.id, role="user", content="x")
    )
    await service.clear_history(chat.id)
    assert await json_repo.list_messages(chat.id) == []
