"""Тесты FSM-сценария AskFlow.

Не запускают aiogram-диспетчер — проверяют, что состояния объявлены
и что FSMContext корректно их хранит / переключает.
"""

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.keyboards.inline import DEFAULT_TOPICS, topics_kb
from bot.states import AskFlow


def test_ask_flow_states_exist():
    assert AskFlow.waiting_for_topic is not None
    assert AskFlow.waiting_for_question is not None
    assert AskFlow.confirming is not None


@pytest.mark.asyncio
async def test_state_transitions():
    storage = MemoryStorage()
    key = StorageKey(bot_id=1, chat_id=1, user_id=1)
    ctx = FSMContext(storage=storage, key=key)

    assert await ctx.get_state() is None

    await ctx.set_state(AskFlow.waiting_for_topic)
    assert await ctx.get_state() == "AskFlow:waiting_for_topic"

    await ctx.update_data(topic="billing")
    data = await ctx.get_data()
    assert data["topic"] == "billing"

    await ctx.set_state(AskFlow.waiting_for_question)
    assert await ctx.get_state() == "AskFlow:waiting_for_question"

    await ctx.clear()
    assert await ctx.get_state() is None


def test_topics_kb_has_buttons():
    kb = topics_kb()
    # Все темы + кнопка «Отмена»
    assert len(kb.inline_keyboard) == len(DEFAULT_TOPICS) + 1
    # callback_data формата "topic:<slug>"
    seen_callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
    for _, slug in DEFAULT_TOPICS:
        assert f"topic:{slug}" in seen_callbacks
    assert "topic:cancel" in seen_callbacks


def test_topics_kb_accepts_custom_topics():
    custom = [("RAG", "rag"), ("Агенты", "agents")]
    kb = topics_kb(custom)
    callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
    assert "topic:rag" in callbacks
    assert "topic:agents" in callbacks
    assert "topic:cancel" in callbacks
