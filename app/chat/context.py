"""Утилиты подсчёта токенов и подгонки контекста под бюджет."""

from functools import lru_cache
from typing import Any

import tiktoken

from app.chat.domain import ChatMessage


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(
    messages: list[dict[str, Any] | ChatMessage],
    model: str = "gpt-5.4-mini",
) -> int:
    """Считаем токены через o200k_base (GPT-4o / GPT-5).

    Поправки на ChatML overhead: +4 на каждое сообщение, +2 итого.
    """
    enc = _encoder()
    total = 0
    for m in messages:
        content = m["content"] if isinstance(m, dict) else m.content
        role = m.get("role", "") if isinstance(m, dict) else m.role
        total += len(enc.encode(content)) + len(enc.encode(role))
        total += 4  # ChatML overhead на сообщение
    return total + 2


def fit_to_budget(
    messages: list[ChatMessage], budget: int, model: str = "gpt-5.4-mini"
) -> list[ChatMessage]:
    """Режет messages с начала под бюджет, сохраняя ВСЕ system-сообщения.

    Возвращает новый список — исходный не мутируется.
    """
    if not messages:
        return []

    # Текущее число токенов
    def _tokens(msgs: list[ChatMessage]) -> int:
        return count_tokens(msgs, model=model)

    msgs = list(messages)
    if _tokens(msgs) <= budget:
        return msgs

    # Разделяем на system и non-system, сохраняя порядок.
    systems = [m for m in msgs if m.role == "system"]
    rest = [m for m in msgs if m.role != "system"]

    # Если даже только system'ы превышают бюджет — отдаём как есть
    # (резать system нельзя по контракту).
    if _tokens(systems) >= budget:
        return systems

    # Режем rest с начала, пока system + rest <= budget.
    while rest and _tokens(systems + rest) > budget:
        rest.pop(0)

    # Восстановить исходный относительный порядок: проходим по messages,
    # включаем то, что осталось в systems/rest.
    survived_ids = {id(m) for m in systems} | {id(m) for m in rest}
    return [m for m in messages if id(m) in survived_ids]
