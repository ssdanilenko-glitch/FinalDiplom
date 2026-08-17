"""Тесты A/B traffic-split для system prompts."""

from uuid import uuid4

from app.chat.domain import SystemPrompt
from app.chat.prompt_selection import choose_by_split


def _mk(version: str, pct: int) -> SystemPrompt:
    return SystemPrompt(
        id=uuid4(), version=version, body=f"body-{version}", traffic_pct=pct
    )


def test_empty_candidates_returns_none():
    assert choose_by_split("u1", []) is None


def test_single_candidate_always_wins():
    c = _mk("v1", 100)
    for i in range(50):
        assert choose_by_split(f"u{i}", [c]) is c


def test_split_is_deterministic_sticky():
    """Один и тот же owner → один и тот же кандидат при одинаковых traffic_pct."""
    cs = [_mk("a", 50), _mk("b", 50)]
    for owner in ("user1", "user2", "user3"):
        first = choose_by_split(owner, cs)
        for _ in range(10):
            assert choose_by_split(owner, cs) is first


def test_split_respects_traffic_proportions():
    """На большом числе owner'ов 80/20 даёт ~80% в первый."""
    cs = [_mk("a", 80), _mk("b", 20)]
    counts = {"a": 0, "b": 0}
    for i in range(1000):
        chosen = choose_by_split(f"owner-{i}", cs)
        counts[chosen.version] += 1
    # Распределение sha256-buckets ≈ uniform; 80/20 даст разброс не более ±5%.
    a_share = counts["a"] / 1000
    assert 0.75 <= a_share <= 0.85


def test_sum_less_than_100_falls_back_to_first():
    """Если sum(traffic_pct) < 100 и bucket в «дыре», падаем на первого."""
    cs = [_mk("a", 10), _mk("b", 10)]
    # Перебираем owner'ы — большинство bucket'ов попадает в «дыру» 20-99
    # и должно вернуть candidates[0]
    seen_first = 0
    total = 100
    for i in range(total):
        chosen = choose_by_split(f"owner-{i}", cs)
        if chosen.version == "a":
            seen_first += 1
    # Должно быть существенно больше половины (включая fallback)
    assert seen_first > total // 2
