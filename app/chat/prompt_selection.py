"""A/B-выбор системного промпта по детерминированному бакету.

Sticky split: один и тот же `owner_external_id` всегда попадает в один и тот
же вариант (пока операторы не меняют traffic_pct). Это важно для интерпретации
конверсий и feedback-метрик.

Алгоритм:
1. sha256(owner_external_id) → первые 8 hex-символов → int → mod 100
2. идём по candidates в порядке repo.list_active(), накапливаем traffic_pct
3. первый кандидат, у которого cumulative > bucket — победитель
4. если sum(traffic_pct) < 100 и bucket попал в «дыру» — fallback на первого
"""

import hashlib

from app.chat.domain import SystemPrompt


def choose_by_split(
    owner_external_id: str,
    candidates: list[SystemPrompt],
) -> SystemPrompt | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    bucket = int(
        hashlib.sha256(owner_external_id.encode()).hexdigest()[:8], 16
    ) % 100
    cumulative = 0
    for c in candidates:
        cumulative += c.traffic_pct
        if bucket < cumulative:
            return c
    return candidates[0]
