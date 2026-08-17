"""Тесты gate'а Phoenix-трейсинга — без тяжёлых зависимостей и сети."""

import app.observability.tracing as tracing
from app.core.config import Settings


def test_tracing_disabled_returns_false() -> None:
    assert tracing.setup_tracing(Settings(phoenix_enabled=False)) is False


def test_tracing_enabled_without_deps_returns_false(monkeypatch) -> None:
    # Флаг включён, но пакеты трейсинга не установлены — find_spec вернёт None,
    # setup_tracing не должен падать, только предупредить и вернуть False.
    monkeypatch.setattr(tracing, "find_spec", lambda name: None)
    assert tracing.setup_tracing(Settings(phoenix_enabled=True)) is False
