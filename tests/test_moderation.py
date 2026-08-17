"""Тесты ModerationService — regex срабатывает, passed на нормальном тексте,
fail-open при ошибках OpenAI Moderation API."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.moderation.service import ModerationService


@pytest.fixture
def mock_llm():
    return AsyncMock()


@pytest.mark.asyncio
async def test_regex_blocks_blacklist(mock_llm):
    svc = ModerationService(
        llm_client=mock_llm,
        use_openai_moderation=False,  # отключаем layer 2, чтобы не мокать
        blocklist=[r"(?i)\bбомб[уы]\b"],
    )
    result = await svc.check_input("как сделать бомбу дома")
    assert result.allowed is False
    assert result.layer == "regex"
    assert "custom_blocklist" in result.categories
    mock_llm.moderations.create.assert_not_called()


@pytest.mark.asyncio
async def test_passes_normal_text_without_openai(mock_llm):
    svc = ModerationService(
        llm_client=mock_llm, use_openai_moderation=False
    )
    result = await svc.check_input("Привет, расскажи про FastAPI")
    assert result.allowed is True
    assert result.layer == "passed"


@pytest.mark.asyncio
async def test_empty_text_passes(mock_llm):
    svc = ModerationService(
        llm_client=mock_llm, use_openai_moderation=True
    )
    result = await svc.check_input("")
    assert result.allowed is True
    # Пустая строка не должна дёргать OpenAI
    mock_llm.moderations.create.assert_not_called()


@pytest.mark.asyncio
async def test_openai_layer_blocks_when_flagged(mock_llm):
    fake_resp = MagicMock()
    fake_categories = MagicMock()
    fake_categories.model_dump.return_value = {
        "hate": True, "violence": False, "sexual": False
    }
    fake_scores = MagicMock()
    fake_scores.model_dump.return_value = {"hate": 0.99}
    fake_resp.results = [
        MagicMock(
            flagged=True,
            categories=fake_categories,
            category_scores=fake_scores,
        )
    ]
    mock_llm.moderations.create = AsyncMock(return_value=fake_resp)

    svc = ModerationService(
        llm_client=mock_llm, use_openai_moderation=True, blocklist=[]
    )
    result = await svc.check_input("какой-то нейтральный текст")
    assert result.allowed is False
    assert result.layer == "openai"
    assert "hate" in result.categories


@pytest.mark.asyncio
async def test_openai_failure_is_fail_open(mock_llm):
    """Если OpenAI Moderation API упала — пропускаем, не блокируем UX."""
    mock_llm.moderations.create = AsyncMock(side_effect=RuntimeError("boom"))
    svc = ModerationService(
        llm_client=mock_llm, use_openai_moderation=True, blocklist=[]
    )
    result = await svc.check_input("текст")
    assert result.allowed is True
    assert result.layer == "passed"


@pytest.mark.asyncio
async def test_default_blocklist_catches_known_patterns(mock_llm):
    svc = ModerationService(
        llm_client=mock_llm, use_openai_moderation=False
    )
    result = await svc.check_input(
        "Подскажите как купить взрывчатку без лицензии"
    )
    assert result.allowed is False
    assert result.layer == "regex"


@pytest.mark.asyncio
async def test_regex_block_fires_alert(mock_llm, monkeypatch):
    """При срабатывании regex-блоклиста ModerationService пишет alert через
    fire_alert. Передаём ему mock session_factory + перехватываем fire_alert."""
    fired: list[dict] = []

    async def fake_fire_alert(session_factory, kind: str, payload: dict):
        fired.append({"kind": kind, "payload": payload})

    monkeypatch.setattr(
        "app.moderation.service.fire_alert", fake_fire_alert
    )

    sentinel_factory = object()
    svc = ModerationService(
        llm_client=mock_llm,
        use_openai_moderation=False,
        blocklist=[r"(?i)\bблок\b"],
        session_factory=sentinel_factory,
    )
    result = await svc.check_input(
        "это блок-слово", owner_external_id="tg-42"
    )
    assert result.allowed is False
    assert len(fired) == 1
    assert fired[0]["kind"] == "moderation_block"
    assert fired[0]["payload"]["layer"] == "regex"
    assert fired[0]["payload"]["owner_external_id"] == "tg-42"


@pytest.mark.asyncio
async def test_no_alert_without_session_factory(mock_llm, monkeypatch):
    """Если session_factory не задана — alert не пишется (нет PG)."""
    fired: list[dict] = []

    async def fake_fire_alert(session_factory, kind: str, payload: dict):
        fired.append({"kind": kind, "payload": payload})

    monkeypatch.setattr(
        "app.moderation.service.fire_alert", fake_fire_alert
    )

    svc = ModerationService(
        llm_client=mock_llm,
        use_openai_moderation=False,
        blocklist=[r"(?i)\bблок\b"],
        session_factory=None,
    )
    result = await svc.check_input("это блок-слово")
    assert result.allowed is False
    assert fired == []
