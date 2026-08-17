"""Тесты increment_and_check + enforce_rate_limit dependency-фабрики."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.ratelimit.dependencies import enforce_rate_limit
from app.ratelimit.service import increment_and_check


def _make_session_with_count(count: int):
    """Сессия, возвращающая заданный count на execute()/scalar()."""
    session = AsyncMock()
    exec_result = MagicMock()
    exec_result.scalar = MagicMock(return_value=count)
    session.execute = AsyncMock(return_value=exec_result)
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_increment_within_limit_allowed():
    session = _make_session_with_count(3)
    allowed, count = await increment_and_check(
        session, owner_id="u1", kind="message", limit=5
    )
    assert allowed is True
    assert count == 3
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_increment_over_limit_blocked():
    session = _make_session_with_count(11)
    allowed, count = await increment_and_check(
        session, owner_id="u1", kind="message", limit=10
    )
    assert allowed is False
    assert count == 11


@pytest.mark.asyncio
async def test_dependency_no_session_factory_allows():
    """Если PG не подключен (session_factory is None) — rate-limit пропускает."""
    request = MagicMock()
    request.app.state.session_factory = None
    response = MagicMock()
    dep = enforce_rate_limit("message", 5)
    # Не должно бросать
    await dep(request=request, response=response, x_owner_external_id="u1")


@pytest.mark.asyncio
async def test_dependency_no_owner_header_allows():
    """Без X-Owner-External-Id rate-limit пропускает (учебный режим)."""
    request = MagicMock()
    response = MagicMock()
    dep = enforce_rate_limit("message", 5)
    await dep(request=request, response=response, x_owner_external_id=None)


@pytest.mark.asyncio
async def test_dependency_raises_429_when_over():
    """Превышение лимита → HTTPException 429."""
    request = MagicMock()
    session = _make_session_with_count(11)

    class _CtxMgr:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *a):
            return False

    session_factory = MagicMock(return_value=_CtxMgr())
    request.app.state.session_factory = session_factory

    response = MagicMock()
    response.headers = {}

    dep = enforce_rate_limit("message", 10)
    with pytest.raises(HTTPException) as exc_info:
        await dep(
            request=request, response=response, x_owner_external_id="u1"
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["code"] == "rate_limit"
    assert "retry_after" in exc_info.value.detail


@pytest.mark.asyncio
async def test_dependency_passes_when_under():
    request = MagicMock()
    session = _make_session_with_count(3)

    class _CtxMgr:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *a):
            return False

    session_factory = MagicMock(return_value=_CtxMgr())
    request.app.state.session_factory = session_factory

    response = MagicMock()
    response.headers = {}

    dep = enforce_rate_limit("message", 10)
    await dep(request=request, response=response, x_owner_external_id="u1")
