import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock

from app.deps.providers import get_cache, get_llm
from app.main import app


def _make_openai_response(content: str = "мок-ответ", model: str = "gpt-5.4-mini"):
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(content=content),
                finish_reason="stop",
            )
        ],
        model=model,
        usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    return llm


@pytest.fixture
def mock_cache():
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.setex = AsyncMock(return_value=True)
    cache.ping = AsyncMock(return_value=True)
    return cache


@pytest.fixture
async def client(mock_llm, mock_cache):
    # ASGITransport не запускает lifespan — выставляем app.state вручную,
    # чтобы health/ready, использующие request.app.state, тоже работали.
    app.state.llm = mock_llm
    app.state.redis = mock_cache

    app.dependency_overrides[get_llm] = lambda: mock_llm
    app.dependency_overrides[get_cache] = lambda: mock_cache

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
