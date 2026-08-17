from fastapi import APIRouter

from app.deps.providers import SettingsDep
from app.schemas.models import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])

CATALOG: dict[str, ModelInfo] = {
    "gpt-5.4-mini": ModelInfo(
        id="gpt-5.4-mini",
        provider="openai",
        input_per_1m=0.15,
        output_per_1m=0.60,
        context_window=128_000,
    ),
    "gpt-5.4": ModelInfo(
        id="gpt-5.4",
        provider="openai",
        input_per_1m=2.50,
        output_per_1m=10.00,
        context_window=128_000,
    ),
}


@router.get("", response_model=list[ModelInfo])
async def list_models(settings: SettingsDep) -> list[ModelInfo]:
    return list(CATALOG.values())
