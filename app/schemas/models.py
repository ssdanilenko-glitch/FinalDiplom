from typing import Literal

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    provider: Literal["openai", "ollama", "anthropic"] = "openai"
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    context_window: int | None = None
