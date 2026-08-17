"""Батчевый клиент embeddings поверх OpenAI."""


import asyncio
import logging
from collections.abc import Sequence

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingsClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Возвращает по вектору на каждый text, в том же порядке."""
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i : i + self._batch_size])
            resp = await self._client.embeddings.create(model=self._model, input=batch)
            out.extend([item.embedding for item in resp.data])
        return out

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]


async def _smoke() -> None:
    import os

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embed = EmbeddingsClient(client)
    vec = await embed.embed_one("Привет, мир")
    print(f"len(vec)={len(vec)}, first 5 dims: {vec[:5]}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(_smoke())
