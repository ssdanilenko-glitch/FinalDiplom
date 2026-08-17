"""ModerationService — двухслойный каскад модерации.

Layer 1: локальный regex-blocklist (моментальная проверка, без сетевых вызовов).
Layer 2: OpenAI Moderation API (omni-moderation-latest) — категории harassment,
hate, sexual, violence и т.п. Fail-open: при ошибке API пропускаем, чтобы не
ронять UX из-за upstream-проблем.

DEFAULT_BLOCKLIST — минимальная учебная заглушка. В production вынести в
YAML/БД, чтобы редактировать без редеплоя.

На любой блок (regex или OpenAI) сервис пишет строку в таблицу `alerts` через
`fire_alert(...)` — этот алерт потом подхватывает bot-drain и кладёт в админ-чат.
Если `session_factory` не передан (нет Postgres) — алерт просто не пишется.
"""

import logging
import re

from openai import AsyncOpenAI

from app.moderation.domain import ModerationResult
from app.services.alerter import fire_alert

log = logging.getLogger(__name__)


DEFAULT_BLOCKLIST: list[str] = [
    r"(?i)\b(как\s+(сделать|собрать|изготовить|купить))\s+(бомб[уы]|взрывчатк)",
]


class ModerationService:
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        use_openai_moderation: bool = True,
        blocklist: list[str] | None = None,
        session_factory=None,
    ):
        self.llm = llm_client
        self.use_openai = use_openai_moderation
        patterns = blocklist if blocklist is not None else DEFAULT_BLOCKLIST
        self._patterns = [re.compile(p) for p in patterns]
        self.session_factory = session_factory

    async def check_input(
        self, text: str, owner_external_id: str | None = None
    ) -> ModerationResult:
        if not text:
            return ModerationResult(allowed=True, layer="passed")

        # Layer 1: regex blocklist
        for pat in self._patterns:
            if pat.search(text):
                result = ModerationResult(
                    allowed=False,
                    categories=["custom_blocklist"],
                    layer="regex",
                )
                await self._raise_alert(result, owner_external_id)
                return result

        # Layer 2: OpenAI Moderation API (fail-open)
        if self.use_openai:
            try:
                resp = await self.llm.moderations.create(
                    model="omni-moderation-latest", input=text,
                )
                r = resp.results[0]
                if r.flagged:
                    flagged_cats = [
                        c for c, on in r.categories.model_dump().items() if on
                    ]
                    result = ModerationResult(
                        allowed=False,
                        categories=flagged_cats,
                        layer="openai",
                        scores=r.category_scores.model_dump(),
                    )
                    await self._raise_alert(result, owner_external_id)
                    return result
            except Exception as exc:
                log.warning("moderation API failed: %s — fail-open", exc)

        return ModerationResult(allowed=True, layer="passed")

    async def _raise_alert(
        self, result: ModerationResult, owner_external_id: str | None
    ) -> None:
        if self.session_factory is None:
            return
        try:
            await fire_alert(
                self.session_factory,
                kind="moderation_block",
                payload={
                    "layer": result.layer,
                    "categories": result.categories,
                    "owner_external_id": owner_external_id,
                },
            )
        except Exception as exc:
            log.warning("fire_alert(moderation_block) failed: %s", exc)
