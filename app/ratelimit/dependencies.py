"""FastAPI Depends-factory для rate-limit на роуты.

Берёт `X-Owner-External-Id` из header'а (бот обязан передавать),
делает атомарный инкремент в Postgres и при превышении лимита кидает
HTTPException 429 с Retry-After.

Если Postgres не подключен (session_factory is None) — rate-limit
молча отключается. Это позволяет тестам/локальной разработке без БД
не падать на каждом запросе.
"""

from datetime import UTC, datetime
from typing import Annotated, Callable

from fastapi import Header, HTTPException, Request, Response

from app.ratelimit.service import increment_and_check


def enforce_rate_limit(kind: str, limit: int) -> Callable:
    """Фабрика Depends. Возвращает coroutine для FastAPI."""

    async def _dep(
        request: Request,
        response: Response,
        x_owner_external_id: Annotated[
            str | None, Header(alias="X-Owner-External-Id")
        ] = None,
    ) -> None:
        if not x_owner_external_id:
            # Без идентификатора владельца rate-limit невозможен;
            # пропускаем (учебный режим). В проде стоит требовать.
            return
        session_factory = getattr(
            request.app.state, "session_factory", None
        )
        if session_factory is None:
            return
        async with session_factory() as session:
            allowed, _count = await increment_and_check(
                session, x_owner_external_id, kind, limit,
            )
        if not allowed:
            retry_after = 60 - datetime.now(UTC).second
            response.headers["Retry-After"] = str(retry_after)
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "rate_limit",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

    return _dep
