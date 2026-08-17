"""Atomic rate-limit counter поверх Postgres.

Идея: PRIMARY KEY (owner, kind, bucket) + INSERT…ON CONFLICT DO UPDATE
гарантируют единственный счётчик на минуту/тип/пользователя. Без блокировок,
без race-condition'ов между параллельными воркерами.

Buckets — minute-precision строки `YYYY-MM-DDTHH:MM`. Старые бакеты живут
в таблице, но не мешают — можно периодически чистить (см. cron / partial index).
"""

from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def increment_and_check(
    session: AsyncSession,
    owner_id: str,
    kind: str,
    limit: int,
) -> tuple[bool, int]:
    """Атомарный инкремент. Возвращает (allowed, current_count)."""
    bucket = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
    stmt = text(
        """
        INSERT INTO rate_limits (owner_external_id, kind, bucket, count)
        VALUES (:o, :k, :b, 1)
        ON CONFLICT (owner_external_id, kind, bucket)
        DO UPDATE SET count = rate_limits.count + 1
        RETURNING count
        """
    )
    result = await session.execute(
        stmt, {"o": owner_id, "k": kind, "b": bucket}
    )
    count = result.scalar() or 0
    await session.commit()
    return count <= limit, count
