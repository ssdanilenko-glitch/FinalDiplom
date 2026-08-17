"""Alerter: упрощённая БД-очередь алертов.

Pattern: fire_alert пишет строку в alerts (jsonb payload), бот периодически
drain'ит pending → шлёт в админ-чат → ack. Это даёт at-least-once delivery
без внешнего message broker'а.

Для PoC хватает; в проде стоит подумать про partitioning по created_at
и cleanup acked-строк cron'ом.
"""

import json

from sqlalchemy import text


async def fire_alert(session_factory, kind: str, payload: dict) -> None:
    if session_factory is None:
        return
    async with session_factory() as s:
        await s.execute(
            text(
                """
                INSERT INTO alerts (kind, payload, created_at)
                VALUES (:k, CAST(:p AS jsonb), NOW())
                """
            ),
            {"k": kind, "p": json.dumps(payload)},
        )
        await s.commit()


async def fetch_pending_alerts(session_factory) -> list[dict]:
    if session_factory is None:
        return []
    async with session_factory() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT id, kind, payload FROM alerts
                    WHERE acked_at IS NULL
                    ORDER BY created_at ASC LIMIT 50
                    """
                )
            )
        ).all()
    return [
        {"id": r.id, "kind": r.kind, "payload": r.payload} for r in rows
    ]


async def ack_alert(session_factory, alert_id: int) -> None:
    if session_factory is None:
        return
    async with session_factory() as s:
        await s.execute(
            text("UPDATE alerts SET acked_at = NOW() WHERE id = :id"),
            {"id": alert_id},
        )
        await s.commit()
