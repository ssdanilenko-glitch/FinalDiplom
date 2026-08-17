"""Broadcaster: серийная рассылка в N пользователей через bot /notify.

THROTTLE ~25 msg/sec — безопасно ниже Telegram-лимита 30/sec на бота.
Логирует ошибки на каждого получателя, но не прерывается — счётчики
sent/failed возвращаются вызывающему.

Для большого N (>1k) лучше выводить в фон-таску и писать прогресс в
таблицу broadcasts. Здесь — простой синхронный вариант для учебного MVP.
"""

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

THROTTLE = 0.04  # ~25 msg/sec


async def broadcast(
    text: str,
    owner_ids: list[int],
    bot_url: str,
    internal_token: str,
) -> dict:
    sent = failed = 0
    async with httpx.AsyncClient(timeout=5.0) as c:
        for owner_id in owner_ids:
            try:
                r = await c.post(
                    f"{bot_url}/notify",
                    json={"chat_id": owner_id, "text": text},
                    headers={"X-Internal-Token": internal_token},
                )
                r.raise_for_status()
                sent += 1
            except httpx.HTTPError as e:
                failed += 1
                log.warning("broadcast: failed for %s: %s", owner_id, e)
            await asyncio.sleep(THROTTLE)
    return {"sent": sent, "failed": failed}
