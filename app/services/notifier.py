"""Notifier: backend → bot/notify.

Тонкий клиент: POST /notify с X-Internal-Token. Используется для одиночных
push'ей (handoff-уведомления оператору, тех-алерты в админ-чат).
"""

import httpx


async def notify_user(
    chat_id_tg: int,
    text: str,
    bot_url: str,
    internal_token: str,
) -> None:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(
            f"{bot_url}/notify",
            json={"chat_id": chat_id_tg, "text": text},
            headers={"X-Internal-Token": internal_token},
        )
        r.raise_for_status()
