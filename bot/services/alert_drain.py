"""Фоновая таска: опрашивает backend /chats/admin/alerts и шлёт в админ-чат.

Periodicity: 10 секунд. При отсутствии admin_chat_id просто спит, чтобы
не сжечь CPU/network. Ошибки логирует, но цикл не падает.

При желании можно добавить exponential backoff на ошибках — здесь
для PoC хватает фиксированного интервала.
"""

import asyncio
import json
import logging

log = logging.getLogger(__name__)

POLL_INTERVAL = 10.0


async def drain_alerts(
    bot, backend, admin_chat_id: int | None
) -> None:
    while True:
        if admin_chat_id is None:
            await asyncio.sleep(60)
            continue
        try:
            alerts = await backend.fetch_pending_alerts()
            for a in alerts:
                payload_s = json.dumps(
                    a.get("payload", {}), ensure_ascii=False, indent=2
                )
                text = (
                    f"⚠️ <b>{a['kind']}</b>\n<pre>{payload_s}</pre>"
                )
                try:
                    await bot.send_message(chat_id=admin_chat_id, text=text)
                    await backend.ack_alert(a["id"])
                except Exception as e:
                    log.warning(
                        "alert delivery failed id=%s: %s", a.get("id"), e
                    )
        except Exception as e:
            log.warning("alert drain loop: %s", e)
        await asyncio.sleep(POLL_INTERVAL)
