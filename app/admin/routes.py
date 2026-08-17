"""Admin-роуты /chats/admin/*.

Защита: один общий X-Admin-Token в Header (см. settings.admin_token).
Это не enterprise-grade auth — учебная заглушка вместо JWT/OAuth. Чтобы
включить в проде, как минимум: ротация токена + аудит-лог + IP allowlist
на ingress'е.

Endpoint'ы:
- GET  /chats/admin/stats          — агрегаты за окно window_hours
- POST /chats/admin/broadcast      — массовая отправка через bot /notify
- GET  /chats/admin/export         — экспорт сообщений с PII-маскированием
- POST /chats/admin/handoff        — поставить чат на паузу / снять
- GET  /chats/admin/alerts         — pending-алерты (для бот-drain'а)
- POST /chats/admin/alerts/{id}/ack — пометить алерт обработанным
"""

import logging
from datetime import datetime
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException

from app.admin.repository import AdminRepository
from app.admin.schemas import (
    AlertOut,
    BroadcastIn,
    BroadcastResult,
    ExportResult,
    HandoffIn,
    StatsOut,
)
from app.core.config import get_settings
from app.deps.providers import SessionFactoryDep, SettingsDep
from app.services.alerter import ack_alert, fetch_pending_alerts, fire_alert
from app.services.broadcaster import broadcast as do_broadcast
from app.services.handoff import set_handoff_status_by_owner
from app.services.notifier import notify_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chats/admin", tags=["admin"])


async def require_admin(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    settings = get_settings()
    expected = settings.admin_token.get_secret_value()
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=403, detail="forbidden")


@router.get(
    "/stats",
    dependencies=[Depends(require_admin)],
    response_model=StatsOut,
)
async def stats(
    session_factory: SessionFactoryDep, window_hours: int = 24
) -> StatsOut:
    repo = AdminRepository(session_factory)
    return await repo.compute_stats(window_hours=window_hours)


@router.post(
    "/broadcast",
    dependencies=[Depends(require_admin)],
    response_model=BroadcastResult,
)
async def broadcast_route(
    body: BroadcastIn,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
) -> BroadcastResult:
    owner_ids = list(body.owner_ids or [])
    if not owner_ids and body.interface:
        repo = AdminRepository(session_factory)
        owner_ids = await repo.list_owner_ids_by_interface(body.interface)
    if not owner_ids:
        raise HTTPException(
            status_code=400,
            detail="provide non-empty `owner_ids` or `interface` with at least one chat",
        )
    result = await do_broadcast(
        text=body.text,
        owner_ids=owner_ids,
        bot_url=settings.bot_url,
        internal_token=settings.internal_token.get_secret_value(),
    )
    return BroadcastResult(**result)


@router.get(
    "/export",
    dependencies=[Depends(require_admin)],
    response_model=ExportResult,
)
async def export(
    session_factory: SessionFactoryDep,
    after: datetime | None = None,
    limit: int = 1000,
) -> ExportResult:
    repo = AdminRepository(session_factory)
    return await repo.export_messages(after=after, limit=limit)


@router.post(
    "/handoff",
    dependencies=[Depends(require_admin)],
)
async def handoff(
    body: HandoffIn,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
) -> dict:
    affected = await set_handoff_status_by_owner(
        session_factory,
        owner_external_id=body.owner_external_id,
        interface=body.interface,
        status=body.status,
    )
    if affected > 0 and body.status == "paused_for_human":
        # Алерт в админ-чат: «нужен оператор для такого-то».
        await fire_alert(
            session_factory,
            kind="handoff_requested",
            payload={
                "owner_external_id": body.owner_external_id,
                "interface": body.interface,
            },
        )
        # Пользователю в Telegram — короткое подтверждение, чтобы он не сидел
        # в пустоте, пока подключается оператор. Если notify падает — лог,
        # 200-ответ всё равно отдаём (handoff в БД зафиксирован).
        try:
            await notify_user(
                chat_id_tg=int(body.owner_external_id),
                text="Передаю запрос оператору. Ожидайте — подключим в ближайшее время.",
                bot_url=settings.bot_url,
                internal_token=settings.internal_token.get_secret_value(),
            )
        except (ValueError, httpx.HTTPError) as exc:
            log.warning("handoff notify failed: %s", exc)
    return {"status": "ok", "updated": affected}


@router.get(
    "/alerts",
    dependencies=[Depends(require_admin)],
    response_model=list[AlertOut],
)
async def list_alerts(session_factory: SessionFactoryDep) -> list[AlertOut]:
    items = await fetch_pending_alerts(session_factory)
    return [AlertOut(**a) for a in items]


@router.post(
    "/alerts/{alert_id}/ack",
    dependencies=[Depends(require_admin)],
)
async def ack(alert_id: int, session_factory: SessionFactoryDep) -> dict:
    await ack_alert(session_factory, alert_id)
    return {"status": "ok"}
