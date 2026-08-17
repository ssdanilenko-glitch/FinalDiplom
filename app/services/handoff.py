"""Handoff: переключение чата между ассистентом и оператором.

status ∈ {'active', 'paused_for_human', 'resolved'}.
'paused_for_human' — ассистент не отвечает, чат ждёт оператора.

Поиск чата:
- set_handoff_status_by_owner — берём ПЕРВЫЙ чат пользователя по interface
  (для telegram у нас по дизайну один чат на owner). Если будут «комнаты»
  типа forum, понадобится явный chat_id.
"""

from uuid import UUID

from sqlalchemy import text


VALID_STATUSES = ("active", "paused_for_human", "resolved")


async def set_handoff_status(
    session_factory, chat_id: UUID, status: str
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(
            f"invalid handoff status: {status}; expected one of {VALID_STATUSES}"
        )
    if session_factory is None:
        return
    async with session_factory() as s:
        await s.execute(
            text("UPDATE chats SET handoff_status = :s WHERE id = :id"),
            {"s": status, "id": chat_id},
        )
        await s.commit()


async def set_handoff_status_by_owner(
    session_factory,
    owner_external_id: str,
    interface: str,
    status: str,
) -> int:
    """UPDATE по owner+interface. Возвращает кол-во затронутых строк."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"invalid handoff status: {status}; expected one of {VALID_STATUSES}"
        )
    if session_factory is None:
        return 0
    async with session_factory() as s:
        result = await s.execute(
            text(
                """
                UPDATE chats SET handoff_status = :s
                WHERE owner_external_id = :o AND interface = :i
                """
            ),
            {"s": status, "o": owner_external_id, "i": interface},
        )
        await s.commit()
        return result.rowcount or 0
