"""AdminRepository: агрегации по таблицам chat_messages / chats / message_feedback.

PII-маскирование применяется только к экспорту, не к сторонним read-API.
Внутренние логи/UI продолжают видеть исходный контент (это полезно для
дебага). Если включить маскировку в /list_messages — пользователь увидит
свои же сообщения с [EMAIL] и не поймёт, что произошло.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select, text

from app.admin.schemas import ExportItem, ExportResult, StatsOut
from app.chat.repositories.pg_models import RagQueryRow
from app.observability.pii import mask_pii


class AdminRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def compute_stats(self, window_hours: int = 24) -> StatsOut:
        """Активность за окно времени. `deleted_at` намеренно НЕ фильтруем:
        soft-delete нужен LLM-контексту (после /clear прошлые сообщения не
        подтягиваются в prompt), но факт активности юзера в окне остался —
        статистика должна это видеть.
        """
        if self.session_factory is None:
            return StatsOut(total_messages=0, active_users=0)
        since = datetime.now(UTC) - timedelta(hours=window_hours)
        async with self.session_factory() as s:
            total = await s.scalar(
                text(
                    "SELECT COUNT(*) FROM chat_messages WHERE created_at >= :since"
                ),
                {"since": since},
            )
            active = await s.scalar(
                text(
                    """
                    SELECT COUNT(DISTINCT c.owner_external_id)
                    FROM chats c
                    JOIN chat_messages cm ON cm.chat_id = c.id
                    WHERE cm.created_at >= :since
                    """
                ),
                {"since": since},
            )
            fb = await s.execute(
                text(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE value='up') AS up,
                        COUNT(*) FILTER (WHERE value='down') AS down
                    FROM message_feedback
                    WHERE created_at >= :since
                    """
                ),
                {"since": since},
            )
            row = fb.first()
            up = (row.up if row else 0) or 0
            down = (row.down if row else 0) or 0
            total_fb = up + down
            ratio = up / total_fb if total_fb > 0 else 0.0
            negative_rate = down / total_fb if total_fb > 0 else 0.0

            rag = (
                await s.execute(
                    text(
                        """
                        SELECT
                            COUNT(*) AS total,
                            COUNT(*) FILTER (WHERE confident = false) AS refused
                        FROM rag_queries
                        WHERE created_at >= :since
                        """
                    ),
                    {"since": since},
                )
            ).first()
            rag_total = (rag.total if rag else 0) or 0
            refused = (rag.refused if rag else 0) or 0
            refusal_rate = refused / rag_total if rag_total > 0 else 0.0

        gaps = await self.knowledge_gaps(limit=10)
        return StatsOut(
            total_messages=total or 0,
            active_users=active or 0,
            feedback_ratio=ratio,
            refusal_rate=refusal_rate,
            negative_feedback_rate=negative_rate,
            knowledge_gaps=gaps,
        )

    async def log_rag_query(
        self, question: str, confident: bool, top_score: float
    ) -> None:
        """Пишет строку лога RAG-запроса. Нормализуем вопрос для группировки пробелов."""
        if self.session_factory is None:
            return
        async with self.session_factory() as s:
            s.add(
                RagQueryRow(
                    question_normalized=question.strip().lower()[:500],
                    confident=confident,
                    top_score=top_score,
                )
            )
            await s.commit()

    async def knowledge_gaps(self, limit: int = 10) -> list[str]:
        """Топ вопросов без уверенного ответа — что добавить в базу знаний.

        Тот же `select().group_by()`, что и весь чат-репозиторий, без сырого SQL.
        """
        if self.session_factory is None:
            return []
        stmt = (
            select(RagQueryRow.question_normalized)
            .where(RagQueryRow.confident.is_(False))
            .group_by(RagQueryRow.question_normalized)
            .order_by(func.count().desc())
            .limit(limit)
        )
        async with self.session_factory() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return list(rows)

    async def list_owner_ids_by_interface(self, interface: str) -> list[int]:
        """Возвращает уникальные owner_external_id для рассылок.

        Для telegram owner_external_id — это str(message.chat.id), поэтому
        корректно интерпретируется как int. Owner'ы, чьи id не приводятся
        к int (другой интерфейс с не-числовым id) — пропускаются.
        """
        if self.session_factory is None:
            return []
        async with self.session_factory() as s:
            rows = (
                await s.execute(
                    text(
                        """
                        SELECT DISTINCT owner_external_id
                        FROM chats
                        WHERE interface = :i
                        """
                    ),
                    {"i": interface},
                )
            ).all()
        out: list[int] = []
        for r in rows:
            try:
                out.append(int(r.owner_external_id))
            except (TypeError, ValueError):
                continue
        return out

    async def export_messages(
        self, after: datetime | None, limit: int
    ) -> ExportResult:
        if self.session_factory is None:
            return ExportResult(items=[], next_after=None)
        async with self.session_factory() as s:
            stmt = text(
                """
                SELECT id, chat_id, role, content, created_at
                FROM chat_messages
                WHERE deleted_at IS NULL
                  AND (CAST(:after AS TIMESTAMPTZ) IS NULL
                       OR created_at > CAST(:after AS TIMESTAMPTZ))
                ORDER BY created_at ASC
                LIMIT :limit
                """
            )
            rows = (
                await s.execute(stmt, {"after": after, "limit": limit})
            ).all()
        items = [
            ExportItem(
                id=str(r.id),
                chat_id=str(r.chat_id),
                role=r.role,
                content=mask_pii(r.content),
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]
        return ExportResult(
            items=items,
            next_after=rows[-1].created_at if rows else None,
        )
