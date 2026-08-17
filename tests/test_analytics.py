"""Тесты аналитики RAG: лог запросов, пробелы в знаниях, поле sources."""

from uuid import uuid4

from app.admin.repository import AdminRepository
from app.chat.domain import ChatMessage
from app.chat.repositories.pg_models import ChatMessageRow, RagQueryRow


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self):
        rows = self._rows

        class _Scalars:
            def all(self):
                return rows

        return _Scalars()


class _FakeSession:
    def __init__(self, rows: list | None = None) -> None:
        self.added: list = []
        self.committed = False
        self.last_stmt = None
        self._rows = rows or []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True

    async def execute(self, stmt):
        self.last_stmt = stmt
        return _FakeResult(self._rows)


def _factory(session: _FakeSession):
    return lambda: session


# --- graceful без PG ---------------------------------------------------------

async def test_knowledge_gaps_empty_without_pg() -> None:
    assert await AdminRepository(None).knowledge_gaps() == []


async def test_log_rag_query_noop_without_pg() -> None:
    await AdminRepository(None).log_rag_query("вопрос", confident=True, top_score=0.5)


# --- log_rag_query пишет нормализованную строку ------------------------------

async def test_log_rag_query_records_normalized_row() -> None:
    session = _FakeSession()
    await AdminRepository(_factory(session)).log_rag_query(
        "  Срок ВОЗВРАТА?  ", confident=False, top_score=0.12
    )
    assert session.committed
    assert len(session.added) == 1
    row = session.added[0]
    assert isinstance(row, RagQueryRow)
    assert row.question_normalized == "срок возврата?"
    assert row.confident is False
    assert row.top_score == 0.12


# --- knowledge_gaps: ORM select, не сырой SQL --------------------------------

async def test_knowledge_gaps_returns_rows_and_builds_grouped_query() -> None:
    session = _FakeSession(rows=["сброс пароля", "оплата картой"])
    gaps = await AdminRepository(_factory(session)).knowledge_gaps(limit=5)
    assert gaps == ["сброс пароля", "оплата картой"]
    sql = str(session.last_stmt).lower()
    assert "rag_queries" in sql
    assert "group by" in sql
    assert "order by" in sql


# --- поле sources у сообщения ассистента -------------------------------------

def test_chat_message_carries_sources() -> None:
    msg = ChatMessage(
        chat_id=uuid4(),
        role="assistant",
        content="Возврат за 14 дней [1].",
        sources=[{"id": 1, "file_name": "policy.pdf"}],
    )
    assert msg.sources[0]["file_name"] == "policy.pdf"


def test_message_row_accepts_sources() -> None:
    row = ChatMessageRow(
        id=uuid4(),
        chat_id=uuid4(),
        role="assistant",
        content="ответ",
        sources=[{"id": 1, "file_name": "a.pdf"}],
    )
    assert row.sources == [{"id": 1, "file_name": "a.pdf"}]
