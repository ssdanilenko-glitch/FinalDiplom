"""RAG-endpoint: ответ по базе знаний с цитатами на источники."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.admin.repository import AdminRepository
from app.deps.providers import RAGServiceDep, SessionFactoryDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RAGSource(BaseModel):
    id: int                       # номер цитаты [1], [2] в тексте ответа
    file_name: str
    page: int | None = None
    score: float
    snippet: str


class RAGAnswer(BaseModel):
    answer: str
    top_score: float
    sources: list[RAGSource]
    confident: bool               # top_score >= rag_score_threshold


@router.post(
    "/query",
    response_model=RAGAnswer,
    summary="Ответ по базе знаний (RAG)",
    description="Ищет релевантные чанки в Qdrant и генерирует ответ строго по контексту.",
    responses={
        200: {"description": "Ответ с источниками"},
        503: {"description": "RAG-индекс недоступен"},
    },
)
async def rag_query(
    req: RAGQuery, rag: RAGServiceDep, session_factory: SessionFactoryDep
) -> RAGAnswer:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG-индекс недоступен")
    result = await rag.answer(req.question)
    # Лог в rag_queries для refusal_rate и пробелов в знаниях; сбой лога не должен
    # ронять ответ пользователю.
    try:
        await AdminRepository(session_factory).log_rag_query(
            req.question, result["confident"], result["top_score"]
        )
    except Exception:
        logger.warning("не записан rag_queries-лог", exc_info=True)
    return RAGAnswer(**result)
