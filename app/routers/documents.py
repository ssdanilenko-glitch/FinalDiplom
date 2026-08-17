"""Офлайн-ручки индексации: загрузка документа и переиндексация корпуса.

Тяжёлая работа (парсинг + эмбеддинг) уходит в `BackgroundTasks` и не блокирует
запрос: ответ 202 Accepted отдаётся сразу со статусом «queued». Для продакшена
вместо `BackgroundTasks` — отдельный воркер на Celery/RQ/Dramatiq.
"""

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.deps.providers import IngestionServiceDep, SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class ReindexRequest(BaseModel):
    mode: Literal["full", "incremental", "files"] = "incremental"
    files: list[str] = Field(default_factory=list)


class QueuedResponse(BaseModel):
    status: Literal["queued"] = "queued"
    detail: str


@router.post(
    "/upload",
    status_code=202,
    response_model=QueuedResponse,
    summary="Загрузить документ в базу знаний",
    description="Сохраняет файл в корпус и запускает индексацию в фоне.",
)
async def upload_document(
    file: UploadFile,
    background: BackgroundTasks,
    ingestion: IngestionServiceDep,
    settings: SettingsDep,
) -> QueuedResponse:
    if ingestion is None:
        raise HTTPException(status_code=503, detail="индексатор недоступен")
    if not file.filename:
        raise HTTPException(status_code=422, detail="имя файла обязательно")
    upload_dir = Path(settings.rag_data_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Path(...).name отбрасывает любые ../ — защита от обхода каталога.
    target = upload_dir / Path(file.filename).name
    target.write_bytes(await file.read())
    background.add_task(ingestion.run_for_file, target)
    return QueuedResponse(detail=f"{target.name} принят, индексация в фоне")


@router.post(
    "/reindex",
    status_code=202,
    response_model=QueuedResponse,
    summary="Переиндексировать корпус",
    description="full — вычистить и заново; incremental — по хешам docstore; files — точечно.",
)
async def reindex(
    req: ReindexRequest,
    background: BackgroundTasks,
    ingestion: IngestionServiceDep,
) -> QueuedResponse:
    if ingestion is None:
        raise HTTPException(status_code=503, detail="индексатор недоступен")
    if req.mode == "files":
        if not req.files:
            raise HTTPException(
                status_code=422, detail="mode=files: список files не должен быть пустым"
            )
        background.add_task(ingestion.ingest_files, req.files)
    elif req.mode == "full":
        background.add_task(ingestion.reindex_all)
    else:
        background.add_task(ingestion.ingest_all)
    return QueuedResponse(detail=f"режим {req.mode}, индексация в фоне")
