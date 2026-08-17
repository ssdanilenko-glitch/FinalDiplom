"""Общие утилиты для скриптов загрузки в Qdrant."""

import json
import uuid
from pathlib import Path

NAMESPACE = uuid.UUID("c0ffee00-0000-0000-0000-c0ffee000000")


def stable_id(source: str, chunk_index: int) -> str:
    """Детерминированный UUID5 на (source, chunk_index) — даёт идемпотентность upsert."""
    return str(uuid.uuid5(NAMESPACE, f"{source}::{chunk_index}"))


def read_jsonl(path: Path) -> list[dict]:
    """Читает JSON Lines, пустые строки пропускает."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
