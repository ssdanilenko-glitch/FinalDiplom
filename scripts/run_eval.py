"""Прогон RAGAS-метрик по golden dataset через текущий RAG (группа eval).

Считает шесть метрик на строку: faithfulness, answer_relevancy, context_precision,
context_recall, factual_correctness (collections) и has_citation (@discrete_metric).
По датасету идём конкурентно через asyncio.gather, агрегаты и per-row собираем в
pandas.DataFrame и пишем в tests/eval/results/{timestamp}_{label}.csv — это audit
log, по которому строится временной ряд метрик.

Судья (claude-sonnet-4-6 по умолчанию) отделён от production-LLM в /rag/query.

Запуск:
    uv run --extra eval python scripts/run_eval.py \
        --golden tests/eval/golden_dataset.json --label baseline
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.eval.metrics import build_judge, build_metrics, eval_row, make_has_citation  # noqa: E402
from app.services.rag import RAGService  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS eval по golden dataset")
    parser.add_argument("--golden", default="tests/eval/golden_dataset.json")
    parser.add_argument(
        "--label", default="baseline", help="метка конфигурации: baseline, chunk_1024 ..."
    )
    parser.add_argument("--out-dir", default="tests/eval/results")
    args = parser.parse_args()

    settings = get_settings()
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    print(f"Загружено {len(golden)} пар из {args.golden}")

    rag = RAGService(settings)
    await asyncio.to_thread(rag.build)

    judge, embeddings = build_judge(settings)
    metrics = build_metrics(judge, embeddings)
    has_citation = make_has_citation(judge)

    try:
        rows = await asyncio.gather(
            *[eval_row(rag, row, metrics, has_citation) for row in golden]
        )
    finally:
        await rag.close()

    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = Path(args.out_dir) / f"{stamp}_{args.label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nРезультат: {out}")
    print("\nАгрегаты:")
    print(df.mean(numeric_only=True))
    print("\nТоп худших по faithfulness:")
    print(df.sort_values("faithfulness").head()[["user_input", "faithfulness"]])


if __name__ == "__main__":
    asyncio.run(main())
