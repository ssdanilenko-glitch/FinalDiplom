"""Сохраняет Mermaid-схему кастомного графа агента в `docs/agent-graph-custom.mmd`.

Структура графа не зависит от реального API-ключа: модель строится офлайн,
сеть не дёргается. Файл открывается в mermaid.live.

    uv run python -m scripts.visualize_graph
"""

import sys
from pathlib import Path

from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.graph import build_custom_graph  # noqa: E402
from app.agents.tools import multiply  # noqa: E402


def main() -> None:
    model = ChatOpenAI(model="gpt-5.4-mini", temperature=0, api_key="sk-placeholder")
    graph = build_custom_graph(model, [multiply])
    mermaid = graph.get_graph().draw_mermaid()

    out = Path("docs/agent-graph-custom.mmd")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mermaid, encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
