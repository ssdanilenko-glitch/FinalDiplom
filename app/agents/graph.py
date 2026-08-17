"""Сборка ReAct-агента на LangGraph.

Два пути одной и той же логики:
- `build_custom_graph` — `StateGraph` руками: узлы `call_model`, `execute_tool`,
  `force_finish` и детерминированный маршрутизатор с лимитом итераций.
- `build_prebuilt_graph` — тот же ReAct через `langchain.agents.create_agent`.

Обе фабрики принимают модель и список инструментов на этапе сборки — состояние
остаётся чистым, а зависимости управляются в lifespan.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from app.agents.state import AgentState

MAX_ITERATIONS = 6


def build_custom_graph(model: BaseChatModel, tools: list[BaseTool]):
    """Компилирует ReAct-граф из трёх узлов и условного ребра."""
    bound_model = model.bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    async def call_model(state: AgentState) -> dict:
        response = await bound_model.ainvoke(state["messages"])
        return {
            "messages": [response],
            "iteration_count": state["iteration_count"] + 1,
        }

    async def execute_tool(state: AgentState) -> dict:
        last = state["messages"][-1]
        messages: list = []
        results: list[dict] = []
        for call in last.tool_calls:
            if call["name"] not in tool_by_name:
                content = f"error: unknown tool '{call['name']}'"
            else:
                content = str(await tool_by_name[call["name"]].ainvoke(call["args"]))
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            results.append(
                {"name": call["name"], "args": call["args"], "result": content}
            )
        return {"messages": messages, "tool_results": results}

    async def force_finish(state: AgentState) -> dict:
        # Последний AIMessage уже в state — прокидываем его как финал.
        return {}

    def route_after_model(
        state: AgentState,
    ) -> Literal["execute_tool", "force_finish"]:
        if state["iteration_count"] >= MAX_ITERATIONS:
            return "force_finish"
        last = state["messages"][-1]
        return "execute_tool" if getattr(last, "tool_calls", None) else "force_finish"

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {"execute_tool": "execute_tool", "force_finish": "force_finish"},
    )
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("force_finish", END)
    return builder.compile()


def build_prebuilt_graph(
    model: BaseChatModel, tools: list[BaseTool], system_prompt: str
):
    """Тот же ReAct через prebuilt-агента `langchain.agents.create_agent`."""
    from langchain.agents import create_agent

    return create_agent(model=model, tools=tools, system_prompt=system_prompt)
