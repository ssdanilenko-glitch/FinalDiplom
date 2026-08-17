"""Состояние ReAct-агента на LangGraph.

Держим только сериализуемые данные: историю сообщений, счётчик итераций и
накопленные результаты инструментов. SDK-клиенты, сессии и ключи в состояние
не кладём — они приходят в узлы через фабрику при сборке графа.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
