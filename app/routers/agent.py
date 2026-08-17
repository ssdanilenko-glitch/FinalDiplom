"""Ручки агентного слоя: прогон персистентного ReAct-графа с HIL.

- `POST /agent/chat` — один шаг диалога; если агент дошёл до опасного действия,
  вернётся `status="interrupted"` с payload для подтверждения.
- `POST /agent/resume` — возобновление после подтверждения человеком.
- `POST /agent/stream` — SSE-поток прогресса по узлам и токенов LLM.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.deps.providers import AgentGraphDep

router = APIRouter(prefix="/agent", tags=["agent"])


def _config(thread_id: str, user_role: str = "write-with-approve") -> dict:
    return {"configurable": {"thread_id": thread_id, "user_role": user_role}}


def _initial_state(message: str) -> dict:
    return {
        "messages": [HumanMessage(message)],
        "iteration_count": 0,
        "tool_results": [],
        "draft": None,
        "sent": False,
    }


class AgentChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class AgentChatResponse(BaseModel):
    status: str  # "done" | "interrupted"
    thread_id: str
    answer: str | None = None
    tool_results: list[dict] = []
    interrupt: dict | None = None


def _to_response(result: dict, thread_id: str) -> AgentChatResponse:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return AgentChatResponse(
            status="interrupted",
            thread_id=thread_id,
            interrupt=payload,
            tool_results=result.get("tool_results", []),
        )
    final = result["messages"][-1]
    return AgentChatResponse(
        status="done",
        thread_id=thread_id,
        answer=final.content or "",
        tool_results=result.get("tool_results", []),
    )


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(req: AgentChatRequest, graph: AgentGraphDep) -> AgentChatResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")
    result = await graph.ainvoke(_initial_state(req.message), _config(req.thread_id))
    return _to_response(result, req.thread_id)


class AgentResumeRequest(BaseModel):
    thread_id: str
    decision: bool | str = True


@router.post("/resume", response_model=AgentChatResponse)
async def agent_resume(
    req: AgentResumeRequest, graph: AgentGraphDep
) -> AgentChatResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")
    result = await graph.ainvoke(Command(resume=req.decision), _config(req.thread_id))
    return _to_response(result, req.thread_id)


class AgentStreamRequest(BaseModel):
    thread_id: str
    input: dict | None = None  # старт: {"messages": [...]}
    resume: bool | str | None = None  # возобновление после interrupt


def _format_event(stream_type: str, payload: Any) -> dict | None:
    if stream_type == "updates":
        if isinstance(payload, dict) and "__interrupt__" in payload:
            interrupts = payload["__interrupt__"]
            value = interrupts[0].value if interrupts else {}
            return {"type": "interrupt", "payload": value}
        return {"type": "update", "nodes": list(payload.keys())}
    if stream_type == "messages":
        chunk, _meta = payload
        text = getattr(chunk, "content", "")
        return {"type": "token", "text": text} if text else None
    return None


@router.post("/stream")
async def agent_stream(
    req: AgentStreamRequest, graph: AgentGraphDep
) -> StreamingResponse:
    if graph is None:
        raise HTTPException(status_code=503, detail="агентный граф не инициализирован")

    if req.resume is not None:
        graph_input: Any = Command(resume=req.resume)
    elif req.input is not None:
        graph_input = {
            "iteration_count": 0,
            "tool_results": [],
            "draft": None,
            "sent": False,
            **req.input,
        }
    else:
        raise HTTPException(status_code=422, detail="нужен input или resume")

    config = _config(req.thread_id)

    async def event_source() -> AsyncIterator[str]:
        async for stream_type, payload in graph.astream(
            graph_input, config, stream_mode=["updates", "messages"]
        ):
            event = _format_event(stream_type, payload)
            if event is not None:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(event_source(), media_type="text/event-stream")
