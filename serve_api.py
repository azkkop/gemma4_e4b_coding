#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_api.py
------------
Cursor-like 로컬 코딩 IDE 백엔드 API.

핵심:
  - 채팅만 아님. 파일/폴더/터미널/검색/웹 등 **도구 전부 HTTP로 실행**
  - 에이전트 모드: 모델이 도구를 고르고 서버가 워크스페이스에서 실행
  - 클라이언트 모드: 모델이 tool_calls 반환 → 클라이언트가 /v1/tools/call

엔드포인트:
  GET  /health
  GET  /v1/models
  GET  /v1/tools
  POST /v1/tools/call              # 도구 1회 실행
  POST /v1/chat/completions        # 채팅 또는 agent/tool_calls
  POST /v1/ide/chat                # Cursor형 통합 (도구 실행 포함, SSE 가능)
  POST /v1/agent/run[/stream]      # 작업 한 줄 → 에이전트
  POST /v1/agent/approve

실행:
  ollama serve
  python serve_api.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from hf_env import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SYSTEM = (
    "당신은 Python, Java, C언어를 전문으로 다루는 시니어 코딩 AI입니다. "
    "코드 작성·리팩토링·디버깅 요청에 대해 정확한 코드와 함께 "
    "한국어로 핵심 원리와 주의사항을 명확히 설명합니다."
)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL_NAME = os.environ.get("API_MODEL_NAME", "gemma4-e4b-coding")
AGENT_MODEL_NAME = os.environ.get("API_AGENT_MODEL_NAME", "gemma4-e4b-coding-agent")
VISION_MODEL = os.environ.get("VISION_MODEL", "gemma4:e4b")
API_KEY = os.environ.get("API_KEY", "")
HOST = os.environ.get("API_HOST", "127.0.0.1")
PORT = int(os.environ.get("API_PORT", "8000"))
DEFAULT_APPROVAL = os.environ.get("AGENT_APPROVAL_MODE", "auto")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = 0.2
    top_p: float | None = 0.9
    max_tokens: int | None = Field(default=2048, alias="max_tokens")
    stream: bool = False
    stop: list[str] | str | None = None
    # Cursor-like extensions
    agent: bool | None = None
    execute_tools: bool | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    workspace: str | None = None
    approval_mode: Literal["auto", "ask", "readonly"] | None = None
    allow_writes: bool = True
    allow_terminal: bool = True
    allow_web: bool = True
    max_rounds: int = 12

    model_config = {"populate_by_name": True}


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    workspace: str | None = None
    allow_writes: bool = True
    allow_terminal: bool = True
    allow_web: bool = True


class AgentAttach(BaseModel):
    filename: str
    content_base64: str


class IdeChatRequest(BaseModel):
    """Cursor형 통합: 대화 + 서버 측 도구 실행."""

    messages: list[ChatMessage]
    workspace: str | None = None
    model: str | None = None
    vision_model: str | None = None
    stream: bool = False
    max_rounds: int = 12
    temperature: float = 0.2
    num_predict: int = 2048
    approval_mode: Literal["auto", "ask", "readonly"] | None = None
    allow_writes: bool = True
    allow_terminal: bool = True
    allow_web: bool = True
    attach: list[AgentAttach] = Field(default_factory=list)
    attach_paths: list[str] = Field(default_factory=list)
    execute_tools: bool = True


class AgentRunRequest(BaseModel):
    task: str
    workspace: str | None = None
    model: str | None = None
    vision_model: str | None = None
    max_rounds: int = 12
    temperature: float = 0.2
    num_predict: int = 2048
    approval_mode: Literal["auto", "ask", "readonly"] | None = None
    allow_writes: bool = True
    allow_terminal: bool = True
    allow_web: bool = True
    attach: list[AgentAttach] = Field(default_factory=list)
    attach_paths: list[str] = Field(default_factory=list)
    stream: bool = False


class AgentApproveRequest(BaseModel):
    run_id: str
    approved: bool = True


app = FastAPI(
    title="Gemma4 Local Cursor API",
    version="2.0.0",
    description=(
        "로컬 Cursor형 코딩 백엔드: 채팅 + 전체 도구 API + 에이전트 루프 "
        "(파일/폴더 생성·수정, 터미널, 검색, 웹, 비전)"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("API_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth / helpers
# ---------------------------------------------------------------------------


def _check_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _workspace_header(x_workspace: str | None, body_workspace: str | None) -> str | None:
    return body_workspace or x_workspace


def _resolve_ws(path: str | None):
    from agent.workspace import WorkspaceError, resolve_workspace

    try:
        return resolve_workspace(path)
    except WorkspaceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _registry(workspace: Path, *, allow_web: bool = True, vision_chat=None):
    from agent.tools import ToolRegistry

    return ToolRegistry(workspace, vision_chat=vision_chat, enable_web=allow_web)


def _guard_tool(
    name: str,
    *,
    allow_writes: bool,
    allow_terminal: bool,
) -> None:
    from agent.loop import RISKY_SHELL_TOOLS, RISKY_WRITE_TOOLS

    if name in RISKY_WRITE_TOOLS and not allow_writes:
        raise HTTPException(status_code=403, detail=f"쓰기 도구 거부: {name}")
    if name in RISKY_SHELL_TOOLS and not allow_terminal:
        raise HTTPException(status_code=403, detail=f"터미널 도구 거부: {name}")


def _extract_ollama_content(data: dict[str, Any]) -> str:
    msg = data.get("message") or {}
    content = msg.get("content")
    if content is None or content == "":
        content = msg.get("thinking") or ""
    if not content:
        content = data.get("response") or ""
    return str(content)


def _normalize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        out.append({"role": "system", "content": DEFAULT_SYSTEM})
    for m in messages:
        content = m.content if m.content is not None else ""
        if m.role == "tool":
            label = m.name or "tool"
            out.append(
                {
                    "role": "user",
                    "content": f"[tool result:{label} id={m.tool_call_id or ''}]\n{content}",
                }
            )
        else:
            item: dict[str, Any] = {"role": m.role, "content": content}
            if m.tool_calls:
                item["tool_calls"] = m.tool_calls
            out.append(item)
    return out


def _messages_to_task(messages: list[ChatMessage]) -> str:
    """대화 히스토리를 에이전트 태스크 문자열로."""
    parts: list[str] = []
    for m in messages:
        if m.role == "system":
            continue
        c = (m.content or "").strip()
        if not c and not m.tool_calls:
            continue
        if m.role == "tool":
            parts.append(f"[도구결과:{m.name or 'tool'}]\n{c}")
        elif m.tool_calls:
            parts.append(f"ASSISTANT(tool_calls): {json.dumps(m.tool_calls, ensure_ascii=False)}")
            if c:
                parts.append(c)
        else:
            parts.append(f"{m.role.upper()}: {c}")
    return "\n\n".join(parts) if parts else (messages[-1].content or "")


def _want_agent(req: ChatCompletionRequest) -> bool:
    if req.execute_tools is True or req.agent is True:
        return True
    if req.execute_tools is False or req.agent is False:
        return False
    model = (req.model or "").lower()
    if model in {AGENT_MODEL_NAME.lower(), "coding-agent", "agent"}:
        return True
    if model.endswith("-agent"):
        return True
    return False


def _want_tool_calls_only(req: ChatCompletionRequest) -> bool:
    """도구는 클라이언트 실행: 모델이 tool_calls만 반환."""
    if _want_agent(req):
        return False
    if req.tools is not None or req.tool_choice not in (None, "none"):
        return True
    return False


def _completion_response(
    *,
    model: str,
    content: str,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = content or None
        finish_reason = "tool_calls"
    body: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if extra:
        body.update(extra)
    return body


def _ollama_options_from_chat(req: ChatCompletionRequest) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.max_tokens is not None:
        opts["num_predict"] = req.max_tokens
    if req.stop is not None:
        opts["stop"] = [req.stop] if isinstance(req.stop, str) else list(req.stop)
    return opts


async def _ollama_chat(payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        try:
            r = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama 연결 실패 ({OLLAMA_BASE}). `ollama serve` 확인.",
            ) from exc
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=r.text)
        return r.json()


async def _ollama_chat_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = payload.get("model", MODEL_NAME)
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n".encode("utf-8")

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        try:
            async with client.stream(
                "POST", f"{OLLAMA_BASE}/api/chat", json=payload
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise HTTPException(
                        status_code=502, detail=body.decode("utf-8", "replace")
                    )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    piece = (data.get("message") or {}).get("content") or ""
                    if piece:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode(
                            "utf-8"
                        )
                    if data.get("done"):
                        done = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        }
                        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode(
                            "utf-8"
                        )
                        yield b"data: [DONE]\n\n"
                        return
        except httpx.ConnectError as exc:
            raise HTTPException(status_code=503, detail=f"Ollama 연결 실패: {exc}") from exc


def _ingest_attaches(workspace: Path, attach: list[AgentAttach], paths: list[str]) -> list[str]:
    from agent.tools import ingest_attachments
    from agent.workspace import WorkspaceError, ingest_upload_bytes

    attached: list[str] = []
    for item in attach:
        try:
            raw = base64.b64decode(item.content_base64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400, detail=f"base64 디코드 실패: {item.filename}"
            ) from exc
        try:
            attached.append(ingest_upload_bytes(workspace, item.filename, raw))
        except WorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if paths:
        try:
            attached.extend(ingest_attachments(workspace, paths))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return attached


def _build_agent(
    *,
    workspace: Path,
    model: str,
    vision_model: str | None,
    max_rounds: int,
    temperature: float,
    num_predict: int,
    approval_mode: str,
    allow_writes: bool,
    allow_terminal: bool,
    allow_web: bool,
    on_structured=None,
    approver=None,
):
    from agent.llm import OllamaClient
    from agent.loop import CodingAgent

    llm = OllamaClient(
        model=model,
        vision_model=vision_model or VISION_MODEL,
        base_url=OLLAMA_BASE,
        temperature=temperature,
        num_predict=num_predict,
    )
    return CodingAgent(
        workspace=workspace,
        llm=llm,
        max_rounds=max_rounds,
        approval_mode=approval_mode,  # type: ignore[arg-type]
        approver=approver,
        allow_writes=allow_writes,
        allow_terminal=allow_terminal,
        allow_web=allow_web,
        on_structured=on_structured,
    )


def _prepare_ask_gate(mode: str):
    from agent.approval import APPROVALS, ApprovalGate

    if mode != "ask":
        return None, None, uuid.uuid4().hex
    gate = ApprovalGate(
        timeout_sec=float(os.environ.get("AGENT_APPROVAL_TIMEOUT", "300"))
    )
    APPROVALS.register(gate)
    return gate, gate.request, gate.run_id


# ---------------------------------------------------------------------------
# Routes: health / models / tools
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    ok = False
    detail = "unreachable"
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            ok = r.status_code == 200
            detail = "ok" if ok else r.text[:200]
        except Exception as exc:  # noqa: BLE001
            detail = str(exc)
    return {
        "status": "ok" if ok else "degraded",
        "ollama": detail,
        "model": MODEL_NAME,
        "agent_model": AGENT_MODEL_NAME,
        "vision_model": VISION_MODEL,
        "features": {
            "chat": True,
            "tools_api": True,
            "agent_loop": True,
            "ide_chat": True,
            "web_search": True,
        },
    }


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _check_api_key(authorization)
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": now, "owned_by": "local"},
            {
                "id": AGENT_MODEL_NAME,
                "object": "model",
                "created": now,
                "owned_by": "local",
                "description": "채팅+서버측 도구 실행(에이전트)",
            },
        ],
    }


@app.get("/v1/tools")
async def list_tools(
    authorization: str | None = Header(default=None),
    x_workspace: str | None = Header(default=None, alias="X-Workspace"),
    workspace: str | None = None,
) -> dict[str, Any]:
    """OpenAI function tools 목록 + 내부 스키마."""
    _check_api_key(authorization)
    from agent.openai_compat import openai_tools_from_registry

    ws = _resolve_ws(workspace or x_workspace)
    reg = _registry(ws)
    return {
        "object": "list",
        "workspace": str(ws),
        "tools": openai_tools_from_registry(reg),
        "native": reg.schemas(),
    }


@app.post("/v1/tools/call")
async def call_tool(
    req: ToolCallRequest,
    authorization: str | None = Header(default=None),
    x_workspace: str | None = Header(default=None, alias="X-Workspace"),
) -> dict[str, Any]:
    """도구를 HTTP로 직접 실행 (Cursor 프론트가 파일/터미널을 돌릴 때)."""
    _check_api_key(authorization)
    from agent.llm import OllamaClient
    from agent.tools import ToolError

    ws = _resolve_ws(_workspace_header(x_workspace, req.workspace))
    _guard_tool(
        req.name, allow_writes=req.allow_writes, allow_terminal=req.allow_terminal
    )
    if req.name == "web_search" and not req.allow_web:
        raise HTTPException(status_code=403, detail="web_search 비활성")

    llm = OllamaClient(model=MODEL_NAME, vision_model=VISION_MODEL, base_url=OLLAMA_BASE)
    reg = _registry(
        ws,
        allow_web=req.allow_web,
        vision_chat=lambda prompt, path: llm.chat_with_image(prompt, path),
    )
    started = time.time()
    try:
        result = await asyncio.to_thread(reg.call, req.name, req.arguments)
        ok = True
        err = None
    except ToolError as exc:
        result = str(exc)
        ok = False
        err = "tool_error"
    except Exception as exc:  # noqa: BLE001
        result = f"{type(exc).__name__}: {exc}"
        ok = False
        err = "internal_error"

    return {
        "id": f"tool-{uuid.uuid4().hex[:16]}",
        "object": "tool.result",
        "name": req.name,
        "arguments": req.arguments,
        "workspace": str(ws),
        "ok": ok,
        "error": err,
        "result": result,
        "duration_ms": int((time.time() - started) * 1000),
    }


# ---------------------------------------------------------------------------
# Chat completions (plain / tool_calls / agent)
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
    x_workspace: str | None = Header(default=None, alias="X-Workspace"),
) -> Any:
    _check_api_key(authorization)
    model = req.model or MODEL_NAME

    # --- A) 서버가 도구까지 실행 (Cursor Agent) ---
    if _want_agent(req):
        from agent.approval import APPROVALS

        ws = _resolve_ws(_workspace_header(x_workspace, req.workspace))
        mode = req.approval_mode or DEFAULT_APPROVAL
        gate, approver, run_id = _prepare_ask_gate(mode)
        coding_model = MODEL_NAME if model.lower().endswith("-agent") else (req.model or MODEL_NAME)
        if coding_model.lower() in {AGENT_MODEL_NAME.lower(), "coding-agent", "agent"}:
            coding_model = MODEL_NAME
        agent = _build_agent(
            workspace=ws,
            model=coding_model,
            vision_model=VISION_MODEL,
            max_rounds=req.max_rounds,
            temperature=req.temperature or 0.2,
            num_predict=req.max_tokens or 2048,
            approval_mode=mode,
            allow_writes=req.allow_writes,
            allow_terminal=req.allow_terminal,
            allow_web=req.allow_web,
            approver=approver,
        )
        task = _messages_to_task(req.messages)
        try:
            if req.stream:
                return _agent_openai_stream(
                    agent=agent,
                    task=task,
                    model=model,
                    run_id=run_id,
                    workspace=str(ws),
                    gate=gate,
                )
            result = await asyncio.to_thread(agent.run, task)
        finally:
            if gate:
                APPROVALS.drop(gate.run_id)
        return _completion_response(
            model=model,
            content=result.answer,
            extra={
                "agent": True,
                "run_id": run_id,
                "workspace": str(ws),
                "steps": result.steps,
                "events": result.events,
                "denied": result.denied,
            },
        )

    # --- B) 1턴만: 도구 필요 시 OpenAI tool_calls 반환 (클라이언트가 /v1/tools/call) ---
    if _want_tool_calls_only(req):
        from agent.loop import SYSTEM_TEMPLATE
        from agent.openai_compat import to_openai_tool_calls
        from agent.tools import tools_prompt_block

        ws = _resolve_ws(_workspace_header(x_workspace, req.workspace))
        reg = _registry(ws, allow_web=req.allow_web)
        system = SYSTEM_TEMPLATE.format(
            workspace=str(ws),
            approval_mode="client",
            tools=tools_prompt_block(reg),
        )
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in req.messages:
            if m.role == "system":
                continue
            c = m.content or ""
            if m.role == "tool":
                msgs.append(
                    {
                        "role": "user",
                        "content": f"도구 결과 ({m.name or 'tool'}):\n```\n{c}\n```",
                    }
                )
            else:
                msgs.append({"role": m.role, "content": c})

        payload = {
            "model": MODEL_NAME if model.lower().endswith("-agent") else model,
            "messages": msgs,
            "stream": False,
            "options": _ollama_options_from_chat(req),
        }
        data = await _ollama_chat(payload)
        content = _extract_ollama_content(data)
        if not content.strip():
            payload["messages"] = msgs + [
                {
                    "role": "user",
                    "content": '응답이 비었습니다. 도구 JSON {"tool":"...","args":{}} 또는 최종 답변을 출력하세요.',
                }
            ]
            data = await _ollama_chat(payload)
            content = _extract_ollama_content(data)
        tool_calls = to_openai_tool_calls(content)
        if tool_calls:
            return _completion_response(
                model=model,
                content="",
                tool_calls=tool_calls,
                extra={"workspace": str(ws), "raw_assistant": content},
            )
        return _completion_response(
            model=model,
            content=content,
            extra={"workspace": str(ws)},
        )

    # --- C) 순수 채팅 ---
    payload = {
        "model": model if not model.lower().endswith("-agent") else MODEL_NAME,
        "messages": _normalize_messages(req.messages),
        "stream": req.stream,
        "options": _ollama_options_from_chat(req),
    }
    if req.stream:
        return StreamingResponse(
            _ollama_chat_stream(payload), media_type="text/event-stream"
        )
    data = await _ollama_chat(payload)
    content = _extract_ollama_content(data)
    if not content.strip():
        payload["messages"] = list(payload["messages"]) + [
            {
                "role": "user",
                "content": "이전 응답이 비어 있었습니다. 한국어로 답변만 출력하세요.",
            }
        ]
        data = await _ollama_chat(payload)
        content = _extract_ollama_content(data)
    return _completion_response(
        model=model,
        content=content,
        usage={
            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
            "completion_tokens": int(data.get("eval_count") or 0),
            "total_tokens": int(data.get("prompt_eval_count") or 0)
            + int(data.get("eval_count") or 0),
        },
    )


def _agent_openai_stream(*, agent, task, model, run_id, workspace, gate):
    from agent.approval import APPROVALS

    q: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def on_structured(ev: dict[str, Any]) -> None:
        q.put(ev)

    def job() -> None:
        try:
            agent.on_structured = on_structured
            result = agent.run(task)
            q.put(
                {
                    "type": "_done",
                    "answer": result.answer,
                    "steps": result.steps,
                    "events": result.events,
                    "denied": result.denied,
                }
            )
        except Exception as exc:  # noqa: BLE001
            q.put({"type": "_error", "message": str(exc)})
        finally:
            if gate:
                APPROVALS.drop(gate.run_id)
            q.put(None)

    threading.Thread(target=job, daemon=True).start()

    async def gen() -> AsyncIterator[bytes]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        # role
        yield _sse_chunk(
            completion_id,
            created,
            model,
            {"role": "assistant", "content": ""},
            None,
        )
        answer = ""
        while True:
            item = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if item is None:
                break
            if item.get("type") == "_done":
                answer = item.get("answer") or ""
                # stream answer as content
                if answer:
                    yield _sse_chunk(
                        completion_id,
                        created,
                        model,
                        {"content": answer},
                        None,
                    )
                # custom event with steps
                meta = {
                    "agent": True,
                    "run_id": run_id,
                    "workspace": workspace,
                    "steps": item.get("steps"),
                    "denied": item.get("denied"),
                }
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': None}], 'agent_meta': meta}, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                )
            elif item.get("type") == "_error":
                yield _sse_chunk(
                    completion_id,
                    created,
                    model,
                    {"content": f"\n[error] {item.get('message')}"},
                    None,
                )
            else:
                # tool progress as SSE comment-like custom field
                yield f"data: {json.dumps({'id': completion_id, 'object': 'agent.event', 'created': created, 'model': model, 'event': item}, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                )
        yield _sse_chunk(completion_id, created, model, {}, "stop")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse_chunk(cid, created, model, delta, finish):
    body = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# IDE chat (Cursor-like primary)
# ---------------------------------------------------------------------------


@app.post("/v1/ide/chat")
async def ide_chat(
    req: IdeChatRequest,
    authorization: str | None = Header(default=None),
    x_workspace: str | None = Header(default=None, alias="X-Workspace"),
) -> Any:
    """Cursor형 메인 API: 대화 + (기본) 서버 도구 실행."""
    _check_api_key(authorization)
    from agent.approval import APPROVALS

    ws = _resolve_ws(_workspace_header(x_workspace, req.workspace))
    attached = _ingest_attaches(ws, req.attach, req.attach_paths)
    hint = ""
    if attached:
        hint = (
            "\n\n[첨부 파일]\n"
            + "\n".join(f"- {p}" for p in attached)
            + "\n필요하면 list_uploads / read_upload / analyze_image 사용."
        )

    if not req.execute_tools:
        # 도구 미실행: chat completions tool_calls 경로로 위임
        chat_req = ChatCompletionRequest(
            model=req.model or MODEL_NAME,
            messages=req.messages,
            temperature=req.temperature,
            max_tokens=req.num_predict,
            stream=req.stream,
            tools=[],  # trigger tool_calls mode
            workspace=str(ws),
            allow_writes=req.allow_writes,
            allow_terminal=req.allow_terminal,
            allow_web=req.allow_web,
            agent=False,
            execute_tools=False,
        )
        return await chat_completions(chat_req, authorization, x_workspace)

    mode = req.approval_mode or DEFAULT_APPROVAL
    gate, approver, run_id = _prepare_ask_gate(mode)
    agent = _build_agent(
        workspace=ws,
        model=req.model or MODEL_NAME,
        vision_model=req.vision_model,
        max_rounds=req.max_rounds,
        temperature=req.temperature,
        num_predict=req.num_predict,
        approval_mode=mode,
        allow_writes=req.allow_writes,
        allow_terminal=req.allow_terminal,
        allow_web=req.allow_web,
        approver=approver,
    )
    task = _messages_to_task(req.messages) + hint

    if req.stream:
        return _ide_sse(
            agent=agent,
            task=task,
            run_id=run_id,
            workspace=str(ws),
            attached=attached,
            mode=mode,
            gate=gate,
        )

    try:
        result = await asyncio.to_thread(agent.run, task)
    finally:
        if gate:
            APPROVALS.drop(gate.run_id)

    return {
        "id": f"ide-{run_id}",
        "object": "ide.chat",
        "created": int(time.time()),
        "run_id": run_id,
        "workspace": str(ws),
        "attached": attached,
        "approval_mode": mode,
        "answer": result.answer,
        "steps": result.steps,
        "events": result.events,
        "denied": result.denied,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
    }


def _ide_sse(*, agent, task, run_id, workspace, attached, mode, gate):
    from agent.approval import APPROVALS

    q: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def on_structured(ev: dict[str, Any]) -> None:
        q.put({"event": "agent", "data": ev})

    def job() -> None:
        try:
            agent.on_structured = on_structured
            result = agent.run(task)
            q.put(
                {
                    "event": "done",
                    "data": {
                        "run_id": run_id,
                        "workspace": workspace,
                        "attached": attached,
                        "approval_mode": mode,
                        "answer": result.answer,
                        "steps": result.steps,
                        "events": result.events,
                        "denied": result.denied,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            q.put({"event": "error", "data": {"message": str(exc)}})
        finally:
            if gate:
                APPROVALS.drop(gate.run_id)
            q.put(None)

    threading.Thread(target=job, daemon=True).start()

    async def gen() -> AsyncIterator[bytes]:
        start = {
            "run_id": run_id,
            "workspace": workspace,
            "attached": attached,
            "approval_mode": mode,
        }
        yield f"event: start\ndata: {json.dumps(start, ensure_ascii=False)}\n\n".encode(
            "utf-8"
        )
        while True:
            item = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if item is None:
                yield b"event: end\ndata: {}\n\n"
                return
            ev = item.get("event", "agent")
            data = item.get("data", {})
            yield (
                f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            ).encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Agent run (task string shortcut)
# ---------------------------------------------------------------------------


@app.post("/v1/agent/run")
async def agent_run(
    req: AgentRunRequest,
    authorization: str | None = Header(default=None),
    x_workspace: str | None = Header(default=None, alias="X-Workspace"),
) -> Any:
    _check_api_key(authorization)
    ide = IdeChatRequest(
        messages=[ChatMessage(role="user", content=req.task)],
        workspace=req.workspace,
        model=req.model,
        vision_model=req.vision_model,
        stream=req.stream,
        max_rounds=req.max_rounds,
        temperature=req.temperature,
        num_predict=req.num_predict,
        approval_mode=req.approval_mode,
        allow_writes=req.allow_writes,
        allow_terminal=req.allow_terminal,
        allow_web=req.allow_web,
        attach=req.attach,
        attach_paths=req.attach_paths,
        execute_tools=True,
    )
    return await ide_chat(ide, authorization, x_workspace)


@app.post("/v1/agent/run/stream")
async def agent_run_stream(
    req: AgentRunRequest,
    authorization: str | None = Header(default=None),
    x_workspace: str | None = Header(default=None, alias="X-Workspace"),
) -> Any:
    req.stream = True
    return await agent_run(req, authorization, x_workspace)


@app.post("/v1/agent/approve")
async def agent_approve(
    req: AgentApproveRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(authorization)
    from agent.approval import APPROVALS

    ok = APPROVALS.resolve(req.run_id, req.approved)
    if not ok:
        raise HTTPException(status_code=404, detail="대기 중인 승인 없음")
    return {"ok": True, "run_id": req.run_id, "approved": req.approved}


# ---------------------------------------------------------------------------
# Compat alias
# ---------------------------------------------------------------------------


@app.get("/v1/agent/tools")
async def agent_tools_alias(
    authorization: str | None = Header(default=None),
    x_workspace: str | None = Header(default=None, alias="X-Workspace"),
) -> dict[str, Any]:
    return await list_tools(authorization, x_workspace, None)


@app.exception_handler(HTTPException)
async def http_exc_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "api_error"}},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local Cursor-like coding API")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--reload", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import uvicorn

    print("=" * 60)
    print("Local Cursor-like Coding API")
    print(f"  listen     : http://{args.host}:{args.port}")
    print(f"  ollama     : {OLLAMA_BASE}")
    print(f"  chat model : {MODEL_NAME}")
    print(f"  agent model: {AGENT_MODEL_NAME}  (chat+tools)")
    print("  tools      : GET /v1/tools  POST /v1/tools/call")
    print("  ide        : POST /v1/ide/chat")
    print("  docs       : /docs")
    print("=" * 60)
    uvicorn.run(
        "serve_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
