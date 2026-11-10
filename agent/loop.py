#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent loop: plan → tool → observe → answer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from agent.llm import OllamaClient
from agent.tools import ToolError, ToolRegistry, parse_tool_call, tools_prompt_block

ApprovalMode = Literal["auto", "ask", "readonly"]

RISKY_WRITE_TOOLS = frozenset(
    {"write_file", "make_dir", "create_tree", "str_replace"}
)
RISKY_SHELL_TOOLS = frozenset({"run_terminal", "run_tests"})
RISKY_TOOLS = RISKY_WRITE_TOOLS | RISKY_SHELL_TOOLS

SYSTEM_TEMPLATE = """당신은 로컬 코딩 에이전트입니다.
워크스페이스: {workspace}
업로드 폴더: {workspace}/uploads
코딩 모델: 텍스트 LoRA(Ollama). 이미지 분석: 비전 모델(별도).
승인 모드: {approval_mode}

목표: 사용자의 **구체적 코딩 작업**을 도구로 수행합니다.
- 폴더/파일 **실제 생성** (make_dir, write_file, create_tree) — 디스크에 바로 씀
- 파일 읽기/수정, 검색, 터미널, 테스트, git 조회
- 웹 검색 (web_search) — 최신 문서/에러 조사
- 업로드된 문서/이미지 분석 (list_uploads, read_upload, analyze_image)

언제 도구를 쓰나요:
- "만들어줘/생성해줘/프로젝트 짜줘" → make_dir / write_file / create_tree 로 **실제로** 생성
- 파일/코드 확인·수정, 검색, 명령, 테스트, 업로드 파일/스크린샷 분석이 필요할 때
- 최신 정보·외부 문서가 필요하면 web_search

언제 도구를 쓰지 않나요 (즉시 한국어로만 답하세요):
- 인사, 감사, 잡담, 자기소개
- 모호한 의도만 있을 때 → 무엇을 할지 짧게 되묻기
- 워크스페이스를 볼 필요 없는 일반 개념 설명

규칙:
1. 코드 작업은 추측하지 말고 도구로 확인하세요.
2. 도구는 한 번에 하나만. (여러 파일은 create_tree 한 번으로)
3. 경로는 워크스페이스 상대 경로만. 절대경로/상위(..)/워크스페이스 밖 금지.
4. 위험 시스템 변경 금지.
5. 최종 답변은 한국어로 짧게. 생성 후에는 만든 경로를 알려주세요.
6. 잡담에 list_dir로 프로젝트를 훑지 마세요.
7. 스크린샷/에러 이미지는 analyze_image 를 쓰세요.
8. 사용자가 올린 파일은 보통 uploads/ 아래에 있습니다.
9. 코드만 채팅에 적지 말고, 요청이면 반드시 write_file/create_tree로 파일로 저장하세요.
10. 도구 호출 시 JSON만 출력하세요. 설명 문구와 섞지 마세요.

{tools}
"""


@dataclass
class AgentResult:
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    denied: list[dict[str, Any]] = field(default_factory=list)


EventCallback = Callable[[dict[str, Any]], None]
Approver = Callable[[str, dict[str, Any]], bool]


class CodingAgent:
    def __init__(
        self,
        workspace: Path,
        llm: OllamaClient | None = None,
        max_rounds: int = 12,
        on_event: Callable[[str], None] | None = None,
        on_structured: EventCallback | None = None,
        approval_mode: ApprovalMode = "auto",
        approver: Approver | None = None,
        allow_writes: bool = True,
        allow_terminal: bool = True,
        allow_web: bool = True,
        empty_retries: int = 2,
    ) -> None:
        self.workspace = workspace.resolve()
        self.llm = llm or OllamaClient()
        self.tools = ToolRegistry(
            self.workspace,
            vision_chat=lambda prompt, path: self.llm.chat_with_image(prompt, path),
            enable_web=allow_web,
        )
        self.max_rounds = max_rounds
        self.on_event = on_event or (lambda _msg: None)
        self.on_structured = on_structured
        self.approval_mode = approval_mode
        self.approver = approver
        self.allow_writes = allow_writes
        self.allow_terminal = allow_terminal
        self.empty_retries = max(0, empty_retries)
        self._events: list[dict[str, Any]] = []

    def _emit_text(self, msg: str) -> None:
        self.on_event(msg)

    def _emit(self, event_type: str, **data: Any) -> dict[str, Any]:
        ev = {"type": event_type, **data}
        self._events.append(ev)
        if self.on_structured:
            self.on_structured(ev)
        return ev

    def _chat_with_retry(self, messages: list[dict[str, str]], round_i: int) -> str:
        last = ""
        attempts = 1 + self.empty_retries
        for attempt in range(1, attempts + 1):
            last = self.llm.chat(messages)
            if last.strip():
                if attempt > 1:
                    self._emit("retry", reason="empty_response_recovered", attempt=attempt)
                return last
            self._emit_text(f"[재시도] 빈 응답 ({attempt}/{attempts})")
            self._emit("retry", reason="empty_response", attempt=attempt, round=round_i)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "이전 응답이 비어 있었습니다. "
                        '도구가 필요하면 {"tool":"...","args":{...}} JSON만, '
                        "아니면 최종 답변만 한국어로 출력하세요."
                    ),
                }
            )
        return last

    def _maybe_fix_tool_json(
        self,
        messages: list[dict[str, str]],
        assistant: str,
        round_i: int,
    ) -> str:
        if parse_tool_call(assistant) is not None:
            return assistant
        if not _looks_like_tool_attempt(assistant):
            return assistant
        self._emit_text("[재시도] 잘못된 도구 JSON")
        self._emit("retry", reason="malformed_tool_json", round=round_i)
        messages.append({"role": "assistant", "content": assistant})
        messages.append(
            {
                "role": "user",
                "content": (
                    "도구 JSON 형식이 올바르지 않습니다. "
                    '설명 없이 정확히 이 형식만 출력하세요: {"tool":"TOOL_NAME","args":{...}} '
                    "도구가 필요 없으면 JSON 없이 최종 답변만 하세요."
                ),
            }
        )
        fixed = self.llm.chat(messages)
        return fixed if fixed.strip() else assistant

    def _check_approval(self, name: str, args: dict[str, Any]) -> str | None:
        """승인 거부 시 오류 메시지, 허용 시 None."""
        if name in RISKY_WRITE_TOOLS and not self.allow_writes:
            return f"쓰기 도구 '{name}' 이 비활성화되어 있습니다 (allow_writes=false)."
        if name in RISKY_SHELL_TOOLS and not self.allow_terminal:
            return f"터미널 도구 '{name}' 이 비활성화되어 있습니다 (allow_terminal=false)."
        if self.approval_mode == "readonly" and name in RISKY_TOOLS:
            return f"readonly 모드에서는 '{name}' 을 실행할 수 없습니다."
        if self.approval_mode == "ask" and name in RISKY_TOOLS:
            self._emit("approval_needed", tool=name, args=_safe_args(args))
            self._emit_text(f"[승인 필요] {name}")
            if self.approver is None:
                return (
                    f"승인 필요: '{name}'. API에서 approve 하거나 "
                    "approval_mode=auto / allow_* 플래그를 사용하세요."
                )
            if not self.approver(name, args):
                return f"사용자가 '{name}' 실행을 거부했습니다."
        return None

    def run(self, user_task: str) -> AgentResult:
        self._events = []
        system = SYSTEM_TEMPLATE.format(
            workspace=str(self.workspace),
            approval_mode=self.approval_mode,
            tools=tools_prompt_block(self.tools),
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_task},
        ]
        steps: list[dict[str, Any]] = []
        denied: list[dict[str, Any]] = []
        self._emit("start", workspace=str(self.workspace), task=user_task[:500])

        for round_i in range(1, self.max_rounds + 1):
            self._emit_text(f"\n[{round_i}/{self.max_rounds}] 모델 생각 중...")
            self._emit("thinking", round=round_i)
            assistant = self._chat_with_retry(messages, round_i)
            if not assistant.strip():
                answer = "모델이 빈 응답만 반환했습니다. Ollama/모델 상태를 확인하세요."
                self._emit("final", answer=answer)
                return AgentResult(
                    answer=answer, steps=steps, events=list(self._events), denied=denied
                )

            assistant = self._maybe_fix_tool_json(messages, assistant, round_i)
            messages.append({"role": "assistant", "content": assistant})
            call = parse_tool_call(assistant)

            if call is None:
                answer = assistant.strip()
                self._emit_text("[완료] 최종 답변")
                self._emit("final", answer=answer)
                return AgentResult(
                    answer=answer, steps=steps, events=list(self._events), denied=denied
                )

            name = call["tool"]
            args = call.get("args") or {}
            self._emit_text(f"[도구] {name}({json_preview(args)})")
            self._emit("tool_call", tool=name, args=_safe_args(args), round=round_i)

            deny_msg = self._check_approval(name, args)
            if deny_msg:
                ok = False
                result = f"DENIED: {deny_msg}"
                denied.append({"tool": name, "args": _safe_args(args), "reason": deny_msg})
                self._emit("tool_denied", tool=name, reason=deny_msg)
            else:
                try:
                    result = self.tools.call(name, args)
                    ok = True
                except ToolError as exc:
                    result = f"ERROR: {exc}"
                    ok = False
                except Exception as exc:  # noqa: BLE001
                    result = f"ERROR: {type(exc).__name__}: {exc}"
                    ok = False

            step = {
                "tool": name,
                "args": _safe_args(args),
                "ok": ok,
                "result": result[:4000],
            }
            steps.append(step)
            self._emit_text(f"[결과] {'OK' if ok else 'FAIL'} ({len(result)} chars)")
            self._emit(
                "tool_result",
                tool=name,
                ok=ok,
                result_preview=result[:1500],
                result_chars=len(result),
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"도구 결과 ({name}):\n```\n{result}\n```\n"
                        "필요하면 다음 도구를 호출하고, 끝났으면 최종 답변만 하세요."
                    ),
                }
            )

        answer = "최대 도구 호출 횟수에 도달했습니다. 부분 결과만 확인하세요."
        self._emit("final", answer=answer)
        return AgentResult(
            answer=answer, steps=steps, events=list(self._events), denied=denied
        )


def json_preview(args: dict) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= 120 else s[:117] + "..."


def _safe_args(args: dict[str, Any]) -> dict[str, Any]:
    """로그용: 긴 content 축약."""
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > 400:
            out[k] = v[:400] + f"...({len(v)} chars)"
        elif k == "items" and isinstance(v, list):
            out[k] = f"[{len(v)} items]"
        else:
            out[k] = v
    return out


def _looks_like_tool_attempt(text: str) -> bool:
    t = text.strip()
    if '"tool"' in t or "'tool'" in t:
        return True
    if re.search(r"\b(write_file|make_dir|create_tree|run_terminal|web_search)\b", t):
        if "{" in t:
            return True
    return False
