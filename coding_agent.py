#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coding_agent.py
---------------
로컬 오프라인 코딩 에이전트 CLI.

예시:
  python coding_agent.py "README에 API 사용법 한 줄 추가"
  python coding_agent.py --attach screenshot.png "이 에러 원인 찾고 관련 파일 고쳐줘"
  python coding_agent.py --approval ask "테스트 실행해줘"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hf_env import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local offline coding agent (Ollama)")
    p.add_argument("task", nargs="?", help="수행할 작업 (없으면 대화 모드)")
    p.add_argument(
        "--workspace",
        default=os.environ.get("AGENT_WORKSPACE", "."),
        help="작업 루트 (기본: AGENT_WORKSPACE 또는 .)",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("API_MODEL_NAME", "gemma4-e4b-coding"),
        help="코딩(텍스트) 모델",
    )
    p.add_argument(
        "--vision-model",
        default=os.environ.get("VISION_MODEL", "gemma4:e4b"),
        help="이미지 분석용 비전 모델 (text-only LoRA GGUF 는 비전 없음)",
    )
    p.add_argument(
        "--ollama",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    p.add_argument(
        "--attach",
        action="append",
        default=[],
        help="업로드할 파일(여러 번 지정 가능). uploads/ 로 복사",
    )
    p.add_argument("--max-rounds", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--num-predict", type=int, default=2048)
    p.add_argument(
        "--approval",
        choices=["auto", "ask", "readonly"],
        default=os.environ.get("AGENT_APPROVAL_MODE", "auto"),
        help="auto=즉시실행, ask=쓰기/터미널 확인, readonly=읽기만",
    )
    p.add_argument("--no-writes", action="store_true", help="파일 쓰기 도구 비활성")
    p.add_argument("--no-terminal", action="store_true", help="터미널 도구 비활성")
    p.add_argument("--no-web", action="store_true", help="웹 검색 비활성")
    return p.parse_args()


def _cli_approver(name: str, args: dict) -> bool:
    preview = ", ".join(f"{k}={str(v)[:60]}" for k, v in list(args.items())[:4])
    try:
        ans = input(f"\n승인할까요? [{name}] {preview}\n[y/N]> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in {"y", "yes", "ㅛ"}


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from agent.llm import OllamaClient
    from agent.loop import CodingAgent
    from agent.tools import ingest_attachments
    from agent.workspace import WorkspaceError, resolve_workspace

    try:
        workspace = resolve_workspace(args.workspace)
    except WorkspaceError as exc:
        raise SystemExit(str(exc)) from exc

    attached: list[str] = []
    if args.attach:
        try:
            attached = ingest_attachments(workspace, args.attach)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    llm = OllamaClient(
        model=args.model,
        vision_model=args.vision_model,
        base_url=args.ollama,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    agent = CodingAgent(
        workspace=workspace,
        llm=llm,
        max_rounds=args.max_rounds,
        on_event=lambda m: print(m, flush=True),
        approval_mode=args.approval,
        approver=_cli_approver if args.approval == "ask" else None,
        allow_writes=not args.no_writes,
        allow_terminal=not args.no_terminal,
        allow_web=not args.no_web,
    )

    print("=" * 60)
    print("Local Coding Agent")
    print(f"  workspace    : {workspace}")
    print(f"  coding model : {args.model}")
    print(f"  vision model : {args.vision_model}")
    print(f"  ollama       : {args.ollama}")
    print(f"  approval     : {args.approval}")
    print(
        f"  flags        : writes={not args.no_writes} "
        f"terminal={not args.no_terminal} web={not args.no_web}"
    )
    if attached:
        print(f"  attached     : {', '.join(attached)}")
    print("=" * 60)

    attach_hint = ""
    if attached:
        attach_hint = (
            "\n\n[첨부 파일]\n"
            + "\n".join(f"- {p}" for p in attached)
            + "\n필요하면 list_uploads / read_upload / analyze_image 를 사용하세요."
        )

    if args.task:
        result = agent.run(args.task + attach_hint)
        print("\n" + "-" * 60)
        print(result.answer)
        if result.steps:
            print("\n[도구 로그]")
            for s in result.steps:
                print(f"  - {s['tool']}: {'OK' if s['ok'] else 'FAIL'}")
        return

    print("대화 모드. 종료: exit / quit")
    print("팁: 파일을 uploads/ 에 넣거나 실행 시 --attach 사용")
    while True:
        try:
            task = input("\nYOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            continue
        if task.lower() in {"exit", "quit", "q"}:
            break
        extra = ""
        if task.startswith("@") and " " in task:
            maybe_path, rest = task[1:].split(" ", 1)
            try:
                newly = ingest_attachments(workspace, [maybe_path])
                extra = (
                    "\n\n[첨부 파일]\n"
                    + "\n".join(f"- {p}" for p in newly)
                    + "\n필요하면 read_upload / analyze_image 사용."
                )
                task = rest
                print(f"[첨부] {', '.join(newly)}")
            except FileNotFoundError as exc:
                print(f"[첨부 실패] {exc}")
                continue
        result = agent.run(task + extra)
        print("\nAGENT>\n" + result.answer)


if __name__ == "__main__":
    main()
