#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_hq_dataset.py
----------------------
NVIDIA Build API (Gemma 4 31B)로 고품질 코딩 샘플을 생성하고,
코드 추출 → 구문/실행 검증 통과분만 JSONL에 저장합니다.

필요:
  .env 에 NVIDIA_API_KEY=nvapi-...

사용 예:
  python generate_hq_dataset.py --target 50 --languages python,java,c
  python generate_hq_dataset.py --target 200 --output dataset_hq.jsonl --append

검증:
  - Python: ast.parse + 선택적 안전한 compile
  - Java: 기본 균형/키워드 휴리스틱 (+ javac 있으면 컴파일)
  - C: 기본 휴리스틱 (+ gcc 있으면 컴파일)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from hf_env import get_nvidia_api_key, load_dotenv
from prepare_dataset import SYSTEM_PROMPT, format_gemma4_chat

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "google/gemma-4-31b-it"

TOPICS = {
    "python": [
        "타입 힌팅과 TypedDict/dataclass",
        "asyncio gather/as_completed",
        "리팩토링(딕셔너리 테이블화)",
        "버그 수정(None 반환/off-by-one)",
        "pytest 단위 테스트 작성",
        "context manager로 자원 정리",
        "제너레이터로 배치 처리",
        "에러 처리와 커스텀 예외",
    ],
    "java": [
        "OOP 클래스 + 불변 식별자",
        "Spring @RestControllerAdvice 예외 처리",
        "record DTO + Jakarta Validation",
        "Service/Repository 유스케이스",
        "Stream API 필터/정렬",
        "equals/hashCode 구현",
        "Optional 기반 null 안전",
        "Enum으로 상태 머신",
    ],
    "c": [
        "포인터 연산으로 배열 합/최대",
        "malloc/free + dangling 방지",
        "문자열 복제 시 NUL/누수 버그 수정",
        "구조체 동적 배열",
        "realloc 성장 시 실패 안전",
        "소유권 create/destroy 패턴",
        "단순 연결 리스트 append",
        "버퍼 오버플로 수정",
    ],
}


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """return list of (lang, code)."""
    blocks = re.findall(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", text, flags=re.S)
    out: list[tuple[str, str]] = []
    for lang, code in blocks:
        out.append(((lang or "").strip().lower(), code.strip()))
    return out


def pick_code(blocks: list[tuple[str, str]], language: str) -> str | None:
    aliases = {
        "python": {"python", "py", ""},
        "java": {"java", ""},
        "c": {"c", ""},
    }[language]
    for lang, code in blocks:
        if lang in aliases and len(code) > 20:
            return code
    # fallback: longest block
    if blocks:
        return max(blocks, key=lambda x: len(x[1]))[1]
    return None


def verify_python(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax: {e}"
    try:
        compile(code, "<hq>", "exec")
    except Exception as e:  # noqa: BLE001
        return False, f"compile: {e}"
    return True, "ok"


def verify_java(code: str) -> tuple[bool, str]:
    if code.count("{") != code.count("}"):
        return False, "brace mismatch"
    if "class " not in code and "record " not in code and "interface " not in code:
        # allow snippets with just methods? prefer class/record
        if "public " not in code and "private " not in code:
            return False, "no class/record"
    javac = shutil.which("javac")
    if not javac:
        return True, "heuristic_ok(no javac)"
    # extract public class name
    m = re.search(r"public\s+(?:final\s+)?(?:class|record)\s+(\w+)", code)
    name = m.group(1) if m else "Main"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"{name}.java"
        # if no public class wrapper, wrap
        src = code
        if "class " not in code and "record " not in code:
            src = f"public class {name} {{\n{code}\n}}\n"
            path = Path(td) / f"{name}.java"
        path.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [javac, str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout)[:300]
    return True, "javac_ok"


def verify_c(code: str) -> tuple[bool, str]:
    if code.count("{") != code.count("}"):
        return False, "brace mismatch"
    if "malloc" in code and "free" not in code and "realloc" not in code:
        # soft fail for leak-demo questions? keep as warning fail for HQ
        if "누수" not in code:
            pass
    gcc = shutil.which("gcc")
    if not gcc:
        # require at least a function or main
        if "main(" not in code and "(" not in code:
            return False, "no function"
        return True, "heuristic_ok(no gcc)"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "snippet.c"
        src = code
        if "main(" not in code:
            src = code + "\nint main(void){return 0;}\n"
        path.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [gcc, "-fsyntax-only", "-std=c11", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return False, (r.stderr or r.stdout)[:300]
    return True, "gcc_ok"


def verify(language: str, code: str) -> tuple[bool, str]:
    if language == "python":
        return verify_python(code)
    if language == "java":
        return verify_java(code)
    if language == "c":
        return verify_c(code)
    return False, "unknown language"


def build_user_prompt(language: str, topic: str, seed: int) -> str:
    return (
        f"당신은 시니어 {language} 엔지니어입니다. 아래 주제에 대해 "
        f"학습용 질의응답 1개를 만들어 주세요.\n"
        f"- 주제: {topic}\n"
        f"- 언어: {language}\n"
        f"- seed: {seed}\n\n"
        "출력 형식(반드시 지키세요):\n"
        "USER:\n"
        "(한국어 질문. 구체적 요구 + 필요시 잘못된 코드 포함)\n\n"
        "ASSISTANT:\n"
        "(한국어 설명 + 완성 코드 블록. 코드는 실행/컴파일 가능한 수준)\n"
    )


def parse_qa(text: str) -> tuple[str, str] | None:
    m = re.search(
        r"USER:\s*(.*?)\s*ASSISTANT:\s*(.*)",
        text,
        flags=re.S | re.I,
    )
    if not m:
        return None
    user, assistant = m.group(1).strip(), m.group(2).strip()
    if len(user) < 20 or len(assistant) < 40:
        return None
    return user, assistant


def nvidia_chat(
    api_key: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    retries: int = 3,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err: Exception | None = None
    for i in range(retries):
        try:
            resp = requests.post(
                NVIDIA_URL,
                headers=headers,
                json=payload,
                timeout=(30, 180),
            )
            if resp.status_code == 429:
                time.sleep(2 ** i + 1)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1 + i)
    raise RuntimeError(f"NVIDIA API 실패: {last_err}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NVIDIA 31B 고품질 데이터 생성")
    p.add_argument("--target", type=int, default=30, help="저장할 통과 샘플 수")
    p.add_argument("--output", default="dataset_hq.jsonl")
    p.add_argument("--append", action="store_true", help="기존 파일에 이어쓰기")
    p.add_argument("--languages", default="python,java,c")
    p.add_argument("--max-attempts", type=int, default=0, help="0이면 target*8")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--sleep", type=float, default=0.5, help="요청 간 대기(초)")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    api_key = get_nvidia_api_key()
    if not api_key:
        raise SystemExit(
            "NVIDIA_API_KEY 가 없습니다. .env 에 NVIDIA_API_KEY=nvapi-... 를 넣으세요."
        )

    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    for lang in langs:
        if lang not in TOPICS:
            raise SystemExit(f"지원 언어 아님: {lang}")

    out_path = Path(args.output)
    seen: set[str] = set()
    saved = 0
    if args.append and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                seen.add(_hash(rec.get("text", "")))
                saved += 1
        print(f"[이어쓰기] 기존 {saved}개 로드")

    target = args.target
    max_attempts = args.max_attempts or max(target * 8, 50)
    rng = random.Random(args.seed)
    attempts = 0
    passed = 0
    failed = 0

    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as out:
        while passed < target and attempts < max_attempts:
            attempts += 1
            language = langs[attempts % len(langs)]
            topic = rng.choice(TOPICS[language])
            user_req = build_user_prompt(language, topic, rng.randint(1, 10_000_000))

            try:
                raw = nvidia_chat(
                    api_key,
                    [
                        {
                            "role": "system",
                            "content": "고품질 코딩 SFT 데이터를 생성합니다. 형식만 지키세요.",
                        },
                        {"role": "user", "content": user_req},
                    ],
                )
            except Exception as e:  # noqa: BLE001
                print(f"[API오류] {e}")
                time.sleep(2)
                continue

            qa = parse_qa(raw)
            if not qa:
                failed += 1
                print(f"[{attempts}] parse_fail")
                time.sleep(args.sleep)
                continue
            user, assistant = qa
            blocks = extract_code_blocks(assistant)
            code = pick_code(blocks, language)
            if not code:
                failed += 1
                print(f"[{attempts}] no_code")
                time.sleep(args.sleep)
                continue

            ok, reason = verify(language, code)
            if not ok:
                failed += 1
                print(f"[{attempts}] verify_fail/{language}: {reason[:80]}")
                time.sleep(args.sleep)
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
            rec = {
                "language": language,
                "topic": f"hq_{topic[:24]}",
                "text": format_gemma4_chat(messages, add_bos=True),
                "messages": messages,
                "meta": {"source": "nvidia_gemma4_31b", "verify": reason},
            }
            h = _hash(rec["text"])
            if h in seen:
                failed += 1
                print(f"[{attempts}] dup")
                time.sleep(args.sleep)
                continue

            seen.add(h)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            passed += 1
            print(f"[{attempts}] PASS {passed}/{target} ({language}) {reason}")
            time.sleep(args.sleep)

    print(
        f"\n[완료] saved_pass={passed} failed_or_skip≈{failed} "
        f"attempts={attempts} output={out_path.resolve()}"
    )
    if passed < target:
        print("[경고] 목표 미달. 쿼터/파싱/검증 실패를 확인하세요.")


if __name__ == "__main__":
    main()
