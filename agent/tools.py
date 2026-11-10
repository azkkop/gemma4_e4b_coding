#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent/tools.py
--------------
로컬 코딩 에이전트 도구.
- 파일/터미널/테스트
- 부분 수정(str_replace), git 조회
- uploads/ 문서·이미지 읽기 (비전은 Ollama 멀티모달)
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


MAX_READ_CHARS = 80_000
MAX_LIST_ENTRIES = 200
DEFAULT_SHELL_TIMEOUT = 120
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
TEXT_EXTS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".css",
    ".html",
    ".xml",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".env",
    ".log",
    ".csv",
}

SKIP_DIRS = {
    ".venv",
    "venv",
    ".git",
    "__pycache__",
    "node_modules",
    "outputs",
    "gemma4_e4b_coding_gguf",
    "gemma4_e4b_coding_merged",
    "gemma4_e4b_coding_lora",
    "unsloth_compiled_cache",
}


class ToolError(Exception):
    pass


class ToolRegistry:
    def __init__(
        self,
        workspace: Path,
        vision_chat: Callable[[str, Path], str] | None = None,
        enable_web: bool = True,
    ) -> None:
        self.workspace = workspace.resolve()
        self.uploads = self.workspace / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.vision_chat = vision_chat
        self.enable_web = enable_web
        self._tools: dict[str, Callable[..., str]] = {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "make_dir": self.make_dir,
            "create_tree": self.create_tree,
            "str_replace": self.str_replace,
            "search_text": self.search_text,
            "run_terminal": self.run_terminal,
            "run_tests": self.run_tests,
            "git_status": self.git_status,
            "git_diff": self.git_diff,
            "list_uploads": self.list_uploads,
            "read_upload": self.read_upload,
            "analyze_image": self.analyze_image,
        }
        if enable_web:
            self._tools["web_search"] = self.web_search

    def names(self) -> list[str]:
        return sorted(self._tools)

    def _all_schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": "list_dir", "description": "디렉터리 목록", "args": {"path": "상대 경로(기본 '.')"}},
            {"name": "read_file", "description": "텍스트 파일 읽기", "args": {"path": "상대 경로", "offset": "시작 줄", "limit": "줄 수"}},
            {
                "name": "write_file",
                "description": "디스크에 텍스트 파일을 실제로 생성/덮어씀 (중간 폴더 자동 생성)",
                "args": {"path": "상대 경로 예: src/app.py", "content": "파일 전체 내용"},
            },
            {
                "name": "make_dir",
                "description": "디스크에 폴더를 실제로 생성 (중첩 경로 OK)",
                "args": {"path": "상대 경로 예: projects/demo/src"},
            },
            {
                "name": "create_tree",
                "description": "여러 폴더/파일을 한 번에 디스크에 생성",
                "args": {
                    "items": '[{"path":"foo/","content":""},{"path":"foo/a.py","content":"print(1)"}] 경로가 /로 끝나면 폴더'
                },
            },
            {"name": "str_replace", "description": "파일 일부 문자열 치환(정확 일치)", "args": {"path": "경로", "old": "찾을 문자열", "new": "바꿀 문자열", "replace_all": "true/false"}},
            {"name": "search_text", "description": "워크스페이스 텍스트 검색", "args": {"query": "검색어", "glob": "*.py", "regex": "false"}},
            {"name": "run_terminal", "description": "PowerShell 명령 실행", "args": {"command": "명령", "timeout_sec": "120"}},
            {"name": "run_tests", "description": "pytest/unittest 실행", "args": {"target": "경로", "extra_args": "추가 인자"}},
            {"name": "git_status", "description": "git status --short", "args": {}},
            {"name": "git_diff", "description": "git diff (읽기 전용)", "args": {"path": "선택 경로"}},
            {"name": "list_uploads", "description": "uploads/ 업로드 파일 목록", "args": {}},
            {"name": "read_upload", "description": "uploads/ 또는 워크스페이스 문서 읽기(텍스트/CSV/JSON/PDF)", "args": {"path": "상대 경로", "question": "선택 질문"}},
            {"name": "analyze_image", "description": "이미지(스크린샷/에러창) 비전 분석", "args": {"path": "이미지 경로", "question": "무엇을 볼지"}},
            {
                "name": "web_search",
                "description": "웹 검색(문서/에러/최신 정보). 오프라인 불가 시 실패 메시지 반환",
                "args": {"query": "검색어", "max_results": "결과 수(기본 5)"},
            },
        ]

    def schemas(self) -> list[dict[str, Any]]:
        all_schemas = self._all_schemas()
        if self.enable_web:
            return all_schemas
        return [s for s in all_schemas if s["name"] != "web_search"]

    def resolve(self, rel: str | None) -> Path:
        raw = (rel or ".").strip() or "."
        p = Path(raw)
        if p.is_absolute():
            raise ToolError("절대 경로는 허용되지 않습니다. 워크스페이스 상대 경로를 쓰세요.")
        target = (self.workspace / raw).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise ToolError("워크스페이스 밖 경로는 허용되지 않습니다.") from exc
        return target

    def call(self, name: str, args: dict[str, Any] | None) -> str:
        if name not in self._tools:
            raise ToolError(f"알 수 없는 도구: {name}. 사용 가능: {', '.join(self.names())}")
        return self._tools[name](**(args or {}))

    def list_dir(self, path: str = ".") -> str:
        d = self.resolve(path)
        if not d.exists():
            raise ToolError(f"없음: {path}")
        if not d.is_dir():
            raise ToolError(f"디렉터리 아님: {path}")
        entries = sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        lines = []
        for i, e in enumerate(entries):
            if i >= MAX_LIST_ENTRIES:
                lines.append(f"... (+{len(entries) - MAX_LIST_ENTRIES} more)")
                break
            kind = "dir" if e.is_dir() else "file"
            size = e.stat().st_size if e.is_file() else "-"
            lines.append(f"{kind:4} {size:>12}  {e.relative_to(self.workspace).as_posix()}")
        return "\n".join(lines) if lines else "(empty)"

    def read_file(self, path: str, offset: int | None = None, limit: int | None = None) -> str:
        f = self.resolve(path)
        if not f.exists() or not f.is_file():
            raise ToolError(f"파일 없음: {path}")
        if f.suffix.lower() in IMAGE_EXTS:
            raise ToolError("이미지입니다. analyze_image 도구를 사용하세요.")
        text = f.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, int(offset)) if offset is not None else 1
        end = start + max(0, int(limit)) if limit is not None else len(lines) + 1
        sliced = lines[start - 1 : end - 1]
        body = "\n".join(f"{i}|{line}" for i, line in enumerate(sliced, start=start))
        if len(body) > MAX_READ_CHARS:
            body = body[:MAX_READ_CHARS] + "\n... [truncated]"
        return body if body else "(empty file)"

    def write_file(self, path: str, content: str = "") -> str:
        f = self.resolve(path)
        if f.suffix == "" and str(path).endswith(("/", "\\")):
            return self.make_dir(path.rstrip("/\\"))
        f.parent.mkdir(parents=True, exist_ok=True)
        # Windows: 경로가 기존 디렉터리면 거부
        if f.exists() and f.is_dir():
            raise ToolError(f"이미 폴더입니다. 파일 경로를 쓰세요: {path}")
        f.write_text(content if content is not None else "", encoding="utf-8")
        rel = f.relative_to(self.workspace).as_posix()
        return f"CREATED/UPDATED file on disk: {rel} ({len(content or '')} chars) @ {f}"

    def make_dir(self, path: str) -> str:
        d = self.resolve(path.rstrip("/\\"))
        if d.exists() and d.is_file():
            raise ToolError(f"같은 이름 파일이 있어 폴더를 만들 수 없습니다: {path}")
        d.mkdir(parents=True, exist_ok=True)
        rel = d.relative_to(self.workspace).as_posix()
        return f"CREATED directory on disk: {rel}/ @ {d}"

    def create_tree(self, items: list | str | None = None) -> str:
        """여러 폴더/파일을 한 번에 생성.
        items 예: [{"path":"app/","content":""},{"path":"app/main.py","content":"..."}]
        """
        if items is None:
            raise ToolError("items 가 필요합니다.")
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except json.JSONDecodeError as exc:
                raise ToolError("items JSON 파싱 실패") from exc
        if not isinstance(items, list) or not items:
            raise ToolError("items 는 비어 있지 않은 객체 리스트여야 합니다.")

        results: list[str] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "path" not in item:
                raise ToolError(f"items[{i}] 에 path 가 필요합니다.")
            path = str(item["path"])
            content = item.get("content", "")
            is_dir = path.endswith(("/", "\\")) or (
                content in ("", None) and Path(path).suffix == "" and item.get("type") == "dir"
            )
            if item.get("type") == "dir" or path.endswith(("/", "\\")):
                results.append(self.make_dir(path))
            elif is_dir and content in ("", None) and Path(path).suffix == "":
                results.append(self.make_dir(path))
            else:
                results.append(self.write_file(path, content if content is not None else ""))
        return "\n".join(results)

    def str_replace(
        self,
        path: str,
        old: str = "",
        new: str = "",
        replace_all: bool | str = False,
    ) -> str:
        if not old:
            raise ToolError("old 가 비어 있습니다.")
        f = self.resolve(path)
        if not f.exists():
            raise ToolError(f"파일 없음: {path}")
        text = f.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ToolError("old 문자열을 찾지 못했습니다.")
        do_all = str(replace_all).lower() in {"1", "true", "yes", "y"}
        if not do_all and count > 1:
            raise ToolError(
                f"old 가 {count}번 등장합니다. replace_all=true 로 하거나 더 긴 고유 문자열을 쓰세요."
            )
        updated = text.replace(old, new) if do_all else text.replace(old, new, 1)
        f.write_text(updated, encoding="utf-8")
        n = count if do_all else 1
        return f"replaced {n} occurrence(s) in {f.relative_to(self.workspace).as_posix()}"

    def search_text(
        self,
        query: str,
        glob: str | None = None,
        regex: bool | str = False,
    ) -> str:
        if not query:
            raise ToolError("query 가 비어 있습니다.")
        use_re = str(regex).lower() in {"1", "true", "yes", "y"}
        pattern = re.compile(query) if use_re else None
        matches: list[str] = []
        root = self.workspace
        paths = root.rglob(glob) if glob else root.rglob("*")
        for p in paths:
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                hit = pattern.search(line) is not None if pattern else (query in line)
                if hit:
                    rel = p.relative_to(root).as_posix()
                    matches.append(f"{rel}:{i}:{line[:200]}")
                    if len(matches) >= 80:
                        matches.append("... [truncated]")
                        return "\n".join(matches)
        return "\n".join(matches) if matches else "(no matches)"

    def run_terminal(self, command: str, timeout_sec: int | float | None = None) -> str:
        if not command or not str(command).strip():
            raise ToolError("command 가 비어 있습니다.")
        blocked = [
            r"\bformat\b",
            r"\bshutdown\b",
            r"\brm\s+-rf\s+/",
            r"Remove-Item\s+-Recurse\s+-Force\s+C:\\",
            r"\bdel\s+/s\s+/q\s+C:\\",
        ]
        for b in blocked:
            if re.search(b, command, flags=re.IGNORECASE):
                raise ToolError(f"위험 명령으로 차단됨: {command}")
        timeout = int(timeout_sec or DEFAULT_SHELL_TIMEOUT)
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=str(self.workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        text = (
            f"exit_code={completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout or ''}\n"
            f"--- stderr ---\n{completed.stderr or ''}"
        )
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n... [truncated]"
        return text

    def run_tests(self, target: str | None = None, extra_args: str | None = None) -> str:
        use_pytest = False
        try:
            import pytest  # noqa: F401

            use_pytest = True
        except Exception:
            use_pytest = False

        if use_pytest:
            cmd = "python -m pytest -q"
            if target:
                cmd += f" {target}"
            if extra_args:
                cmd += f" {extra_args}"
        else:
            cmd = "python -m unittest"
            if target:
                cmd += f" {target}"
            elif extra_args:
                cmd += f" {extra_args}"
            else:
                cmd += ' discover -s . -p "test_*.py"'
        return self.run_terminal(cmd, timeout_sec=300)

    def git_status(self) -> str:
        return self.run_terminal("git status --short", timeout_sec=30)

    def git_diff(self, path: str | None = None) -> str:
        cmd = "git --no-pager diff"
        if path:
            # keep relative path quoted for PowerShell
            cmd += f" -- '{path}'"
        return self.run_terminal(cmd, timeout_sec=60)

    def list_uploads(self) -> str:
        self.uploads.mkdir(parents=True, exist_ok=True)
        files = sorted(self.uploads.rglob("*"))
        files = [p for p in files if p.is_file()]
        if not files:
            return "(uploads/ 비어 있음) 파일을 uploads/ 에 넣거나 --attach 로 추가하세요."
        lines = []
        for p in files[:MAX_LIST_ENTRIES]:
            rel = p.relative_to(self.workspace).as_posix()
            lines.append(f"{p.stat().st_size:>12}  {rel}")
        return "\n".join(lines)

    def read_upload(self, path: str, question: str | None = None) -> str:
        f = self.resolve(path)
        if not f.exists() or not f.is_file():
            raise ToolError(f"파일 없음: {path}")
        suf = f.suffix.lower()
        if suf in IMAGE_EXTS:
            return self.analyze_image(path, question or "이 이미지를 설명해 주세요.")
        if suf == ".pdf":
            text = _read_pdf(f)
        elif suf == ".csv":
            text = _read_csv(f)
        elif suf == ".json":
            text = f.read_text(encoding="utf-8", errors="replace")
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
        elif suf in TEXT_EXTS or suf == "":
            text = f.read_text(encoding="utf-8", errors="replace")
        else:
            # best-effort text
            text = f.read_text(encoding="utf-8", errors="replace")
        if question:
            text = f"[질문] {question}\n\n[파일 내용]\n{text}"
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n... [truncated]"
        return text

    def analyze_image(self, path: str, question: str | None = None) -> str:
        f = self.resolve(path)
        if not f.exists() or not f.is_file():
            raise ToolError(f"이미지 없음: {path}")
        if f.suffix.lower() not in IMAGE_EXTS:
            raise ToolError(f"이미지 확장자 아님: {f.suffix}")
        if self.vision_chat is None:
            raise ToolError("비전 클라이언트가 없습니다. VISION_MODEL/Ollama 설정을 확인하세요.")
        q = question or (
            "이 이미지(스크린샷/에러/UI)를 코딩 관점에서 분석하세요. "
            "보이는 에러 메시지, 코드, UI 상태를 한국어로 정리하세요."
        )
        return self.vision_chat(q, f)

    def web_search(self, query: str, max_results: int | str = 5) -> str:
        if not self.enable_web:
            raise ToolError("web_search 가 비활성화되어 있습니다.")
        q = (query or "").strip()
        if not q:
            raise ToolError("query 가 비어 있습니다.")
        try:
            n = max(1, min(10, int(max_results)))
        except (TypeError, ValueError):
            n = 5
        return _duckduckgo_search(q, n)


def ingest_attachments(workspace: Path, paths: list[str]) -> list[str]:
    """CLI --attach 경로를 uploads/ 로 복사. 워크스페이스 밖 절대경로 허용."""
    uploads = workspace.resolve() / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for raw in paths:
        src = Path(raw).expanduser().resolve()
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"첨부 실패: {raw}")
        dest = uploads / src.name
        # 이름 충돌 시 번호
        if dest.exists():
            stem, suf = src.stem, src.suffix
            i = 1
            while True:
                cand = uploads / f"{stem}_{i}{suf}"
                if not cand.exists():
                    dest = cand
                    break
                i += 1
        shutil.copy2(src, dest)
        saved.append(dest.relative_to(workspace.resolve()).as_posix())
    return saved


def _duckduckgo_search(query: str, max_results: int) -> str:
    """의존성 없이 DuckDuckGo Instant Answer + HTML 결과 수집."""
    import html as html_lib
    import urllib.parse

    try:
        import httpx
    except ImportError as exc:
        raise ToolError("web_search 에 httpx 가 필요합니다: pip install httpx") from exc

    lines: list[str] = [f"query: {query}"]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GemmaCodingAgent/1.0)",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
            r = client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
            )
            if r.status_code == 200:
                data = r.json()
                abstract = (data.get("AbstractText") or "").strip()
                abs_url = (data.get("AbstractURL") or "").strip()
                if abstract:
                    lines.append(f"[abstract] {abstract}")
                    if abs_url:
                        lines.append(f"  url: {abs_url}")
                related = data.get("RelatedTopics") or []
                for item in related:
                    if len(lines) >= max_results + 3:
                        break
                    if isinstance(item, dict) and item.get("Text"):
                        lines.append(f"- {item['Text']}")
                        if item.get("FirstURL"):
                            lines.append(f"  url: {item['FirstURL']}")
                    elif isinstance(item, dict) and "Topics" in item:
                        for sub in item.get("Topics") or []:
                            if isinstance(sub, dict) and sub.get("Text"):
                                lines.append(f"- {sub['Text']}")
                                if sub.get("FirstURL"):
                                    lines.append(f"  url: {sub['FirstURL']}")
                                if len(lines) >= max_results + 5:
                                    break

            # HTML 결과 (Instant Answer가 빈약한 경우 보강)
            if len(lines) <= 2:
                hr = client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                )
                if hr.status_code == 200:
                    body = hr.text
                    # result links
                    for m in re.finditer(
                        r'uddg=([^&"]+).*?class="result__a"[^>]*>(.*?)</a>',
                        body,
                        flags=re.IGNORECASE | re.DOTALL,
                    ):
                        url = urllib.parse.unquote(m.group(1))
                        title = re.sub(r"<[^>]+>", "", m.group(2))
                        title = html_lib.unescape(title).strip()
                        if not title:
                            continue
                        lines.append(f"- {title}\n  url: {url}")
                        if sum(1 for x in lines if x.startswith("- ")) >= max_results:
                            break
                    if sum(1 for x in lines if x.startswith("- ")) == 0:
                        for m in re.finditer(
                            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                            body,
                            flags=re.IGNORECASE | re.DOTALL,
                        ):
                            href = html_lib.unescape(m.group(1))
                            title = re.sub(r"<[^>]+>", "", m.group(2))
                            title = html_lib.unescape(title).strip()
                            if "uddg=" in href:
                                qs = urllib.parse.parse_qs(
                                    urllib.parse.urlparse(href).query
                                )
                                href = urllib.parse.unquote(qs.get("uddg", [href])[0])
                            if title:
                                lines.append(f"- {title}\n  url: {href}")
                            if sum(1 for x in lines if x.startswith("- ")) >= max_results:
                                break
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"웹 검색 실패: {type(exc).__name__}: {exc}") from exc

    if len(lines) <= 1:
        return f"query: {query}\n(검색 결과 없음 - 네트워크/차단 가능)"
    text = "\n".join(lines)
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n... [truncated]"
    return text


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ToolError(
            "PDF 읽기에는 pypdf 가 필요합니다: pip install pypdf"
        ) from exc
    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages[:30], start=1):
        parts.append(f"--- page {i} ---\n{(page.extract_text() or '').strip()}")
    text = "\n\n".join(parts)
    return text if text.strip() else "(PDF 텍스트 추출 결과 없음)"


def _read_csv(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(raw))
    rows = []
    for i, row in enumerate(reader):
        if i >= 80:
            rows.append("... [truncated]")
            break
        rows.append(",".join(row))
    return "\n".join(rows)


def tools_prompt_block(registry: ToolRegistry) -> str:
    lines = ["사용 가능한 도구:"]
    for s in registry.schemas():
        args = ", ".join(f"{k}: {v}" for k, v in s["args"].items()) or "(없음)"
        lines.append(f"- {s['name']}: {s['description']} ({args})")
    lines.append(
        "\n도구를 쓰려면 아래 JSON만 출력하세요. 설명과 섞지 마세요.\n"
        '{"tool":"TOOL_NAME","args":{...}}\n'
        "이미지/스크린샷은 analyze_image, 업로드 문서는 read_upload/list_uploads.\n"
        "최신 정보·에러 검색은 web_search.\n"
        "작업이 끝났으면 도구 JSON 없이 최종 답변만 한국어로."
    )
    return "\n".join(lines)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    def _as_call(obj: Any) -> dict[str, Any] | None:
        if not isinstance(obj, dict) or "tool" not in obj:
            return None
        args = obj.get("args") or obj.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        return {"tool": str(obj["tool"]), "args": args}

    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text):
        try:
            parsed = _as_call(json.loads(m.group(1)))
            if parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        parsed = _as_call(obj)
        if parsed:
            return parsed
    return None
