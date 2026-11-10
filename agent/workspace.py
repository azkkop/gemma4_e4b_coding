#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace path validation for agent API/CLI."""

from __future__ import annotations

import os
from pathlib import Path


class WorkspaceError(ValueError):
    pass


def default_workspace() -> Path:
    raw = os.environ.get("AGENT_WORKSPACE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve()


def allowed_roots() -> list[Path]:
    """AGENT_ALLOWED_ROOTS=C:\\Users\\USER;D:\\work  (세미콜론/쉼표 구분).
    비어 있으면 제한 없음(로컬 기본).
    """
    raw = os.environ.get("AGENT_ALLOWED_ROOTS", "").strip()
    if not raw:
        return []
    roots: list[Path] = []
    for part in re_split(raw):
        p = Path(part).expanduser().resolve()
        if p.exists():
            roots.append(p)
    return roots


def re_split(raw: str) -> list[str]:
    parts: list[str] = []
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip().strip('"').strip("'")
        if chunk:
            parts.append(chunk)
    return parts


def resolve_workspace(path: str | Path | None) -> Path:
    if path is None or str(path).strip() == "":
        ws = default_workspace()
    else:
        ws = Path(str(path)).expanduser().resolve()
    if not ws.exists() or not ws.is_dir():
        raise WorkspaceError(f"워크스페이스 디렉터리가 없습니다: {ws}")
    roots = allowed_roots()
    if roots:
        ok = False
        for root in roots:
            try:
                ws.relative_to(root)
                ok = True
                break
            except ValueError:
                continue
        if not ok:
            raise WorkspaceError(
                f"워크스페이스가 AGENT_ALLOWED_ROOTS 밖입니다: {ws}"
            )
    return ws


def ingest_upload_bytes(workspace: Path, filename: str, data: bytes) -> str:
    """API 업로드 바이트를 uploads/ 에 저장."""
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise WorkspaceError("잘못된 파일 이름")
    uploads = workspace.resolve() / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / name
    if dest.exists():
        stem, suf = dest.stem, dest.suffix
        i = 1
        while True:
            cand = uploads / f"{stem}_{i}{suf}"
            if not cand.exists():
                dest = cand
                break
            i += 1
    dest.write_bytes(data)
    return dest.relative_to(workspace.resolve()).as_posix()
