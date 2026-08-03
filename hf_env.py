#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""프로젝트 루트의 .env 에서 HF_TOKEN 등을 읽어 os.environ 에 주입합니다."""

from __future__ import annotations

import os
from pathlib import Path

# 시스템/유저 환경에 잘못된 값이 있어도 .env 를 우선 적용할 키
_FORCE_FROM_DOTENV = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NVIDIA_API_KEY",
    "NVAPI_KEY",
}


def load_dotenv(dotenv_path: Path | None = None) -> None:
    """
    간단한 .env 로더 (python-dotenv 의존성 없음).
    일반 키: 이미 설정된 환경변수는 유지.
    HF_TOKEN 계열: .env 값이 있으면 항상 덮어씀 (잘못된 시스템 토큰 방지).
    """
    path = dotenv_path or Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if key in _FORCE_FROM_DOTENV or key not in os.environ:
            os.environ[key] = value


def get_hf_token(cli_token: str | None = None) -> str | None:
    """CLI 인자 > .env(강제 주입된 환경변수) 순으로 HF 토큰을 반환합니다."""
    load_dotenv()
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )


def get_nvidia_api_key(cli_key: str | None = None) -> str | None:
    """NVIDIA Build/NIM API 키."""
    load_dotenv()
    return (
        cli_key
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVAPI_KEY")
    )
