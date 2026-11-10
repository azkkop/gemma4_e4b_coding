#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ollama chat + vision client for the coding agent."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx


class OllamaClient:
    def __init__(
        self,
        model: str = "gemma4-e4b-coding",
        vision_model: str | None = None,
        base_url: str = "http://127.0.0.1:11434",
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_predict: int = 2048,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        # Fine-tuned GGUF is text-only → vision은 베이스 멀티모달 모델 사용
        self.vision_model = vision_model or "gemma4:e4b"
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.timeout = timeout

    def _options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
        }

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            try:
                r = client.post(f"{self.base_url}/api/chat", json=payload)
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Ollama 연결 실패 ({self.base_url}). `ollama serve` 확인."
                ) from exc
            if r.status_code >= 400:
                raise RuntimeError(f"Ollama 오류 {r.status_code}: {r.text[:500]}")
            return r.json()

    @staticmethod
    def _content_of(data: dict[str, Any]) -> str:
        msg = data.get("message") or {}
        content = msg.get("content")
        if content is None or content == "":
            content = msg.get("thinking") or ""
        return str(content)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(),
        }
        return self._content_of(self._post_chat(payload))

    def chat_with_image(
        self,
        prompt: str,
        image_path: Path,
        system: str | None = None,
    ) -> str:
        """비전 모델로 이미지+텍스트 질의."""
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        )
        payload = {
            "model": self.vision_model,
            "messages": messages,
            "stream": False,
            "options": self._options(),
        }
        return self._content_of(self._post_chat(payload))
