#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI-compatible tool schemas + one-shot tool-call parsing."""

from __future__ import annotations

import json
from typing import Any

from agent.tools import ToolRegistry, parse_tool_call


# OpenAI function parameters 대략적 스키마 (모델/클라이언트가 도구 목록을 읽을 때 사용)
TOOL_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_dir": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "상대 경로"}},
    },
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
    "make_dir": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    "create_tree": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["path"],
                },
            }
        },
        "required": ["items"],
    },
    "str_replace": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old", "new"],
    },
    "search_text": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "glob": {"type": "string"},
            "regex": {"type": "boolean"},
        },
        "required": ["query"],
    },
    "run_terminal": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_sec": {"type": "integer"},
        },
        "required": ["command"],
    },
    "run_tests": {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "extra_args": {"type": "string"},
        },
    },
    "git_status": {"type": "object", "properties": {}},
    "git_diff": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    },
    "list_uploads": {"type": "object", "properties": {}},
    "read_upload": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["path"],
    },
    "analyze_image": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "question": {"type": "string"},
        },
        "required": ["path"],
    },
    "web_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
}


def openai_tools_from_registry(registry: ToolRegistry) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in registry.schemas():
        name = s["name"]
        params = TOOL_JSON_SCHEMAS.get(
            name,
            {"type": "object", "properties": {}, "additionalProperties": True},
        )
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": s["description"],
                    "parameters": params,
                },
            }
        )
    return out


def to_openai_tool_calls(assistant_text: str) -> list[dict[str, Any]] | None:
    """모델 텍스트 JSON 도구호출 → OpenAI tool_calls 배열."""
    call = parse_tool_call(assistant_text)
    if not call:
        return None
    tid = f"call_{abs(hash(json.dumps(call, ensure_ascii=False))) % 10**10}"
    return [
        {
            "id": tid,
            "type": "function",
            "function": {
                "name": call["tool"],
                "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
            },
        }
    ]


def last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""
