#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive / API approval gate for risky tools."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApprovalGate:
    """ask 모드에서 위험 도구 실행 전 대기."""

    timeout_sec: float = 300.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pending: dict[str, Any] | None = None
    _event: threading.Event = field(default_factory=threading.Event)
    _approved: bool = False
    created_at: float = field(default_factory=time.time)

    def request(self, tool: str, args: dict[str, Any]) -> bool:
        self.pending = {"tool": tool, "args": args, "run_id": self.run_id}
        self._approved = False
        self._event.clear()
        ok = self._event.wait(timeout=self.timeout_sec)
        self.pending = None
        if not ok:
            return False
        return self._approved

    def resolve(self, approved: bool) -> None:
        self._approved = bool(approved)
        self._event.set()


class ApprovalRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gates: dict[str, ApprovalGate] = {}

    def register(self, gate: ApprovalGate) -> str:
        with self._lock:
            self._gates[gate.run_id] = gate
            return gate.run_id

    def get(self, run_id: str) -> ApprovalGate | None:
        with self._lock:
            return self._gates.get(run_id)

    def resolve(self, run_id: str, approved: bool) -> bool:
        gate = self.get(run_id)
        if gate is None:
            return False
        gate.resolve(approved)
        return True

    def drop(self, run_id: str) -> None:
        with self._lock:
            self._gates.pop(run_id, None)


APPROVALS = ApprovalRegistry()
