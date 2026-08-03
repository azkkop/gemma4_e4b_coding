#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_datasets.py
-----------------
여러 JSONL을 합치고 exact/near 중복을 제거합니다.

예:
  python merge_datasets.py --inputs dataset.jsonl dataset_hq.jsonl --output dataset.jsonl --backup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d+\.?\d*", "<n>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _user(rec: dict[str, Any]) -> str:
    for m in rec.get("messages") or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", default="dataset.jsonl")
    p.add_argument("--backup", action="store_true")
    p.add_argument("--near-dup", action="store_true", default=True)
    args = p.parse_args()

    out = Path(args.output)
    if args.backup and out.exists():
        bak = out.with_suffix(out.suffix + ".bak")
        bak.write_bytes(out.read_bytes())
        print(f"[백업] {bak}")

    seen_e: set[str] = set()
    seen_n: set[str] = set()
    merged: list[dict[str, Any]] = []

    for path in args.inputs:
        fp = Path(path)
        if not fp.exists():
            print(f"[스킵] 없음: {fp}")
            continue
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                eh = _h(rec.get("text") or "")
                if eh in seen_e:
                    continue
                if args.near_dup:
                    nh = _h(_norm(_user(rec)))
                    if nh in seen_n:
                        continue
                    seen_n.add(nh)
                seen_e.add(eh)
                merged.append(rec)
        print(f"[로드] {fp}")

    with out.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    c = Counter(r.get("language", "?") for r in merged)
    print(f"[OK] {out} n={len(merged)}")
    for k, v in sorted(c.items()):
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
