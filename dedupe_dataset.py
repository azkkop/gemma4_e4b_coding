#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedupe_dataset.py
-----------------
dataset.jsonl 에서 중복 샘플을 제거하고 고유 샘플만 남깁니다.

중복 판정:
  1) text 전체 해시 (완전 동일)
  2) user 메시지 정규화 해시 (숫자만 다른 near-dup 도 제거 가능)

사용:
  python dedupe_dataset.py
  python dedupe_dataset.py --input dataset.jsonl --output dataset_unique.jsonl --near-dup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _user_content(rec: dict[str, Any]) -> str:
    for m in rec.get("messages") or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    # text 필드에서 대략 추출 실패 시 빈 문자열
    return ""


def _normalize(s: str) -> str:
    """숫자/공백 차이를 무시하는 near-dup 키."""
    s = s.lower()
    s = re.sub(r"\d+\.?\d*", "<n>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSONL 데이터셋 중복 제거")
    p.add_argument("--input", default="dataset.jsonl")
    p.add_argument("--output", default="dataset_unique.jsonl")
    p.add_argument(
        "--near-dup",
        action="store_true",
        help="숫자만 다른 near-duplicate user 도 제거",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="output 결과를 input 경로에도 복사(원본 교체)",
    )
    p.add_argument("--backup", action="store_true", help="교체 전 input.bak 백업")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"입력 없음: {in_path}")

    rows: list[dict[str, Any]] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    unique: list[dict[str, Any]] = []
    dropped_exact = 0
    dropped_near = 0

    for rec in rows:
        text = rec.get("text") or ""
        h = _hash(text)
        if h in seen_exact:
            dropped_exact += 1
            continue
        if args.near_dup:
            nh = _hash(_normalize(_user_content(rec)))
            if nh in seen_near:
                dropped_near += 1
                continue
            seen_near.add(nh)
        seen_exact.add(h)
        unique.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in unique:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    langs = Counter(r.get("language", "?") for r in unique)
    print(f"[OK] {in_path} -> {out_path}")
    print(f"  input={len(rows)} unique={len(unique)}")
    print(f"  dropped_exact={dropped_exact} dropped_near={dropped_near}")
    for lang, cnt in sorted(langs.items()):
        pct = 100.0 * cnt / max(len(unique), 1)
        print(f"  - {lang}: {cnt} ({pct:.1f}%)")

    if args.replace:
        if args.backup:
            bak = in_path.with_suffix(in_path.suffix + ".bak")
            bak.write_bytes(in_path.read_bytes())
            print(f"  backup -> {bak}")
        in_path.write_bytes(out_path.read_bytes())
        print(f"  replaced {in_path}")


if __name__ == "__main__":
    main()
