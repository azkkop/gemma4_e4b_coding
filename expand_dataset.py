#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expand_dataset.py
-----------------
다양성 강화된 로컬 합성 데이터 생성기.

개선점:
  - 주제/시나리오/식별자 조합 공간 확대
  - 생성 중 exact + near-dup 해시로 중복 차단
  - --unique-target 만큼 "고유 샘플"을 채울 때까지 생성

사용:
  python expand_dataset.py --unique-target 3000 --include-seed --backup
  python expand_dataset.py --unique-target 5000 --output dataset.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator

from prepare_dataset import SYSTEM_PROMPT, build_raw_examples, format_gemma4_chat


def _msg(user: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _record(language: str, topic: str, user: str, assistant: str) -> dict[str, Any]:
    messages = _msg(user, assistant)
    return {
        "language": language,
        "topic": topic,
        "text": format_gemma4_chat(messages, add_bos=True),
        "messages": messages,
    }


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d+\.?\d*", "<n>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 어휘 풀 (조합 공간 확대)
# ---------------------------------------------------------------------------
VERBS = [
    "작성", "구현", "리팩토링", "수정", "디버깅", "최적화", "리뷰", "단순화", "일반화"
]
GOALS = [
    "타입 안정성", "가독성", "예외 안전성", "메모리 안전", "테스트 용이성",
    "성능", "확장성", "동시성 안전", "API 명확성",
]
DOMAINS = [
    "결제", "주문", "재고", "회원", "알림", "로그", "캐시", "파일 업로드",
    "채팅", "검색", "인증", "권한", "배치 작업", "리포트",
]
PY_NAMES = [
    "normalize", "aggregate", "paginate", "serialize", "deserialize", "validate",
    "throttle", "debounce", "backoff", "checkpoint", "hydrate", "project",
    "compose", "pipeline", "reduce_tree", "flatten", "partition", "windowed",
]
JAVA_NAMES = [
    "Invoice", "Shipment", "Coupon", "Ledger", "Device", "Channel", "Workspace",
    "Subscription", "AuditLog", "Permission", "Catalog", "Reservation",
]
C_NAMES = [
    "ring_buffer", "hashmap", "vector", "arena", "bitset", "queue", "stack",
    "string_builder", "pool", "slab",
]


def gen_python(rng: random.Random) -> Iterator[dict[str, Any]]:
    while True:
        kind = rng.choice([
            "type_hint", "async", "refactor", "debug", "dataclass",
            "contextmgr", "generator", "pytest",
        ])
        domain = rng.choice(DOMAINS)
        goal = rng.choice(GOALS)
        verb = rng.choice(VERBS)
        fn = rng.choice(PY_NAMES) + "_" + rng.choice(["items", "rows", "events", "nodes"])
        n = rng.randint(2, 20)
        timeout = rng.choice([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0])

        if kind == "type_hint":
            user = (
                f"{domain} 도메인의 `{fn}` 함수를 {verb}해 주세요. "
                f"목표: {goal}. TypedDict/dataclass 반환과 타입 힌팅을 넣고 한국어 설명을 추가하세요.\n\n"
                f"```python\ndef {fn}(rows):\n"
                f"    total = 0\n"
                f"    for r in rows:\n"
                f"        total += r.get('amount', 0)\n"
                f"    return {{'total': total, 'n': len(rows), 'domain': '{domain}'}}\n```"
            )
            cls = "".join(p.capitalize() for p in fn.split("_")) + "Result"
            assistant = (
                f"`{fn}` 에 타입을 고정해 {goal}을(를) 노립니다.\n\n"
                f"```python\nfrom typing import Mapping, Sequence, TypedDict\n\n"
                f"class {cls}(TypedDict):\n"
                f"    total: float\n    n: int\n    domain: str\n\n"
                f"def {fn}(rows: Sequence[Mapping[str, object]]) -> {cls}:\n"
                f"    if not rows:\n        raise ValueError('rows가 비어 있습니다.')\n"
                f"    total = 0.0\n"
                f"    for r in rows:\n"
                f"        amount = r.get('amount', 0)\n"
                f"        total += float(amount)  # type: ignore[arg-type]\n"
                f"    return {{'total': total, 'n': len(rows), 'domain': '{domain}'}}\n```\n\n"
                f"**포인트**: 런타임 검증이 필요하면 pydantic을 검토하세요."
            )
            yield _record("python", "type_hinting", user, assistant)

        elif kind == "async":
            user = (
                f"{domain} 관련 비동기 작업 {n}개를 동시에 실행하고 "
                f"`asyncio.as_completed`로 완료 순 출력을 {verb}해 주세요. "
                f"delay≈{timeout}s, 타입 힌팅 + 한국어 설명."
            )
            assistant = (
                f"완료 순 처리에는 `as_completed`가 적합합니다. ({domain})\n\n"
                f"```python\nfrom __future__ import annotations\nimport asyncio\n"
                f"from typing import Any\n\n"
                f"async def job(i: int, delay: float = {timeout}) -> dict[str, Any]:\n"
                f"    await asyncio.sleep(delay)\n"
                f"    return {{'i': i, 'domain': '{domain}', 'ok': True}}\n\n"
                f"async def main(n: int = {n}) -> None:\n"
                f"    tasks = [asyncio.create_task(job(i, {timeout} + (i % 5) * 0.01)) for i in range(n)]\n"
                f"    for fut in asyncio.as_completed(tasks):\n"
                f"        print(await fut)\n\n"
                f"if __name__ == '__main__':\n    asyncio.run(main())\n```\n\n"
                f"**주의**: 순서 보존이 필요하면 gather를 사용하세요."
            )
            yield _record("python", "async_await", user, assistant)

        elif kind == "refactor":
            a, b = round(rng.uniform(0.6, 0.9), 2), round(rng.uniform(0.85, 0.99), 2)
            user = (
                f"{domain} 할인 로직을 {verb}해 주세요. 목표: {goal}. "
                f"if-elif를 테이블로 바꾸세요.\n\n"
                f"```python\ndef price_of(tier, price):\n"
                f"    if tier == 'gold': return price * {a}\n"
                f"    elif tier == 'silver': return price * {b}\n"
                f"    elif tier == 'bronze': return price\n"
                f"    return price\n```"
            )
            assistant = (
                f"정책은 데이터로 분리합니다.\n\n"
                f"```python\nfrom typing import Mapping\n"
                f"RATES: Mapping[str, float] = {{'gold': {a}, 'silver': {b}, 'bronze': 1.0}}\n\n"
                f"def price_of(tier: str, price: float) -> float:\n"
                f"    if price < 0: raise ValueError('price < 0')\n"
                f"    rate = RATES.get(tier.lower())\n"
                f"    if rate is None: raise ValueError(f'unknown tier: {{tier!r}}')\n"
                f"    return price * rate\n```\n\n"
                f"**개선**: 새 등급은 딕셔너리만 수정하면 됩니다. ({goal})"
            )
            yield _record("python", "refactoring", user, assistant)

        elif kind == "debug":
            user = (
                f"다음 {domain} 코드 버그를 {verb}하고 한국어로 설명해 주세요.\n\n"
                f"```python\ndef {fn}(xs):\n"
                f"    out = []\n"
                f"    for x in xs:\n"
                f"        if x not in out: out.append(x)\n"
                f"    return out.sort()\n```"
            )
            assistant = (
                "`list.sort()`는 in-place 후 None을 반환합니다.\n\n"
                f"```python\nfrom typing import Hashable, Sequence, TypeVar\n"
                f"T = TypeVar('T', bound=Hashable)\n\n"
                f"def {fn}(xs: Sequence[T]) -> list[T]:\n"
                f"    seen: set[T] = set()\n    out: list[T] = []\n"
                f"    for x in xs:\n"
                f"        if x not in seen:\n            seen.add(x); out.append(x)\n"
                f"    return sorted(out)\n```\n\n"
                f"**원인**: sort 반환값 오해가 버그입니다. ({domain}/{goal})"
            )
            yield _record("python", "debugging", user, assistant)

        elif kind == "dataclass":
            name = rng.choice(["Event", "Task", "Job", "Metric", "Signal"]) + str(rng.randint(1, 99))
            user = (
                f"{domain}용 dataclass `{name}` 를 {verb}해 주세요. "
                f"`__post_init__` 검증과 default_factory를 포함하고 목표({goal})를 설명하세요."
            )
            assistant = (
                f"```python\nfrom dataclasses import dataclass, field\nfrom datetime import datetime, timezone\n\n"
                f"@dataclass\nclass {name}:\n"
                f"    id: int\n    title: str\n    domain: str = '{domain}'\n"
                f"    tags: list[str] = field(default_factory=list)\n"
                f"    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))\n\n"
                f"    def __post_init__(self) -> None:\n"
                f"        if self.id <= 0: raise ValueError('id')\n"
                f"        if not self.title.strip(): raise ValueError('title')\n```\n\n"
                f"**포인트**: 가변 기본값은 default_factory. ({goal})"
            )
            yield _record("python", "dataclass", user, assistant)

        elif kind == "contextmgr":
            user = (
                f"{domain} 자원 관리용 context manager를 {verb}해 주세요. "
                f"예외 시에도 정리되고, 목표={goal}."
            )
            assistant = (
                f"```python\nfrom contextlib import contextmanager\nfrom typing import Iterator\n\n"
                f"@contextmanager\ndef {domain.replace(' ', '_')}_resource(name: str) -> Iterator[str]:\n"
                f"    handle = f'open:{{name}}'\n    try:\n        yield handle\n"
                f"    finally:\n        # 예외가 나도 정리\n        _ = handle\n```\n\n"
                f"**원리**: finally에서 해제하면 {goal}에 도움이 됩니다."
            )
            yield _record("python", "context_manager", user, assistant)

        elif kind == "generator":
            user = (
                f"{domain} 로그를 {n}개씩 묶는 generator `{fn}` 를 {verb}해 주세요. 타입 힌팅 포함."
            )
            assistant = (
                f"```python\nfrom typing import Iterable, Iterator, TypeVar\nT = TypeVar('T')\n\n"
                f"def {fn}(items: Iterable[T], size: int = {n}) -> Iterator[list[T]]:\n"
                f"    if size <= 0: raise ValueError('size')\n"
                f"    batch: list[T] = []\n"
                f"    for x in items:\n"
                f"        batch.append(x)\n"
                f"        if len(batch) == size:\n            yield batch\n            batch = []\n"
                f"    if batch: yield batch\n```\n\n"
                f"**용도**: 대량 {domain} 데이터를 메모리 친화적으로 처리."
            )
            yield _record("python", "generator", user, assistant)

        else:
            user = (
                f"{domain}의 `{fn}` 에 대한 pytest 예제를 {verb}해 주세요. "
                f"정상/예외 케이스 포함, 목표={goal}."
            )
            assistant = (
                f"```python\nimport pytest\n\n"
                f"def {fn}(x: int) -> int:\n"
                f"    if x < 0: raise ValueError('x')\n    return x * 2\n\n"
                f"def test_{fn}_ok() -> None:\n    assert {fn}(2) == 4\n\n"
                f"def test_{fn}_error() -> None:\n"
                f"    with pytest.raises(ValueError):\n        {fn}(-1)\n```\n\n"
                f"**포인트**: 예외 경로 테스트가 {goal}을(를) 지켜줍니다."
            )
            yield _record("python", "pytest", user, assistant)


def gen_java(rng: random.Random) -> Iterator[dict[str, Any]]:
    while True:
        kind = rng.choice(["oop", "spring_ex", "dto", "validation", "service", "stream"])
        cls = rng.choice(JAVA_NAMES)
        domain = rng.choice(DOMAINS)
        goal = rng.choice(GOALS)
        verb = rng.choice(VERBS)
        code = rng.choice([
            "NOT_FOUND", "CONFLICT", "BAD_REQUEST", "FORBIDDEN", "GONE", "UNPROCESSABLE"
        ])
        status = {
            "NOT_FOUND": 404, "CONFLICT": 409, "BAD_REQUEST": 400,
            "FORBIDDEN": 403, "GONE": 410, "UNPROCESSABLE": 422,
        }[code]
        http = {
            400: "BAD_REQUEST", 403: "FORBIDDEN", 404: "NOT_FOUND",
            409: "CONFLICT", 410: "GONE", 422: "UNPROCESSABLE_ENTITY",
        }[status]

        if kind == "oop":
            user = (
                f"{domain} 도메인 `{cls}` 클래스를 {verb}해 주세요. "
                f"식별자는 final, 상태 전이 검증 포함. 목표={goal}. 한국어 설명."
            )
            assistant = (
                f"```java\npublic final class {cls} {{\n"
                f"    private final String id;\n    private String name;\n    private boolean active;\n\n"
                f"    public {cls}(String id, String name) {{\n"
                f"        if (id == null || id.isBlank()) throw new IllegalArgumentException(\"id\");\n"
                f"        this.id = id; this.name = name; this.active = true;\n    }}\n\n"
                f"    public void deactivate() {{\n"
                f"        if (!active) throw new IllegalStateException(\"already inactive\");\n"
                f"        active = false;\n    }}\n"
                f"    public String getId() {{ return id; }}\n"
                f"    public boolean isActive() {{ return active; }}\n}}\n```\n\n"
                f"**설계**: 불변식 보장으로 {goal}을(를) 확보합니다."
            )
            yield _record("java", "oop_class", user, assistant)

        elif kind == "spring_ex":
            ex = f"{cls}{code.title().replace('_','')}Exception"
            user = (
                f"Spring에서 `{ex}` + `@RestControllerAdvice`를 {verb}해 주세요. "
                f"HTTP {status}, code={code}, 도메인={domain}."
            )
            assistant = (
                f"```java\npublic class {ex} extends RuntimeException {{\n"
                f"    public {ex}(String id) {{ super(\"{domain} error: \" + id); }}\n}}\n\n"
                f"public record ErrorResponse(String code, String message) {{}}\n\n"
                f"@RestControllerAdvice\npublic class GlobalExceptionHandler {{\n"
                f"    @ExceptionHandler({ex}.class)\n"
                f"    @ResponseStatus(HttpStatus.{http})\n"
                f"    public ErrorResponse handle({ex} ex) {{\n"
                f"        return new ErrorResponse(\"{code}\", ex.getMessage());\n    }}\n}}\n```\n\n"
                f"**포인트**: 도메인 예외와 HTTP 매핑 분리."
            )
            yield _record("java", "spring_exception", user, assistant)

        elif kind == "dto":
            user = (
                f"`Create{cls}Request/Response` DTO를 record+Validation으로 {verb}해 주세요. "
                f"도메인={domain}, 목표={goal}."
            )
            assistant = (
                f"```java\nimport jakarta.validation.constraints.*;\n\n"
                f"public record Create{cls}Request(\n"
                f"    @NotBlank @Size(max=100) String name,\n"
                f"    @NotBlank @Size(max=64) String code\n) {{}}\n\n"
                f"public record Create{cls}Response(Long id, String name, String code) {{}}\n```\n\n"
                f"**팁**: Entity 직접 노출 금지. ({goal})"
            )
            yield _record("java", "dto", user, assistant)

        elif kind == "validation":
            field = rng.choice(["email", "password", "code", "slug", "token"])
            user = (
                f"{domain}의 `{field}` 검증 유틸을 {verb}해 주세요. 실패 시 IAE."
            )
            assistant = (
                f"```java\npublic final class {cls}Validators {{\n"
                f"    private {cls}Validators() {{}}\n"
                f"    public static String require{field.capitalize()}(String v) {{\n"
                f"        if (v == null || v.isBlank()) throw new IllegalArgumentException(\"{field}\");\n"
                f"        if (v.length() < 3) throw new IllegalArgumentException(\"{field} too short\");\n"
                f"        return v.trim();\n    }}\n}}\n```\n\n"
                f"**원칙**: fail-fast. ({goal})"
            )
            yield _record("java", "validation", user, assistant)

        elif kind == "service":
            user = (
                f"`{cls}Service` 업데이트 유스케이스를 {verb}해 주세요. "
                f"없으면 예외, 있으면 name 변경. 도메인={domain}."
            )
            assistant = (
                f"```java\n@Service\npublic class {cls}Service {{\n"
                f"    private final {cls}Repository repo;\n"
                f"    public {cls}Service({cls}Repository repo) {{ this.repo = repo; }}\n\n"
                f"    @Transactional\n"
                f"    public {cls} rename(String id, String name) {{\n"
                f"        {cls} e = repo.findById(id).orElseThrow(() -> new IllegalArgumentException(id));\n"
                f"        e.rename(name);\n        return e;\n    }}\n}}\n```\n\n"
                f"**포인트**: 트랜잭션 경계는 서비스."
            )
            yield _record("java", "service", user, assistant)

        else:
            user = (
                f"Java Stream으로 {domain} `{cls}` 리스트에서 active만 남겨 name 정렬하는 코드를 {verb}해 주세요."
            )
            assistant = (
                f"```java\nimport java.util.*;\nimport java.util.stream.*;\n\n"
                f"public final class {cls}Queries {{\n"
                f"    private {cls}Queries() {{}}\n"
                f"    public static List<{cls}> activeSorted(List<{cls}> src) {{\n"
                f"        return src.stream()\n"
                f"            .filter({cls}::isActive)\n"
                f"            .sorted(Comparator.comparing({cls}::getName))\n"
                f"            .toList();\n    }}\n}}\n```\n\n"
                f"**주의**: 원본 리스트를 직접 sort하지 않습니다."
            )
            yield _record("java", "stream", user, assistant)


def gen_c(rng: random.Random) -> Iterator[dict[str, Any]]:
    while True:
        kind = rng.choice([
            "pointer", "malloc", "leak", "string", "struct", "realloc", "ownership"
        ])
        name = rng.choice(C_NAMES)
        domain = rng.choice(DOMAINS)
        goal = rng.choice(GOALS)
        verb = rng.choice(VERBS)
        n = rng.choice([4, 7, 8, 15, 16, 31, 32, 64, 100, 128, 256])
        fill = rng.randint(-5, 200)

        if kind == "pointer":
            user = (
                f"{domain}용 int 배열 `{name}_sum` 을 포인터로 {verb}해 주세요. "
                f"len={n} 가정 설명 포함, 목표={goal}."
            )
            assistant = (
                f"`arr[i]` == `*(arr+i)` 입니다.\n\n"
                f"```c\n#include <stddef.h>\n"
                f"long {name}_sum(const int *arr, size_t len) {{\n"
                f"    long total = 0; const int *p = arr; const int *end = arr + len;\n"
                f"    while (p < end) total += *p++;\n    return total;\n}}\n```\n\n"
                f"**주의**: end는 역참조 금지. ({goal})"
            )
            yield _record("c", "pointer_arithmetic", user, assistant)

        elif kind == "malloc":
            user = (
                f"`{name}` 버퍼 {n}개를 malloc하고 {fill}로 채운 뒤 free하는 코드를 {verb}해 주세요. "
                f"NULL/dangling 처리 포함. 도메인={domain}."
            )
            assistant = (
                f"```c\n#include <stdio.h>\n#include <stdlib.h>\n"
                f"int *{name}_new(size_t n, int fill) {{\n"
                f"    int *p = (int*)malloc(n * sizeof(int));\n"
                f"    if (!p) return NULL;\n"
                f"    for (size_t i=0;i<n;i++) p[i]=fill;\n    return p;\n}}\n"
                f"void {name}_free(int **pp) {{\n"
                f"    if (!pp || !*pp) return; free(*pp); *pp=NULL;\n}}\n"
                f"int main(void){{\n"
                f"    int *b={name}_new({n},{fill});\n"
                f"    if(!b){{fprintf(stderr,\"oom\\n\"); return 1;}}\n"
                f"    {name}_free(&b); return 0;\n}}\n```\n\n"
                f"**규칙**: malloc/free 짝. ({goal})"
            )
            yield _record("c", "malloc_free", user, assistant)

        elif kind == "leak":
            label = rng.choice(["alpha", "beta", "gamma", "delta", "omega", "sigma"])
            user = (
                f"아래 {domain} C 코드의 누수/오버플로를 {verb}해 주세요.\n\n"
                f"```c\n#include <stdlib.h>\n#include <string.h>\n"
                f"char *dup_{name}(const char *s){{\n"
                f"  size_t n=strlen(s); char *b=(char*)malloc(n); memcpy(b,s,n); return b;\n}}\n"
                f"void demo(void){{ char *a=dup_{name}(\"{label}\"); char *b=dup_{name}(\"x\"); free(a); }}\n```"
            )
            assistant = (
                "1) n+1 미할당 2) NUL 미기록 3) b 미해제\n\n"
                f"```c\n#include <stdlib.h>\n#include <string.h>\n"
                f"char *dup_{name}(const char *s){{\n"
                f"  if(!s) return NULL; size_t n=strlen(s);\n"
                f"  char *b=(char*)malloc(n+1); if(!b) return NULL;\n"
                f"  memcpy(b,s,n); b[n]='\\0'; return b;\n}}\n"
                f"void demo(void){{ char *a=dup_{name}(\"{label}\"); char *b=dup_{name}(\"x\"); free(a); free(b); a=b=NULL; }}\n```\n\n"
                f"**팁**: valgrind. ({goal})"
            )
            yield _record("c", "memory_leak_debug", user, assistant)

        elif kind == "string":
            user = (
                f"`{name}_strlen` 을 포인터로 {verb}해 주세요. 표준 strlen과 차이 설명. 도메인={domain}."
            )
            assistant = (
                f"```c\n#include <stddef.h>\nsize_t {name}_strlen(const char *s){{\n"
                f"  const char *p=s; if(!p) return 0; while(*p) p++; return (size_t)(p-s);\n}}\n```\n\n"
                f"**차이**: 표준 구현은 최적화됨. NULL 정책은 명시하세요."
            )
            yield _record("c", "string_pointer", user, assistant)

        elif kind == "struct":
            user = (
                f"{domain}용 `{name}` 구조체 동적 배열({n}) 할당/해제를 {verb}해 주세요. 목표={goal}."
            )
            assistant = (
                f"```c\n#include <stdlib.h>\ntypedef struct {{ int x; int y; }} {name}_point;\n"
                f"{name}_point *{name}_points_new(size_t n){{ return ({name}_point*)calloc(n, sizeof(*({name}_point*)0)); }}\n"
                f"/* 위 sizeof는 가독성 위해 아래처럼 쓰는 편: */\n"
                f"{name}_point *{name}_new(size_t n){{ return ({name}_point*)calloc(n, sizeof({name}_point)); }}\n"
                f"void {name}_free({name}_point **pp){{ if(!pp||!*pp) return; free(*pp); *pp=NULL; }}\n```\n\n"
                f"**팁**: calloc 0초기화. ({goal})"
            )
            yield _record("c", "struct_alloc", user, assistant)

        elif kind == "realloc":
            user = (
                f"`{name}` 동적 배열을 {n}에서 2배 grow 하는 realloc 래퍼를 {verb}해 주세요. 실패 시 원본 보존."
            )
            assistant = (
                f"```c\n#include <stdlib.h>\nint *{name}_grow(int *p, size_t *cap){{\n"
                f"  size_t ncap = (*cap ? *cap*2 : {n});\n"
                f"  int *np = (int*)realloc(p, ncap*sizeof(int));\n"
                f"  if(!np) return NULL; /* p는 그대로 유효 */\n"
                f"  *cap = ncap; return np;\n}}\n```\n\n"
                f"**중요**: realloc 실패 시 원본 포인터를 잃지 마세요."
            )
            yield _record("c", "realloc_grow", user, assistant)

        else:
            user = (
                f"{domain}에서 `{name}_create/destroy` 소유권 규칙을 {verb}해 주세요. "
                f"생성자/파괴자 쌍과 문서화 포함. 목표={goal}."
            )
            assistant = (
                f"```c\n#include <stdlib.h>\ntypedef struct {{ int *data; size_t n; }} {name}_t;\n"
                f"{name}_t *{name}_create(size_t n){{\n"
                f"  {name}_t *o=({name}_t*)malloc(sizeof(*o)); if(!o) return NULL;\n"
                f"  o->data=(int*)calloc(n,sizeof(int)); if(!o->data){{free(o); return NULL;}}\n"
                f"  o->n=n; return o;\n}}\n"
                f"void {name}_destroy({name}_t **po){{\n"
                f"  if(!po||!*po) return; free((*po)->data); free(*po); *po=NULL;\n}}\n```\n\n"
                f"**규칙**: create한 쪽이 destroy. ({goal})"
            )
            yield _record("c", "ownership", user, assistant)


def allocate_counts(total: int) -> dict[str, int]:
    py = int(total * 0.35)
    java = int(total * 0.35)
    c = total - py - java
    return {"python": py, "java": java, "c": c}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="다양성 강화 데이터셋 생성")
    p.add_argument("--unique-target", type=int, default=3000, help="목표 고유 샘플 수")
    p.add_argument("--output", default="dataset.jsonl")
    p.add_argument("--include-seed", action="store_true")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--backup", action="store_true")
    p.add_argument("--max-attempts", type=int, default=200000, help="중복 많아질 때 상한")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.output)

    if args.backup and out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        bak.write_bytes(out_path.read_bytes())
        print(f"[백업] {out_path} -> {bak}")

    records: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_near: set[str] = set()

    def try_add(rec: dict[str, Any]) -> bool:
        eh = _h(rec["text"])
        if eh in seen_exact:
            return False
        user = ""
        for m in rec["messages"]:
            if m["role"] == "user":
                user = m["content"]
                break
        nh = _h(_norm(user))
        if nh in seen_near:
            return False
        seen_exact.add(eh)
        seen_near.add(nh)
        records.append(rec)
        return True

    if args.include_seed:
        for ex in build_raw_examples():
            rec = {
                "language": ex["language"],
                "topic": ex["topic"],
                "text": format_gemma4_chat(ex["messages"], add_bos=True),
                "messages": ex["messages"],
            }
            try_add(rec)

    target = args.unique_target
    remaining = max(0, target - len(records))
    counts = allocate_counts(remaining)
    print(f"[목표] unique={target} 추가할당={counts}")

    gens = {
        "python": gen_python(rng),
        "java": gen_java(rng),
        "c": gen_c(rng),
    }
    need = dict(counts)
    attempts = 0
    while sum(need.values()) > 0 and attempts < args.max_attempts:
        attempts += 1
        # 아직 필요한 언어 중 하나 선택
        langs = [k for k, v in need.items() if v > 0]
        if not langs:
            break
        lang = rng.choice(langs)
        rec = next(gens[lang])
        if try_add(rec):
            need[lang] -= 1
        if attempts % 2000 == 0:
            print(f"  ... attempts={attempts} unique={len(records)} need={need}")

    rng.shuffle(records)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    lang_counts = Counter(r["language"] for r in records)
    print(f"[OK] {out_path} unique={len(records)} attempts={attempts}")
    for lang, cnt in sorted(lang_counts.items()):
        print(f"  - {lang}: {cnt} ({100*cnt/len(records):.1f}%)")
    if sum(need.values()) > 0:
        print(f"[경고] 목표 미달 잔여 need={need} (템플릿 고갈). HQ 파이프라인을 함께 쓰세요.")


if __name__ == "__main__":
    main()
