#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py
------------------
Python / Java / C언어 코딩 인스트럭션 샘플을 Gemma 4 Official Chat Template
포맷으로 변환하여 dataset.jsonl 을 생성합니다.

Gemma 4 채팅 규칙 (공식):
  - 역할: system / user / assistant
  - 턴 마커: <start_of_turn>…<end_of_turn>
  - 파일 시작에 <bos> 를 둡니다 (학습 시 토크나이저가 중복 삽입하지 않도록 주의)

언어 비중 목표: Python 35% / Java 35% / C 30%
(샘플 단계에서는 언어별 3개씩 = 총 9개로 균형 시드 데이터를 구성합니다.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Gemma 4 Chat Template 포맷터
# ---------------------------------------------------------------------------
def format_gemma4_chat(messages: list[dict[str, str]], add_bos: bool = True) -> str:
    """
    messages = [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    를 Gemma 4 공식 턴 포맷 문자열로 변환합니다.

    학습용 text 필드에는 전체 대화(응답 포함)를 넣습니다.
    추론 시에는 assistant 턴을 빼고 add_generation_prompt=True 로 생성합니다.
    """
    parts: list[str] = []
    if add_bos:
        parts.append("<bos>")

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        # Gemma 4: system / user / assistant 역할을 그대로 턴 이름으로 사용
        parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# 시스템 프롬프트 (코딩 AI 페르소나)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "당신은 Python, Java, C언어를 전문으로 다루는 시니어 코딩 AI입니다. "
    "코드 작성·리팩토링·디버깅 요청에 대해 정확한 코드와 함께 "
    "한국어로 핵심 원리와 주의사항을 명확히 설명합니다."
)


# ---------------------------------------------------------------------------
# 시드 데이터셋 (언어별 3개)
# ---------------------------------------------------------------------------
def build_raw_examples() -> list[dict[str, Any]]:
    """언어·주제 태그가 붙은 원시 (messages) 예제 목록을 반환합니다."""

    examples: list[dict[str, Any]] = []

    # ===== Python (35%) — 타입 힌팅 / 비동기 / 리팩토링 =====
    examples.append(
        {
            "language": "python",
            "topic": "type_hinting",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "다음 함수에 타입 힌팅을 추가하고, TypedDict 또는 dataclass로 "
                        "반환 구조를 명확히 해 주세요. 한국어로 변경 이유도 설명해 주세요.\n\n"
                        "```python\n"
                        "def summarize_scores(scores):\n"
                        "    total = 0\n"
                        "    for s in scores:\n"
                        "        total += s\n"
                        "    avg = total / len(scores)\n"
                        "    return {'total': total, 'average': avg, 'count': len(scores)}\n"
                        "```"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "반환 키가 고정되어 있으므로 `TypedDict`로 스키마를 고정하고, "
                        "`Sequence[float]`로 입력 타입을 열어 두는 편이 안전합니다. "
                        "빈 리스트 나눗셈은 `ZeroDivisionError`가 나므로 가드도 추가합니다.\n\n"
                        "```python\n"
                        "from typing import Sequence, TypedDict\n\n"
                        "class ScoreSummary(TypedDict):\n"
                        "    total: float\n"
                        "    average: float\n"
                        "    count: int\n\n"
                        "def summarize_scores(scores: Sequence[float]) -> ScoreSummary:\n"
                        "    if not scores:\n"
                        "        raise ValueError('scores는 비어 있을 수 없습니다.')\n"
                        "    total = float(sum(scores))\n"
                        "    count = len(scores)\n"
                        "    return {\n"
                        "        'total': total,\n"
                        "        'average': total / count,\n"
                        "        'count': count,\n"
                        "    }\n"
                        "```\n\n"
                        "**포인트**: TypedDict는 런타임 검증이 아니라 정적 분석(mypy/pyright)용입니다. "
                        "런타임 검증이 필요하면 pydantic 모델을 고려하세요."
                    ),
                },
            ],
        }
    )

    examples.append(
        {
            "language": "python",
            "topic": "async_await",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "여러 URL에서 JSON을 동시에 받아오는 async 함수를 작성해 주세요. "
                        "`aiohttp` 또는 `httpx.AsyncClient`를 쓰고, 개별 실패는 결과에 남기되 "
                        "전체는 중단되지 않게 해 주세요. 한국어로 설명해 주세요."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "`asyncio.gather(..., return_exceptions=True)`로 개별 예외를 수집하면 "
                        "한 요청 실패가 전체를 죽이지 않습니다. 세션/클라이언트는 한 번만 열어 "
                        "연결 풀을 재사용하세요.\n\n"
                        "```python\n"
                        "from __future__ import annotations\n\n"
                        "import asyncio\n"
                        "from typing import Any\n\n"
                        "import httpx\n\n"
                        "async def fetch_json(\n"
                        "    client: httpx.AsyncClient, url: str\n"
                        ") -> dict[str, Any]:\n"
                        "    response = await client.get(url, timeout=10.0)\n"
                        "    response.raise_for_status()\n"
                        "    return {'url': url, 'ok': True, 'data': response.json()}\n\n"
                        "async def fetch_all(urls: list[str]) -> list[dict[str, Any]]:\n"
                        "    async with httpx.AsyncClient() as client:\n"
                        "        tasks = [fetch_json(client, url) for url in urls]\n"
                        "        results = await asyncio.gather(*tasks, return_exceptions=True)\n\n"
                        "    output: list[dict[str, Any]] = []\n"
                        "    for url, result in zip(urls, results):\n"
                        "        if isinstance(result, Exception):\n"
                        "            output.append({'url': url, 'ok': False, 'error': str(result)})\n"
                        "        else:\n"
                        "            output.append(result)\n"
                        "    return output\n"
                        "```\n\n"
                        "**주의**: 동기 `requests`를 async 함수 안에서 호출하면 이벤트 루프가 "
                        "막힙니다. 반드시 async HTTP 클라이언트를 사용하세요."
                    ),
                },
            ],
        }
    )

    examples.append(
        {
            "language": "python",
            "topic": "refactoring",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "아래 코드를 가독성과 확장성 관점에서 리팩토링해 주세요. "
                        "중복 if-elif를 줄이고, 한국어로 개선 이유를 설명해 주세요.\n\n"
                        "```python\n"
                        "def apply_discount(price, user_type):\n"
                        "    if user_type == 'vip':\n"
                        "        return price * 0.8\n"
                        "    elif user_type == 'member':\n"
                        "        return price * 0.9\n"
                        "    elif user_type == 'guest':\n"
                        "        return price\n"
                        "    else:\n"
                        "        return price\n"
                        "```"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "할인율 테이블을 딕셔너리로 분리하면 새 등급 추가 시 분기문을 "
                        "늘리지 않아도 됩니다. 가격·등급 검증도 함수 입구에서 한 번에 합니다.\n\n"
                        "```python\n"
                        "from typing import Mapping\n\n"
                        "DISCOUNT_RATES: Mapping[str, float] = {\n"
                        "    'vip': 0.8,\n"
                        "    'member': 0.9,\n"
                        "    'guest': 1.0,\n"
                        "}\n\n"
                        "def apply_discount(price: float, user_type: str) -> float:\n"
                        "    if price < 0:\n"
                        "        raise ValueError('price는 0 이상이어야 합니다.')\n"
                        "    rate = DISCOUNT_RATES.get(user_type.lower())\n"
                        "    if rate is None:\n"
                        "        raise ValueError(f'알 수 없는 user_type: {user_type!r}')\n"
                        "    return price * rate\n"
                        "```\n\n"
                        "**개선 요약**: 제어 흐름(if)과 데이터(할인율)를 분리해 테스트·확장이 쉬워집니다."
                    ),
                },
            ],
        }
    )

    # ===== Java (35%) — OOP / Spring 예외 / DTO =====
    examples.append(
        {
            "language": "java",
            "topic": "oop_class",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "은행 계좌를 나타내는 Java 클래스를 OOP로 설계해 주세요. "
                        "입금/출금, 잔액 조회, 불변 계좌번호 필드를 포함하고 "
                        "출금 시 잔액 부족이면 예외를 던지도록 해 주세요. 한국어로 설명해 주세요."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "계좌번호는 생성 후 변하지 않으므로 `final`로 두고, 잔액 변경은 "
                        "검증된 메서드로만 허용해 캡슐화를 지킵니다.\n\n"
                        "```java\n"
                        "public final class BankAccount {\n"
                        "    private final String accountNumber;\n"
                        "    private long balance; // 원 단위\n\n"
                        "    public BankAccount(String accountNumber, long initialBalance) {\n"
                        "        if (accountNumber == null || accountNumber.isBlank()) {\n"
                        "            throw new IllegalArgumentException(\"계좌번호가 비어 있습니다.\");\n"
                        "        }\n"
                        "        if (initialBalance < 0) {\n"
                        "            throw new IllegalArgumentException(\"초기 잔액은 0 이상이어야 합니다.\");\n"
                        "        }\n"
                        "        this.accountNumber = accountNumber;\n"
                        "        this.balance = initialBalance;\n"
                        "    }\n\n"
                        "    public void deposit(long amount) {\n"
                        "        if (amount <= 0) {\n"
                        "            throw new IllegalArgumentException(\"입금액은 양수여야 합니다.\");\n"
                        "        }\n"
                        "        balance += amount;\n"
                        "    }\n\n"
                        "    public void withdraw(long amount) {\n"
                        "        if (amount <= 0) {\n"
                        "            throw new IllegalArgumentException(\"출금액은 양수여야 합니다.\");\n"
                        "        }\n"
                        "        if (amount > balance) {\n"
                        "            throw new IllegalStateException(\"잔액이 부족합니다.\");\n"
                        "        }\n"
                        "        balance -= amount;\n"
                        "    }\n\n"
                        "    public String getAccountNumber() { return accountNumber; }\n"
                        "    public long getBalance() { return balance; }\n"
                        "}\n"
                        "```\n\n"
                        "**설계 이유**: 필드를 private로 두고 상태 전이를 메서드로만 허용하면 "
                        "불변식(잔액 ≥ 0)을 클래스 내부에서 보장할 수 있습니다."
                    ),
                },
            ],
        }
    )

    examples.append(
        {
            "language": "java",
            "topic": "spring_exception",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Spring Boot 스타일로 주문 미발견 예외와 "
                        "`@RestControllerAdvice` 전역 예외 핸들러 예제를 작성해 주세요. "
                        "HTTP 404와 에러 본식을 JSON으로 반환하고 한국어로 설명해 주세요."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "도메인 예외와 HTTP 매핑을 분리하면 서비스 계층은 비즈니스 의미만 "
                        "표현하고, 웹 계층에서 상태 코드를 결정합니다.\n\n"
                        "```java\n"
                        "public class OrderNotFoundException extends RuntimeException {\n"
                        "    private final Long orderId;\n\n"
                        "    public OrderNotFoundException(Long orderId) {\n"
                        "        super(\"주문을 찾을 수 없습니다: \" + orderId);\n"
                        "        this.orderId = orderId;\n"
                        "    }\n\n"
                        "    public Long getOrderId() { return orderId; }\n"
                        "}\n\n"
                        "public record ErrorResponse(String code, String message) {}\n\n"
                        "@RestControllerAdvice\n"
                        "public class GlobalExceptionHandler {\n\n"
                        "    @ExceptionHandler(OrderNotFoundException.class)\n"
                        "    @ResponseStatus(HttpStatus.NOT_FOUND)\n"
                        "    public ErrorResponse handleOrderNotFound(OrderNotFoundException ex) {\n"
                        "        return new ErrorResponse(\"ORDER_NOT_FOUND\", ex.getMessage());\n"
                        "    }\n"
                        "}\n"
                        "```\n\n"
                        "**포인트**: `@ResponseStatus` 또는 `ResponseEntity`로 상태 코드를 "
                        "명시하고, 클라이언트에 내부 스택트레이스는 노출하지 마세요."
                    ),
                },
            ],
        }
    )

    examples.append(
        {
            "language": "java",
            "topic": "dto",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "회원 가입 API용 Request/Response DTO를 Java record로 작성하고, "
                        "Jakarta Validation 애노테이션을 붙여 주세요. 한국어로 설명해 주세요."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "요청 DTO는 검증 애노테이션으로 입력을 방어하고, 응답 DTO는 "
                        "비밀번호 같은 민감 정보를 제외해 노출 범위를 최소화합니다.\n\n"
                        "```java\n"
                        "import jakarta.validation.constraints.Email;\n"
                        "import jakarta.validation.constraints.NotBlank;\n"
                        "import jakarta.validation.constraints.Size;\n\n"
                        "public record SignUpRequest(\n"
                        "    @NotBlank @Email String email,\n"
                        "    @NotBlank @Size(min = 8, max = 64) String password,\n"
                        "    @NotBlank @Size(max = 50) String nickname\n"
                        ") {}\n\n"
                        "public record SignUpResponse(\n"
                        "    Long userId,\n"
                        "    String email,\n"
                        "    String nickname\n"
                        ") {}\n"
                        "```\n\n"
                        "컨트롤러에서는 `@Valid @RequestBody SignUpRequest request`로 받고, "
                        "서비스에서 엔티티 저장 후 `SignUpResponse`만 반환하면 됩니다.\n\n"
                        "**팁**: Entity를 API에 직접 노출하지 말고 DTO로 변환하세요 "
                        "(지연 로딩·순환 참조·과도한 필드 노출을 막습니다)."
                    ),
                },
            ],
        }
    )

    # ===== C언어 (30%) — 포인터 / malloc·free / 누수 디버깅 =====
    examples.append(
        {
            "language": "c",
            "topic": "pointer_arithmetic",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "C언어에서 int 배열의 합을 포인터 연산으로 구하는 함수를 작성하고, "
                        "`arr[i]`와 `*(arr + i)`의 관계를 한국어로 설명해 주세요."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "`arr[i]`는 문법 설탕이고, 의미상 `*(arr + i)`와 같습니다. "
                        "`arr + i`는 `i * sizeof(int)` 바이트만큼 전진한 주소입니다.\n\n"
                        "```c\n"
                        "#include <stddef.h>\n\n"
                        "long sum_array(const int *arr, size_t len) {\n"
                        "    long total = 0;\n"
                        "    const int *p = arr;\n"
                        "    const int *end = arr + len; /* 끝 바로 다음(past-the-end) */\n\n"
                        "    while (p < end) {\n"
                        "        total += *p; /* 역참조로 값 읽기 */\n"
                        "        p++;         /* 다음 int 요소로 이동 */\n"
                        "    }\n"
                        "    return total;\n"
                        "}\n"
                        "```\n\n"
                        "**주의**: `end`는 역참조하지 마세요. 유효 구간은 `[arr, arr+len)` 입니다."
                    ),
                },
            ],
        }
    )

    examples.append(
        {
            "language": "c",
            "topic": "malloc_free",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "동적으로 int 배열을 할당·초기화·해제하는 C 코드를 작성해 주세요. "
                        "`malloc` 실패 처리와 `free` 후 포인터 NULL 설정도 포함해 "
                        "한국어로 설명해 주세요."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "`malloc`은 실패 시 `NULL`을 반환하므로 반드시 검사합니다. "
                        "`free` 후에는 dangling pointer를 막기 위해 `NULL`을 대입하는 습관이 안전합니다.\n\n"
                        "```c\n"
                        "#include <stdio.h>\n"
                        "#include <stdlib.h>\n\n"
                        "int *create_int_array(size_t n, int fill) {\n"
                        "    int *arr = (int *)malloc(n * sizeof(int));\n"
                        "    if (arr == NULL) {\n"
                        "        return NULL; /* 호출자가 실패를 처리 */\n"
                        "    }\n"
                        "    for (size_t i = 0; i < n; i++) {\n"
                        "        arr[i] = fill;\n"
                        "    }\n"
                        "    return arr;\n"
                        "}\n\n"
                        "void destroy_int_array(int **arr_ptr) {\n"
                        "    if (arr_ptr == NULL || *arr_ptr == NULL) {\n"
                        "        return;\n"
                        "    }\n"
                        "    free(*arr_ptr);\n"
                        "    *arr_ptr = NULL; /* dangling 방지 */\n"
                        "}\n\n"
                        "int main(void) {\n"
                        "    int *nums = create_int_array(5, 42);\n"
                        "    if (nums == NULL) {\n"
                        "        fprintf(stderr, \"메모리 할당 실패\\n\");\n"
                        "        return 1;\n"
                        "    }\n"
                        "    /* ... 사용 ... */\n"
                        "    destroy_int_array(&nums);\n"
                        "    return 0;\n"
                        "}\n"
                        "```\n\n"
                        "**규칙**: `malloc`한 횟수만큼 `free`하고, 이미 free한 포인터를 "
                        "다시 free하지 마세요(double-free)."
                    ),
                },
            ],
        }
    )

    examples.append(
        {
            "language": "c",
            "topic": "memory_leak_debug",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "아래 C 코드의 메모리 누수/버그를 찾아 수정해 주세요. "
                        "무엇이 잘못됐는지 한국어로 설명해 주세요.\n\n"
                        "```c\n"
                        "#include <stdlib.h>\n"
                        "#include <string.h>\n\n"
                        "char *dup_upper(const char *s) {\n"
                        "    size_t n = strlen(s);\n"
                        "    char *buf = (char *)malloc(n);  /* 버그 있음 */\n"
                        "    for (size_t i = 0; i < n; i++) {\n"
                        "        char c = s[i];\n"
                        "        if (c >= 'a' && c <= 'z') c = (char)(c - 32);\n"
                        "        buf[i] = c;\n"
                        "    }\n"
                        "    return buf;\n"
                        "}\n\n"
                        "void demo(void) {\n"
                        "    char *a = dup_upper(\"hello\");\n"
                        "    char *b = dup_upper(\"world\");\n"
                        "    /* a만 free하고 b는 잊음 */\n"
                        "    free(a);\n"
                        "}\n"
                        "```"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "문제점 세 가지입니다.\n"
                        "1) `malloc(n)`은 NUL 종단(`'\\0'`) 공간을 빼먹었습니다 → 버퍼 오버플로.\n"
                        "2) `'\\0'`을 쓰지 않아 C 문자열로 안전하지 않습니다.\n"
                        "3) `demo`에서 `b`를 free하지 않아 누수가 납니다.\n\n"
                        "수정본:\n\n"
                        "```c\n"
                        "#include <stdlib.h>\n"
                        "#include <string.h>\n\n"
                        "char *dup_upper(const char *s) {\n"
                        "    if (s == NULL) return NULL;\n"
                        "    size_t n = strlen(s);\n"
                        "    char *buf = (char *)malloc(n + 1); /* +1 for '\\0' */\n"
                        "    if (buf == NULL) return NULL;\n\n"
                        "    for (size_t i = 0; i < n; i++) {\n"
                        "        char c = s[i];\n"
                        "        if (c >= 'a' && c <= 'z') c = (char)(c - 32);\n"
                        "        buf[i] = c;\n"
                        "    }\n"
                        "    buf[n] = '\\0';\n"
                        "    return buf;\n"
                        "}\n\n"
                        "void demo(void) {\n"
                        "    char *a = dup_upper(\"hello\");\n"
                        "    char *b = dup_upper(\"world\");\n"
                        "    free(a);\n"
                        "    free(b); /* 할당과 해제를 짝으로 */\n"
                        "    a = b = NULL;\n"
                        "}\n"
                        "```\n\n"
                        "**디버깅 팁**: Linux/WSL에서는 `valgrind --leak-check=full ./a.out` 으로 "
                        "누수와 invalid write를 빠르게 확인할 수 있습니다."
                    ),
                },
            ],
        }
    )

    return examples


def main() -> None:
    """시드 예제를 Gemma 4 템플릿 text로 변환해 dataset.jsonl에 저장합니다."""
    raw = build_raw_examples()
    out_path = Path(__file__).resolve().parent / "dataset.jsonl"

    # 언어 비중 확인용 카운트
    lang_counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as f:
        for ex in raw:
            text = format_gemma4_chat(ex["messages"], add_bos=True)
            record = {
                "language": ex["language"],
                "topic": ex["topic"],
                # SFTTrainer dataset_text_field="text" 용
                "text": text,
                # 필요 시 재포맷/검수용 원본 메시지
                "messages": ex["messages"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            lang_counts[ex["language"]] = lang_counts.get(ex["language"], 0) + 1

    total = sum(lang_counts.values())
    # Windows cp949 콘솔 호환을 위해 ASCII 하이픈만 사용
    print(f"[OK] {out_path} 생성 완료 - 총 {total}개 샘플")
    for lang, cnt in sorted(lang_counts.items()):
        pct = 100.0 * cnt / total
        print(f"  - {lang}: {cnt}개 ({pct:.1f}%)")
    print(
        "\n참고: 실제 학습 전에는 언어 비중 Python 35% / Java 35% / C 30%에 맞게 "
        "데이터를 확장하는 것을 권장합니다."
    )


if __name__ == "__main__":
    main()
