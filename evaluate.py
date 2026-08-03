#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py
-----------
학습이 끝난 뒤, 저장된 LoRA 어댑터를 불러와
"실제로 코딩 질문에 잘 답하는지" 눈으로 확인하는 추론 테스트 스크립트입니다.

train.py  = 공부(가중치 학습)
evaluate.py = 시험(샘플 질문 3개 던져보고 답변 출력)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hf_env import get_hf_token, load_dotenv

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


SYSTEM_PROMPT = (
    "당신은 Python, Java, C언어를 전문으로 다루는 시니어 코딩 AI입니다. "
    "코드 작성·리팩토링·디버깅 요청에 대해 정확한 코드와 함께 "
    "한국어로 핵심 원리와 주의사항을 명확히 설명합니다."
)


EVAL_PROMPTS: list[dict[str, str]] = [
    {
        "name": "C - 포인터 스왑",
        "language": "c",
        "user": (
            "C언어에서 두 int 변수의 값을 포인터로 교환하는 swap 함수를 작성하고, "
            "호출 예제와 함께 한국어로 설명해 주세요. "
            "왜 포인터(또는 주소)가 필요한지도 포함해 주세요."
        ),
    },
    {
        "name": "Java - 클래스 구현",
        "language": "java",
        "user": (
            "도서(Book)를 나타내는 Java 클래스를 작성해 주세요. "
            "제목·저자·ISBN(불변)·대여 가능 여부 필드를 두고, "
            "대여(borrow)/반납(returnBook) 메서드에서 상태 전이를 검증하세요. "
            "한국어로 설계 이유를 설명해 주세요."
        ),
    },
    {
        "name": "Python - 비동기 함수",
        "language": "python",
        "user": (
            "asyncio를 사용해 N개의 작업을 동시에 실행하고, "
            "완료되는 순서대로 결과를 출력하는 예제를 작성해 주세요. "
            "`asyncio.as_completed`를 쓰고 타입 힌팅을 포함해 한국어로 설명해 주세요."
        ),
    },
]


def build_prompt(tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts = ["<bos>"]
    for msg in messages:
        parts.append(
            f"<start_of_turn>{msg['role']}\n{msg['content'].strip()}<end_of_turn>\n"
        )
    parts.append("<start_of_turn>assistant\n")
    return "".join(parts)


def _ensure_architectures(model) -> None:
    """
    text_only 로드 시 config.architectures 가 None 이 되는 경우가 있습니다.
    Unsloth generate 패치가 `any(... for x in architectures)` 를 돌리므로
    None 이면 TypeError 가 납니다. CausalLM 로 명시해 둡니다.
    """
    candidates = [model]
    for attr in ("base_model", "model", "get_base_model"):
        obj = getattr(model, attr, None)
        if callable(obj):
            try:
                obj = obj()
            except Exception:  # noqa: BLE001
                obj = None
        if obj is not None:
            candidates.append(obj)

    for obj in candidates:
        cfg = getattr(obj, "config", None)
        if cfg is None:
            continue
        if getattr(cfg, "architectures", None) is None:
            cfg.architectures = ["Gemma4ForCausalLM"]


def _resolve_device(model) -> str:
    try:
        return str(model.device)
    except Exception:  # noqa: BLE001
        try:
            return str(next(model.parameters()).device)
        except Exception:  # noqa: BLE001
            return "cuda:0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습된 LoRA로 Python/Java/C 추론 품질을 확인합니다."
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        help="베이스 모델 ID",
    )
    parser.add_argument(
        "--adapter",
        default="gemma4_e4b_coding_lora",
        help="학습된 LoRA 디렉터리",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    # 첫 생성은 CUDA/Unsloth 워밍업으로 느림. 기본 512면 체감이 훨씬 빠름.
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"{adapter_path} 이 없습니다. 먼저 `python train.py` 로 학습하세요."
        )
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"{adapter_path}/adapter_config.json 이 없습니다. "
            "LoRA 저장이 완료됐는지 확인하세요."
        )

    token = get_hf_token(args.hf_token)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[경고] huggingface_hub.login 실패: {exc}")
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    else:
        print("[경고] HF_TOKEN 이 없습니다. .env 를 확인하세요.")

    from unsloth import FastLanguageModel, FastModel
    from peft import PeftModel
    import torch

    print(f"[1/3] 베이스 4-bit 로드: {args.base_model}")
    load_kwargs = dict(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=token,
        device_map={"": 0},
        text_only=True,
    )
    try:
        model, tokenizer = FastModel.from_pretrained(**load_kwargs)
    except TypeError:
        load_kwargs.pop("text_only", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    print(f"[2/3] LoRA 어댑터 결합: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    _ensure_architectures(model)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[경고] 어댑터 토크나이저 로드 실패, 베이스 토크나이저 유지: {exc}")

    print("[3/3] 추론 모드 활성화")
    try:
        FastModel.for_inference(model)
    except Exception:
        FastLanguageModel.for_inference(model)

    device = _resolve_device(model)

    print("\n" + "=" * 72)
    print("Gemma 4 E4B Coding LoRA - 추론 평가")
    print("목적: 학습된 모델이 C/Java/Python 질문에 한국어로 잘 답하는지 확인")
    print("=" * 72)

    for i, case in enumerate(EVAL_PROMPTS, start=1):
        print(f"\n{'-' * 72}")
        print(f"[{i}/{len(EVAL_PROMPTS)}] {case['name']} ({case['language']})")
        print(f"{'-' * 72}")
        print(f"USER>\n{case['user']}\n")

        prompt = build_prompt(tokenizer, case["user"])
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        print(
            f"[생성 중] max_new_tokens={args.max_new_tokens} "
            f"(첫 질문은 CUDA 워밍업으로 1~3분 걸릴 수 있습니다)...",
            flush=True,
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            # Unsloth/HF 가 config 의 거대한 max_length 를 물려받아 경고가 뜨는 것 방지
            max_length=None,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )

        with torch.no_grad():
            try:
                outputs = model.generate(**gen_kwargs)
            except TypeError as exc:
                # Unsloth vision generate 패치 호환 문제 시 HF 원본 generate 로 폴백
                print(f"[경고] Unsloth generate 실패, HF 경로로 재시도: {exc}", flush=True)
                gen_kwargs.pop("max_length", None)
                from transformers import GenerationMixin

                gen_model = model.get_base_model() if hasattr(model, "get_base_model") else model
                outputs = GenerationMixin.generate(gen_model, **gen_kwargs)

        print("[생성 완료]", flush=True)
        generated = outputs[0][input_ids.shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"ASSISTANT>\n{text}\n", flush=True)

    print("=" * 72)
    print("평가 완료. 코드 정확성 / 한국어 설명 / 환각 여부를 직접 검토하세요.")


if __name__ == "__main__":
    main()
