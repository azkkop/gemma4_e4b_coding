#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_gguf.py
--------------
학습된 LoRA를 베이스에 얹은 뒤:
  1) (선택) HF merged 16bit 저장  — 비전/재학습용 백업
  2) GGUF(q4_k_m 기본) 내보내기 — Ollama / llama.cpp 용
  3) Ollama Modelfile 안내

중요: gemma4_e4b_coding_lora/ 는 삭제하지 마세요. 비전은 나중에
멀티모달 베이스 + 이 LoRA 경로로 다시 열 수 있습니다.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA → (optional merge) → GGUF export")
    p.add_argument(
        "--base-model",
        default="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    )
    p.add_argument("--adapter", default="gemma4_e4b_coding_lora")
    p.add_argument("--gguf-dir", default="gemma4_e4b_coding_gguf")
    p.add_argument("--merged-dir", default="gemma4_e4b_coding_merged")
    p.add_argument(
        "--quant",
        default="q4_k_m",
        help="GGUF quant: q4_k_m | q5_k_m | q8_0 | f16 ...",
    )
    p.add_argument(
        "--save-merged",
        action="store_true",
        help="HF merged_16bit 도 저장 (디스크 ~수 GB, 비전/HF 재사용용)",
    )
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--hf-token", default=None)
    p.add_argument(
        "--skip-gguf",
        action="store_true",
        help="GGUF 생략 (merged만)",
    )
    return p.parse_args()


def _ensure_architectures(model) -> None:
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
        if cfg is not None and getattr(cfg, "architectures", None) is None:
            cfg.architectures = ["Gemma4ForCausalLM"]


def _write_modelfile(gguf_dir: Path, gguf_file: Path | None) -> Path:
    """Unsloth가 Modelfile을 안 만들면 최소 템플릿을 둡니다."""
    out = gguf_dir / "Modelfile"
    if out.exists():
        return out

    from_line = f"FROM ./{gguf_file.name}" if gguf_file else "FROM ."
    # Gemma 4 chat template (Ollama)
    content = f"""# Auto-generated for gemma4-e4b-coding
{from_line}

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop <end_of_turn>
PARAMETER stop <start_of_turn>

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

TEMPLATE \"\"\"{{{{- if .System }}}}<start_of_turn>system
{{{{ .System }}}}<end_of_turn>
{{{{- end }}}}{{{{- range .Messages }}}}{{{{- if eq .Role "user" }}}}<start_of_turn>user
{{{{ .Content }}}}<end_of_turn>
{{{{- else if eq .Role "assistant" }}}}<start_of_turn>assistant
{{{{ .Content }}}}<end_of_turn>
{{{{- end }}}}{{{{- end }}}}<start_of_turn>assistant
\"\"\"
"""
    out.write_text(content, encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"[중단] LoRA 없음: {adapter}/adapter_config.json")

    print("=" * 60)
    print("1단계: LoRA merge / GGUF export")
    print(f"  adapter : {adapter.resolve()}")
    print(f"  quant   : {args.quant}")
    print(f"  merged  : {'yes' if args.save_merged else 'no (LoRA 유지로 충분)'}")
    print(f"  gguf    : {'skip' if args.skip_gguf else args.gguf_dir}")
    print("=" * 60)
    print("※ gemma4_e4b_coding_lora/ 는 그대로 둡니다 (비전용).")

    token = get_hf_token(args.hf_token)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[경고] HF login: {exc}")
    else:
        print("[경고] HF_TOKEN 없음 — gated/속도 제한 가능")

    from unsloth import FastLanguageModel, FastModel
    from unsloth.save import patch_saving_functions
    from peft import PeftModel
    import torch

    print(f"\n[1/4] 베이스 로드: {args.base_model}")
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

    print(f"[2/4] LoRA 결합: {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter))
    _ensure_architectures(model)
    model = patch_saving_functions(model)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    except Exception as exc:  # noqa: BLE001
        print(f"[경고] 어댑터 토크나이저: {exc}")

    if args.save_merged:
        merged = Path(args.merged_dir)
        merged.mkdir(parents=True, exist_ok=True)
        print(f"[3/4] HF merged_16bit → {merged.resolve()}")
        try:
            model.save_pretrained_merged(
                str(merged),
                tokenizer,
                save_method="merged_16bit",
            )
            print("  merged 저장 완료")
        except Exception as exc:  # noqa: BLE001
            print(f"[경고] merged 저장 실패: {exc}")
            print("  → GGUF만으로도 Ollama는 가능합니다.")
    else:
        print("[3/4] merged_16bit 생략 (--save-merged 로 켤 수 있음)")

    if args.skip_gguf:
        print("[4/4] GGUF 생략")
        return

    gguf_dir = Path(args.gguf_dir)
    if gguf_dir.exists():
        print(f"[안내] 기존 {gguf_dir} 내용을 유지한 채 덮어씁니다.")
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] GGUF export ({args.quant}) → {gguf_dir.resolve()}")
    print("  (처음이면 llama.cpp 준비 + 변환으로 수십 분~1시간+ 걸릴 수 있음)")
    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=args.quant,
        )
        print("  GGUF 저장 완료")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[실패] GGUF 저장: {exc}\n"
            "  대안: cmake/빌드 도구 설치 후 재시도, 또는 --quant f16 시도.\n"
            "  LoRA는 그대로 있으니 evaluate.py 는 계속 사용 가능합니다."
        ) from exc

    gguf_files = sorted(gguf_dir.glob("*.gguf"), key=lambda p: p.stat().st_mtime)
    latest = gguf_files[-1] if gguf_files else None
    modelfile = _write_modelfile(gguf_dir, latest)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "=" * 60)
    print("완료")
    if latest:
        print(f"  GGUF     : {latest.resolve()} ({latest.stat().st_size / 1e9:.2f} GB)")
    print(f"  Modelfile: {modelfile.resolve()}")
    print(f"  LoRA 보관: {adapter.resolve()}  ← 지우지 마세요")
    print("\nOllama 등록:")
    print(f"  cd {gguf_dir.resolve()}")
    print("  ollama create gemma4-e4b-coding -f Modelfile")
    print("  ollama run gemma4-e4b-coding")
    print("=" * 60)


if __name__ == "__main__":
    main()
