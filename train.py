#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
--------
google/gemma-4-E4B-it 를 Unsloth + 4-bit QLoRA 로 SFT 학습합니다.

RTX 3060 12GB 기본값 (OOM 방지):
  - text_only=True
  - max_seq_length = 1024
  - batch=1, grad_accum=8
  - lora_r = 8
  - fused CE 최소 VRAM 가드
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
from pathlib import Path

from hf_env import get_hf_token, load_dotenv

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# padding-free 는 긴 샘플에서 peak VRAM 이 더 튈 수 있음
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma 4 E4B QLoRA SFT (Unsloth)")
    parser.add_argument(
        "--model-name",
        default="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        help="베이스 모델 ID",
    )
    parser.add_argument("--dataset", default="dataset.jsonl")
    parser.add_argument("--output-dir", default="gemma4_e4b_coding_lora")
    # 12GB + E4B 4bit: 2048 은 fused CE 단계에서 OOM 나기 쉬움 → 1024 기본
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="디버그용. >0 이면 epochs 대신 max_steps 사용",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="학습 후 LoRA를 병합하고 GGUF(q4_k_m)로도 저장 시도",
    )
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--full-multimodal",
        action="store_true",
        help="비전/오디오 타워까지 로드 (12GB에서는 비권장)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="outputs/ 최신 checkpoint 에서 이어서 학습",
    )
    return parser.parse_args()


def _apply_hf_token(token: str | None) -> str | None:
    if not token:
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if key in os.environ and not os.environ[key].startswith("hf_"):
                os.environ.pop(key, None)
        return None

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login, whoami

        login(token=token, add_to_git_credential=False)
        info = whoami(token=token)
        print(f"[HF] 로그인 확인: {info.get('name') or info.get('fullname') or 'ok'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[경고] HF 토큰 무효/만료: {exc}")
        print("  → .env 의 HF_TOKEN 을 재발급 키로 교체하세요.")
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        return None
    return token


def _patch_fused_ce_vram_guard() -> None:
    """
    Unsloth fused CE 가 free VRAM≈0 이면 RuntimeError 를 냅니다.
    12GB 꽉 찬 학습 1스텝에서 자주 발생하므로, 최소 target_gb 바닥을 둡니다.
    """
    try:
        import unsloth_zoo.fused_losses.cross_entropy_loss as ce

        if getattr(ce, "_gemma4_ft_patched", False):
            return

        original = ce._get_chunk_multiplier

        def _safe_get_chunk_multiplier(vocab_size, target_gb=None):
            import torch

            if target_gb is None:
                free, _total = torch.cuda.mem_get_info(0)
                free_gb = free / 1024 / 1024 / 1024
                # 기존: free*0.5 가 0이면 즉시 사망 → 최소 0.35GB 보장
                target_gb = max(min(free_gb * 0.5, 4.0), 0.35)
            if target_gb <= 1e-9:
                target_gb = 0.35
            # cache 키에 쓰이므로 반올림
            target_gb = round(float(target_gb), 3)
            return original(vocab_size, target_gb)

        # functools.cache 원본을 우회
        ce._get_chunk_multiplier = _safe_get_chunk_multiplier  # type: ignore[assignment]
        if hasattr(original, "cache_clear"):
            original.cache_clear()
        ce._gemma4_ft_patched = True
        print("[패치] fused CE VRAM 가드 적용 (min target_gb=0.35)")
    except Exception as exc:  # noqa: BLE001
        print(f"[경고] fused CE 패치 실패(계속): {exc}")


def _find_latest_checkpoint(output_dir: str = "outputs") -> str | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    ckpts = sorted(
        [p for p in root.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(ckpts[-1]) if ckpts else None


def _read_adapter_r(checkpoint_dir: str) -> int | None:
    cfg = Path(checkpoint_dir) / "adapter_config.json"
    if not cfg.exists():
        return None
    import json

    data = json.loads(cfg.read_text(encoding="utf-8"))
    return int(data.get("r")) if data.get("r") is not None else None


def main() -> None:
    args = parse_args()
    token = _apply_hf_token(get_hf_token(args.hf_token))
    if not token:
        print("[경고] HF_TOKEN 이 없습니다. gated 모델/속도 제한에 걸릴 수 있습니다.")

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        free_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        print(f"[GPU] free VRAM ≈ {free_gb:.2f} GB")
        if free_gb < 8.0:
            print(
                "[경고] 시작 전 여유 VRAM 이 적습니다. "
                "다른 Python/게임/브라우저 GPU 사용을 종료하세요."
            )

    # Unsloth import 전에 env 가 잡혀 있어야 함
    from unsloth import FastLanguageModel, FastModel
    from datasets import load_dataset

    _patch_fused_ce_vram_guard()

    text_only = not args.full_multimodal
    device_map = {"": 0}

    print(
        f"[1/5] 모델 로드: {args.model_name} "
        f"(4-bit, seq={args.max_seq_length}, text_only={text_only}, r={args.lora_r})"
    )

    load_kwargs = dict(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
        token=token,
        device_map=device_map,
        text_only=text_only,
        use_gradient_checkpointing="unsloth",
    )

    try:
        model, tokenizer = FastModel.from_pretrained(**load_kwargs)
    except TypeError:
        load_kwargs.pop("text_only", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[GPU] 로드 후 free ≈ {torch.cuda.mem_get_info()[0] / (1024**3):.2f} GB")

    print("[2/5] LoRA 어댑터 구성 (QLoRA)")
    peft_common = dict(
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    try:
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            **peft_common,
        )
    except TypeError:
        model = FastLanguageModel.get_peft_model(model, **peft_common)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} 이 없습니다. 먼저 데이터셋 스크립트를 실행하세요."
        )

    print(f"[3/5] 데이터셋 로드: {dataset_path}")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    from trl import SFTTrainer, SFTConfig

    print("[4/5] SFT 학습 시작")
    sft_kwargs: dict = {
        "output_dir": "outputs",
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": 5,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": args.seed,
        "bf16": True,
        "fp16": False,
        "report_to": "none",
        # 자주 저장 → 중간에 멈춰도 --resume 가능
        "save_strategy": "steps",
        "save_steps": 250,
        "save_total_limit": 3,
        "dataset_text_field": "text",
        "packing": False,
        "dataset_num_proc": None,
    }

    config_params = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in config_params:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in config_params:
        sft_kwargs["max_length"] = args.max_seq_length

    if args.max_steps > 0:
        sft_kwargs["max_steps"] = args.max_steps
    else:
        sft_kwargs["num_train_epochs"] = args.epochs

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict = {
        "model": model,
        "train_dataset": dataset,
        "args": SFTConfig(**sft_kwargs),
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    resume_path = None
    if args.resume:
        resume_path = _find_latest_checkpoint("outputs")
        if resume_path:
            ckpt_r = _read_adapter_r(resume_path)
            if ckpt_r is not None and ckpt_r != args.lora_r:
                raise SystemExit(
                    f"[중단] checkpoint LoRA r={ckpt_r} 인데 현재 --lora-r={args.lora_r} 입니다.\n"
                    f"  재개하려면: python train.py --resume --lora-r {ckpt_r} --lora-alpha {ckpt_r}\n"
                    f"  처음부터:   python train.py   ( --resume 없이 )\n"
                    f"  옛 ckpt는 outputs_incompatible_r16/ 등으로 옮겨 두세요."
                )
            print(f"[재개] {resume_path} (r={ckpt_r})")
        else:
            print("[재개] checkpoint 없음 → 처음부터 학습")
            resume_path = None

    # 학습 직전 한 번 더 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    train_result = trainer.train(resume_from_checkpoint=resume_path)
    print(f"학습 완료 - metrics: {train_result.metrics}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[5/5] LoRA 저장 → {out_dir.resolve()}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if args.export_gguf:
        gguf_dir = Path("gemma4_e4b_coding_gguf")
        gguf_dir.mkdir(parents=True, exist_ok=True)
        print(f"[선택] GGUF 내보내기 시도 → {gguf_dir.resolve()}")
        try:
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method="q4_k_m",
            )
            print("GGUF 저장 완료")
        except Exception as exc:  # noqa: BLE001
            print(f"[경고] GGUF 저장 실패: {exc}\n  LoRA 는 저장되었습니다.")

    print("\n완료. 다음 단계: python evaluate.py")


if __name__ == "__main__":
    main()
