#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_hf_to_gguf.py
---------------------
이미 만들어진 HF 16bit 폴더(model.safetensors)를 GGUF로 변환합니다.
Ollama가 safetensors 직접 변환하다 죽는 경우를 피합니다.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM_PROMPT = (
    "당신은 Python, Java, C언어를 전문으로 다루는 시니어 코딩 AI입니다. "
    "코드 작성·리팩토링·디버깅 요청에 대해 정확한 코드와 함께 "
    "한국어로 핵심 원리와 주의사항을 명확히 설명합니다."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-dir",
        default="gemma4_e4b_coding_merged",
        help="HF safetensors 폴더 (없으면 gguf 폴더에서 승격)",
    )
    p.add_argument("--gguf-dir", default="gemma4_e4b_coding_gguf")
    p.add_argument(
        "--outtype",
        default="f16",
        help="convert_hf_to_gguf outtype: f16|bf16|q8_0 (q4 quantize는 f16/bf16에서만)",
    )
    p.add_argument(
        "--quant",
        default="Q4_K_M",
        help="llama-quantize 타입. 빈 문자열이면 quantize 생략",
    )
    p.add_argument("--model-name", default="gemma4-e4b-coding")
    return p.parse_args()


def _promote_partial_gguf_folder(hf_dir: Path, gguf_dir: Path) -> None:
    """이전 export가 safetensors만 gguf 폴더에 남긴 경우 merged로 옮김."""
    st = gguf_dir / "model.safetensors"
    cfg = gguf_dir / "config.json"
    if hf_dir.exists() and (hf_dir / "model.safetensors").exists():
        return
    if st.exists() and cfg.exists():
        print(f"[정리] {gguf_dir} → {hf_dir} (HF 16bit 승격)")
        if hf_dir.exists():
            shutil.rmtree(hf_dir)
        hf_dir.mkdir(parents=True)
        for name in (
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
        ):
            src = gguf_dir / name
            if src.exists():
                shutil.move(str(src), str(hf_dir / name))


def _write_modelfile(gguf_dir: Path, gguf_file: Path) -> Path:
    out = gguf_dir / "Modelfile"
    content = f"""FROM ./{gguf_file.name}

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
    hf_dir = Path(args.hf_dir)
    gguf_dir = Path(args.gguf_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    _promote_partial_gguf_folder(hf_dir, gguf_dir)

    if not (hf_dir / "model.safetensors").exists():
        raise SystemExit(
            f"[중단] {hf_dir}/model.safetensors 없음. "
            "먼저 export_gguf.py 또는 merge가 필요합니다."
        )

    print("[1/4] llama.cpp 설치/확인 (Unsloth prebuilt)...")
    import unsloth  # noqa: F401 — must import before unsloth_zoo helpers
    from unsloth_zoo.llama_cpp import (
        LLAMA_CPP_DEFAULT_DIR,
        check_llama_cpp,
        install_llama_cpp,
    )

    convert_py = None
    quantize = None
    llama_dir = Path(LLAMA_CPP_DEFAULT_DIR)

    try:
        checked = check_llama_cpp()
        print(f"  check: {checked}")
    except Exception:
        print("  설치 중... (수 분 소요)")
        installed = install_llama_cpp(print_output=True)
        print(f"  설치 결과: {installed}")
        # Unsloth may return (quantize_exe, convert_script) or a folder path
        if isinstance(installed, (tuple, list)):
            for item in installed:
                p = Path(str(item))
                if p.name.startswith("convert_hf_to_gguf") and p.suffix == ".py":
                    convert_py = p
                if "quantize" in p.name.lower():
                    quantize = p
            if convert_py is not None:
                llama_dir = convert_py.parent
        elif installed:
            llama_dir = Path(str(installed))

    if convert_py is None:
        convert_py = llama_dir / "convert_hf_to_gguf.py"
        if not convert_py.exists():
            found = list(Path(LLAMA_CPP_DEFAULT_DIR).rglob("convert_hf_to_gguf.py"))
            if not found:
                raise SystemExit(
                    f"[중단] convert_hf_to_gguf.py 를 못 찾음: {LLAMA_CPP_DEFAULT_DIR}"
                )
            convert_py = found[0]
            llama_dir = convert_py.parent

    if quantize is None:
        for name in ("llama-quantize.exe", "llama-quantize", "quantize.exe"):
            hits = list(Path(LLAMA_CPP_DEFAULT_DIR).rglob(name))
            if hits:
                quantize = hits[0]
                break

    print(f"  convert : {convert_py}")
    print(f"  quantize: {quantize}")

    mid_name = f"gemma4-e4b-coding-{args.outtype}.gguf"
    mid_path = gguf_dir / mid_name

    print(f"[2/4] HF → GGUF ({args.outtype})")
    print(f"  in : {hf_dir.resolve()}")
    print(f"  out: {mid_path.resolve()}")
    cmd = [
        sys.executable,
        str(convert_py),
        str(hf_dir.resolve()),
        "--outfile",
        str(mid_path.resolve()),
        "--outtype",
        args.outtype,
    ]
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)

    final_path = mid_path
    if args.quant and quantize is not None:
        qname = f"gemma4-e4b-coding-{args.quant.lower()}.gguf"
        qpath = gguf_dir / qname
        print(f"[3/4] quantize → {args.quant}")
        qcmd = [str(quantize), str(mid_path), str(qpath), args.quant]
        print(" ", " ".join(qcmd))
        subprocess.run(qcmd, check=True)
        final_path = qpath
        # keep mid file too (disk); user can delete later
    elif args.quant and quantize is None:
        print("[경고] llama-quantize 없음 → outtype GGUF만 사용")
        print("[3/4] quantize 생략")
    else:
        print("[3/4] quantize 생략")

    print("[4/4] Modelfile 작성")
    mf = _write_modelfile(gguf_dir, final_path)

    print("\n완료.")
    print(f"  GGUF     : {final_path.resolve()} ({final_path.stat().st_size/1e9:.2f} GB)")
    print(f"  Modelfile: {mf.resolve()}")
    print("\nOllama:")
    print(f"  cd {gguf_dir.resolve()}")
    print(f"  ollama create {args.model_name} -f Modelfile")
    print(f"  ollama run {args.model_name}")


if __name__ == "__main__":
    main()
