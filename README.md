# Gemma 4 E4B Coding AI — Unsloth QLoRA 파인튜닝

RTX 3060 (VRAM 12GB) 환경에서 `google/gemma-4-E4B-it` 를 **Unsloth + 4-bit QLoRA** 로 파인튜닝하는 프로젝트입니다.

목표: **Python 35% / Java 35% / C 30%** 코딩 지능과 **한국어 설명·디버깅·코드 수정** 능력 강화.

---

## 하드웨어 프로파일 (기본값)

| 항목 | 값 |
|------|-----|
| GPU | RTX 3060 12GB |
| Max Sequence Length | 4096 |
| Batch Size | 2 |
| Gradient Accumulation | 4 (effective batch = 8) |
| Optim | `adamw_8bit` |
| Gradient Checkpointing | `unsloth` |
| Quantization | 4-bit NF4 (bitsandbytes) |

---

## 디렉터리 구조

```
gemma4-e4b-coding-ft/
├── prepare_dataset.py       # 시드 9개
├── expand_dataset.py        # 다양성 로컬 합성 (고유 보장)
├── dedupe_dataset.py        # exact/near 중복 제거
├── generate_hq_dataset.py   # NVIDIA 31B 생성 + 검증
├── merge_datasets.py        # JSONL 병합
├── train.py / evaluate.py / export_gguf.py
├── dataset.jsonl
└── README.md
```

---

## 1. 환경 설정

### 사전 요구

- Windows 10/11 (네이티브 가능) / WSL2도 가능
- NVIDIA Driver (CUDA 12.6 이상 표시 권장)
- Python **3.10 ~ 3.12**
- Hugging Face 계정 + **Gemma 4 gated 모델 액세스 승인**
- 프로젝트 `.env` 의 `HF_TOKEN` (이미 설정됨)

### 가상환경 & 설치 (Windows)

`pip install -r requirements.txt` 만 하면 **CPU용 torch**로 덮일 수 있습니다. 순서를 지키세요.

**방법 A — 원클릭**

```bat
setup_windows.bat
```

**방법 B — 수동**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install "fsspec>=2023.1.0,<=2025.9.0"
python -c "import torch; from unsloth import FastLanguageModel; print(torch.__version__, torch.cuda.is_available())"
```

정상 시: `2.11.0+cu126 True`

### Hugging Face 로그인 (선택)

`.env`에 토큰이 있으면 생략 가능합니다.

```powershell
huggingface-cli login
```

Gemma 4 모델 페이지에서 라이선스/액세스에 동의한 뒤 사용하세요:  
https://huggingface.co/google/gemma-4-E4B-it

---

## 2. 데이터셋 준비 (품질 파이프라인)

### A. 시드
```bash
python prepare_dataset.py
```

### B. 로컬 다양성 확장 (고유 샘플 보장)
```bash
python expand_dataset.py --unique-target 3000 --include-seed --backup
```

### C. 중복 제거
```bash
python dedupe_dataset.py --near-dup --replace --backup
```

### D. 고품질 생성 (NVIDIA Gemma 4 31B + 검증)
`.env`에 `NVIDIA_API_KEY=nvapi-...` 설정 후:
```bash
python generate_hq_dataset.py --target 50 --output dataset_hq.jsonl
# 더 모으려면
python generate_hq_dataset.py --target 200 --output dataset_hq.jsonl --append
```

### E. 병합
```bash
python merge_datasets.py --inputs dataset.jsonl dataset_hq.jsonl --output dataset.jsonl --backup
```

학습:
```bash
python train.py --dataset dataset.jsonl
```

각 라인은 Gemma 4 Official Chat Template 형식의 `text` 필드를 포함합니다.

```text
<bos><start_of_turn>system
...<end_of_turn>
<start_of_turn>user
...<end_of_turn>
<start_of_turn>assistant
...<end_of_turn>
```

시드 구성 (언어별 3개):

- **Python**: 타입 힌팅, async/await, 리팩토링
- **Java**: OOP 클래스, Spring 스타일 예외 처리, DTO
- **C**: 포인터 연산, `malloc`/`free`, 메모리 누수 디버깅

> 실제 품질을 올리려면 동일 스키마로 수천~수만 건으로 확장하세요.  
> 목표 비중: Python 35% / Java 35% / C 30%.

---

## 3. 학습 (QLoRA SFT)

```bash
python train.py
```

주요 기본 인자:

```bash
python train.py \
  --model-name google/gemma-4-E4B-it \
  --dataset dataset.jsonl \
  --output-dir gemma4_e4b_coding_lora \
  --max-seq-length 4096 \
  --batch-size 2 \
  --grad-accum 4 \
  --epochs 3 \
  --lora-r 16 \
  --lora-alpha 16
```

스모크 테스트 (짧은 스텝):

```bash
python train.py --max-steps 10
```

선택: LoRA 학습 후 GGUF도 내보내기

```bash
python train.py --export-gguf
```

저장 위치:

- LoRA + tokenizer → `gemma4_e4b_coding_lora/`
- (선택) GGUF → `gemma4_e4b_coding_gguf/`

VRAM이 부족하면 `--batch-size 1`, `--max-seq-length 2048`, `--lora-r 8` 순으로 낮추세요.

---

## 4. 평가 (추론)

```bash
python evaluate.py
```

기본 평가 케이스:

1. **C** — 포인터로 두 변수 스왑
2. **Java** — Book 클래스 (대여/반납 상태 전이)
3. **Python** — `asyncio.as_completed` 비동기 결과 출력

옵션 예시:

```bash
python evaluate.py \
  --adapter gemma4_e4b_coding_lora \
  --max-new-tokens 1024 \
  --temperature 0.2
```

---

## 5. GGUF / Ollama 내보내기 (1단계)

학습이 **끝난 뒤** LoRA는 그대로 두고 GGUF만 만듭니다 (비전용으로 LoRA 필수 보관).

```powershell
.\.venv\Scripts\Activate.ps1
python export_gguf.py
# 선택: HF 16bit 병합본도 보관
# python export_gguf.py --save-merged
```

완료 후:

```powershell
cd gemma4_e4b_coding_gguf
ollama create gemma4-e4b-coding -f Modelfile
ollama run gemma4-e4b-coding
```

- LoRA: `gemma4_e4b_coding_lora/` (삭제 금지)
- GGUF: `gemma4_e4b_coding_gguf/`
- (선택) merged: `gemma4_e4b_coding_merged/`

처음 변환은 llama.cpp 준비 때문에 **수십 분 이상** 걸릴 수 있습니다.

---

## 6. Cursor형 로컬 API (채팅 + 전체 도구)

채팅만 프록시하는 API가 아닙니다. **파일/폴더 생성·수정, 터미널, 검색, 웹, 비전**까지 HTTP로 호출합니다.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-api.txt
python serve_api.py
# 문서 UI: http://127.0.0.1:8000/docs
```

### 아키텍처 (Cursor처럼)

| 모드 | 엔드포인트 | 동작 |
|------|------------|------|
| **IDE 에이전트** | `POST /v1/ide/chat` | 모델이 도구 선택 → **서버가 워크스페이스에서 실행** |
| **도구 직접 호출** | `POST /v1/tools/call` | 프론트가 `write_file` 등 개별 실행 |
| **tool_calls 프로토콜** | `POST /v1/chat/completions` + `tools` | 모델이 `tool_calls` 반환 → 클라이언트가 `/v1/tools/call` |
| **에이전트 모델** | `model: gemma4-e4b-coding-agent` | chat completions에서 서버 측 도구 루프 |
| 순수 채팅 | `model: gemma4-e4b-coding` | 도구 없음 (설명용) |

### 도구 API

```powershell
# 도구 목록 (OpenAI function schema)
curl http://127.0.0.1:8000/v1/tools -H "X-Workspace: C:/Users/USER/gemma4-e4b-coding-ft"

# 폴더/파일 실제로 생성
curl http://127.0.0.1:8000/v1/tools/call -H "Content-Type: application/json" -d "{\"name\":\"create_tree\",\"arguments\":{\"items\":[{\"path\":\"demo_ide/\",\"content\":\"\"},{\"path\":\"demo_ide/main.py\",\"content\":\"print(1)\\n\"}]},\"workspace\":\".\"}"

# 디렉터리 목록
curl http://127.0.0.1:8000/v1/tools/call -H "Content-Type: application/json" -d "{\"name\":\"list_dir\",\"arguments\":{\"path\":\".\"},\"workspace\":\".\"}"
```

### IDE 채팅 (추천)

```powershell
curl http://127.0.0.1:8000/v1/ide/chat -H "Content-Type: application/json" -d "{\"workspace\":\".\",\"messages\":[{\"role\":\"user\",\"content\":\"demo_app 폴더 만들고 hello.py 작성\"}],\"execute_tools\":true}"
```

SSE:
```powershell
curl -N http://127.0.0.1:8000/v1/ide/chat -H "Content-Type: application/json" -d "{\"workspace\":\".\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"README 요약\"}]}"
```

### OpenAI SDK로 에이전트

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")

# 서버가 도구까지 실행
r = client.chat.completions.create(
    model="gemma4-e4b-coding-agent",
    messages=[{"role": "user", "content": "utils 폴더에 hello.py 만들어줘"}],
    extra_body={"workspace": "."},  # 또는 헤더 X-Workspace
)
print(r.choices[0].message.content)
print(r.model_extra)  # steps 등
```

헤더 `X-Workspace` 또는 body `workspace`로 작업 폴더 지정.  
환경변수: `API_PORT`, `API_KEY`, `OLLAMA_BASE_URL`, `API_MODEL_NAME`, `API_AGENT_MODEL_NAME`, `VISION_MODEL`, `AGENT_WORKSPACE`, `AGENT_ALLOWED_ROOTS`, `AGENT_APPROVAL_MODE`

---

## 6.1 데스크톱 IDE 클라이언트 (Electron)

로컬 Cursor형 UI: 파일 트리 + Monaco + 에이전트 채팅. API 서버가 켜져 있어야 합니다.

```powershell
# 터미널 1
python serve_api.py

# 터미널 2
cd ide
npm install
npm run dev
```

설정에서 API Base(`http://127.0.0.1:8000`), `.env`의 `API_KEY`, Workspace 폴더를 지정하세요.  
자세한 내용: [`ide/README.md`](ide/README.md)

---

## 7. 로컬 코딩 에이전트 (CLI)

파일 읽기/수정/생성, 터미널, 테스트, 웹 검색 도구가 붙은 조수입니다. Ollama `gemma4-e4b-coding` 사용.

```powershell
.\.venv\Scripts\Activate.ps1
# 한 번 작업
python coding_agent.py "프로젝트 구조 확인하고 README 상단에 에이전트 사용법 한 줄 추가"

# 대화 모드
python coding_agent.py

# 위험 작업 전 승인
python coding_agent.py --approval ask "테스트 돌려줘"

# 읽기 전용
python coding_agent.py --approval readonly "이 프로젝트 구조 설명해줘"
```

옵션: `--workspace`, `--model`, `--max-rounds`, `--approval`, `--no-writes`, `--no-terminal`, `--no-web`

도구: `list_dir`, `read_file`, `write_file`, `str_replace`, `make_dir`, `create_tree`, `search_text`, `run_terminal`, `run_tests`, `git_status`, `git_diff`, `list_uploads`, `read_upload`, `analyze_image`, `web_search`  
워크스페이스 밖 경로·위험 명령은 차단됩니다. `AGENT_ALLOWED_ROOTS`로 API/CLI 워크스페이스 허용 루트를 제한할 수 있습니다.

### 파일 첨부 / 비전

파인튜닝 GGUF(`gemma4-e4b-coding`)는 텍스트 전용입니다. 이미지는 Ollama 베이스 `gemma4:e4b`로 분석합니다.

```powershell
# 스크린샷/문서 첨부
python coding_agent.py --attach error.png "이 에러 원인 찾고 관련 코드 수정해줘"
python coding_agent.py --attach notes.pdf --attach data.csv "핵심만 요약"

# 대화 중: @C:\path\to\file.png 이 화면 설명해줘
```

첨부 파일은 `uploads/` 로 복사됩니다. PDF는 `pip install pypdf` 필요.

---

## 8. 학습 원리 요약

1. **4-bit 양자화**: 베이스 가중치를 NF4로 로드해 12GB VRAM에 맞춤  
2. **LoRA**: `q/k/v/o_proj` + `gate/up/down_proj` 에 저랭크 어댑터만 학습  
3. **Unsloth checkpointing**: 활성 활성화를 다시 계산해 메모리 절약  
4. **adamw_8bit**: 옵티마이저 상태 메모리 추가 절약  
5. **SFT**: Gemma 4 chat template 텍스트로 supervised fine-tuning

---

## 9. 참고 / 문제 해결

| 증상 | 조치 |
|------|------|
| `401` / gated 모델 거부 | HF 액세스 승인 + `HF_TOKEN` 확인 |
| CUDA OOM | batch=1, seq=2048, lora_r=8 |
| Windows에서 bitsandbytes 오류 | WSL2 Ubuntu에서 재설치 권장 |
| GGUF 저장 실패 | LoRA만 사용하거나 llama.cpp convert로 후처리 |
| 모델 ID 대소문자 | 공식 ID는 `google/gemma-4-E4B-it` |

Unsloth Gemma 4 문서: https://unsloth.ai/docs/models/gemma-4

---

## 라이선스 고지

베이스 모델(`google/gemma-4-E4B-it`) 사용 시 Google Gemma 라이선스 및 Hugging Face gated 약관을 준수해야 합니다. 이 저장소의 스크립트·시드 데이터는 학습 파이프라인 예제입니다.
