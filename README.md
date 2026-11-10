# gemma4_e4b_coding

Gemma 4 E4B 기반 자체 파인튜닝 코딩 모델과 학습·평가·배포 프로젝트입니다. 한국어 코딩 설명, Python·Java·C 코드 작성과 수정에 초점을 맞추며, 두원 AI 캠퍼스의 코딩 캔버스에 연결해 사용했습니다.

**코드만 남기지 않고 실제 가중치까지 복구할 수 있도록 백업했습니다.** 모델 파일은 큰 용량 때문에 GitHub Releases에 분할 저장하고, 이 저장소에는 코드·데이터·토크나이저·설정·SHA-256 검증 정보와 복원 도구를 보관합니다.

- [모델 백업 릴리스](https://github.com/azkkop/gemma4_e4b_coding/releases/tag/backup-2026-11-11)
- [파일별 크기·해시·다운로드 주소](backup-manifest.json)
- [기존 학습/IDE 사용 안내](docs/TRAINING_GUIDE_LEGACY.md)

## 가장 빠르게 실행하기

Python 3.10 이상과 Ollama가 필요합니다. 다음 명령은 검증된 GGUF 모델만 받습니다. 현재 복원 도구는 표준 라이브러리만 사용하므로 별도 pip 설치가 필요하지 않습니다.

```powershell
git clone --branch cursor/export-gguf-ollama https://github.com/azkkop/gemma4_e4b_coding.git
cd gemma4_e4b_coding
python tools/restore_model.py --only gguf
ollama create gemma4_e4b_coding -f ollama/Modelfile
ollama run gemma4_e4b_coding
```

분할 파일은 각각 단독 모델이 아닙니다. 복원 도구가 순서대로 이어 붙이고 각 조각과 전체 파일의 SHA-256을 확인합니다. GGUF를 복원하려면 약 6GB의 여유 공간이 필요합니다. 이미 동일한 파일이 있으면 검증 후 건너뛰며, 다른 내용의 기존 모델은 덮어쓰지 않습니다.

## 원본 가중치까지 모두 복원하기

```powershell
python tools/restore_model.py --only all
```

- 실행용 GGUF: `artifacts/gemma4_e4b_coding-Q4_Q8.gguf` — 5,991,516,448바이트.
- 원본 가중치: `artifacts/hf/model.safetensors` — 15,992,595,884바이트.
- 원본의 `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`는 `artifacts/hf/`에 함께 보존합니다.
- 두 가중치를 모두 복원할 때는 약 22GB와 코드·설정 저장 공간이 필요합니다. 각 분할 조각은 1GiB 이하이며, GGUF 6개와 원본 15개입니다.
- 원본은 발견 당시 바이트 그대로 보관했습니다. 임의로 베이스 모델을 다시 받아 원본 대신 저장하지 않았습니다. 원본 Hugging Face 직접 로딩은 별도의 호환성 확인이 필요합니다.

## 학습 자료와 확인 범위

`dataset.jsonl`에는 코딩 예제 10,000행이 있고, `dataset_unique.jsonl`에는 118행이 있습니다. 데이터 수는 보유 파일의 행 수이며 최종 학습에 전부 사용됐다는 증거나 성능 점수가 아닙니다.

`train.py`에는 Unsloth·QLoRA 학습과 LoRA 저장 코드가 있습니다. 자체 파인튜닝 모델이라는 설명은 개발자의 모델 이력을 기준으로 합니다. 이번 백업 시점에는 과거 LoRA 어댑터와 최종 학습 로그를 찾지 못했으므로, 특정 학습 설정·학습 완료율·기본 모델 대비 개선율을 재현한 것으로 표시하지 않습니다. 새 학습이나 재파인튜닝을 수행한 백업도 아닙니다.

검증한 것은 복구된 모델의 Ollama 텍스트 추론, 코딩 요청과 수정, 브라우저 HTML·CSS·JavaScript 캔버스 실행입니다. GGUF는 텍스트 경로를 변환한 것이며 이미지 입력 지원까지 검증한 모델로 설명하지 않습니다.

## 저장소 구성

```text
train.py / evaluate.py           학습·평가
prepare_dataset.py 외           데이터 준비·생성·정리
dataset*.jsonl                  보유 코딩 학습 데이터
agent/ / coding_agent.py         코딩 에이전트
serve_api.py                     Ollama 기반 API
ide/                             데스크톱 IDE 클라이언트
artifacts/hf/                    원본 설정·토크나이저
ollama/Modelfile                 복원한 GGUF의 Ollama 등록 설정
tools/restore_model.py           릴리스 다운로드·복원·해시 검증
tools/*gemma*.py                 당시 복구 변환·경량화 코드
backup-manifest.json             정확한 크기·해시·21개 조각 주소
licenses/                       Gemma 4 기반 모델 라이선스 사본
docs/                           복구 이력·기존 사용법
```

실패하거나 불완전했던 중간 GGUF, 중복 변환 캐시와 0바이트 임시 파일은 복원 원본으로 배포하지 않습니다. 원본 safetensors와 실행 검증한 GGUF를 백업 기준으로 삼습니다.

## 환경 변수

`.env.example`을 `.env`로 복사한 후 필요한 값을 로컬에서 입력합니다. 실제 `.env`, 토큰, 서비스 인증키는 Git과 릴리스 모두에 포함하지 않습니다. `.env`의 HF/NVIDIA 키는 학습·데이터 생성용이고, 이미 복원한 GGUF의 Ollama 로컬 추론에는 필요하지 않습니다.

예전에 공개된 인증키는 파일 삭제와 별개로 발급 서비스에서 폐기·재발급해야 합니다. 이 저장소의 현재 브랜치 이력에서는 `.env`를 제거했지만 외부 복제본이나 과거 커밋 캐시의 삭제를 보장하지 않습니다.

## 학습 또는 API 사용

기존 상세 명령은 [학습 안내](docs/TRAINING_GUIDE_LEGACY.md)를 참고하되, 과거 안내에 적힌 기본값보다 현재 `train.py --help`가 우선합니다. CUDA·PyTorch·Unsloth 버전 호환성은 사용 환경에서 확인해야 합니다. 모델 복원은 학습 환경 설치와 별개입니다.

API·IDE 예전 기본 모델명은 하이픈을 사용합니다. API 설정의 `API_MODEL_NAME=gemma4_e4b_coding`을 지정하거나 모델명을 직접 맞춰 사용하십시오. 캠퍼스 서비스에서 사용했던 로컬 별칭은 `gemma4-e4b-coding-recovered`이며, 백업의 공식 모델명은 `gemma4_e4b_coding`입니다.

## 라이선스와 변경 고지

Gemma 4의 공식 모델 라이선스는 [Apache License 2.0](https://ai.google.dev/gemma/apache_2)입니다. 사본은 [licenses/GEMMA4-APACHE-2.0.txt](licenses/GEMMA4-APACHE-2.0.txt)에 포함합니다. 이전 Gemma 세대의 약관과 구분합니다. 파인튜닝·변환·양자화 변경 내용은 [모델 백업 기록](docs/MODEL_BACKUP.md)과 [NOTICE](NOTICE)에 표시했습니다.

학습 데이터의 예제·생성 코드와 제3자 코드의 기존 저작권 고지는 유지합니다. 이 문서는 출처가 확정되지 않은 모든 학습 데이터에 새 이용 허락을 부여하는 선언이 아닙니다.
