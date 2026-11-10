# Gemma IDE (로컬 Cursor형 클라이언트)

Electron + React + Monaco 미니 IDE.  
백엔드 [`serve_api.py`](../serve_api.py)의 도구/에이전트 API를 사용합니다.

## 사전 준비

1. Ollama에 `gemma4-e4b-coding` 모델
2. 프로젝트 루트에서 API 서버:

```powershell
cd C:\Users\USER\gemma4-e4b-coding-ft
.\.venv\Scripts\Activate.ps1
python serve_api.py
```

3. `.env`의 `API_KEY` 값을 앱 설정에 입력

## 실행

```powershell
cd ide
npm install
npm run dev
```

창이 뜨면 **설정**에서:

- API Base: `http://127.0.0.1:8000`
- API Key: `.env`의 `API_KEY`
- Workspace: 작업할 프로젝트 폴더

## 사용

- 왼쪽: 파일 트리 (클릭으로 열기)
- 가운데: Monaco 에디터 (`Ctrl+S` 저장 → `write_file`)
- 오른쪽: 에이전트 채팅 (서버가 도구로 파일 생성/수정)

연결이 안 되면 상단 상태바에서 `serve_api.py` 실행 여부를 확인하세요.
