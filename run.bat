@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Gemma IDE - API + Client
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv 이 없습니다. 먼저 가상환경을 만드세요.
  pause
  exit /b 1
)

if not exist "ide\package.json" (
  echo [ERROR] ide\ 폴더가 없습니다.
  pause
  exit /b 1
)

if not exist "ide\node_modules\" (
  echo [INFO] ide 의존성 설치 중 (npm install)...
  pushd ide
  call npm install
  if errorlevel 1 (
    echo [ERROR] npm install 실패
    popd
    pause
    exit /b 1
  )
  popd
)

echo [1/2] API 서버 시작 (serve_api.py :8000)
start "Gemma API" cmd /k "cd /d "%~dp0" && .venv\Scripts\activate.bat && python serve_api.py"

timeout /t 2 /nobreak >nul

echo [2/2] IDE 클라이언트 시작 (npm run dev)
start "Gemma IDE" cmd /k "cd /d "%~dp0ide" && npm run dev"

echo.
echo 두 창이 열렸습니다.
echo   - API : http://127.0.0.1:8000/docs
echo   - IDE : Electron 창 (Vite)
echo.
echo 종료하려면 각 창에서 Ctrl+C 후 닫으세요.
echo.
endlocal
