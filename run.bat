@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM 현재 폴더로 이동
cd /d "%~dp0"

REM 1) Python 찾기 (py 런처 우선, 없으면 python)
where py >nul 2>&1 && (set "PY=py -3.12") || (set "PY=python")
%PY% --version >nul 2>&1 || (
  echo [오류] Python 3.12 실행 불가. py 런처 또는 Python 3.12가 설치되어 있는지 확인하세요.
  pause & exit /b 1
)

REM 2) 가상환경 준비 (괄호 블록으로 안전하게)
if not exist ".venv312\Scripts\python.exe" (
  echo === 가상환경 생성: .venv312 ===
  %PY% -m venv ".venv312" || (echo [오류] venv 생성 실패 & pause & exit /b 1)
)

call ".venv312\Scripts\activate.bat" || (echo [오류] venv 활성화 실패 & pause & exit /b 1)

REM 3) 필수 패키지(버전 고정) 설치 — 한 번만 설치됨
python -m pip install -q --upgrade pip wheel setuptools
python -m pip install -q "streamlit==1.38.0" "protobuf<6" "altair<6" "pandas<3" "numpy<2.4" "pyarrow>=15,<19" || (
  echo [오류] 패키지 설치 실패
  pause & exit /b 1
)

REM 4) app.py 실행
if not exist "app.py" (echo [오류] app.py 없음 & pause & exit /b 1)
echo === Streamlit 실행 시작 ===
streamlit run "app.py" --server.headless=false || (
  echo [오류] 실행 중 오류 발생
  pause & exit /b 1
)
