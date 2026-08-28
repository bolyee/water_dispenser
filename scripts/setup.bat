@echo off
chcp 65001 >nul
:: 이 스크립트는 scripts/ 안에 있으므로 저장소 루트로 이동한 뒤 실행한다.
cd /d "%~dp0.."
echo ============================================
echo   Water Dispenser 자동 환경 설정
echo ============================================
echo.

:: Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo        https://www.python.org/downloads/ 에서 Python 3.10을 설치해 주세요.
    pause
    exit /b 1
)

:: venv 생성
if not exist "venv" (
    echo [1/2] 가상환경 생성 중...
    python -m venv venv
    echo      완료!
) else (
    echo [1/2] 가상환경이 이미 존재합니다. 건너뜁니다.
)

:: 패키지 설치
echo [2/2] 필요한 패키지 설치 중... (시간이 좀 걸립니다)
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo ============================================
echo   설치 완료! 이제 scripts\run.bat 으로 실행하세요.
echo ============================================
pause
