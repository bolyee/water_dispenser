@echo off
chcp 65001 >nul
:: 이 스크립트는 scripts/ 안에 있으므로 저장소 루트로 이동한 뒤 실행한다.
cd /d "%~dp0.."
if not exist "venv" (
    echo [오류] 먼저 scripts\setup.bat 을 실행해 주세요!
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python realtime\realtime_mic.py
pause
