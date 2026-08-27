@echo off
chcp 65001 >nul
if not exist "venv" (
    echo [오류] 먼저 setup.bat 을 실행해 주세요!
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python realtime_mic.py
pause
