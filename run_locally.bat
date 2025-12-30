@echo off
echo Starting Bee Detection Web App...
cd /d "%~dp0"
.\yolovenv\Scripts\python.exe -m uvicorn app:app --reload
if %errorlevel% neq 0 (
    echo.
    echo Server stopped or crashed.
    pause
)
