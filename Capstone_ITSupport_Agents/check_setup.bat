@echo off
rem Double-click to check whether the project can run and whether the web app is running.
title Bayview IT Assistant - setup check
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [X]  The project's Python environment ^(.venv^) was not found. Set it up once with:
    echo       python -m venv .venv
    echo       .venv\Scripts\pip install -r requirements.txt
) else (
    ".venv\Scripts\python.exe" scripts\check_setup.py
)
echo.
pause
