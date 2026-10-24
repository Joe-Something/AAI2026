@echo off
rem Double-click to start the Bayview IT Assistant web app.
rem Uses the project's own Python (.venv), starts Ollama if it is installed, opens the browser,
rem and keeps this window open so status and errors stay visible. Close the window to stop the app.
title Bayview IT Assistant
cd /d "%~dp0"
set PYTHONUNBUFFERED=1

if not exist ".venv\Scripts\python.exe" (
    echo [X] The project's Python environment ^(.venv^) was not found. Set it up once with:
    echo       python -m venv .venv
    echo       .venv\Scripts\pip install -r requirements.txt
    goto :end
)

set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if exist "%OLLAMA%" (
    "%OLLAMA%" list >nul 2>&1
    if errorlevel 1 (
        echo [..] Starting the local AI model server ^(Ollama^)...
        start "" /min "%OLLAMA%" serve
        timeout /t 5 /nobreak >nul
    )
) else (
    echo [i] Ollama is not installed: the app will use Gemini if GEMINI_API_KEY is set, otherwise offline rules.
)

echo [..] Starting the app. The first start loads the AI model and can take up to a minute.
echo      When you see "Running on local URL", it is ready and your browser opens at http://127.0.0.1:7860
echo      Keep this window open while you use the app. Close it ^(or press Ctrl+C^) to stop.
echo.
".venv\Scripts\python.exe" app.py %*

:end
echo.
echo The app has stopped. If there is an error above, copy it and ask for help.
pause
