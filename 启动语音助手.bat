@echo off
rem ============================================
rem  Desktop Voice Assistant Launcher
rem  Always runs with the project venv python.
rem  Double-click this file to start the app.
rem ============================================
cd /d "%~dp0"

set "VENV_PY=%~dp0venv_assistant\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] venv python not found: %VENV_PY%
    echo Please rebuild the virtual environment first.
    pause
    exit /b 1
)

"%VENV_PY%" "%~dp0src\main.py"

if errorlevel 1 (
    echo.
    echo [ERROR] App exited with an error. See messages above.
    pause
)
