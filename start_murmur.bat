@echo off
REM Launch Murmur using the project venv (no console window).

set "MURMUR_DIR=%~dp0"
set "PYTHONW=%MURMUR_DIR%.venv\Scripts\pythonw.exe"

if not exist "%PYTHONW%" (
    echo ERROR: venv not found at %PYTHONW%
    echo Run: py -3.14 -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt nvidia-cudnn-cu12 nvidia-cublas-cu12
    pause
    exit /b 1
)

start "" "%PYTHONW%" "%MURMUR_DIR%murmur.py"
