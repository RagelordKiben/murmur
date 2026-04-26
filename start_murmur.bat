@echo off
REM Launch Murmur. Uses pythonw.exe (no console window).
REM Looks for Python on PATH first, then a typical Windows user install.

set "MURMUR_DIR=%~dp0"
set "PYTHONW="

where pythonw.exe >nul 2>nul
if %ERRORLEVEL%==0 (
    for /f "delims=" %%P in ('where pythonw.exe') do (
        set "PYTHONW=%%P"
        goto :found
    )
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe" (
    set "PYTHONW=%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe" (
    set "PYTHONW=%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\pythonw.exe" (
    set "PYTHONW=%LOCALAPPDATA%\Programs\Python\Python313\pythonw.exe"
    goto :found
)

echo ERROR: pythonw.exe not found. Install Python 3.11+ from python.org and re-run.
pause
exit /b 1

:found
start "" "%PYTHONW%" "%MURMUR_DIR%murmur.py"
