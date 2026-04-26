@echo off
REM Install Murmur dependencies. Uses python.exe from PATH first,
REM then a typical Windows user install.

set "MURMUR_DIR=%~dp0"
set "PYTHON="

where python.exe >nul 2>nul
if %ERRORLEVEL%==0 (
    for /f "delims=" %%P in ('where python.exe') do (
        set "PYTHON=%%P"
        goto :found
    )
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto :found
)

echo ERROR: python.exe not found. Install Python 3.11+ from python.org and re-run.
pause
exit /b 1

:found
echo Installing Murmur dependencies with %PYTHON% ...
"%PYTHON%" -m pip install -r "%MURMUR_DIR%requirements.txt"
echo.
echo Done. Launch with start_murmur.bat
pause
