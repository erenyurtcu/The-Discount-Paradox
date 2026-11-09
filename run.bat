@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py -3.10"
) else (
    set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto error
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo Starting The Discount Paradox...
".venv\Scripts\python.exe" -m streamlit run profit_arena_v2.py
if errorlevel 1 goto error
exit /b 0

:error
echo.
echo Setup or application startup failed.
pause
exit /b 1
