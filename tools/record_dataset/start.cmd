@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PY=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo venv python not found: %PY%
  pause
  exit /b 1
)
"%PY%" "%~dp0record.py" %*
if "%~1"=="" pause
endlocal
