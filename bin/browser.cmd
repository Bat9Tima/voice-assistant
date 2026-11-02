@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m app.browser_cli %*
) else (
  py -3.12 -m app.browser_cli %*
)
exit /b %ERRORLEVEL%
