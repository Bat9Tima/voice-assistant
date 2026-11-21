@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\sync.ps1" -Pull
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
