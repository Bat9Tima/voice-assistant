@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0coverage.ps1" %*
exit /b %ERRORLEVEL%