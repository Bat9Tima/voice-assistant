@echo off
chcp 65001 >nul
cd /d "%TEMP%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\uninstall.ps1" %*
exit /b %ERRORLEVEL%
