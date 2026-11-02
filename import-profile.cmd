@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\import-chrome-profile.ps1" %*
exit /b %ERRORLEVEL%
