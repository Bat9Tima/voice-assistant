@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Получение изменений из GitHub ===
git pull --ff-only
if errorlevel 1 (
    echo.
    echo Не удалось обновить: есть локальные изменения или конфликт.
    echo Сначала выполните sync-push.cmd или разберитесь с git status.
    pause
    exit /b 1
)
echo.
echo Готово. Перезапустите ассистента, если он уже запущен.
pause
