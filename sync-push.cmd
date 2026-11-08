@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Отправка изменений в GitHub ===
git add -A
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Синхронизация: %COMPUTERNAME% %DATE% %TIME%"
)
git pull --rebase --autostash
if errorlevel 1 (
    echo.
    echo Не удалось синхронизироваться: конфликт или нет сети.
    echo Разберитесь с git status и повторите.
    pause
    exit /b 1
)
git push
if errorlevel 1 (
    echo.
    echo Push не удался. Проверьте вход в GitHub и интернет.
    pause
    exit /b 1
)
echo.
echo Готово.
pause
