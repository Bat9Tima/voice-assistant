#Requires -Version 5.1
param(
    [switch]$Pull,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

if ($Pull) {
    Write-Host "=== Получение изменений из GitHub ==="
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Не удалось обновить: есть локальные изменения или конфликт."
        Write-Host "Сначала выполните sync-push.cmd или разберитесь с git status."
        exit 1
    }
    Write-Host ""
    Write-Host "Готово. Перезапустите ассистента, если он уже запущен."
    exit 0
}

if ($Push) {
    Write-Host "=== Отправка изменений в GitHub ==="
    git add -A
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
        git commit -m "Синхронизация: $env:COMPUTERNAME $stamp"
    }
    git pull --rebase --autostash
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Не удалось синхронизироваться: конфликт или нет сети."
        Write-Host "Разберитесь с git status и повторите."
        exit 1
    }
    git push
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Push не удался. Проверьте вход в GitHub и интернет."
        exit 1
    }
    Write-Host ""
    Write-Host "Готово."
    exit 0
}

Write-Host "Укажите -Pull или -Push."
exit 2
