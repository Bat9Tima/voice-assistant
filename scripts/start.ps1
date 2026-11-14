$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"

if (-not (Test-Path $pythonw)) {
    throw "Окружение не найдено. Сначала запустите scripts\setup.ps1"
}

$running = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*app.main*" }
if ($running) {
    Write-Host "Ассистент уже запущен (PID $($running.ProcessId -join ', '))" -ForegroundColor Yellow
    return
}

Start-Process -FilePath $pythonw -ArgumentList "-m", "app.main" -WorkingDirectory $root -WindowStyle Hidden
Write-Host "Ассистент запущен в фоне. Лог: $root\logs\assistant.log" -ForegroundColor Green
