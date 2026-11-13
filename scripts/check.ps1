# Локальный прогон тех же проверок, что делает CI.
# Сообщения намеренно ASCII: консоль Windows иначе показывает кракозябры.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "venv not found: run scripts\setup.ps1 first"
}

$failed = $false

Write-Host "== ruff check =="
& $python -m ruff check app tests
if ($LASTEXITCODE -ne 0) { $failed = $true }

Write-Host "== ruff format --check =="
& $python -m ruff format --check app tests
if ($LASTEXITCODE -ne 0) { $failed = $true }

Write-Host "== unittest =="
& $python -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) { $failed = $true }

if ($failed) {
    Write-Host "FAILED: checks did not pass"
    exit 1
}
Write-Host "OK: all checks passed"
exit 0