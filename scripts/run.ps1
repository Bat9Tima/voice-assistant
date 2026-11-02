$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Окружение не найдено. Сначала запустите scripts\setup.ps1"
}

& $python -m app.main @args
exit $LASTEXITCODE
