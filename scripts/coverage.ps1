# Локальный отчёт о покрытии. Это измерение, а не проверка: порога нет.
# Сообщения намеренно ASCII: консоль Windows иначе показывает кракозябры.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "venv not found: run scripts\setup.ps1 first"
}

& $python -c "import coverage" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "coverage not found: pip install -r requirements-dev.txt"
}

& $python -m coverage erase
& $python -m coverage run -m unittest discover -s tests
$testExit = $LASTEXITCODE

Write-Host "== coverage report =="
& $python -m coverage report --show-missing
$reportExit = $LASTEXITCODE

& $python -m coverage html -d coverage_html
$htmlExit = $LASTEXITCODE

if ($testExit -ne 0) { exit $testExit }
if ($reportExit -ne 0) { exit $reportExit }
if ($htmlExit -ne 0) { exit $htmlExit }
Write-Host "HTML: coverage_html\index.html"
exit 0