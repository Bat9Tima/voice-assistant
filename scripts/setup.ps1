#Requires -Version 5.1
param(
    [switch]$English,
    [switch]$Whisper,
    [switch]$NoEnglish,
    [ValidateSet("tiny", "base", "small", "medium")]
    [string]$WhisperModel = "tiny"
)

$installEnglish = $English -and -not $NoEnglish

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

Write-Host "=== Установка Voice Assistant ===" -ForegroundColor Cyan
Write-Host "Папка проекта: $root"

# --- Python 3.12 ---
$py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) {
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $candidate = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate) {
            $py = $candidate.Trim()
        }
    }
}
if (-not (Test-Path $py)) {
    throw "Python 3.12 не найден. Установите: winget install -e --id Python.Python.3.12 --scope user"
}
Write-Host "Python: $py"

# --- venv ---
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Создаю виртуальное окружение..."
    & $py -m venv .venv
}
$vpy = Join-Path $root ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install -r requirements.txt

# --- Vosk wake word model ---
$dataDir = Join-Path $env:LOCALAPPDATA "VoiceAssistant"
$modelsDir = Join-Path $dataDir "models"
$modelDir = Join-Path $modelsDir "vosk-ru-small"
if (-not (Test-Path (Join-Path $modelDir "am"))) {
    New-Item -ItemType Directory -Force $modelsDir | Out-Null
    $zip = Join-Path $modelsDir "vosk-ru-small.zip"
    Write-Host "Скачиваю модель Vosk (~45 МБ)..." -ForegroundColor Yellow
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $modelsDir -Force
    $extracted = Join-Path $modelsDir "vosk-model-small-ru-0.22"
    if (Test-Path $extracted) {
        if (Test-Path $modelDir) { Remove-Item -Recurse -Force $modelDir }
        Move-Item $extracted $modelDir
    }
    Remove-Item $zip -Force
}
if (-not (Test-Path (Join-Path $modelDir "am"))) {
    throw "Модель Vosk распаковалась не полностью: $modelDir"
}
Write-Host "Модель Vosk: OK ($modelDir)"

# --- Silero VAD: локальное определение конца команды ---
Push-Location $root
& $vpy -c "import logging; logging.basicConfig(level=logging.INFO, format='%(message)s'); from app.vad import ensure_model; p = ensure_model(); print('Silero VAD:', p or 'не найден, останется порог громкости')"
Pop-Location

# --- Vosk EN: опциональный локатор английских слов в смешанных фразах ---
if ($installEnglish) {
    $enModelDir = Join-Path $modelsDir "vosk-en-small"
    if (-not (Test-Path (Join-Path $enModelDir "am"))) {
        $zip = Join-Path $modelsDir "vosk-en-small.zip"
        Write-Host "Скачиваю модель Vosk EN (~40 МБ)..." -ForegroundColor Yellow
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $modelsDir -Force
        $extracted = Join-Path $modelsDir "vosk-model-small-en-us-0.15"
        if (Test-Path $extracted) {
            if (Test-Path $enModelDir) { Remove-Item -Recurse -Force $enModelDir }
            Move-Item $extracted $enModelDir
        }
        Remove-Item $zip -Force
    }
    if (-not (Test-Path (Join-Path $enModelDir "am"))) {
        throw "Модель Vosk EN распаковалась не полностью: $enModelDir"
    }
    Write-Host "Модель Vosk EN: OK ($enModelDir)"
}

# --- GigaAM v3 (основной движок распознавания) ---
$gigaamName = "sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16"
$gigaamDir = Join-Path $modelsDir $gigaamName
if (-not (Test-Path (Join-Path $gigaamDir "encoder.int8.onnx"))) {
    Write-Host "Скачиваю GigaAM v3 (~160 МБ)..." -ForegroundColor Yellow
    $tar = Join-Path $modelsDir "gigaam-v3.tar.bz2"
    & curl.exe -L --retry 3 --retry-delay 2 -o $tar "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$gigaamName.tar.bz2"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $tar)) {
        throw "Не удалось скачать модель GigaAM. Проверьте интернет и повторите."
    }
    & tar.exe -xjf $tar -C $modelsDir
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось распаковать $tar (нужен tar.exe из Windows 10 1803+)"
    }
    Remove-Item $tar -Force
}
if (-not (Test-Path (Join-Path $gigaamDir "encoder.int8.onnx"))) {
    throw "Модель GigaAM распаковалась не полностью: $gigaamDir"
}
Write-Host "Модель GigaAM: OK ($gigaamDir)"

# --- Whisper tiny: опциональные английские слова в поисковых запросах ---
if ($installEnglish) {
    Write-Host "Устанавливаю faster-whisper..." -ForegroundColor Yellow
    & $vpy -m pip install -r requirements-whisper.txt
    $whisperDir = (Join-Path $modelsDir "whisper") -replace "\\", "/"
    Write-Host "Скачиваю Whisper $WhisperModel (~75 МБ для tiny)..." -ForegroundColor Yellow
    & $vpy -c "from faster_whisper import WhisperModel; WhisperModel('$WhisperModel', device='cpu', compute_type='int8', download_root=r'$whisperDir'); print('Whisper: OK')"
    Write-Host "Для английских слов включите stt.parallel_english = true в config.json." -ForegroundColor DarkGray
} else {
    Write-Host "Английское распознавание не устанавливается (по умолчанию только GigaAM)." -ForegroundColor DarkGray
    Write-Host "Нужно включить: .\scripts\setup.cmd -English" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "=== Готово ===" -ForegroundColor Green
Write-Host "Проверка без микро:  .\scripts\run.cmd --text `"открой ютуб`""
Write-Host "Диагностика:         .\scripts\doctor.cmd --record 5 --wake --stt"
Write-Host "Запуск:              .\scripts\run.cmd"
Write-Host "Автозапуск:          .\scripts\install-autostart.cmd"
