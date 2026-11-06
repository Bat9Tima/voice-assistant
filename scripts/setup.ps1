#Requires -Version 5.1
param(
    [switch]$English,
    [switch]$Whisper,
    [switch]$NoEnglish,
    [ValidateSet("tiny", "base", "small", "medium")]
    [string]$WhisperModel = "tiny",
    [switch]$InstallPython,
    [string]$ModelsDir = ""
)

$installEnglish = $English -and -not $NoEnglish
# Параметр $ModelsDir и локальная $modelsDir — в PowerShell одна переменная
# (регистр не важен), поэтому значение параметра фиксируем сразу.
$archiveDirParam = $ModelsDir

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

# SHA-256 рабочих файлов моделей, снятые с проверенной установки.
# Если upstream поменяет версию по тем же ссылкам — обновите хэши.
$SHA_VOSK_RU = "36D67B53DC90D96EA48097F9E17E7C5937689AD76CB8BED25091317BBAF25919"
$SHA_GIGAAM_ENCODER = "0EC54E5F130A91C0F228D21795EDE10B31C992A1F73E45DCD827E843A71A9B50"


function Get-SizeMB {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [math]::Round((Get-Item -LiteralPath $Path).Length / 1MB, 1)
}


# Качает файл с ретраями; готовый архив из папки переноса не трогает.
function Invoke-Download {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [double]$MinMB = 0
    )
    $parent = Split-Path -Parent $OutFile
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force $parent | Out-Null
    }
    if (Test-Path -LiteralPath $OutFile) {
        $cachedMB = Get-SizeMB $OutFile
        if ($MinMB -le 0 -or $cachedMB -ge $MinMB) {
            Write-Host "  уже есть: $OutFile ($cachedMB МБ)"
            return
        }
        Write-Host "  файл слишком мал ($cachedMB МБ) — перекачиваю" -ForegroundColor Yellow
        Remove-Item -LiteralPath $OutFile -Force
    }
    Write-Host "  скачиваю: $Url" -ForegroundColor Yellow
    & curl.exe -L --fail --retry 5 --retry-delay 2 --retry-connrefused -o $OutFile $Url
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutFile)) {
        Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
        throw "Не удалось скачать: $Url"
    }
    $actualMB = Get-SizeMB $OutFile
    if ($MinMB -gt 0 -and $actualMB -lt $MinMB) {
        Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
        throw "Скачано $actualMB МБ, ожидалось не меньше $MinMB МБ: $Url"
    }
    Write-Host "  готово: $OutFile ($actualMB МБ)"
}


# Проверяет SHA-256 файла модели после распаковки.
function Assert-FileHash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [string]$Label = "файл модели"
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Нет файла для проверки: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ne $Expected.ToUpperInvariant()) {
        throw ("$Label повреждён: SHA-256 $actual`n" +
               "Ожидался: $Expected`n" +
               "Удалите архивы моделей и повторите установку.")
    }
}


function Find-Python312 {
    $candidates = @()
    $candidates += (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe")
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        $candidate = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $candidate) {
            $candidates += ($candidate | Select-Object -First 1).Trim()
        }
    }
    foreach ($path in $candidates) {
        if ($path -and (Test-Path -LiteralPath $path)) { return $path }
    }
    return $null
}


Write-Host "=== Установка Voice Assistant ===" -ForegroundColor Cyan
Write-Host "Папка проекта: $root"

# --- Python 3.12 ---
$py = Find-Python312
if (-not $py -and $InstallPython) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget не найден. Установите Python вручную: winget install -e --id Python.Python.3.12 --scope user"
    }
    Write-Host "Python 3.12 не найден — устанавливаю через winget..." -ForegroundColor Yellow
    & winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget не смог установить Python 3.12 (код $LASTEXITCODE)."
    }
    $py = Find-Python312
}
if (-not $py) {
    throw ("Python 3.12 не найден.`n" +
           "Установите: winget install -e --id Python.Python.3.12 --scope user`n" +
           "Или запустите: install.cmd -InstallPython")
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

# --- Модели: рабочие файлы и папка-«флешка» с архивами ---
$dataDir = Join-Path $env:LOCALAPPDATA "VoiceAssistant"
$modelsDir = Join-Path $dataDir "models"
New-Item -ItemType Directory -Force $modelsDir | Out-Null

# Внимание: значение -ModelsDir уже сохранено в $archiveDirParam выше,
# потому что $modelsDir ниже перезапишет переменную параметра.
if ([string]::IsNullOrWhiteSpace($archiveDirParam)) {
    $archiveDir = $modelsDir
} else {
    $archiveDir = [System.IO.Path]::GetFullPath($archiveDirParam)
}
New-Item -ItemType Directory -Force $archiveDir | Out-Null
Write-Host "Архивы моделей: $archiveDir" -ForegroundColor DarkGray
if ($archiveDir -ne $modelsDir) {
    Write-Host "  готовые архивы отсюда будут использованы без интернета" -ForegroundColor DarkGray
}

# --- Vosk wake word model ---
$modelDir = Join-Path $modelsDir "vosk-ru-small"
$zip = Join-Path $archiveDir "vosk-model-small-ru-0.22.zip"
$needModel = -not (Test-Path (Join-Path $modelDir "am"))
$needArchive = ($archiveDir -ne $modelsDir) -and -not (Test-Path -LiteralPath $zip)
if ($needModel -or $needArchive) {
    Invoke-Download -Url "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip" -OutFile $zip -MinMB 40
}
if ($needModel) {
    Write-Host "Распаковываю Vosk RU..."
    Expand-Archive -Path $zip -DestinationPath $modelsDir -Force
    $extracted = Join-Path $modelsDir "vosk-model-small-ru-0.22"
    if (Test-Path $extracted) {
        if (Test-Path $modelDir) { Remove-Item -Recurse -Force $modelDir }
        Move-Item $extracted $modelDir
    }
}
Assert-FileHash -Path (Join-Path $modelDir "am\final.mdl") -Expected $SHA_VOSK_RU -Label "Vosk RU"
Write-Host "Модель Vosk: OK ($modelDir)"

# --- Silero VAD: локальное определение конца команды ---
Push-Location $root
& $vpy -c "import logging; logging.basicConfig(level=logging.INFO, format='%(message)s'); from app.vad import ensure_model; p = ensure_model(); print('Silero VAD:', p or 'не найден, останется порог громкости')"
Pop-Location

# --- Vosk EN: опциональный локатор английских слов в смешанных фразах ---
if ($installEnglish) {
    $enModelDir = Join-Path $modelsDir "vosk-en-small"
    $enZip = Join-Path $archiveDir "vosk-model-small-en-us-0.15.zip"
    $needEnModel = -not (Test-Path (Join-Path $enModelDir "am"))
    $needEnArchive = ($archiveDir -ne $modelsDir) -and -not (Test-Path -LiteralPath $enZip)
    if ($needEnModel -or $needEnArchive) {
        Invoke-Download -Url "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" -OutFile $enZip -MinMB 35
    }
    if ($needEnModel) {
        Write-Host "Распаковываю Vosk EN..."
        Expand-Archive -Path $enZip -DestinationPath $modelsDir -Force
        $extractedEn = Join-Path $modelsDir "vosk-model-small-en-us-0.15"
        if (Test-Path $extractedEn) {
            if (Test-Path $enModelDir) { Remove-Item -Recurse -Force $enModelDir }
            Move-Item $extractedEn $enModelDir
        }
    }
    if (-not (Test-Path (Join-Path $enModelDir "am"))) {
        throw "Модель Vosk EN распаковалась не полностью: $enModelDir"
    }
    Write-Host "Модель Vosk EN: OK ($enModelDir)"
}

# --- GigaAM v3 (основной движок распознавания) ---
$gigaamName = "sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16"
$gigaamDir = Join-Path $modelsDir $gigaamName
$tar = Join-Path $archiveDir "gigaam-v3.tar.bz2"
$needGigaam = -not (Test-Path (Join-Path $gigaamDir "encoder.int8.onnx"))
$needGigaamArchive = ($archiveDir -ne $modelsDir) -and -not (Test-Path -LiteralPath $tar)
if ($needGigaam -or $needGigaamArchive) {
    Invoke-Download -Url "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$gigaamName.tar.bz2" -OutFile $tar -MinMB 100
}
if ($needGigaam) {
    Write-Host "Распаковываю GigaAM v3..."
    & tar.exe -xjf $tar -C $modelsDir
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось распаковать $tar (нужен tar.exe из Windows 10 1803+)"
    }
}
Assert-FileHash -Path (Join-Path $gigaamDir "encoder.int8.onnx") -Expected $SHA_GIGAAM_ENCODER -Label "GigaAM v3"
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
Write-Host "Архивы моделей остаются в: $archiveDir" -ForegroundColor DarkGray
Write-Host "Для установки на другом ПК без интернета скопируйте эту папку и запустите:" -ForegroundColor DarkGray
Write-Host "  install.cmd -ModelsDir <папка с архивами>" -ForegroundColor DarkGray
Write-Host ""
Write-Host "=== Готово ===" -ForegroundColor Green
Write-Host "Проверка без микро:  .\scripts\run.cmd --text `"открой ютуб`""
Write-Host "Диагностика:         .\scripts\doctor.cmd --record 5 --wake --stt"
Write-Host "Запуск:              .\scripts\run.cmd"
Write-Host "Автозапуск:          .\scripts\install-autostart.cmd"
