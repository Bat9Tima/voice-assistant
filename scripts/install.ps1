#Requires -Version 5.1
param(
    [switch]$English,
    [switch]$Whisper,
    [switch]$NoEnglish,
    [ValidateSet("tiny", "base", "small", "medium")]
    [string]$WhisperModel = "tiny",
    [switch]$Clean,
    [switch]$NoAutostart,
    [switch]$NoStart,
    [switch]$DryRun,
    [switch]$InstallPython,
    [string]$SourceUrl = "",
    [string]$ModelsDir = "",
    [switch]$SkipDoctor
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

$setup = Join-Path $PSScriptRoot "setup.ps1"
$autostart = Join-Path $PSScriptRoot "install-autostart.ps1"
$start = Join-Path $PSScriptRoot "start.ps1"
$doctor = Join-Path $PSScriptRoot "doctor.cmd"
$uninstall = Join-Path $PSScriptRoot "uninstall.ps1"

foreach ($required in @($setup, $autostart, $start, $doctor, $uninstall)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Не найден обязательный файл установщика: $required"
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


function Read-MenuChoice {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [Parameter(Mandatory = $true)][string[]]$Allowed
    )
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        $answer = (Read-Host $Prompt).Trim()
        if ($Allowed -contains $answer) { return $answer }
        Write-Host "  Введите одно из: $($Allowed -join ', ')" -ForegroundColor Yellow
    }
    throw "Не удалось получить ответ на вопрос: $Prompt"
}


# ---------------------------------------------------------------- меню
# Запуск без аргументов (двойной клик по install.cmd) — интерактивный режим.
if ($PSBoundParameters.Count -eq 0) {
    Write-Host ""
    Write-Host "=== Установщик Voice Assistant ===" -ForegroundColor Cyan
    Write-Host "  1. Полная установка (эта машина или новая)"
    Write-Host "  2. Обновить исходники из ZIP"
    Write-Host "  3. Только окружение и модели (ремонт)"
    Write-Host "  4. Диагностика"
    Write-Host "  5. Удаление"
    Write-Host "  0. Выход"
    $choice = Read-MenuChoice "Выбор" @("0", "1", "2", "3", "4", "5")
    switch ($choice) {
        "0" { exit 0 }
        "4" { & $doctor; exit $LASTEXITCODE }
        "5" { & $uninstall; exit $LASTEXITCODE }
        "2" {
            if (-not $SourceUrl) {
                $SourceUrl = (Read-Host "Путь к ZIP или http(s)-ссылка").Trim().Trim('"')
            }
            if (-not $SourceUrl) {
                Write-Host "Нужен путь к ZIP или ссылка." -ForegroundColor Yellow
                exit 1
            }
        }
        "3" {
            $NoAutostart = $true
            $NoStart = $true
        }
        "1" {
            if (-not (Find-Python312)) {
                $answer = Read-MenuChoice "Python 3.12 не найден. Поставить через winget? (y/n)" @("y", "n", "д", "н")
                if ($answer -in @("y", "д")) {
                    $InstallPython = $true
                } else {
                    Write-Host "Установите Python 3.12 и запустите установщик снова." -ForegroundColor Yellow
                    exit 1
                }
            }
            $modelsAnswer = (Read-Host "Папка с архивами моделей для переноса (Enter — без переноса)").Trim().Trim('"')
            if ($modelsAnswer) { $ModelsDir = $modelsAnswer }
        }
    }
}

# ---------------------------------------------------------------- dry-run
if ($DryRun) {
    Write-Host ""
    Write-Host "Режим проверки: ничего не будет изменено." -ForegroundColor Yellow
    if ($Clean) {
        Write-Host "  - удалить .venv и создать окружение заново"
    } else {
        Write-Host "  - использовать существующее .venv или создать его"
    }
    if ($SourceUrl) {
        Write-Host "  - скачать исходники и обновить папку проекта: $SourceUrl"
    }
    if ($InstallPython) {
        Write-Host "  - при отсутствии Python 3.12 установить его через winget"
    }
    Write-Host "  - установить зависимости из requirements.lock"
    Write-Host "  - скачать Vosk RU для wake word"
    Write-Host "  - скачать GigaAM v3, если модели нет"
    if ($ModelsDir) {
        Write-Host "  - брать и складывать архивы моделей в: $ModelsDir"
    } else {
        Write-Host "  - архивы моделей: %LOCALAPPDATA%\VoiceAssistant\models"
    }
    if ($English -and -not $NoEnglish) {
        Write-Host "  - установить Vosk EN, faster-whisper и скачать модель $WhisperModel"
    }
    if ($NoAutostart) {
        Write-Host "  - автозапуск не трогать"
    } else {
        Write-Host "  - установить автозапуск в Startup"
    }
    if ($NoStart) {
        Write-Host "  - не запускать ассистента"
    } else {
        Write-Host "  - запустить ассистента в фоне"
    }
    if ($SkipDoctor) {
        Write-Host "  - не запускать финальную диагностику"
    } else {
        Write-Host "  - в конце проверить установку через doctor"
    }
    Write-Host ""
    Write-Host "Проверка окружения:"
    $py = Find-Python312
    if ($py) {
        Write-Host "  Python 3.12: $py"
    } else {
        Write-Host "  Python 3.12: НЕ НАЙДЕН (setup попробует через py -3.12 или winget)" -ForegroundColor Yellow
    }
    $gigaam = Join-Path $env:LOCALAPPDATA "VoiceAssistant\models\sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16\encoder.int8.onnx"
    $vosk = Join-Path $env:LOCALAPPDATA "VoiceAssistant\models\vosk-ru-small\am"
    Write-Host "  Vosk: $(if (Test-Path $vosk) { 'уже установлен' } else { 'будет скачан (~87 МБ)' })"
    Write-Host "  GigaAM: $(if (Test-Path $gigaam) { 'уже установлен' } else { 'будет скачан (~160 МБ)' })"
    Write-Host ""
    Write-Host "Для реальной установки запустите без -DryRun." -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------- журнал
$transcriptStarted = $false
$logPath = ""
try {
    $logDir = Join-Path $root "logs"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $logPath = Join-Path $logDir ("install-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
    Start-Transcript -Path $logPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Журнал установки: $logPath" -ForegroundColor DarkGray
} catch {
    Write-Warning "Не удалось включить журнал установки: $($_.Exception.Message)"
}

$failed = $false
try {
    # ------------------------------------------------------------ исходники
    if ($SourceUrl) {
        Write-Host ""
        Write-Host "Скачиваю исходники..."
        $tempZip = Join-Path $env:TEMP ("va-source-{0}.zip" -f [guid]::NewGuid())
        $tempDir = Join-Path $env:TEMP ("va-source-{0}" -f [guid]::NewGuid())
        try {
            if ($SourceUrl -match '^https?://') {
                & curl.exe -L --fail --retry 5 --retry-delay 2 -o $tempZip $SourceUrl
                if ($LASTEXITCODE -ne 0) {
                    throw "Не удалось скачать ZIP: $SourceUrl"
                }
            } elseif (Test-Path -LiteralPath $SourceUrl) {
                Copy-Item -LiteralPath $SourceUrl -Destination $tempZip -Force
            } else {
                throw "Укажите http(s)-ссылку на ZIP или путь к существующему ZIP-файлу."
            }

            Expand-Archive -LiteralPath $tempZip -DestinationPath $tempDir -Force
            $sourceRoot = $tempDir
            if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "app\main.py"))) {
                $mainFile = Get-ChildItem -LiteralPath $tempDir -Recurse -Filter "main.py" -ErrorAction SilentlyContinue |
                    Where-Object { $_.Directory -and $_.Directory.Name -eq "app" } |
                    Select-Object -First 1
                if ($mainFile) {
                    $sourceRoot = Split-Path -Parent $mainFile.Directory.FullName
                }
            }
            if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot "app\main.py"))) {
                throw "В архиве не найден app\main.py. Это должен быть ZIP с исходниками Voice Assistant."
            }

            foreach ($dirName in @("app", "bin", ".opencode", "scripts", "tests")) {
                $from = Join-Path $sourceRoot $dirName
                if (Test-Path -LiteralPath $from) {
                    Copy-Item -LiteralPath $from -Destination $root -Recurse -Force
                    Write-Host "  - обновлено: $dirName"
                }
            }
            foreach ($fileName in @("config.example.json", "requirements.txt", "requirements.lock", "requirements-whisper.txt", "requirements-dev.txt", "pyproject.toml", "README.md")) {
                $from = Join-Path $sourceRoot $fileName
                if (Test-Path -LiteralPath $from) {
                    Copy-Item -LiteralPath $from -Destination $root -Force
                    Write-Host "  - обновлено: $fileName"
                }
            }
            foreach ($fileName in @("config.example.json", "opencode.json", "AGENTS.md", ".gitignore", "install.cmd", "uninstall.cmd")) {
                $from = Join-Path $sourceRoot $fileName
                $to = Join-Path $root $fileName
                if ((Test-Path -LiteralPath $from) -and -not (Test-Path -LiteralPath $to)) {
                    Copy-Item -LiteralPath $from -Destination $root -Force
                    Write-Host "  - добавлено: $fileName"
                }
            }
            Get-ChildItem -LiteralPath $root -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
                Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "Исходники обновлены." -ForegroundColor Green
        } finally {
            Remove-Item -LiteralPath $tempZip -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    if ($Clean -and (Test-Path -LiteralPath ".venv")) {
        Write-Host "Удаляю старое виртуальное окружение..." -ForegroundColor Yellow
        Remove-Item -LiteralPath ".venv" -Recurse -Force
    }

    # ------------------------------------------------------------ окружение и модели
    $setupArgs = @{
        English = $English
        Whisper = $Whisper
        NoEnglish = $NoEnglish
        WhisperModel = $WhisperModel
        InstallPython = $InstallPython
    }
    if ($ModelsDir) { $setupArgs.ModelsDir = $ModelsDir }
    & $setup @setupArgs

    # ------------------------------------------------------------ автозапуск и запуск
    if (-not $NoAutostart) {
        try {
            & $autostart
        } catch {
            Write-Warning "Автозапуск не настроен: $($_.Exception.Message)"
        }
    }

    if (-not $NoStart) {
        try {
            & $start
        } catch {
            Write-Warning "Ассистент не запущен автоматически: $($_.Exception.Message)"
        }
    }

    # ------------------------------------------------------------ финальная проверка
    if (-not $SkipDoctor) {
        Write-Host ""
        Write-Host "=== Проверка установки ===" -ForegroundColor Cyan
        try {
            & $doctor
        } catch {
            Write-Warning "Диагностика не запустилась: $($_.Exception.Message)"
        }
    }
} catch {
    Write-Host ""
    Write-Host "ОШИБКА: $($_.Exception.Message)" -ForegroundColor Red
    $failed = $true
} finally {
    if ($transcriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
    }
}

if ($failed) {
    if ($logPath) { Write-Host "Журнал: $logPath" -ForegroundColor Yellow }
    exit 1
}

Write-Host ""
Write-Host "=== Установка завершена ===" -ForegroundColor Green
if ($logPath) { Write-Host "Журнал:      $logPath" }
Write-Host "Запуск:      .\scripts\start.cmd"
Write-Host "Проверка:    .\scripts\doctor.cmd"
Write-Host "Удаление:    .\uninstall.cmd"
