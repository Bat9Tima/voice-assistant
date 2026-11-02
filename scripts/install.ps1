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
    [string]$SourceUrl = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = "1"

Write-Host "=== Установка Voice Assistant ===" -ForegroundColor Cyan
Write-Host "Папка проекта: $root"

$setup = Join-Path $PSScriptRoot "setup.ps1"
$autostart = Join-Path $PSScriptRoot "install-autostart.ps1"
$start = Join-Path $PSScriptRoot "start.ps1"

foreach ($required in @($setup, $autostart, $start)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Не найден обязательный файл установщика: $required"
    }
}

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
    Write-Host "  - установить зависимости из requirements.txt"
    Write-Host "  - скачать Vosk RU для wake word"
    Write-Host "  - скачать GigaAM v3, если модели нет"
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
    Write-Host ""
    Write-Host "Проверка окружения:"
    $py = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path $py) {
        Write-Host "  Python 3.12: $py"
    } else {
        Write-Host "  Python 3.12: НЕ НАЙДЕН (setup попробует найти через py -3.12)" -ForegroundColor Yellow
    }
    $gigaam = Join-Path $env:LOCALAPPDATA "VoiceAssistant\models\sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16\encoder.int8.onnx"
    $vosk = Join-Path $env:LOCALAPPDATA "VoiceAssistant\models\vosk-ru-small\am"
    Write-Host "  Vosk: $(if (Test-Path $vosk) { 'уже установлен' } else { 'будет скачан (~87 МБ)' })"
    Write-Host "  GigaAM: $(if (Test-Path $gigaam) { 'уже установлен' } else { 'будет скачан (~160 МБ)' })"
    Write-Host ""
    Write-Host "Для реальной установки запустите без -DryRun." -ForegroundColor Green
    exit 0
}

if ($SourceUrl) {
    Write-Host ""
    Write-Host "Скачиваю исходники..."
    $tempZip = Join-Path $env:TEMP ("va-source-{0}.zip" -f [guid]::NewGuid())
    $tempDir = Join-Path $env:TEMP ("va-source-{0}" -f [guid]::NewGuid())
    try {
        if ($SourceUrl -match '^https?://') {
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri $SourceUrl -OutFile $tempZip -UseBasicParsing
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
        foreach ($fileName in @("requirements.txt", "requirements-whisper.txt", "README.md")) {
            $from = Join-Path $sourceRoot $fileName
            if (Test-Path -LiteralPath $from) {
                Copy-Item -LiteralPath $from -Destination $root -Force
                Write-Host "  - обновлено: $fileName"
            }
        }
        foreach ($fileName in @("config.json", "opencode.json", "AGENTS.md", ".gitignore", "install.cmd", "uninstall.cmd")) {
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
    } catch {
        Write-Error "Не удалось получить исходники: $($_.Exception.Message)"
        exit 1
    } finally {
        Remove-Item -LiteralPath $tempZip -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($Clean -and (Test-Path -LiteralPath ".venv")) {
    Write-Host "Удаляю старое виртуальное окружение..." -ForegroundColor Yellow
    Remove-Item -LiteralPath ".venv" -Recurse -Force
}

try {
    & $setup -English:$English -NoEnglish:$NoEnglish -WhisperModel $WhisperModel
} catch {
    Write-Error "Установка зависимостей и моделей не удалась: $($_.Exception.Message)"
    exit 1
}

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

Write-Host ""
Write-Host "=== Установка завершена ===" -ForegroundColor Green
Write-Host "Запуск:      .\scripts\start.cmd"
Write-Host "Проверка:    .\scripts\doctor.cmd"
Write-Host "Удаление:    .\uninstall.cmd"
