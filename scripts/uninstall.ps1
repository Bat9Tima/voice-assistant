#Requires -Version 5.1
param(
    [switch]$All,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$NoBackup,
    [string]$DataDir = "",
    [string]$StartupDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $DataDir) {
    $DataDir = Join-Path $env:LOCALAPPDATA "VoiceAssistant"
}
if (-not $StartupDir) {
    $StartupDir = [Environment]::GetFolderPath("Startup")
}
$shortcut = Join-Path $StartupDir "VoiceAssistant.lnk"

function Remove-Target {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "  - $Label`: нет"
        return
    }
    if ($DryRun) {
        Write-Host "  - [dry-run] $Label`: $Path"
        return
    }
    try {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
        Write-Host "  - удалено: $Label"
    } catch {
        Write-Warning "Не удалось удалить $Label ($Path): $($_.Exception.Message)"
    }
}

Write-Host "=== Удаление Voice Assistant ===" -ForegroundColor Cyan
Write-Host "Папка проекта: $root"
Write-Host "Данные и модели: $DataDir"
if ($All) {
    Write-Host "Режим: ПОЛНОЕ удаление, включая папку проекта." -ForegroundColor Yellow
} else {
    Write-Host "Режим: удаление программы и данных, исходники остаются для переустановки." -ForegroundColor Yellow
}
if ($DryRun) {
    Write-Host "Режим проверки: ничего не будет изменено." -ForegroundColor Yellow
}

if (-not $DryRun -and -not $Force) {
    Write-Host ""
    Write-Warning "Действие необратимо."
    if ($All) {
        $answer = Read-Host "Введите УДАЛИТЬ ВСЁ для полного удаления"
        if ($answer.Trim().ToUpper() -ne "УДАЛИТЬ ВСЁ") {
            Write-Host "Отменено."
            exit 0
        }
    } else {
        $answer = Read-Host "Введите ДА для продолжения"
        if ($answer.Trim().ToLower() -ne "да") {
            Write-Host "Отменено."
            exit 0
        }
    }
}

Write-Host ""
Write-Host "Останавливаю процессы..."
$pythonProcesses = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' or Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*app.main*" }
foreach ($process in $pythonProcesses) {
    if ($DryRun) {
        Write-Host "  - [dry-run] остановить python PID $($process.ProcessId)"
    } else {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "  - остановлен python PID $($process.ProcessId)"
    }
}

$chromeProcesses = Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like "*VoiceAssistant*chrome-profile*" -or
            $_.CommandLine -like "*VoiceAssistant/chrome-profile*"
        )
    }
foreach ($process in $chromeProcesses) {
    if ($DryRun) {
        Write-Host "  - [dry-run] закрыть Chrome ассистента PID $($process.ProcessId)"
    } else {
        $savedEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        taskkill /PID $process.ProcessId /T /F 2>&1 | Out-Null
        $ErrorActionPreference = $savedEap
        Write-Host "  - закрыт Chrome ассистента PID $($process.ProcessId)"
    }
}
if (-not $DryRun -and ($pythonProcesses -or $chromeProcesses)) {
    Start-Sleep -Seconds 2
}

if ($All) {
    Write-Host ""
    Write-Host "Сохраняю установщик для повторной установки..."
    if ($NoBackup) {
        Write-Host "  - пропущено (-NoBackup)"
    } else {
        $downloads = Join-Path $env:USERPROFILE "Downloads"
        if (-not (Test-Path -LiteralPath $downloads)) {
            $downloads = [Environment]::GetFolderPath("MyDocuments")
        }
        $kit = Join-Path $downloads ("VoiceAssistant-Setup-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        if ($DryRun) {
            Write-Host "  - [dry-run] создать установщик: $kit"
        } else {
            try {
                New-Item -ItemType Directory -Force $kit | Out-Null
                $kitItems = @(
                    "app", "bin", ".opencode", "scripts", "tests",
                    "install.cmd", "uninstall.cmd", "import-profile.cmd", "config.json", "config.example.json", "opencode.json",
                    "requirements.txt", "requirements-whisper.txt",
                    "AGENTS.md", "README.md", ".gitignore"
                )
                foreach ($item in $kitItems) {
                    $source = Join-Path $root $item
                    if (Test-Path -LiteralPath $source) {
                        Copy-Item -LiteralPath $source -Destination $kit -Recurse -Force
                    }
                }
                Get-ChildItem -LiteralPath $kit -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
                    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "  - установщик сохранён: $kit" -ForegroundColor Green
            } catch {
                Write-Warning "Не удалось сохранить установщик: $($_.Exception.Message)"
                Write-Warning "Полное удаление отменено, чтобы вы не потеряли install.cmd."
                exit 1
            }
        }
    }
}

Write-Host ""
Write-Host "Удаляю автозапуск..."
Remove-Target -Path $shortcut -Label "ярлык в Startup"
if (-not $DryRun) {
    $savedEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    schtasks /Delete /TN "VoiceAssistant" /F 2>&1 | Out-Null
    $ErrorActionPreference = $savedEap
}

Write-Host ""
Write-Host "Удаляю данные, модели и профиль Chrome..."
Remove-Target -Path $DataDir -Label "папка %LOCALAPPDATA%\VoiceAssistant"

Write-Host ""
Write-Host "Удаляю окружение и временные файлы проекта..."
$projectTargets = @(
    (Join-Path $root ".venv"),
    (Join-Path $root "logs"),
    (Join-Path $root "state"),
    (Join-Path $root "models"),
    (Join-Path $root ".pytest_cache")
)
foreach ($target in $projectTargets) {
    Remove-Target -Path $target -Label $target
}
$cacheDirs = @(
    (Join-Path $root "__pycache__"),
    (Join-Path $root "app\__pycache__"),
    (Join-Path $root "tests\__pycache__"),
    (Join-Path $root "scripts\__pycache__")
) | Where-Object { Test-Path -LiteralPath $_ }
foreach ($cache in $cacheDirs) {
    Remove-Target -Path $cache -Label $cache
}

if ($All) {
    Write-Host ""
    Write-Host "Удаляю папку проекта..."
    if ($DryRun) {
        Write-Host "  - [dry-run] $root"
    } else {
        Set-Location $env:TEMP
        # Нельзя удалять папку, пока из неё выполняется .cmd: Windows потеряет
        # текущий batch-файл. Поэтому удаление делает отдельный процесс.
        $cleanupPath = Join-Path $env:TEMP ("va-cleanup-{0}.ps1" -f [guid]::NewGuid())
        $cleanupCode = @'
param([string]$Target)
Start-Sleep -Seconds 2
Remove-Item -LiteralPath $Target -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
'@
        Set-Content -LiteralPath $cleanupPath -Value $cleanupCode -Encoding UTF8
        $argLine = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Target "{1}"' -f $cleanupPath, $root
        Start-Process -FilePath "powershell.exe" -ArgumentList $argLine -WindowStyle Hidden
        Write-Host "  - удаление запущено в фоне: $root"
    }
} else {
    Write-Host ""
    Write-Host "Исходники оставлены: $root" -ForegroundColor Green
    Write-Host "Переустановка: .\install.cmd" -ForegroundColor Green
}

Write-Host ""
if ($DryRun) {
    Write-Host "=== Проверка завершена, ничего не удалено ===" -ForegroundColor Green
} else {
    Write-Host "=== Voice Assistant удалён ===" -ForegroundColor Green
}
