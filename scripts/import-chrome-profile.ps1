#Requires -Version 5.1
param(
    [string]$SourceDir = "",
    [string]$SourceProfile = "Default",
    [string]$TargetDir = "",
    [switch]$Force,
    [switch]$NoBackup,
    [switch]$NoStart,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $SourceDir) {
    $SourceDir = Join-Path $env:LOCALAPPDATA "Google\Chrome\User Data"
}
if (-not $TargetDir) {
    $TargetDir = Join-Path $env:LOCALAPPDATA "VoiceAssistant\chrome-profile"
}
$SourceDir = [IO.Path]::GetFullPath($SourceDir)
$TargetDir = [IO.Path]::GetFullPath($TargetDir)
$defaultTarget = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "VoiceAssistant\chrome-profile"))
$isDefaultTarget = ($TargetDir.TrimEnd("\") -eq $defaultTarget.TrimEnd("\"))

$localState = Join-Path $SourceDir "Local State"
$profileSrc = Join-Path $SourceDir $SourceProfile
$profileDst = Join-Path $TargetDir "Default"

if (-not (Test-Path -LiteralPath $localState)) {
    throw "Не найден файл Local State: $localState"
}
if (-not (Test-Path -LiteralPath $profileSrc)) {
    throw "Не найден профиль Chrome: $profileSrc"
}
if (-not (Test-Path -LiteralPath (Join-Path $profileSrc "Preferences"))) {
    throw "В профиле нет Preferences: $profileSrc"
}
if ($SourceDir.TrimEnd("\") -eq $TargetDir.TrimEnd("\")) {
    throw "Исходный и целевой профиль совпадают. Укажите другой TargetDir."
}

$profileLabel = $SourceProfile
$profileUser = ""
try {
    $state = Get-Content -LiteralPath $localState -Raw -Encoding UTF8 | ConvertFrom-Json
    $entry = $state.profile.info_cache.PSObject.Properties[$SourceProfile]
    if ($entry -and $entry.Value) {
        if ($entry.Value.name) { $profileLabel = [string]$entry.Value.name }
        if ($entry.Value.user_name) { $profileUser = [string]$entry.Value.user_name }
    }
} catch {
    Write-Warning "Не удалось прочитать имя профиля: $($_.Exception.Message)"
}

Write-Host "=== Импорт профиля Chrome в ассистента ===" -ForegroundColor Cyan
Write-Host "Источник:  $profileSrc"
Write-Host "Профиль:   $profileLabel $(if ($profileUser) { "($profileUser)" })"
Write-Host "Назначение: $TargetDir"

$chrome = @(Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" -ErrorAction SilentlyContinue)
$assistantChrome = @($chrome | Where-Object { $_.CommandLine -and $_.CommandLine -like "*$TargetDir*" })
$otherChrome = @($chrome | Where-Object { -not ($_.CommandLine -and $_.CommandLine -like "*$TargetDir*") })
$python = @(Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' or Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*app.main*" })
$wasRunning = ($python.Count -gt 0 -or $assistantChrome.Count -gt 0)

if ($otherChrome.Count -gt 0 -and -not $Force) {
    Write-Host ""
    Write-Warning "Обычный Chrome ещё запущен (PID: $($otherChrome.ProcessId -join ', '))."
    Write-Warning "Закройте все окна Chrome, включая фоновые процессы, и повторите."
    Write-Warning "Скопировать профиль на работающем Chrome можно с -Force, но часть данных может быть повреждена."
    exit 1
}

if ($DryRun) {
    Write-Host ""
    Write-Host "Режим проверки: ничего не будет изменено." -ForegroundColor Yellow
    Write-Host "  - ассистент: $(if ($wasRunning) { 'будет остановлен и запущен заново' } else { 'не запущен' })"
    Write-Host "  - обычный Chrome: $(if ($otherChrome.Count) { 'ЗАПУЩЕН — закройте его' } else { 'закрыт' })"
    Write-Host "  - резервная копия: $(if ($NoBackup) { 'не создавать' } else { 'создать рядом с профилем ассистента' })"
    Write-Host "  - будут скопированы cookies, сессии, пароли, закладки и расширения"
    Write-Host "  - кэш и история не копируются"
    Write-Host ""
    Write-Host "Для реального импорта запустите без -DryRun." -ForegroundColor Green
    exit 0
}

if (-not $Force) {
    Write-Host ""
    $answer = Read-Host "Скопировать профиль в браузер ассистента? Введите ДА"
    if ($answer.Trim().ToLower() -ne "да") {
        Write-Host "Отменено."
        exit 0
    }
}

if ($isDefaultTarget -and $wasRunning) {
    Write-Host ""
    Write-Host "Останавливаю ассистента и его Chrome..."
    foreach ($process in $python) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    foreach ($process in $assistantChrome) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

if (Test-Path -LiteralPath $TargetDir) {
    if ($NoBackup) {
        Write-Host "Удаляю старый профиль ассистента (-NoBackup)..."
        Remove-Item -LiteralPath $TargetDir -Recurse -Force
    } else {
        $backup = "$TargetDir-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        $suffix = 1
        while (Test-Path -LiteralPath $backup) {
            $backup = "$TargetDir-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')-$suffix"
            $suffix++
        }
        Write-Host "Сохраняю старый профиль: $backup"
        Move-Item -LiteralPath $TargetDir -Destination $backup
    }
}

New-Item -ItemType Directory -Force $TargetDir | Out-Null
New-Item -ItemType Directory -Force $profileDst | Out-Null

Write-Host ""
Write-Host "Копирую Local State и профиль..."
Copy-Item -LiteralPath $localState -Destination (Join-Path $TargetDir "Local State") -Force

$excludeDirs = @(
    "Cache", "Code Cache", "GPUCache", "ShaderCache", "GrShaderCache",
    "DawnGraphiteCache", "DawnWebGPUCache", "Media Cache", "Crashpad",
    "BrowserMetrics", "CacheStorage", "ScriptCache",
    "JumpListIconsMostVisited", "JumpListIconsRecentClosed"
) | ForEach-Object { Join-Path $profileSrc $_ }

$excludeFiles = @(
    "LOCK", "LOG", "LOG.old", "SingletonLock", "SingletonCookie", "SingletonSocket",
    "History", "History-journal", "Favicons", "Favicons-journal",
    "Top Sites", "Top Sites-journal", "Shortcuts", "Shortcuts-journal"
)

$robocopyArgs = @(
    $profileSrc, $profileDst, "/E", "/R:1", "/W:1",
    "/NFL", "/NDL", "/NJH", "/NJS", "/NP"
)
if ($excludeDirs.Count -gt 0) {
    $robocopyArgs += "/XD"
    $robocopyArgs += $excludeDirs
}
if ($excludeFiles.Count -gt 0) {
    $robocopyArgs += "/XF"
    $robocopyArgs += $excludeFiles
}

& robocopy @robocopyArgs | Out-Null
$robocopyCode = $LASTEXITCODE
if ($robocopyCode -ge 8) {
    throw "Robocopy не смог скопировать профиль (код $robocopyCode)."
}

if (-not (Test-Path -LiteralPath (Join-Path $profileDst "Preferences"))) {
    throw "Профиль скопирован не полностью: нет Preferences."
}

Write-Host "Готово: профиль скопирован." -ForegroundColor Green
if ($isDefaultTarget) {
    Write-Host "Chrome ассистента будет использовать его при следующем запуске."
}
if ($isDefaultTarget -and $wasRunning -and -not $NoStart) {
    Write-Host "Запускаю ассистента..."
    & (Join-Path $PSScriptRoot "start.ps1")
} else {
    Write-Host "Запуск: .\scripts\start.cmd"
}
