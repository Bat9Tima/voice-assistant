param(
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$startup = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startup "VoiceAssistant.lnk"

if (Test-Path $shortcutPath) {
    Remove-Item $shortcutPath -Force
    Write-Host "Ярлык автозапуска удалён."
} else {
    Write-Host "Ярлык автозапуска не найден."
}

$savedEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
schtasks /Delete /TN "VoiceAssistant" /F 2>&1 | Out-Null
$taskRemoved = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $savedEap
if ($taskRemoved) {
    Write-Host "Задача планировщика удалена."
}

if ($Stop) {
    $processes = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*app.main*" }
    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId -Force
        Write-Host "Остановлен процесс $($process.ProcessId)."
    }
}
