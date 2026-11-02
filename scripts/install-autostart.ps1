param(
    [switch]$Task
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"

if (-not (Test-Path $pythonw)) {
    throw "Окружение не найдено. Сначала запустите scripts\setup.ps1"
}

$startup = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startup "VoiceAssistant.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "-m app.main"
$shortcut.WorkingDirectory = $root
$shortcut.WindowStyle = 7
$shortcut.Description = "Голосовой ассистент Voice Assistant"
$shortcut.Save()

Write-Host "Автозапуск добавлен: $shortcutPath" -ForegroundColor Green
Write-Host "Ассистент будет стартовать при входе пользователя в Windows."

if ($Task) {
    Write-Host "Создаю задачу в Планировщике (может потребоваться запуск от администратора)..."
    $action = "`"$pythonw`" -m app.main"
    $savedEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    schtasks /Create /TN "VoiceAssistant" /TR $action /SC ONLOGON /RL LIMITED /F 2>&1 | Out-Null
    $taskCreated = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $savedEap
    if ($taskCreated) {
        Write-Host "Задача VoiceAssistant создана." -ForegroundColor Green
    } else {
        Write-Warning "Не удалось создать задачу. Ярлык в автозагрузке всё равно работает."
    }
}
