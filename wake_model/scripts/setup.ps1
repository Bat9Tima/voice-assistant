# Восстановление окружения этапа 0: учебный venv, Piper и русские голоса.
#
#   powershell -ExecutionPolicy Bypass -File wake_model\scripts\setup.ps1
#
# Piper и голоса кладутся в ASCII-каталог: piper.exe 2023.11.14-2 падает
# на кириллических путях, а путь проекта содержит «Проект по умолчанию».

param(
    [string]$Python = "",
    [string]$ToolsDir = (Join-Path $HOME "Documents\VoiceAssistantTools"),
    [switch]$SkipPiper,
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$piperVersion = "2023.11.14-2"
$voices = @("denis", "dmitri", "irina", "ruslan")

function Get-File([string]$Url, [string]$OutFile) {
    $dir = Split-Path -Parent $OutFile
    New-Item -ItemType Directory -Force $dir | Out-Null
    curl.exe -L --fail --silent --show-error -o $OutFile $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Не скачалось: $Url"
    }
}

if (-not $SkipPiper) {
    New-Item -ItemType Directory -Force $ToolsDir | Out-Null
    $piperExe = Join-Path $ToolsDir "piper\piper.exe"
    if (-not (Test-Path $piperExe)) {
        Write-Host "Скачиваю Piper $piperVersion..."
        $zip = Join-Path $ToolsDir "piper_windows_amd64.zip"
        Get-File "https://github.com/rhasspy/piper/releases/download/$piperVersion/piper_windows_amd64.zip" $zip
        Expand-Archive -Path $zip -DestinationPath $ToolsDir -Force
        if (-not (Test-Path $piperExe)) {
            throw "Piper распаковался не туда: нет $piperExe"
        }
    }
    Write-Host "Piper: OK ($piperExe)"

    $base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU"
    foreach ($voice in $voices) {
        $dir = Join-Path $ToolsDir "voices\ru_RU-$voice-medium"
        $onnx = Join-Path $dir "ru_RU-$voice-medium.onnx"
        $json = "$onnx.json"
        if (-not (Test-Path $onnx)) {
            Write-Host "Скачиваю голос $voice (~60 МБ)..."
            Get-File "$base/$voice/medium/ru_RU-$voice-medium.onnx" $onnx
        }
        if (-not (Test-Path $json)) {
            Get-File "$base/$voice/medium/ru_RU-$voice-medium.onnx.json" $json
        }
        $size = (Get-Item $onnx).Length / 1MB
        if ($size -lt 50) {
            throw "Голос $voice скачался не полностью: $([math]::Round($size, 1)) МБ"
        }
        Write-Host "Голос ${voice}: OK ($([math]::Round($size, 1)) МБ)"
    }
}

if (-not $SkipVenv) {
    $venv = Join-Path $root ".venv"
    if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
        Write-Host "Создаю учебный venv..."
        if ($Python) {
            & $Python -m venv $venv
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -m venv $venv
        } else {
            & python -m venv $venv
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось создать venv. Укажите -Python путь к python 3.12"
        }
    }
    $py = Join-Path $venv "Scripts\python.exe"
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install -r (Join-Path $PSScriptRoot "..\requirements-train.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Не установились зависимости обучения"
    }
    & $py -c "import torch, openwakeword; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    Write-Host "Учебный venv: OK ($venv)"
}

Write-Host ""
Write-Host "Дальше:"
Write-Host "  python wake_model\scripts\synthesize_silero.py --count 20"
Write-Host "  python wake_model\scripts\check_qc.py --dir wake_model\data\silero_probe"
Write-Host "  python wake_model\scripts\synthesize_supertonic.py --count 30 --clean"
Write-Host "  python wake_model\scripts\check_qc.py --dir wake_model\data\supertonic_probe"
