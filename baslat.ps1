#Requires -Version 5.1
<#
.SYNOPSIS
  Eksik ortami tamamlar ve Osmanli HTR uygulamasini acar.
.DESCRIPTION
  Python 3.11 sanal ortami, paketler ve uygulama modelleri yoksa kurulur.
  NVIDIA karti varsa okuma CUDA uzerinde acilir. Kurulu parcalar atlanir.
#>
param(
    [switch]$Denetle,
    [switch]$Hazirla,
    [switch]$Api,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:GRADIO_ANALYTICS_ENABLED = "False"
$env:PYTHONPATH = $PSScriptRoot
$env:CONFIG_PATH = Join-Path $PSScriptRoot "config.yaml"

function Stop-Launch([string]$Message) {
    Write-Host $Message
    exit 1
}

function Find-Python311 {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if (-not $launcher) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
    }
    if ($launcher) {
        $out = & $launcher.Source -3.11 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $exe = & $launcher.Source -3.11 -c "import sys; print(sys.executable)"
            if ($LASTEXITCODE -eq 0 -and $exe) {
                return ($exe | Select-Object -Last 1).ToString().Trim()
            }
        }
    }
    $candidates = @(
        (Join-Path $env:LocalAppData "Programs\Python\Python311\python.exe"),
        "C:\Program Files\Python311\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            & $candidate -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)"
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        }
    }
    return $null
}

function Read-DotEnv([string]$Name, [string]$Default) {
    foreach ($fileName in @(".env", ".env.example")) {
        $path = Join-Path $PSScriptRoot $fileName
        if (-not (Test-Path -LiteralPath $path)) {
            continue
        }
        foreach ($line in Get-Content -LiteralPath $path -Encoding UTF8) {
            if ($line -match "^$Name=(.*)$") {
                $value = $Matches[1].Trim()
                if ($value) {
                    return $value
                }
            }
        }
    }
    return $Default
}

function Test-PortBusy([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $wait = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $wait.AsyncWaitHandle.WaitOne(400)) {
            return $false
        }
        $client.EndConnect($wait)
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    if ($Denetle) {
        Stop-Launch "Eksik: .venv yok. Python 3.11 kurup baslat.cmd calistirin."
    }
    $base = Find-Python311
    if (-not $base) {
        Stop-Launch "Python 3.11 (64-bit) yok. https://www.python.org/downloads/release/python-3119/ adresinden kurun."
    }
    Write-Host "Sanal ortam olusturuluyor."
    & $base -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        Stop-Launch "Sanal ortam olusturulamadi."
    }
}

& $venvPython -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)"
if ($LASTEXITCODE -ne 0) {
    Stop-Launch ".venv Python 3.11 degil. .venv klasorunu silip baslat.cmd calistirin."
}

$prepare = @(Join-Path $PSScriptRoot "scripts\hazirla.py")
if ($Denetle) {
    $prepare += "--denetle"
}
if ($Cpu) {
    $prepare += "--cpu"
}
& $venvPython @prepare
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
if ($Denetle -or $Hazirla) {
    exit 0
}

$deviceFile = Join-Path $PSScriptRoot ".venv\.device"
if (-not (Test-Path -LiteralPath $deviceFile)) {
    Stop-Launch "Cihaz secilemedi."
}
$device = (Get-Content -LiteralPath $deviceFile -Encoding UTF8 -Raw).Trim()
if ($device -ne "cuda" -and $device -ne "cpu") {
    Stop-Launch "Bilinmeyen cihaz: $device"
}
$env:TARIHHTR_DEVICE = $device
Write-Host "Okuma cihazi: $device"

if ($Api) {
    $port = [int](Read-DotEnv "API_PORT" "8000")
    if (Test-PortBusy $port) {
        Stop-Launch "Port $port dolu."
    }
    Write-Host "API: http://localhost:$port/docs"
    & $venvPython -m uvicorn app.api.main:app --host 0.0.0.0 --port $port
} else {
    $port = [int](Read-DotEnv "GRADIO_PORT" "7860")
    if (Test-PortBusy $port) {
        Stop-Launch "Port $port dolu."
    }
    Write-Host "Arayuz: http://localhost:$port"
    & $venvPython -m app.ui.gradio_app
}
exit $LASTEXITCODE
