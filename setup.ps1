# Project-local Windows PowerShell setup. No WSL, Administrator access, or system Python required.
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root
if ($env:OS -ne 'Windows_NT') { throw 'setup.ps1 is for native Windows; use setup.sh on macOS/Linux.' }
$Tools = Join-Path $Root '.tools\bin'
$Uv = Join-Path $Tools 'uv.exe'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $Tools | Out-Null
function Run-Native([string] $Executable, [string[]] $Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}
if (-not (Test-Path $Uv)) {
    $SystemUv = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($SystemUv) {
        $Uv = $SystemUv.Source
    } else {
        Write-Host '[setup] Downloading official uv Windows installer (internet required).'
        $Installer = Join-Path $env:TEMP ('uv-install-' + [guid]::NewGuid().ToString('N') + '.ps1')
        $OldTarget = $env:UV_UNMANAGED_INSTALL
        try {
            # Older Windows PowerShell installations may require an explicit TLS 1.2 setting.
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile $Installer -UseBasicParsing
            $env:UV_UNMANAGED_INSTALL = $Tools
            & $Installer
            if ($LASTEXITCODE -ne 0) { throw "uv installer returned $LASTEXITCODE" }
        } finally {
            $env:UV_UNMANAGED_INSTALL = $OldTarget
            Remove-Item -LiteralPath $Installer -ErrorAction SilentlyContinue
        }
        if (-not (Test-Path $Uv)) { throw "uv installer did not create $Uv" }
    }
}
Run-Native $Uv @('--version')
if (-not (Test-Path $Python)) {
    Write-Host '[setup] Creating Python 3.11 .venv (uv will download Python if needed).'
    Run-Native $Uv @('venv', '--python', '3.11', '--managed-python', (Join-Path $Root '.venv'))
}
Run-Native $Python @('-c', 'import sys; assert sys.version_info[:2] == (3, 11), "Expected Python 3.11"')
Run-Native $Uv @('pip', 'install', '--python', $Python, '-r', (Join-Path $Root 'requirements.txt'))
Write-Host "[setup] Ready: $Python"
