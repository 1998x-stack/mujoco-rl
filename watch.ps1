# Native Windows 3D live viewer. For macOS/Linux use watch.sh.
[CmdletBinding()]
param([string] $Model = '', [ValidateRange(1, 1000000)] [int] $Episodes = 3)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Run .\run.ps1 first.' }
$Arguments = @('-m', 'src.play', '--episodes', [string]$Episodes)
if ($Model) { $Arguments += @('--model', $Model) }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Viewer failed with exit code $LASTEXITCODE. See docs/TROUBLESHOOTING.md." }
