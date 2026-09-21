$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { throw 'Run .\run.ps1 first.' }
& $Python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed: $LASTEXITCODE" }
& $Python -m src.check_env --steps 20
if ($LASTEXITCODE -ne 0) { throw "MuJoCo smoke test failed: $LASTEXITCODE" }
