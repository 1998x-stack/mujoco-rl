# Windows PowerShell 5.1+ entry point. Example: .\run.ps1 -Full -Resume
[CmdletBinding()]
param(
    [switch] $Full,
    [ValidateRange(1, 2147483647)] [int] $Steps = 2048,
    [switch] $Resume,
    [switch] $Fresh,
    [switch] $NoVideo,
    [switch] $NoOpen,
    [switch] $Headless,
    [switch] $Help
)
$ErrorActionPreference = 'Stop'
if ($Help) {
    Write-Host @'
Usage: .\run.ps1 [-Full | -Steps N] [-Resume | -Fresh] [-NoVideo] [-NoOpen] [-Headless]
Default: 2048-step smoke test; evaluation; MP4 when a desktop is available.
-Full       Train 1,000,000 additional steps (takes substantially longer).
-Steps N    Train N additional steps.
-Resume     Continue latest saved SAC model and replay buffer.
-Fresh      Archive previous training outputs and start a new experiment.
-NoVideo    Skip MP4 generation.
-NoOpen     Save video without opening player.
-Headless   Skip rendering and opening; ideal for remote sessions.
Alternatively: run.cmd [same PowerShell flags]. No WSL necessary.
'@
    exit 0
}
if ($env:OS -ne 'Windows_NT') { throw 'run.ps1 is for native Windows. Use ./run.sh on Linux/macOS.' }
if ($Full -and $PSBoundParameters.ContainsKey('Steps')) { throw 'Choose -Full or -Steps, not both.' }
if ($Full) { $Steps = 1000000 }
if ($Resume -and $Fresh) { throw 'Choose -Resume or -Fresh, not both.' }
if ($Headless) { $NoVideo = $true; $NoOpen = $true }
$Root = $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = if ($env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS } else { '1' }
function Run-Native([string] $Executable, [string[]] $Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}
Write-Host '[1/5] Set up Python + packages.'
& (Join-Path $Root 'setup.ps1')
if (-not (Test-Path $Python)) { throw "Missing Python: $Python" }
Write-Host '[2/5] Check MuJoCo environment.'
Run-Native $Python @('-m', 'src.check_env', '--steps', '32')
Write-Host "[3/5] Train SAC for $Steps additional steps."
$TrainArgs = @('-m', 'src.train', '--steps', [string]$Steps)
if ($Resume) { $TrainArgs += '--resume' }
if ($Fresh) { $TrainArgs += '--fresh' }
Run-Native $Python $TrainArgs
Write-Host '[4/5] Evaluate and plot.'
Run-Native $Python @('-m', 'src.evaluate', '--episodes', '3')
& $Python -m src.plot
if ($LASTEXITCODE -ne 0) { Write-Warning 'Reward plot unavailable (no completed episodes).' }
if (-not $NoVideo) {
    Write-Host '[5/5] Render MP4.'
    try {
        Run-Native $Python @('-m', 'src.record', '--frames', '250')
    } catch {
        Write-Warning "Training/evaluation succeeded; MP4 rendering failed. Use -NoVideo in headless sessions. Details: $_"
        $NoVideo = $true
    }
    $Video = Join-Path $Root 'videos\ant_demo.mp4'
    if (-not $NoVideo -and -not $NoOpen) {
        try { Start-Process -FilePath $Video } catch { Write-Warning "MP4 saved at $Video, but video player could not open: $_" }
    }
} else {
    Write-Host '[5/5] Video skipped.'
}
Write-Host 'SUCCESS: models/latest.json (atomic snapshot pointer) and logs/evaluation.json'
if (-not $NoVideo) { Write-Host 'SUCCESS: videos/ant_demo.mp4' }
Write-Host 'Train more: .\run.ps1 -Full -Resume'
Write-Host 'Watch live: .\watch.ps1'
