param(
    [string]$ReleaseRoot,
    [string]$Output,
    [int]$Epochs = 1,
    [int]$Rounds = 80
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $root = $PSScriptRoot
    if (-not $ReleaseRoot) {
        if ($env:TFB_RELEASE_ROOT) { $ReleaseRoot = $env:TFB_RELEASE_ROOT }
        else { $ReleaseRoot = Join-Path $root 'data\kaggle_public' }
    }
    if (-not $Output) {
        if ($env:TFB_OFFICIAL_DIR) { $Output = $env:TFB_OFFICIAL_DIR }
        else { $Output = Join-Path $root 'artifacts\official' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ReleaseRoot 'corridors'))) {
        throw 'Official release is not downloaded. Use scripts/download_data.py with the secure Kaggle credential file first.'
    }
    if (Test-Path -LiteralPath (Join-Path $ReleaseRoot 'FIXTURE_ONLY.json')) {
        throw 'run.ps1 refuses fixture data. Use the explicit fixture commands in README.md.'
    }
    python -m src.pipeline inspect --release-root $ReleaseRoot --output $Output
    if ($LASTEXITCODE) { throw 'Data inspection failed' }
    python -m src.pipeline train --release-root $ReleaseRoot --output $Output --epochs $Epochs --rounds $Rounds --full-train --resume
    if ($LASTEXITCODE) { throw 'Training failed' }
    python scripts/finish_run.py --release-root $ReleaseRoot --output $Output
    if ($LASTEXITCODE) { throw 'Completion failed. Inspect completion_status.json and stage logs in the output folder.' }
    Write-Host "Submission and results are in $Output; the compiled manuscript is in paper/main.pdf."
} finally { Pop-Location }
