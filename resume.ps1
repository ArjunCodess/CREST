param(
    [string]$DataDirectory,
    [int]$Epochs = 1,
    [int]$Rounds = 80
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $root = $PSScriptRoot
    if (-not $DataDirectory) {
        if ($env:TFB_DATA_DIR) { $DataDirectory = $env:TFB_DATA_DIR }
        else { $DataDirectory = Join-Path $root 'data' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $DataDirectory 'download_manifest.json'))) {
        if (-not $env:KAGGLE_API_TOKEN) {
            throw 'A new download requires KAGGLE_API_TOKEN in this process. Existing verified data requires no credential.'
        }
        python scripts/download_data.py --essential --destination $DataDirectory
        if ($LASTEXITCODE) { throw 'Data download failed; see the download log.' }
    }
    & ./run.ps1 -ReleaseRoot (Join-Path $DataDirectory 'kaggle_public') -Epochs $Epochs -Rounds $Rounds
    if ($LASTEXITCODE) { throw 'Training or prediction did not finish. See artifacts/official.' }
} finally { Pop-Location }
