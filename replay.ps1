param(
    [string]$ReleaseRoot,
    [string]$Output,
    [ValidateSet('unet', 'cat')][string]$Variant = 'unet',
    [ValidateSet('CPU', 'GPU')][string]$Device = 'GPU',
    [string]$Reference,
    [string]$VerifiedOwn,
    [switch]$CheckOnly
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
        if ($env:TFB_V8_REPLAY_DIR) { $Output = $env:TFB_V8_REPLAY_DIR }
        else { $Output = Join-Path $root 'artifacts\v8_replay' }
    }
    python scripts/validate_v8_prereqs.py --package-root $root --release-root $ReleaseRoot --variant $Variant
    if ($LASTEXITCODE) { throw 'Prerequisite check failed; install requirements-v8.lock.txt and unpack checkpoints.' }
    if ($CheckOnly) {
        Write-Host "v8 replay prerequisites satisfied for release $ReleaseRoot"
        return
    }
    if (Test-Path -LiteralPath $Output) {
        throw "Output directory already exists ($Output). Use a fresh path for replay."
    }
    $cmd = @(
        'scripts/reproduce_v8_submission.py',
        '--release-root', $ReleaseRoot,
        '--package-root', '.',
        '--output', $Output,
        '--variant', $Variant,
        '--device', $Device
    )
    if ($Reference) { $cmd += @('--reference', $Reference) }
    if ($VerifiedOwn) { $cmd += @('--verified-own', $VerifiedOwn) }
    python @cmd
    if ($LASTEXITCODE) { throw 'v8 checkpoint replay failed' }
    Write-Host "Replay finished; see $Output\reproduction_audit.json"
} finally { Pop-Location }
