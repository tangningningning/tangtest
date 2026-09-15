param(
    [switch]$IncludeEpisodePool
)

$ErrorActionPreference = "Stop"

# ============================================================
# Paths
# ============================================================

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$DistDir = Join-Path $RepoRoot "dist_server"
$StageDir = Join-Path $DistDir "stage_$Timestamp"

$PackageName = "adaptive_cache_mec_server_$Timestamp.zip"
$PackagePath = Join-Path $DistDir $PackageName

$EpisodePoolRel = "runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911"
$EpisodePoolSrc = Join-Path $RepoRoot $EpisodePoolRel
$EpisodeSeedStart = 20001
$EpisodeSeedEnd = 20300

Write-Host "============================================================"
Write-Host "Adaptive Cache MEC - Server Packaging"
Write-Host "============================================================"
Write-Host "Repo      : $RepoRoot"
Write-Host "Output    : $PackagePath"
Write-Host "Pool      : $IncludeEpisodePool"
Write-Host ""

# ============================================================
# Pre-checks
# ============================================================

$RequiredPaths = @(
    "src",
    "configs",
    "scripts",
    "configs\proposed\system_config_ppo_v10_branchwise_cache_uav.toml",
    "configs\lr_sweep\v10_branchwise_actor_lr_6e4_kl.toml",
    "configs\lr_sweep\v10_branchwise_actor_lr_1e3_kl.toml",
    "scripts\run_v10_branchwise_three_lr.sh",
    "scripts\run_v10_branchwise_three_lr.ps1"
)

foreach ($RelPath in $RequiredPaths) {
    $FullPath = Join-Path $RepoRoot $RelPath

    if (-not (Test-Path $FullPath)) {
        throw "Required path missing: $FullPath"
    }
}

if ($IncludeEpisodePool -and -not (Test-Path $EpisodePoolSrc)) {
    throw "Episode pool not found: $EpisodePoolSrc"
}
if ($IncludeEpisodePool) {
    $EpisodeDirectories = @(
        Get-ChildItem -LiteralPath $EpisodePoolSrc -Directory -Filter "episode_*" |
            Sort-Object Name
    )
    $ExpectedEpisodeNames = @(
        $EpisodeSeedStart..$EpisodeSeedEnd |
            ForEach-Object { "episode_{0}" -f $_ }
    )
    $ActualEpisodeNames = @($EpisodeDirectories | ForEach-Object Name)
    $PoolDifference = Compare-Object `
        -ReferenceObject $ExpectedEpisodeNames `
        -DifferenceObject $ActualEpisodeNames
    if ($EpisodeDirectories.Count -ne 300 -or $null -ne $PoolDifference) {
        throw "Episode pool must contain exactly episode_20001 through episode_20300"
    }
}

# ============================================================
# Clean staging area
# ============================================================

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}

New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

# ============================================================
# Copy repository
# ============================================================

Write-Host "[1/5] Copying repository..."

$ExcludeDirs = @(
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "runs",
    "dist_server",
    "build",
    "dist"
)

$ExcludeFiles = @(
    "*.pyc",
    "*.pyo",
    "*.log",
    "*.tmp",
    "*.zip",
    "*.7z",
    "*.png",
    "*.pt",
    "*.pth",
    "*.ckpt"
)

$RoboArgs = @(
    $RepoRoot,
    $StageDir,
    "/E",
    "/R:1",
    "/W:1",
    "/NFL",
    "/NDL",
    "/NJH",
    "/NJS",
    "/NP"
)

foreach ($Dir in $ExcludeDirs) {
    $RoboArgs += "/XD"
    $RoboArgs += (Join-Path $RepoRoot $Dir)
}

foreach ($Pattern in $ExcludeFiles) {
    $RoboArgs += "/XF"
    $RoboArgs += $Pattern
}

& robocopy @RoboArgs | Out-Null

# Robocopy exit codes 0-7 are success states.
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

# Robocopy can leave empty excluded directories in the destination. Remove
# only those directories from the temporary stage so caches and IDE metadata
# cannot enter the archive; the source checkout is never touched.
$StageExcludedDirNames = @(
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "runs",
    "dist_server",
    "build",
    "dist"
)

$StageExcludedDirs = @(
    Get-ChildItem -LiteralPath $StageDir -Directory -Recurse -Force |
        Where-Object { $_.Name -in $StageExcludedDirNames } |
        Sort-Object { $_.FullName.Length } -Descending
)

foreach ($StageExcludedDir in $StageExcludedDirs) {
    Remove-Item -LiteralPath $StageExcludedDir.FullName -Recurse -Force
}

# ============================================================
# Copy exact episode pool if requested
# ============================================================

if ($IncludeEpisodePool) {

    Write-Host "[2/5] Copying formal 300-episode pool..."

    $EpisodePoolDst = Join-Path $StageDir $EpisodePoolRel
    New-Item -ItemType Directory -Force -Path $EpisodePoolDst | Out-Null

    & robocopy `
        $EpisodePoolSrc `
        $EpisodePoolDst `
        /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null

    if ($LASTEXITCODE -gt 7) {
        throw "Episode-pool robocopy failed with exit code $LASTEXITCODE"
    }

} else {
    Write-Host "[2/5] Episode pool not included."
}

# ============================================================
# Build information
# ============================================================

Write-Host "[3/5] Writing build metadata..."

$GitCommit = "UNKNOWN"
$GitStatus = "UNKNOWN"

try {
    Push-Location $RepoRoot

    $GitCommit = (git rev-parse HEAD 2>$null).Trim()
    $GitStatus = (git status --short 2>$null) -join "`r`n"

    Pop-Location
} catch {
    try { Pop-Location } catch {}
}

$GitBranch = "UNKNOWN"
try {
    Push-Location $RepoRoot
    $GitBranch = (git branch --show-current 2>$null).Trim()
    Pop-Location
} catch {
    try { Pop-Location } catch {}
}

$GitStatusValue = if ([string]::IsNullOrWhiteSpace($GitStatus)) { "CLEAN" } else { $GitStatus }

$BuildInfo = @"
PACKAGE_TIMESTAMP=$Timestamp
GIT_BRANCH=$GitBranch
GIT_COMMIT=$GitCommit
GIT_STATUS=$GitStatusValue
EPISODE_POOL_INCLUDED=$IncludeEpisodePool

V10_B_CONFIG_5E4=configs/proposed/system_config_ppo_v10_branchwise_cache_uav.toml
V10_B_CONFIG_6E4=configs/lr_sweep/v10_branchwise_actor_lr_6e4_kl.toml
V10_B_CONFIG_1E3=configs/lr_sweep/v10_branchwise_actor_lr_1e3_kl.toml

EPISODE_POOL=runs/ppo_v6_popart_actor1e4_lazy_episode_pool_20260911
EPISODE_COUNT=300
EPISODE_SEED_START=$EpisodeSeedStart
EPISODE_SEED_END=$EpisodeSeedEnd
TARGET_EPOCHS_PER_RUN=300
LR_5E4=0.0005
LR_6E4=0.0006
LR_1E3=0.001
CRITIC_LR=0.0003
TARGET_KL=0.10

"@

Set-Content `
    -Path (Join-Path $StageDir "SERVER_BUILD_INFO.txt") `
    -Value $BuildInfo `
    -Encoding UTF8

# ============================================================
# Server instructions
# ============================================================

$ServerReadme = @'
Adaptive Cache MEC - Linux V10-B Server Run
============================================

The server already provides the required container image. Do NOT run
`conda activate`, create a virtual environment, install packages, or install
system packages from this deployment bundle.

After extracting the ZIP:

  cd <project_root>
  export PYTHONPATH="$PWD/src"

Verify the container runtime:

  python -c "import sys, torch, ppo_simple; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(ppo_simple.__file__)"

Start the three formal runs strictly serially:

  bash scripts/run_v10_branchwise_three_lr.sh

The order is 5e-4 -> 6e-4 -> 1e-3. Each run uses 300 epochs, critic LR
3e-4, target KL 0.10, and the included episode pool episode_20001 through
episode_20300. The script refuses to overwrite a non-empty run directory and
stops before the next LR if any run fails. It does not use --resume and does
not generate missing episodes.

Never use final unseen seeds 40001 through 40020 for this training package.
'@

Set-Content `
    -Path (Join-Path $StageDir "SERVER_RUN_README.txt") `
    -Value $ServerReadme `
    -Encoding UTF8

# ============================================================
# Compress
# ============================================================

Write-Host "[4/5] Creating archive..."

if (Test-Path $PackagePath) {
    Remove-Item -Force $PackagePath
}

Compress-Archive `
    -Path (Join-Path $StageDir "*") `
    -DestinationPath $PackagePath `
    -CompressionLevel Optimal

# ============================================================
# SHA256 + cleanup
# ============================================================

Write-Host "[5/5] Calculating SHA256..."

$Hash = Get-FileHash -Algorithm SHA256 $PackagePath

$HashFile = "$PackagePath.sha256"

"$($Hash.Hash)  $PackageName" |
    Set-Content -Path $HashFile -Encoding ASCII

Remove-Item -Recurse -Force $StageDir

$SizeMB = [Math]::Round(
    (Get-Item $PackagePath).Length / 1MB,
    2
)

Write-Host ""
Write-Host "============================================================"
Write-Host "PACKAGE_COMPLETE = YES"
Write-Host "PACKAGE = $PackagePath"
Write-Host "SIZE_MB = $SizeMB"
Write-Host "SHA256 = $($Hash.Hash)"
Write-Host "HASH_FILE = $HashFile"
Write-Host "EPISODE_POOL_INCLUDED = $IncludeEpisodePool"
Write-Host "TEMP_STAGE_REMOVED = YES"
Write-Host "============================================================"
