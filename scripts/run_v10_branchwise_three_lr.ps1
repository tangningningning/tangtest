$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "configs\system_config.toml"))) {
    throw "Not an adaptive_cache_mec repository root: $repoRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "src\ppo_simple\trainer.py"))) {
    throw "PPO trainer is missing from repository root: $repoRoot"
}
Set-Location -LiteralPath $repoRoot

$pythonCommand = Get-Command python -CommandType Application -ErrorAction Stop
$pythonPath = [IO.Path]::GetFullPath($pythonCommand.Path)
if ($pythonPath -notmatch "(?i)[\\/]envs[\\/]adaptive_cache_mec[\\/]python\.exe$") {
    throw "python is not from the adaptive_cache_mec Conda environment: $pythonPath"
}
if ($env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne "adaptive_cache_mec") {
    throw "CONDA_DEFAULT_ENV is not adaptive_cache_mec: $env:CONDA_DEFAULT_ENV"
}

$env:PYTHONPATH = Join-Path $repoRoot "src"
$episodePoolRelative = "runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911"
$episodePoolPath = Join-Path $repoRoot $episodePoolRelative
if (-not (Test-Path -LiteralPath $episodePoolPath -PathType Container)) {
    throw "Episode pool is missing: $episodePoolPath"
}
$episodeDirectories = @(
    Get-ChildItem -LiteralPath $episodePoolPath -Directory -Filter "episode_*" |
        Sort-Object Name
)
$expectedEpisodeNames = @(20001..20300 | ForEach-Object { "episode_{0:D5}" -f $_ })
$actualEpisodeNames = @($episodeDirectories | ForEach-Object Name)
$poolDifference = Compare-Object -ReferenceObject $expectedEpisodeNames -DifferenceObject $actualEpisodeNames
if ($episodeDirectories.Count -ne 300 -or $null -ne $poolDifference) {
    throw "Episode pool must contain exactly episode_20001 through episode_20300"
}

$runs = @(
    [ordered]@{
        Label = "5e-4"
        Config = "configs\proposed\system_config_ppo_v10_branchwise_cache_uav.toml"
        RunDir = "runs\ppo_v10_branchwise_cache_uav_5e4_kl_formal"
    }
    [ordered]@{
        Label = "6e-4"
        Config = "configs\lr_sweep\v10_branchwise_actor_lr_6e4_kl.toml"
        RunDir = "runs\ppo_v10_branchwise_cache_uav_6e4_kl_formal"
    }
    [ordered]@{
        Label = "1e-3"
        Config = "configs\lr_sweep\v10_branchwise_actor_lr_1e3_kl.toml"
        RunDir = "runs\ppo_v10_branchwise_cache_uav_1e3_kl_formal"
    }
)

function Assert-RunDirectoryAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $runPath = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $runPath)) {
        return
    }
    $items = @(Get-ChildItem -LiteralPath $runPath -Force)
    if ($items.Count -ne 0) {
        throw "Refusing to overwrite non-empty run directory: $runPath"
    }
}

foreach ($run in $runs) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $run.Config) -PathType Leaf)) {
        throw "Configuration is missing for $($run.Label): $($run.Config)"
    }
    Assert-RunDirectoryAvailable -RelativePath $run.RunDir
}

Write-Host "V10-B serial training preflight passed."
Write-Host "Repository: $repoRoot"
Write-Host "Python: $pythonPath"
Write-Host "Episode pool: $episodePoolRelative (episode_20001..episode_20300)"

foreach ($run in $runs) {
    # Re-check immediately before each run so a newly created artifact cannot
    # be silently overwritten after the initial preflight.
    Assert-RunDirectoryAvailable -RelativePath $run.RunDir
    Write-Host "Starting V10-B actor LR $($run.Label)"
    $arguments = @(
        "-B",
        "-m",
        "ppo_simple.trainer",
        "--config",
        $run.Config,
        "--epochs",
        "300",
        "--run-dir",
        $run.RunDir,
        "--episode-pool",
        $episodePoolRelative,
        "--episode-seed-start",
        "20001"
    )
    & $pythonPath @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "V10-B training failed for actor LR $($run.Label); remaining runs were not started. Exit code: $exitCode"
    }
    Write-Host "Completed V10-B actor LR $($run.Label)"
}

Write-Host "All three V10-B serial training runs completed successfully."
