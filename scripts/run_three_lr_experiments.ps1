[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repo

$python = "C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe"
$env:PYTHONPATH = Join-Path $repo "src"
$env:PYTHONNOUSERSITE = "1"

$episodePool = Join-Path $repo "runs\ppo_v6_popart_actor1e4_lazy_episode_pool_20260911"
$experiments = @(
    [pscustomobject]@{
        Name = "actor_5e4_kl_protection"
        Config = Join-Path $repo "configs\lr_sweep\v9_actor_lr_5e4_kl_protection.toml"
        RunDir = Join-Path $repo "runs\lr_sweep_v9_actor_5e4_kl_protection"
    }
    [pscustomobject]@{
        Name = "actor_7e4"
        Config = Join-Path $repo "configs\lr_sweep\v9_actor_lr_7e4.toml"
        RunDir = Join-Path $repo "runs\lr_sweep_v9_actor_7e4"
    }
    [pscustomobject]@{
        Name = "actor_1e3_kl_protection"
        Config = Join-Path $repo "configs\lr_sweep\v9_actor_lr_1e3_kl_protection.toml"
        RunDir = Join-Path $repo "runs\lr_sweep_v9_actor_1e3_kl_protection"
    }
)

try {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "adaptive_cache_mec conda Python does not exist: $python"
    }
    if (-not (Test-Path -LiteralPath $episodePool -PathType Container)) {
        throw "Episode pool does not exist: $episodePool"
    }
    foreach ($experiment in $experiments) {
        if (-not (Test-Path -LiteralPath $experiment.Config -PathType Leaf)) {
            throw "Experiment config does not exist: $($experiment.Config)"
        }
        if (Test-Path -LiteralPath $experiment.RunDir) {
            if (-not (Test-Path -LiteralPath $experiment.RunDir -PathType Container)) {
                throw "Run path is not a directory: $($experiment.RunDir)"
            }
            if (@(Get-ChildItem -LiteralPath $experiment.RunDir -Force).Count -ne 0) {
                throw "Run directory is not empty: $($experiment.RunDir)"
            }
        }
    }
}
catch {
    Write-Error $_
    exit 1
}

foreach ($experiment in $experiments) {
    Write-Host "Starting $($experiment.Name)"
    $arguments = @(
        "-s"
        "-B"
        "-m"
        "ppo_simple.trainer"
        "--config"
        $experiment.Config
        "--epochs"
        "300"
        "--run-dir"
        $experiment.RunDir
        "--episode-pool"
        $episodePool
        "--episode-seed-start"
        "20001"
    )
    & $python @arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Error "$($experiment.Name) exited with code $exitCode; stopping before the next experiment."
        exit 1
    }
    Write-Host "Completed $($experiment.Name)"
}

Write-Host "All three experiments completed."
