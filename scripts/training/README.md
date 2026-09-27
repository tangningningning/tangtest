# Training entrypoint and gate status

`src/ppo_simple/trainer.py` is the single PPO training implementation and CLI.
The current second-stage joint-repair gate is **BLOCKED**: hybrid quality and
exhaustive Cache proposal coverage failed. Do not start a short or formal
training run until those gates pass and the registry status is updated.

After all gates pass, activate the project environment and start a fresh
20-epoch proposed-policy run from the repository root:

```powershell
conda activate adaptive_cache_mec
$env:PYTHONPATH = 'src'
python -m ppo_simple.trainer --config configs/proposed/system_config.toml --epochs 20 --run-dir runs/proposed_fresh20
```

Do not pass `--resume`; the required run uses fresh initialization. Use the run
directory for its checkpoints, manifest, and training metrics. Extend to at
most 30 epochs only if the 20-epoch checkpoint review calls for it. Never use
this entrypoint for the formal 300-epoch run until the separate formal-readiness
decision is approved in the current stage report.
