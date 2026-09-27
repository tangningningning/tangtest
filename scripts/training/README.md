# One-epoch runner gate

The shared one-epoch implementation is `one_epoch_runner_common.py`. Four
small entrypoints define the actor learning rate and run directory:

- `run_one_epoch_actor_lr_1e_minus_4.py`
- `run_one_epoch_actor_lr_5e_minus_4.py`
- `run_one_epoch_actor_lr_1e_minus_3.py`
- `run_one_epoch_actor_lr_5e_minus_3.py`

Verify fresh initialization without rollout, optimizer steps, or run-directory
creation:

```powershell
conda activate adaptive_cache_mec
$env:PYTHONPATH = 'src'
python scripts/training/run_one_epoch_actor_lr_1e_minus_4.py --verify-initialization
python scripts/training/run_one_epoch_actor_lr_5e_minus_4.py --verify-initialization
python scripts/training/run_one_epoch_actor_lr_1e_minus_3.py --verify-initialization
python scripts/training/run_one_epoch_actor_lr_5e_minus_3.py --verify-initialization
```

Current status: the proposal-coverage gate failed. Do not execute a training
runner until the required proposal, Hybrid quality and efficiency, Candidate
teacher, runner, and regression gates pass. If they pass, execute only the
5e-4 entrypoint for one complete epoch, with a fresh run directory and no
resume. The shared entrypoint checks the required PASS fields in
`reports/second_stage_joint_repair_gate.md` and refuses training otherwise;
initialization verification remains available while gates are blocked.
