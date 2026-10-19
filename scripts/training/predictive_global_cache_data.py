"""Training Pool loader shared by the non-RL Cache labeler and runner."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts" / "training"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common import project_config
from ppo_simple.interface import prepare_aligned_rollout_steps
import training_pool_contract as pool_contract


def training_context_and_steps(
    project_root: Path,
    episode_seed: int,
    slot_count: int,
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Load one provenance-validated consecutive Training window only."""

    root = project_root.resolve()
    pool, pool_info = pool_contract._validate_pool(
        root,
        pool_contract.TRAIN_POOL_RELATIVE,
        expected_split="train",
        expected_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
        expected_episode_count=300,
    )
    seed = int(episode_seed)
    pool_contract._validate_episode_snapshot(pool, seed, pool_info)
    context = pool_contract._episode_context(root, pool, seed, pool_info)
    if context.realization is None:
        raise ValueError("global Cache solver requires the reviewed Training realization")
    start = int(context.train_origin_start)
    slots = int(slot_count)
    if slots <= 0 or start + slots - 1 > int(context.train_origin_end):
        raise ValueError("requested Training window is outside reviewed Training origins")
    horizon = int(context.observation_spec.prediction_horizon)
    dt_s = float(project_config.get_config_value(context.cfg, "simulation.dt_s"))
    steps = prepare_aligned_rollout_steps(
        context.root,
        context.scenarios,
        start,
        slots,
        context.realization.vehicle_predictions,
        context.realization.popularity_predictions,
        context.realization.tasks,
        context.realization.trajectory,
        include_bootstrap_step=False,
        prediction_horizon=horizon,
        prediction_dt_s=dt_s,
        prediction_terminal_origin=int(context.train_origin_end),
        expected_vehicle_ids=context.observation_spec.vehicle_ids,
        expected_task_vehicle_ids=context.observation_spec.task_vehicle_ids,
        expected_service_ids=context.observation_spec.service_ids,
        include_prediction_horizon_mask=(context.observation_spec.horizon_mask_slice is not None),
    )
    if len(steps) != slots:
        raise ValueError("aligned Training window returned a different Slot count")
    slot_ids = [int(step.slot) for step in steps]
    if slot_ids != list(range(slot_ids[0], slot_ids[0] + slots)):
        raise ValueError("Training window Slots are not consecutive")
    return context, steps, pool_info

