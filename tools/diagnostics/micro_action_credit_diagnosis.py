"""Read-only stochastic Validation diagnosis of Micro UAV PPO credit signals."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import project_config
from ppo_simple.evaluator import _assert_realization_context_compatible, _load_checkpoint_metadata, _seed_policy_sampling
from ppo_simple.interface import aligned_train_steps, load_environment_context
from ppo_simple.policy import _group_balance_candidate_logits
from ppo_simple.rollout import MicroTransition, collect_dual_timescale_episode, compute_gae
from ppo_simple.trainer import PPOTrainer, _normalize_advantages


RUNS = (
    ("5e-4", "formal_cache_economic_v2_lr_5e_minus_4"),
    ("1e-3", "formal_cache_economic_v2_lr_1e_minus_3"),
)
INITIAL_SEEDS = (30003, 30004)
ACTION_SEED_OFFSET = 70000
REPORT_PATH = ROOT / "reports" / "micro_action_credit_diagnosis.md"
TEMP_CACHE_PATH = ROOT / "tools" / "diagnostics" / ".micro_action_credit_diagnosis_cache.pkl"
SEGMENTS = (("前段", 0, 100), ("中段", 100, 200), ("后段", 200, 300))
JOINT_NAMES = {
    (0, 0): "U0 STAY / U1 STAY",
    (0, 1): "U0 STAY / U1 MOVE",
    (1, 0): "U0 MOVE / U1 STAY",
    (1, 1): "U0 MOVE / U1 MOVE",
}
EPS = 1.0e-12


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: Any, digits: int = 5) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):+.{digits}f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows)
    return "\n".join(lines)


def summarize(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "positive": int(np.count_nonzero(array > EPS)),
        "negative": int(np.count_nonzero(array < -EPS)),
        "zero": int(np.count_nonzero(np.abs(array) <= EPS)),
    }


def raw_candidate_scores(policy: Any, transition: MicroTransition) -> tuple[list[np.ndarray], list[np.ndarray]]:
    score_rows: list[np.ndarray] = []
    move_ranks: list[np.ndarray] = []
    with torch.no_grad():
        for head in range(2):
            features = np.asarray(transition.uav_candidate_features[head], dtype=np.float32)
            mask = np.asarray(transition.uav_candidate_masks[head], dtype=bool)
            condition = (
                np.zeros(2, dtype=np.float32)
                if head == 0
                else np.asarray(transition.uav0_position_condition, dtype=np.float32)
            )
            raw = policy._micro_candidate_raw_logits(
                transition.observation,
                features,
                mask,
                condition,
            )
            balanced = _group_balance_candidate_logits(
                raw,
                features,
                mask,
                slice(policy.micro_uav_feature_dim - 2, policy.micro_uav_feature_dim),
            )
            probabilities = torch.softmax(balanced, dim=-1).detach().cpu().numpy()
            recorded = np.asarray(transition.uav_candidate_probabilities[head], dtype=np.float64)
            count = int(transition.valid_candidate_count[head])
            if not np.allclose(probabilities[:count], recorded[:count], rtol=0.0, atol=6.0e-6):
                raise AssertionError(
                    f"recomputed behavior probabilities differ at slot {transition.slot}, UAV{head}"
                )
            raw_np = raw.detach().cpu().numpy().astype(np.float64)
            valid_indices = np.flatnonzero(mask)
            move_indices = [int(index) for index in valid_indices if int(index) != 0]
            ranks = np.zeros(len(mask), dtype=np.int32)
            ordered = sorted(move_indices, key=lambda index: (-raw_np[index], index))
            for rank, index in enumerate(ordered, start=1):
                ranks[index] = rank
            score_rows.append(raw_np)
            move_ranks.append(ranks)
    return score_rows, move_ranks


def compact_transition(transition: MicroTransition, advantage: float, return_value: float) -> dict[str, Any]:
    return {
        "slot": int(transition.slot),
        "origin_index": int(transition.origin_index),
        "actions": np.asarray(transition.actions, dtype=np.int64).copy(),
        "old_joint_log_prob": float(transition.old_joint_log_prob),
        "old_uav_log_probs": np.asarray(transition.old_uav_log_probs, dtype=np.float32).copy(),
        "old_value": float(transition.old_value),
        "reward": float(transition.reward),
        "done": bool(transition.done),
        "observation": np.asarray(transition.observation, dtype=np.float32).copy(),
        "candidate_features": np.asarray(transition.uav_candidate_features, dtype=np.float32).copy(),
        "candidate_masks": np.asarray(transition.uav_candidate_masks, dtype=bool).copy(),
        "candidate_positions": np.asarray(transition.uav_candidate_positions, dtype=np.float32).copy(),
        "candidate_probabilities": np.asarray(transition.uav_candidate_probabilities, dtype=np.float32).copy(),
        "candidate_sources": transition.uav_candidate_sources,
        "condition": np.asarray(transition.uav0_position_condition, dtype=np.float32).copy(),
        "valid_candidate_count": np.asarray(transition.valid_candidate_count, dtype=np.int64).copy(),
        "advantage": float(advantage),
        "return_value": float(return_value),
    }


def collect_rollout(policy: Any, trainer: Any, context: Any, seed: int, action_seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    steps = aligned_train_steps(context)
    if len(steps) != 300 or len(steps) != int(context.episode_slots):
        raise AssertionError(f"Validation seed {seed} is not a full 300-Slot episode")
    # Reuse the frozen evaluator's reproducible Python/NumPy/Torch seed helper.
    _seed_policy_sampling(action_seed)
    started = time.perf_counter()
    with torch.inference_mode():
        collection = collect_dual_timescale_episode(
            policy,
            context,
            steps,
            deterministic=False,
            episode_seed=seed,
        )
    elapsed = time.perf_counter() - started
    transitions = list(collection.micro.transitions)
    if len(transitions) != 300 or any(not isinstance(item, MicroTransition) for item in transitions):
        raise AssertionError("stochastic Micro collection did not return 300 formal transitions")
    dones = np.asarray([item.done for item in transitions], dtype=bool)
    if not dones[-1] or np.any(dones[:-1]):
        raise AssertionError("full Validation Micro rollout must terminate only at Slot 299")
    gamma = float(trainer.config["gamma"])
    gae_lambda = float(trainer.config["gae_lambda"])
    advantages, returns = compute_gae(
        [item.reward for item in transitions],
        [item.old_value for item in transitions],
        dones,
        0.0,
        gamma,
        gae_lambda,
    )
    records = [compact_transition(item, float(advantages[i]), float(returns[i])) for i, item in enumerate(transitions)]
    summary = {
        "seed": int(seed),
        "action_seed": int(action_seed),
        "slots": len(transitions),
        "episode_return": float(collection.summary["episode_return"]),
        "uav0_move_rate": float(1.0 - collection.summary["uav0_stay_rate"]),
        "uav1_move_rate": float(1.0 - collection.summary["uav1_stay_rate"]),
        "joint_counts": dict(Counter(tuple(int(value != 0) for value in item.actions) for item in transitions)),
        "micro_gamma": gamma,
        "micro_gae_lambda": gae_lambda,
        "matching_kkt_seconds": float(sum(item.matching_kkt_seconds for item in transitions)),
        "slot_runtime_seconds": float(sum(item.slot_runtime_seconds for item in transitions)),
        "wall_seconds": elapsed,
    }
    return records, summary


def build_action_data(records: list[dict[str, Any]], policy: Any) -> dict[str, Any]:
    scores_and_ranks: list[tuple[list[np.ndarray], list[np.ndarray]]] = []
    for record in records:
        # Reconstruct only the transition fields needed by the existing scorer.
        class ScoreTransition:
            pass
        transition = ScoreTransition()
        transition.observation = record["observation"]
        transition.uav_candidate_features = record["candidate_features"]
        transition.uav_candidate_masks = record["candidate_masks"]
        transition.uav_candidate_probabilities = record["candidate_probabilities"]
        transition.uav_candidate_sources = record["candidate_sources"]
        transition.uav0_position_condition = record["condition"]
        transition.valid_candidate_count = record["valid_candidate_count"]
        transition.slot = record["slot"]
        scores_and_ranks.append(raw_candidate_scores(policy, transition))
    return {"scores_and_ranks": scores_and_ranks}


def gradient_diagnostics(policy: Any, records: list[dict[str, Any]], normalized_advantages: np.ndarray) -> dict[str, Any]:
    parameters = tuple(policy.micro_actor_parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    count = len(records)
    accum = {"u0": np.zeros(parameter_count, dtype=np.float64), "u1": np.zeros(parameter_count, dtype=np.float64), "joint": np.zeros(parameter_count, dtype=np.float64)}
    max_log_ratio = 0.0
    def flatten(gradients: tuple[Any, ...]) -> np.ndarray:
        pieces = []
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is None:
                pieces.append(np.zeros(parameter.numel(), dtype=np.float64))
            else:
                pieces.append(gradient.detach().cpu().numpy().reshape(-1).astype(np.float64))
        return np.concatenate(pieces)

    chunk_size = 64
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk = records[start:stop]
        observations = np.stack([row["observation"] for row in chunk]).astype(np.float32)
        features = np.stack([row["candidate_features"] for row in chunk]).astype(np.float32)
        masks = np.stack([row["candidate_masks"] for row in chunk]).astype(bool)
        actions = np.stack([row["actions"] for row in chunk]).astype(np.int64)
        conditions = np.stack([row["condition"] for row in chunk]).astype(np.float32)
        old_log_probs = np.asarray([row["old_uav_log_probs"] for row in chunk], dtype=np.float32)
        adv = torch.as_tensor(normalized_advantages[start:stop], dtype=torch.float32, device=next(policy.parameters()).device)
        evaluation = policy.evaluate_micro_joint(observations, features, masks, actions, conditions)
        new_head_log_probs = torch.stack((evaluation.uav0_log_prob, evaluation.uav1_log_prob), dim=-1)
        old_head_log_probs = torch.as_tensor(old_log_probs, dtype=new_head_log_probs.dtype, device=new_head_log_probs.device)
        max_log_ratio = max(max_log_ratio, float(torch.max(torch.abs(new_head_log_probs - old_head_log_probs)).detach().cpu()))
        scale = 1.0 / count
        loss0 = -(evaluation.uav0_log_prob * adv).sum() * scale
        loss1 = -(evaluation.uav1_log_prob * adv).sum() * scale
        joint_loss = -(evaluation.log_prob * adv).sum() * scale
        g0 = torch.autograd.grad(loss0, parameters, retain_graph=True, allow_unused=True)
        g1 = torch.autograd.grad(loss1, parameters, retain_graph=True, allow_unused=True)
        gj = torch.autograd.grad(joint_loss, parameters, allow_unused=True)
        accum["u0"] += flatten(g0)
        accum["u1"] += flatten(g1)
        accum["joint"] += flatten(gj)

    g0, g1, gj = accum["u0"], accum["u1"], accum["joint"]
    norm0, norm1, norm_joint = float(np.linalg.norm(g0)), float(np.linalg.norm(g1)), float(np.linalg.norm(gj))
    denominator = norm0 * norm1
    cosine = float(np.dot(g0, g1) / denominator) if denominator > 0.0 else None
    sum_gap = float(np.linalg.norm(gj - (g0 + g1)))
    cancellation = float(norm_joint / (norm0 + norm1)) if norm0 + norm1 > 0.0 else None
    return {
        "count": count,
        "parameters": parameter_count,
        "u0_norm": norm0,
        "u1_norm": norm1,
        "joint_norm": norm_joint,
        "cosine": cosine,
        "sum_gap": sum_gap,
        "cancellation_ratio": cancellation,
        "max_head_logprob_error": max_log_ratio,
        "normalized_adv_mean": float(np.mean(normalized_advantages)),
        "normalized_adv_std": float(np.std(normalized_advantages)),
    }


def generate_report(results: dict[str, dict[str, Any]], elapsed: float) -> None:
    lines = [
        "# Micro PPO 动作学习信号诊断",
        "",
        "日期：2026-10-09  ",
        "范围：冻结策略的 stochastic Validation 完整 300-Slot Rollout；未训练、未调用任何 optimizer.step()，未修改正式策略、奖励、环境、Checkpoints、日志或数据。",
        "",
        "## 代码链路和诊断口径",
        "",
        "- 正式 `collect_dual_timescale_episode(deterministic=False)` 强制输入覆盖 `context.episode_slots` 的完整 Episode；每 Slot 依次采样 UAV0，再按该 UAV0 位置条件化生成/采样 UAV1，记录两项 per-head log-probability、联合 log-probability、候选支持、动作、critic value 和系统 reward，并调用正式 `execute_configured_slot()` 完成环境执行。",
        "- `compute_gae()` 使用 Micro 每 Slot reward/value/done 计算普通 GAE；正式 Micro 更新对每个 episode buffer 计算 GAE，再拼接并标准化。原始优势表仍报告 GAE 标准化前数值；策略梯度表按此批验证 rollout 联合标准化的优势计算，公式与训练器相同。",
        "- Micro actor 的 PPO clipped surrogate 对联合概率 `logπ(U0|s)+logπ(U1|s,U0)` 使用同一 Advantage。两个 UAV head 共享 Micro Actor encoder 与候选 scorer 参数；entropy 与组熵 floor 是额外 actor 项。下方 `g0/g1/joint` 只分解共享 Advantage 的 log-prob PPO actor surrogate，不包含 entropy/group-floor 梯度。当前策略 ratio=1，clip 未触发，因此该梯度就是当前策略点的未裁剪 PPO surrogate 梯度。报告方向为 actor loss 的梯度；梯度下降更新方向取负号。",
        "- UAV0、UAV1 动作状态分布会不同，特别是 UAV1 的支持和候选分数以 UAV0 已采样位置为条件。因此按组的优势差只是行为策略下的相关性，不能解释为同状态下的动作因果价值。",
        "",
        "## Checkpoint 和 Rollout",
        "",
        "本次先使用 Validation seeds 30003、30004。两组 checkpoint 按各自文件核对格式、策略结构、配置哈希和 episode/update 计数；不以不同训练进度做学习率因果比较。对同一个 Validation seed，两 checkpoint 使用相同固定动作 RNG seed `70000 + seed`，分别从其冻结策略 stochastic 采样；这不是训练采样，也不更新参数。",
        "",
        "| Macro Actor LR | Completed Episodes / Updates | Checkpoint / Architecture | Config SHA-256 | Checkpoint SHA-256 | 每 seed 完整 rollout |",
        "|---|---:|---:|---|---|---:|",
    ]
    for lr in ("5e-4", "1e-3"):
        info = results[lr]["checkpoint"]
        lines.append(f"| {lr} | {info['episodes']} / {info['updates']} | {info['version']} / {info['architecture']} | `{info['config_sha']}` | `{info['checkpoint_sha']}` | 300 Slot |")
    lines.extend(["", "| LR | Validation seed | stochastic RNG seed | Slot 数 | Return | UAV0 MOVE | UAV1 MOVE | Matching/KKT 计时(s) | 完整 rollout 墙钟(s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for lr in ("5e-4", "1e-3"):
        for episode in results[lr]["episodes"]:
            lines.append(f"| {lr} | {episode['seed']} | {episode['action_seed']} | {episode['slots']} | {episode['episode_return']:.3f} | {episode['uav0_move_rate']:.1%} | {episode['uav1_move_rate']:.1%} | {episode['matching_kkt_seconds']:.2f} | {episode['wall_seconds']:.2f} |")
    gae_gamma = results["5e-4"]["episodes"][0]["micro_gamma"]
    gae_lambda = results["5e-4"]["episodes"][0]["micro_gae_lambda"]
    lines.extend(["", f"Micro GAE 参数（来自正式 PPO config）：`gamma={gae_gamma:g}`、`gae_lambda={gae_lambda:g}`，terminal bootstrap=0；每条 rollout 确认 300 条 transition，且仅第 300 条 terminal。", "" ])

    for lr in ("5e-4", "1e-3"):
        data = results[lr]
        lines.extend([f"## {lr} Validation 样本", ""])
        seed_rows = []
        for episode in data["episodes"]:
            counts = episode["joint_counts"]
            seed_rows.append([episode["seed"], episode["slots"], *(counts.get(key, 0) for key in ((0, 0), (0, 1), (1, 0), (1, 1)))])
        lines.append(md_table(["seed", "Slot", "STAY/STAY", "STAY/MOVE", "MOVE/STAY", "MOVE/MOVE"], seed_rows))
        lines.extend(["", "### UAV 动作组与共享原始 Advantage 分布", ""])
        rows = []
        for name in ("UAV0 STAY", "UAV0 MOVE", "UAV1 STAY", "UAV1 MOVE", *JOINT_NAMES.values()):
            stat = data["group_stats"][name]
            rows.append([name, stat["n"], f"{stat.get('positive', 0)}/{stat.get('n', 0)} ({stat.get('positive', 0)/stat.get('n', 1):.1%})", stat.get("negative", 0), stat.get("zero", 0), fmt(stat.get("mean")), fmt(stat.get("median")), f"{stat.get('std', math.nan):.5f}", f"[{fmt(stat.get('q25'))}, {fmt(stat.get('q75'))}]", f"[{fmt(stat.get('min'))}, {fmt(stat.get('max'))}]"])
        lines.append(md_table(["动作组", "n", "正 Advantage", "负", "零", "mean", "median", "std", "Q1–Q3", "min–max"], rows))
        lines.extend(["", "### 按 Slot 前、中、后三段", "", "分段为 Slot 0–99、100–199、200–299。每行报告对应子集中共享原始 Micro Advantage；小样本单元格保留 n，不作因果解释。", ""])
        segment_rows = []
        for segment_name, _, _ in SEGMENTS:
            for name in ("UAV0 STAY", "UAV0 MOVE", "UAV1 STAY", "UAV1 MOVE", *JOINT_NAMES.values()):
                stat = data["segment_stats"][(segment_name, name)]
                segment_rows.append([segment_name, name, stat["n"], f"{stat.get('positive', 0)}/{stat.get('n', 0)}", fmt(stat.get("mean")), fmt(stat.get("median")), f"{stat.get('std', math.nan):.5f}"])
        lines.append(md_table(["Slot 段", "动作组", "n", "+Adv/n", "mean", "median", "std"], segment_rows))
        lines.extend(["", "### MOVE 候选来源与组内 raw-logit 排名", "", "排名是每个 sampled MOVE 对应 slot、UAV 条件下，在当时合法 MOVE 候选中按冻结策略 raw logit 降序的名次（1 为最高）；其分数仅用于记录采样覆盖/排序，不是 Q 值。", ""])
        move_rows = []
        for head in (0, 1):
            for source, stat in sorted(data["move_sources"][head].items()):
                move_rows.append([f"UAV{head}", source, stat["count"], f"{stat['share']:.1%}", f"{stat['rank_mean']:.2f}", f"{stat['rank_median']:.1f}", f"{stat['rank_q25']:.1f}–{stat['rank_q75']:.1f}", f"{stat['top1_fraction']:.1%}", f"{stat['candidate_count_min']}–{stat['candidate_count_max']}"])
        lines.append(md_table(["head", "proposal source", "MOVE 样本数", "MOVE 内占比", "raw 名次 mean", "median", "Q1–Q3", "top-1 率", "合法 MOVE 候选数范围"], move_rows))
        grad = data["gradient"]
        lines.extend(["", "### 共享 Micro Actor 的 PPO log-prob 梯度分解", ""])
        lines.append(md_table(["样本数", "Actor 参数数", "||g0||", "||g1||", "||g_joint||", "cos(g0,g1)", "抵消比 ||sum||/(||g0||+||g1||)", "||g_joint−(g0+g1)||", "最大 per-head log-prob 误差"], [[grad["count"], grad["parameters"], f"{grad['u0_norm']:.7g}", f"{grad['u1_norm']:.7g}", f"{grad['joint_norm']:.7g}", f"{grad['cosine']:+.5f}" if grad["cosine"] is not None else "N/A", f"{grad['cancellation_ratio']:.5f}" if grad["cancellation_ratio"] is not None else "N/A", f"{grad['sum_gap']:.3g}", f"{grad['max_head_logprob_error']:.3g}"]]))
        lines.append(f"\n用于策略梯度的 episode-batch Advantage 按正式 `_normalize_advantages()` 跨本次 {len(data['episodes'])} 条 Validation Episode 标准化：均值 {grad['normalized_adv_mean']:+.3g}、标准差 {grad['normalized_adv_std']:.5f}。`g0` 和 `g1` 分别是 `−mean(A_norm × logπ0)` 与 `−mean(A_norm × logπ1)` 对全部 Micro Actor 参数的梯度；`g_joint` 来自 `−mean(A_norm × (logπ0+logπ1))`。分解求和误差应接近浮点误差。Entropy 与 group-entropy-floor 正则项未混入该 head-to-head 对照。")

    lines.extend(["", "## 综合判断", ""])
    lines.extend([
        "1. **有益联合动作是否采样不足：** 以四类实际 sampled joint action 数量为准；稀少或为零的类别只能判定覆盖不足。结合此前 reports/uav_multislot_counterfactual.md 中已检查的 C2 五步正收益，若 MOVE/MOVE 在此处很少，说明当前 stochastic 行为数据对该联合分支的 advantage 估计覆盖有限，但不能推断若增加它就会因果提升 Return。",
        "2. **已采样但 Advantage 信号是否清晰：** 看四种联合组与各 head STAY/MOVE 的原始 GAE 均值、中位数和正负比例。若同一组中符号混杂或跨时段反转，说明在当前状态分布与 GAE 下信用信号分散；各组状态分布不同，所以均值差不等于动作价值差。",
        "3. **UAV0/UAV1 梯度方向是否冲突：** 看共享参数空间中的 cos(g0,g1) 及 joint 范数相对两项范数的抵消比。负 cosine 表示本批 surrogate 梯度局部拉向不同方向；接近零表示近似正交；正值表示同向。单批验证梯度只描述当前网络点与采样 batch，不能证明训练长期冲突或失效。",
        "4. **候选覆盖与学习机制：** MOVE source、合法候选组内 raw 排名显示哪些 MOVE 实际被抽到及其当时相对排序。若某些候选/联合组采样很少，无法由现有样本判断其 Advantage 是否可靠；若已采样但 GAE 符号高度混杂，当前信号表现为弱/噪声大，但不能仅靠相关性把原因定位到共享 Advantage、Critic 或更新算法。",
        "",
        "## 限制与只读校验",
        "",
        f"本轮完整 stochastic Validation rollout 数：{sum(len(item['episodes']) for item in results.values())}；总 Micro transition 数：{sum(sum(len(episode) for episode in item['records']) for item in results.values())}；没有访问 Test seed。总分析耗时（含 300-slot 环境 Matching/KKT）：{elapsed:.1f}s。各 checkpoint 文件 SHA-256、模型参数快照和 optimizer step 计数在 rollout、GAE、评分与 `autograd.grad` 前后核对不变；未调用 optimizer 清梯度或更新。",
        "当前每 checkpoint 先有 2 个独立 Validation seed；如 MOVE/MOVE 等关键联合类别覆盖不足，本报告明确标为证据不足，不将两组 checkpoint 或不同状态分布解释成学习率、动作或梯度因果效应。",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def aggregate_records(records: list[dict[str, Any]], policy: Any) -> dict[str, Any]:
    action_data = build_action_data(records, policy)
    score_rank = action_data["scores_and_ranks"]
    advantage = np.asarray([row["advantage"] for row in records], dtype=np.float32)
    normalized, norm_mean, norm_std = _normalize_advantages(advantage)
    group_values: dict[str, list[float]] = defaultdict(list)
    segment_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    probabilities: dict[str, list[float]] = defaultdict(list)
    joint_counts = Counter()
    move_entries: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for index, row in enumerate(records):
        action_bits = tuple(int(value != 0) for value in row["actions"])
        joint_counts[action_bits] += 1
        group_values[f"UAV0 {'MOVE' if action_bits[0] else 'STAY'}"].append(row["advantage"])
        group_values[f"UAV1 {'MOVE' if action_bits[1] else 'STAY'}"].append(row["advantage"])
        group_values[JOINT_NAMES[action_bits]].append(row["advantage"])
        segment = next(name for name, start, stop in SEGMENTS if start <= int(row["slot"]) < stop)
        segment_values[(segment, f"UAV0 {'MOVE' if action_bits[0] else 'STAY'}")].append(row["advantage"])
        segment_values[(segment, f"UAV1 {'MOVE' if action_bits[1] else 'STAY'}")].append(row["advantage"])
        segment_values[(segment, JOINT_NAMES[action_bits])].append(row["advantage"])
        for head in range(2):
            count = int(row["valid_candidate_count"][head])
            p = row["candidate_probabilities"][head, :count]
            probabilities[f"UAV{head} STAY"].append(float(p[0]))
            probabilities[f"UAV{head} MOVE"].append(float(p[1:].sum()))
            if action_bits[head] == 0:
                continue
            action_index = int(row["actions"][head])
            source = str(row["candidate_sources"][head][action_index])
            rank = int(score_rank[index][1][head][action_index])
            move_count = count - 1
            legal_move_group_size = int(row["candidate_masks"][head].sum()) - 1
            move_entries[head].append({"source": source, "rank": rank, "move_count": move_count, "group_size": legal_move_group_size})
    group_stats = {name: summarize(np.asarray(group_values.get(name, []), dtype=np.float64)) for name in ("UAV0 STAY", "UAV0 MOVE", "UAV1 STAY", "UAV1 MOVE", *JOINT_NAMES.values())}
    segment_stats = {(segment, name): summarize(np.asarray(segment_values.get((segment, name), []), dtype=np.float64)) for segment, _, _ in SEGMENTS for name in ("UAV0 STAY", "UAV0 MOVE", "UAV1 STAY", "UAV1 MOVE", *JOINT_NAMES.values())}
    source_stats: dict[int, dict[str, Any]] = {0: {}, 1: {}}
    for head in range(2):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in move_entries[head]:
            grouped[item["source"]].append(item)
        total = max(len(move_entries[head]), 1)
        for source, items in grouped.items():
            ranks = np.asarray([item["rank"] for item in items], dtype=float)
            source_stats[head][source] = {
                "count": len(items),
                "share": len(items) / total,
                "rank_mean": float(np.mean(ranks)),
                "rank_median": float(np.median(ranks)),
                "rank_q25": float(np.quantile(ranks, 0.25)),
                "rank_q75": float(np.quantile(ranks, 0.75)),
                "top1_fraction": float(np.mean(ranks == 1.0)),
                "candidate_count_min": min(item["move_count"] for item in items),
                "candidate_count_max": max(item["move_count"] for item in items),
            }
    gradient = gradient_diagnostics(policy, records, normalized)
    gradient["normalized_adv_mean_pre"] = norm_mean
    gradient["normalized_adv_std_pre"] = norm_std
    return {
        "group_stats": group_stats,
        "segment_stats": segment_stats,
        "move_sources": source_stats,
        "joint_counts": dict(joint_counts),
        "mean_group_probabilities": {name: float(np.mean(values)) if values else math.nan for name, values in probabilities.items()},
        "gradient": gradient,
    }


def run(seeds: tuple[int, ...] = INITIAL_SEEDS) -> None:
    started = time.perf_counter()
    checkpoint_states: dict[str, dict[str, Any]] = {}
    references: dict[str, Any] = {}
    trainers: dict[str, Any] = {}
    policies: dict[str, Any] = {}
    params_before: dict[str, list[torch.Tensor]] = {}
    optimizer_steps_before: dict[str, tuple[int, int]] = {}
    checkpoints_before: dict[str, str] = {}
    results: dict[str, dict[str, Any]] = {}

    for lr, run_name in RUNS:
        run_dir = ROOT / "runs" / run_name
        config_path = run_dir / "run_config.toml"
        checkpoint = run_dir / "checkpoints" / "latest_checkpoint.pt"
        config_sha = file_sha(config_path)
        checkpoint_sha = file_sha(checkpoint)
        metadata = _load_checkpoint_metadata(checkpoint)
        reference = load_environment_context(str(config_path))
        semantic_sha = project_config.semantic_config_hash(reference.cfg)
        if metadata.get("config_raw_sha256") != config_sha or metadata.get("config_semantic_sha256") != semantic_sha:
            raise ValueError(f"checkpoint config hash mismatch for {lr}")
        trainer = PPOTrainer(
            reference,
            device="cpu",
            macro_actor_learning_rate_override=float(metadata["learning_rates"]["macro_actor"]),
            unified_cache_action=True,
            cardinality_corrected_cache_action=True,
        )
        trainer.load_checkpoint(checkpoint, config_raw_sha256=config_sha, config_semantic_sha256=semantic_sha, episode_pool=None)
        trainer.policy.eval()
        references[lr], trainers[lr], policies[lr] = reference, trainer, trainer.policy
        params_before[lr] = [parameter.detach().clone() for parameter in trainer.policy.parameters()]
        optimizer_steps_before[lr] = (int(trainer.training_counters["micro_optimizer_steps"]), int(trainer.training_counters["macro_optimizer_steps"]))
        checkpoints_before[lr] = checkpoint_sha
        checkpoint_states[lr] = {
            "path": checkpoint,
            "metadata": metadata,
            "checkpoint": {
                "episodes": int(trainer.training_counters["completed_episodes"]),
                "updates": int(trainer.training_counters["policy_update_count"]),
                "version": int(metadata["checkpoint_version"]),
                "architecture": int(metadata["architecture_version"]),
                "config_sha": config_sha,
                "checkpoint_sha": checkpoint_sha,
            },
        }
        results[lr] = {"checkpoint": checkpoint_states[lr]["checkpoint"], "episodes": [], "records": [], "run": None}

    cache_header = {
        "version": 1,
        "checkpoints": {lr: checkpoints_before[lr] for lr, _ in RUNS},
        "action_seed_offset": ACTION_SEED_OFFSET,
    }
    cached_payload: dict[str, Any] = {"header": cache_header, "episodes": {}}
    if TEMP_CACHE_PATH.is_file():
        try:
            existing = pickle.loads(TEMP_CACHE_PATH.read_bytes())
            if isinstance(existing, dict) and existing.get("header") == cache_header and isinstance(existing.get("episodes"), dict):
                cached_payload = existing
        except Exception:
            # A partial/corrupt task-local cache is ignored and replaced on first completed episode.
            pass

    def persist_completed_episode() -> None:
        temporary_path = TEMP_CACHE_PATH.with_suffix(".tmp")
        temporary_path.write_bytes(pickle.dumps(cached_payload, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temporary_path, TEMP_CACHE_PATH)

    for seed in seeds:
        realization_dir = ROOT / "environment_data" / "proposed_joint_validation" / f"episode_{seed}"
        if not realization_dir.is_dir():
            raise FileNotFoundError(f"Validation realization missing: {realization_dir}")
        context = load_environment_context(str(ROOT / "runs" / RUNS[0][1] / "run_config.toml"), realization_dir=realization_dir)
        # Resolve each run config explicitly when checking the same realization context.
        for lr, run_name in RUNS:
            reference_for_lr = references[lr]
            _assert_realization_context_compatible(reference_for_lr, context)
        steps = aligned_train_steps(context)
        if len(steps) != 300:
            raise AssertionError("Validation aligned rollout input must be exactly 300 slots")
        for lr, run_name in RUNS:
            trainer = trainers[lr]
            policy = policies[lr]
            trainer.context = context
            action_seed = ACTION_SEED_OFFSET + int(seed)
            cache_key = f"{lr}:{seed}"
            cached_episode = cached_payload["episodes"].get(cache_key)
            if (
                isinstance(cached_episode, dict)
                and cached_episode.get("checkpoint_sha") == checkpoints_before[lr]
                and cached_episode.get("action_seed") == action_seed
                and cached_episode.get("slots") == 300
            ):
                episode_records = cached_episode["records"]
                episode_summary = cached_episode["summary"]
                print(f"[{lr}] seed={seed}: reusing completed temporary rollout cache", flush=True)
            else:
                print(f"[{lr}] stochastic Validation seed={seed}, action_seed={action_seed}: starting full 300-slot rollout", flush=True)
                episode_records, episode_summary = collect_rollout(policy, trainer, context, seed, action_seed)
                cached_payload["episodes"][cache_key] = {
                    "checkpoint_sha": checkpoints_before[lr],
                    "action_seed": action_seed,
                    "slots": len(episode_records),
                    "records": episode_records,
                    "summary": episode_summary,
                }
                persist_completed_episode()
            results[lr]["episodes"].append(episode_summary)
            results[lr]["records"].append(episode_records)
            print(f"[{lr}] seed={seed} completed: Return={episode_summary['episode_return']:.3f}, MOVE={episode_summary['uav0_move_rate']:.1%}/{episode_summary['uav1_move_rate']:.1%}, matching={episode_summary['matching_kkt_seconds']:.2f}s", flush=True)

    for lr, run_name in RUNS:
        all_records = [row for episode in results[lr]["records"] for row in episode]
        aggregate = aggregate_records(all_records, policies[lr])
        results[lr].update(aggregate)
        results[lr]["run"] = {"config_path": str(ROOT / "runs" / run_name / "run_config.toml")}

    for lr, run_name in RUNS:
        checkpoint = checkpoint_states[lr]["path"]
        trainer = trainers[lr]
        if any(not torch.equal(before, after.detach()) for before, after in zip(params_before[lr], trainer.policy.parameters(), strict=True)):
            raise AssertionError(f"frozen policy parameters changed for {lr}")
        if optimizer_steps_before[lr] != (int(trainer.training_counters["micro_optimizer_steps"]), int(trainer.training_counters["macro_optimizer_steps"])):
            raise AssertionError(f"optimizer step counters changed for {lr}")
        if file_sha(checkpoint) != checkpoints_before[lr]:
            raise AssertionError(f"checkpoint changed for {lr}")
        if any(parameter.grad is not None for parameter in trainer.policy.micro_actor_parameters()):
            raise AssertionError(f"autograd.grad unexpectedly accumulated parameter .grad for {lr}")
    elapsed = time.perf_counter() - started
    generate_report(results, elapsed)
    TEMP_CACHE_PATH.unlink(missing_ok=True)
    TEMP_CACHE_PATH.with_suffix(".tmp").unlink(missing_ok=True)
    for lr in ("5e-4", "1e-3"):
        print(f"[{lr}] joint counts={results[lr]['joint_counts']} gradient cos={results[lr]['gradient']['cosine']:+.5f} norms={results[lr]['gradient']['u0_norm']:.5g}/{results[lr]['gradient']['u1_norm']:.5g}/{results[lr]['gradient']['joint_norm']:.5g}", flush=True)
    print(f"report written: {REPORT_PATH}; total elapsed={elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(INITIAL_SEEDS))
    args = parser.parse_args()
    if any(seed not in (30003, 30004, 30005, 30006) for seed in args.seeds):
        raise ValueError("only Validation seeds 30003-30006 are allowed")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Validation seeds must be unique")
    torch.set_num_threads(1)
    run(tuple(args.seeds))


if __name__ == "__main__":
    main()
