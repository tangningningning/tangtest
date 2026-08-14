# -*- coding: utf-8 -*-
"""
因果预测与场景生成。

用途：
    在时隙 t 做缓存/UAV 配置决策之前，根据历史 H_{t-1}
    生成同一组预测场景 Xi_t，供 P1/P2/Pjoint 的所有候选配置共用。

严格因果要求：
1. 只使用 slot <= t-1 的车辆位置；
2. 只使用 slot <= t-1 的已实现任务；
3. 不读取未来真实 popularity；
4. 不读取未来真实 SUMO 位置；
5. 同一个 t 的所有候选配置必须复用同一个 ScenarioSet。

第一版预测器：
1. TV/SV 位置：常速度外推；
2. 任务到达：使用模型中已知的 Bernoulli(lambda)；
3. 服务流行度：历史请求频数 + Laplace 平滑；
4. 任务属性：优先从历史同服务任务中有放回重采样；
   若该服务尚无历史样本，则退回 service_catalog 中的服务均值。

这只是第一版因果预测器。后续可替换成 Kalman/EWMA 等预测方法，
但 ScenarioSet 与 P1/P2 的接口不需要变化。
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import config


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Scenario:
    """
    单个预测场景。

    tasks:
        预测时域内生成的任务列表，格式与 p3_solver.load_tasks() 一致。

    positions:
        positions[(slot, vehicle_id)] = (x, y)
        只包含预测时域内 P3 需要的 TV/SV 位置。
    """

    scenario_id: int
    tasks: list[dict]
    positions: dict[tuple[int, str], tuple[float, float]]


@dataclass
class ScenarioSet:
    """
    同一个决策时刻 t 的公共随机场景集合 Xi_t。
    """

    decision_slot: int
    horizon: int
    scenarios: list[Scenario]
    estimated_popularity: list[float]


# =============================================================================
# 历史数据读取
# =============================================================================

def load_realized_task_history(
    decision_slot: int,
) -> list[dict]:
    """
    只读取已经发生的任务：

        slot < decision_slot

    即在线决策 t 只能看到 H_{t-1}。
    """
    history = []

    with config.TASK_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            slot = int(row["slot"])

            if slot >= decision_slot:
                continue

            history.append(
                {
                    "task_id": int(row["task_id"]),
                    "slot": slot,
                    "vehicle_id": row["vehicle_id"],
                    "service_id": int(row["service_id"]),
                    "D": float(row["D_bits"]),
                    "L": float(row["L_cycles"]),
                    "Tmax": float(row["deadline_s"]),
                    "cycles_per_bit": float(
                        row["cycles_per_bit"]
                    ),
                }
            )

    return history


def load_position_history(
    decision_slot: int,
    vehicle_ids: list[str],
) -> dict[str, list[tuple[int, float, float]]]:
    """
    只读取 slot <= t-1 的位置。

    注意：
    mobility CSV 虽然离线保存了完整真实轨迹，但本函数明确忽略
    decision_slot 及其之后的所有行，因此预测器没有未来真值输入。
    """
    vehicle_set = set(vehicle_ids)

    history = defaultdict(list)

    with config.MOBILITY_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            slot = int(row["slot"])
            vehicle_id = row["vehicle_id"]

            if vehicle_id not in vehicle_set:
                continue

            if slot >= decision_slot:
                continue

            history[vehicle_id].append(
                (
                    slot,
                    float(row["x"]),
                    float(row["y"]),
                )
            )

    for vehicle_id in vehicle_ids:
        history[vehicle_id].sort(
            key=lambda x: x[0]
        )

    return dict(history)


def load_service_catalog() -> dict[int, dict]:
    """读取服务均值参数，作为历史不足时的回退分布中心。"""
    catalog = {}

    with config.SERVICE_CATALOG_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            service_id = int(row["service_id"])

            catalog[service_id] = {
                "D_mean_bits":
                    float(row["D_mean_Mbit"]) * 1e6,
                "chi_mean":
                    float(row["chi_mean_cycles_per_bit"]),
                "deadline_mean_s":
                    float(row["deadline_mean_s"]),
            }

    if len(catalog) != config.NUM_SERVICES:
        raise RuntimeError(
            f"服务目录数量错误：读取到 {len(catalog)}，"
            f"配置要求 {config.NUM_SERVICES}"
        )

    return catalog


# =============================================================================
# 流行度估计
# =============================================================================

def estimate_popularity(
    realized_history: list[dict],
) -> list[float]:
    """
    根据已经发生的服务请求估计当前流行度。

    使用 Laplace 平滑：
        rho_hat_f = (n_f + 1) / (N + F)

    优点：
    1. 完全因果；
    2. 不需要额外预测参数；
    3. 即使某服务历史中尚未出现，也不会得到零概率。

    当前先作为简单基线预测器。
    """
    counts = Counter(
        task["service_id"]
        for task in realized_history
    )

    total = len(realized_history)
    denominator = total + config.NUM_SERVICES

    return [
        (counts.get(service_id, 0) + 1)
        / denominator
        for service_id in range(
            config.NUM_SERVICES
        )
    ]


# =============================================================================
# 位置预测
# =============================================================================

def predict_vehicle_positions(
    decision_slot: int,
    horizon: int,
    vehicle_ids: list[str],
    position_history: dict[
        str,
        list[tuple[int, float, float]],
    ],
) -> dict[
    tuple[int, str],
    tuple[float, float],
]:
    """
    常速度预测：

        q_hat(t+k)
        = q(t-1)
          + (k+1) * [q(t-1)-q(t-2)]

    其中基础时隙为 1 s。

    若车辆历史只有一个位置点，则使用零速度保持当前位置。
    若没有任何历史位置，则明确报错，不伪造数据。
    """
    predicted = {}

    for vehicle_id in vehicle_ids:
        history = position_history.get(
            vehicle_id,
            [],
        )

        if not history:
            raise RuntimeError(
                f"车辆 {vehicle_id} 在 slot < {decision_slot} "
                "没有任何历史位置，无法做因果预测。"
            )

        _, x_last, y_last = history[-1]

        if len(history) >= 2:
            slot_prev, x_prev, y_prev = history[-2]
            slot_last, _, _ = history[-1]

            delta_slots = slot_last - slot_prev

            if delta_slots <= 0:
                raise RuntimeError(
                    f"车辆 {vehicle_id} 历史位置时间顺序异常。"
                )

            vx_per_slot = (
                x_last - x_prev
            ) / delta_slots

            vy_per_slot = (
                y_last - y_prev
            ) / delta_slots

        else:
            vx_per_slot = 0.0
            vy_per_slot = 0.0

        for k in range(horizon):
            future_slot = decision_slot + k

            step_ahead = k + 1

            predicted[
                (future_slot, vehicle_id)
            ] = (
                x_last
                + step_ahead * vx_per_slot,
                y_last
                + step_ahead * vy_per_slot,
            )

    return predicted


# =============================================================================
# 任务属性经验分布
# =============================================================================

def build_service_task_history(
    realized_history: list[dict],
) -> dict[int, list[dict]]:
    """
    将历史任务按 service_id 分类。

    后续场景采样时，同一服务的 D/L/deadline 优先从其真实历史
    样本中重采样，避免人为再增加一组分布参数。
    """
    grouped = defaultdict(list)

    for task in realized_history:
        grouped[task["service_id"]].append(task)

    return dict(grouped)


def sample_task_attributes(
    service_id: int,
    service_history: dict[int, list[dict]],
    service_catalog: dict[int, dict],
    rng: random.Random,
) -> tuple[float, float, float]:
    """
    返回：
        D_bits, L_cycles, Tmax

    有该服务历史样本：
        从历史已实现任务中有放回重采样。

    无历史样本：
        使用服务目录均值。
    """
    samples = service_history.get(
        service_id,
        [],
    )

    if samples:
        sample = rng.choice(samples)

        return (
            sample["D"],
            sample["L"],
            sample["Tmax"],
        )

    profile = service_catalog[service_id]

    D = profile["D_mean_bits"]
    chi = profile["chi_mean"]
    L = D * chi
    Tmax = profile["deadline_mean_s"]

    return D, L, Tmax


# =============================================================================
# 单个场景生成
# =============================================================================

def _sample_service_id(
    popularity: list[float],
    rng: random.Random,
) -> int:
    """按预测流行度采样服务类型。"""
    return rng.choices(
        population=list(
            range(config.NUM_SERVICES)
        ),
        weights=popularity,
        k=1,
    )[0]


def generate_one_scenario(
    scenario_id: int,
    decision_slot: int,
    horizon: int,
    tv_ids: list[str],
    predicted_positions: dict[
        tuple[int, str],
        tuple[float, float],
    ],
    estimated_popularity: list[float],
    service_history: dict[int, list[dict]],
    service_catalog: dict[int, dict],
    rng: random.Random,
) -> Scenario:
    """
    生成一个未来 H-slot 场景。

    每个 TV 每个预测时隙：
        a_v ~ Bernoulli(lambda)

    若任务到达：
        service ~ Categorical(rho_hat)
        task attributes ~ 历史经验分布
    """
    tasks = []
    task_id = 0

    for slot in range(
        decision_slot,
        decision_slot + horizon,
    ):
        for vehicle_id in tv_ids:
            if (
                rng.random()
                >= config.TASK_ARRIVAL_LAMBDA
            ):
                continue

            service_id = _sample_service_id(
                estimated_popularity,
                rng,
            )

            D, L, Tmax = sample_task_attributes(
                service_id=service_id,
                service_history=service_history,
                service_catalog=service_catalog,
                rng=rng,
            )

            tasks.append(
                {
                    "task_id": task_id,
                    "slot": slot,
                    "vehicle_id": vehicle_id,
                    "service_id": service_id,
                    "D": D,
                    "L": L,
                    "Tmax": Tmax,
                }
            )

            task_id += 1

    # 位置预测当前是确定性的，因此同一决策时刻的各 scenario
    # 共用同一套预测位置；不影响任务随机性的 CRN 语义。
    return Scenario(
        scenario_id=scenario_id,
        tasks=tasks,
        positions=dict(predicted_positions),
    )


# =============================================================================
# 公共随机场景集合 Xi_t
# =============================================================================

def generate_scenario_set(
    decision_slot: int,
    tv_ids: list[str],
    sv_ids: list[str],
    horizon: int | None = None,
    num_scenarios: int | None = None,
) -> ScenarioSet:
    """
    在决策时刻 t 生成一次 Xi_t。

    之后 P1/P2/Pjoint 的所有候选配置必须复用返回的同一个对象，
    不能为不同候选重新随机生成场景。

    参数
    ----
    decision_slot:
        当前配置决策时隙 t。

    tv_ids:
        当前参与任务生成的 TV。

    sv_ids:
        当前移动边缘节点 SV。
        位置预测需要同时预测 TV 和 SV。

    horizon:
        默认使用 config.PREDICTION_HORIZON。

    num_scenarios:
        默认使用 config.NUM_SCENARIOS。
    """
    if decision_slot <= 0:
        raise ValueError(
            "第一版因果预测器要求 decision_slot >= 1。"
            "t=0 使用外生初始配置，不执行预测更新。"
        )

    if horizon is None:
        horizon = config.PREDICTION_HORIZON

    if num_scenarios is None:
        num_scenarios = config.NUM_SCENARIOS

    if horizon <= 0:
        raise ValueError("prediction horizon 必须大于 0")

    if num_scenarios <= 0:
        raise ValueError("scenario 数量必须大于 0")

    # ------------------------------------------------------------------
    # 只构造 H_{t-1}
    # ------------------------------------------------------------------
    realized_history = load_realized_task_history(
        decision_slot
    )

    all_vehicle_ids = list(
        dict.fromkeys(tv_ids + sv_ids)
    )

    position_history = load_position_history(
        decision_slot,
        all_vehicle_ids,
    )

    # ------------------------------------------------------------------
    # 因果预测
    # ------------------------------------------------------------------
    estimated_popularity = estimate_popularity(
        realized_history
    )

    predicted_positions = predict_vehicle_positions(
        decision_slot=decision_slot,
        horizon=horizon,
        vehicle_ids=all_vehicle_ids,
        position_history=position_history,
    )

    service_history = build_service_task_history(
        realized_history
    )

    service_catalog = load_service_catalog()

    # ------------------------------------------------------------------
    # 同一 t 使用固定的公共随机场景集合 Xi_t。
    #
    # seed 随 decision_slot 改变：
    # - 同一个 t 可复现；
    # - 不同 t 不重复使用完全相同的随机流。
    # ------------------------------------------------------------------
    base_seed = (
        config.SCENARIO_SEED
        + 100003 * decision_slot
    )

    scenarios = []

    for scenario_id in range(num_scenarios):
        rng = random.Random(
            base_seed + scenario_id
        )

        scenarios.append(
            generate_one_scenario(
                scenario_id=scenario_id,
                decision_slot=decision_slot,
                horizon=horizon,
                tv_ids=tv_ids,
                predicted_positions=predicted_positions,
                estimated_popularity=estimated_popularity,
                service_history=service_history,
                service_catalog=service_catalog,
                rng=rng,
            )
        )

    return ScenarioSet(
        decision_slot=decision_slot,
        horizon=horizon,
        scenarios=scenarios,
        estimated_popularity=estimated_popularity,
    )
