# -*- coding: utf-8 -*-
"""
正式 P3 求解器。

当前模型：
    TV 只产生任务，不进行本地执行。

任务执行位置：
    SV / RSU / UAV / Cloud

当前可直接运行的阶段：
    SV + Cloud

但 P3 主体已经写成通用边缘节点接口，后续加入 RSU/UAV 时，
不需要重新设计卸载和资源分配逻辑。

核心输入：
1. 当前时隙任务
2. TV 与边缘节点位置
3. 边缘节点计算能力
4. 当前缓存状态 c_{m,f}
5. 当前节点类型 SV / RSU / UAV

核心输出：
1. 每个任务选择哪个执行节点
2. CPU 资源份额 w
3. 无线资源份额 b
4. 时延、终端传输能耗和归一化代价 J
"""

from __future__ import annotations

import csv
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import config
import channel
import cache_model
import rsu_state


# =============================================================================
# 数据读取
# =============================================================================

def load_positions() -> dict[tuple[int, str], tuple[float, float]]:
    """
    读取统一的 400-slot mobility episode。

    返回：
        positions[(slot, vehicle_id)] = (x, y)
    """
    positions = {}

    with config.MOBILITY_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            positions[
                (int(row["slot"]), row["vehicle_id"])
            ] = (
                float(row["x"]),
                float(row["y"]),
            )

    return positions


def load_tasks() -> list[dict]:
    """读取已经生成的任务轨迹。"""
    tasks = []

    with config.TASK_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            tasks.append(
                {
                    "task_id": int(row["task_id"]),
                    "slot": int(row["slot"]),
                    "vehicle_id": row["vehicle_id"],
                    "service_id": int(row["service_id"]),
                    "D": float(row["D_bits"]),
                    "L": float(row["L_cycles"]),
                    "Tmax": float(row["deadline_s"]),
                }
            )

    return tasks


def load_ranked_vehicle_ids(
    role_name: str,
    k: int,
) -> list[str]:
    """
    从固定 role pool 中读取排名前 k 的车辆。

    这样车辆数量 sweep 时天然满足嵌套关系。
    """
    ranked = []

    with config.ROLE_POOL_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            if row["role_pool"].upper() == role_name.upper():
                ranked.append(
                    (int(row["role_rank"]), row["vehicle_id"])
                )

    ranked.sort()
    ids = [vehicle_id for _, vehicle_id in ranked[:k]]

    if len(ids) != k:
        raise RuntimeError(
            f"{role_name} 需要 {k} 辆，但只找到 {len(ids)} 辆"
        )

    return ids


# =============================================================================
# 节点与缓存状态
# =============================================================================

def build_current_sv_nodes() -> list[dict]:
    """
    构造当前开发阶段使用的 SV 节点。

    后续 RSU/UAV 只需要按同样格式加入：
        {
            "node_id": ...,
            "node_type": "RSU"/"UAV",
            "cpu_hz": ...,
        }

    P3 主体无需变化。
    """
    sv_ids = load_ranked_vehicle_ids("SV", config.NUM_SV)

    return [
        {
            "node_id": sv_id,
            "node_type": "SV",
            "cpu_hz": config.SV_CPU_HZ,
        }
        for sv_id in sv_ids
    ]



def build_uav_nodes(
    uav_positions: dict[str, tuple[float, float]],
) -> list[dict]:
    """
    把外部给定的 UAV 水平位置状态 Q^U 转换为 P3 节点。

    本函数不决定 UAV 数量或初始坐标。
    """
    return [
        {
            "node_id": str(uav_id),
            "node_type": "UAV",
            "cpu_hz": config.UAV_CPU_HZ,
            "position": (float(position[0]), float(position[1])),
        }
        for uav_id, position in sorted(uav_positions.items())
    ]


def build_edge_nodes(
    uav_positions:
        dict[str, tuple[float, float]]
        | None = None,
) -> list[dict]:
    """
    构造当前完整动态缓存/边缘计算节点集合：
        M = V_S ∪ R ∪ U
    """
    nodes = build_current_sv_nodes()
    nodes.extend(rsu_state.load_rsu_nodes())

    if uav_positions:
        nodes.extend(build_uav_nodes(uav_positions))

    return nodes


def build_all_service_cache_state(
    edge_nodes: list[dict],
) -> dict[tuple[str, int], bool]:
    """
    仅用于当前 P3 核心验证的临时缓存状态。

    当前暂时令所有 SV 都拥有全部服务。
    注意：
        这不是最终缓存模型。

    正式 P1 接入后，只需把该字典替换为实际：
        cache_state[(node_id, service_id)] = c_{m,f}

    P3 求解器本身不需要修改。
    """
    cache_state = {}

    for node in edge_nodes:
        for service_id in range(config.NUM_SERVICES):
            cache_state[(node["node_id"], service_id)] = True

    return cache_state


# =============================================================================
# Cloud 代价
# =============================================================================

def cloud_metrics(task: dict) -> dict:
    """
    计算 Cloud 执行的时延、TV 发射能耗和归一化代价。

    当前没有本地执行，因此采用：
        E_ref = p_v * Tmax

    Cloud 时延：
        T_cloud = psi_c*D/R_c + L/F_c

    Cloud 终端发射能耗：
        E_cloud = p_v*D/R_c

    归一化代价：
        J_cloud =
            omega_T*T_cloud/Tmax
            + omega_E*E_cloud/E_ref
    """
    tx_time = (
        config.CLOUD_PSI
        * task["D"]
        / config.CLOUD_RATE_BPS
    )

    compute_time = (
        task["L"]
        / config.CLOUD_CPU_HZ
    )

    delay = tx_time + compute_time

    energy_j = (
        config.TX_POWER_W
        * task["D"]
        / config.CLOUD_RATE_BPS
    )

    energy_ref = (
        config.TX_POWER_W
        * task["Tmax"]
    )

    cost = (
        config.OMEGA_T
        * delay
        / task["Tmax"]
        + config.OMEGA_E
        * energy_j
        / energy_ref
    )

    return {
        "delay": delay,
        "energy_J": energy_j,
        "cost": cost,
    }


# =============================================================================
# 边缘节点候选生成
# =============================================================================

def _node_position(
    slot: int,
    node: dict,
    positions: dict,
) -> tuple[float, float]:
    """
    获取边缘节点二维水平位置。

    SV：从实际/预测车辆位置字典读取；
    RSU：使用固定 position；
    UAV：使用当前候选 Q^U 中的 position。
    """
    node_type = node["node_type"].upper()

    if node_type == "SV":
        key = (slot, node["node_id"])
        if key not in positions:
            raise RuntimeError(f"缺少 SV 位置: {key}")
        return positions[key]

    if node_type in {"RSU", "UAV"}:
        if "position" not in node:
            raise RuntimeError(
                f"{node_type} {node['node_id']} 缺少 position。"
            )
        return (
            float(node["position"][0]),
            float(node["position"][1]),
        )

    raise NotImplementedError(f"{node_type} 位置接口尚未接入。")


def make_edge_candidate(
    task: dict,
    node: dict,
    positions: dict,
    cache_state: dict[tuple[str, int], bool],
) -> dict | None:
    """
    判断某个边缘节点是否是任务的合法候选。

    必须依次满足：
    1. 节点缓存了任务所需服务；
    2. 无线链路可用；
    3. 即使给满 CPU 和满带宽，任务也能满足 deadline。

    对应：
        alpha = L/F_m
        beta  = D/R_bar

        T_min = alpha + beta <= Tmax
    """

    node_id = node["node_id"]
    node_type = node["node_type"].upper()

    # ---------- 1. 缓存门 ----------
    if not cache_state.get(
        (node_id, task["service_id"]),
        False,
    ):
        return None

    # ---------- 2. 计算 TV 与节点距离 ----------
    tv_key = (task["slot"], task["vehicle_id"])

    if tv_key not in positions:
        raise RuntimeError(f"缺少 TV 位置: {tv_key}")

    tx, ty = positions[tv_key]
    nx, ny = _node_position(
        task["slot"],
        node,
        positions,
    )

    # TV 到节点投影点的二维水平距离。
    horizontal_distance_m = math.hypot(
        tx - nx,
        ty - ny,
    )

    if node_type == "SV":
        if not channel.sv_link_eligible(horizontal_distance_m):
            return None
        link_distance_m = horizontal_distance_m

    elif node_type == "RSU":
        if not channel.rsu_link_eligible(horizontal_distance_m):
            return None
        link_distance_m = horizontal_distance_m

    elif node_type == "UAV":
        link_distance_m = math.sqrt(
            horizontal_distance_m ** 2
            + config.UAV_HEIGHT_M ** 2
        )

    else:
        raise ValueError(f"未知边缘节点类型：{node_type}")

    # ---------- 3. 满资源基础速率 ----------
    rbar_bps = channel.full_rate_bps(
        node_type,
        horizontal_distance_m,
    )

    # ---------- 4. 满资源 deadline 筛选 ----------
    alpha = task["L"] / node["cpu_hz"]
    beta = task["D"] / rbar_bps

    if (
        alpha + beta
        > task["Tmax"] + config.FEASIBILITY_TOL
    ):
        return None

    # 固定任务分配到节点后：
    #
    # J = A/w + B/b
    #
    # A = omega_T * L/(Tmax*F_m)
    #
    # 由于 E_ref = p*Tmax 且 omega_T+omega_E=1：
    #
    # B = D/(R_bar*Tmax)
    A = (
        config.OMEGA_T
        * task["L"]
        / (task["Tmax"] * node["cpu_hz"])
    )

    Bcoef = (
        task["D"]
        / (rbar_bps * task["Tmax"])
    )

    return {
        "task_id": task["task_id"],
        "node_id": node_id,
        "node_type": node_type,
        "distance_m": link_distance_m,
        "horizontal_distance_m": horizontal_distance_m,
        "rbar_bps": rbar_bps,
        "alpha": alpha,
        "beta": beta,
        "A": A,
        "Bcoef": Bcoef,
        "Tmax": task["Tmax"],
    }


# =============================================================================
# 固定节点任务集合下的 CPU / 带宽联合分配
# =============================================================================

def closed_form_allocation(
    candidates: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    不考虑激活 deadline 时的闭式最优解：

        w_i ∝ sqrt(A_i)
        b_i ∝ sqrt(B_i)
    """
    sqrt_a = np.sqrt(
        np.array(
            [c["A"] for c in candidates],
            dtype=float,
        )
    )

    sqrt_b = np.sqrt(
        np.array(
            [c["Bcoef"] for c in candidates],
            dtype=float,
        )
    )

    w = sqrt_a / sqrt_a.sum()
    b = sqrt_b / sqrt_b.sum()

    return w, b


def allocation_feasible(
    candidates: list[dict],
    w: np.ndarray,
    b: np.ndarray,
) -> bool:
    """检查资源约束和所有任务 deadline。"""

    if np.any(w <= 0.0) or np.any(b <= 0.0):
        return False

    if (
        w.sum()
        > 1.0 + config.FEASIBILITY_TOL
    ):
        return False

    if (
        b.sum()
        > 1.0 + config.FEASIBILITY_TOL
    ):
        return False

    for i, c in enumerate(candidates):
        delay = (
            c["alpha"] / w[i]
            + c["beta"] / b[i]
        )

        if (
            delay
            > c["Tmax"] + config.FEASIBILITY_TOL
        ):
            return False

    return True


def solve_node_allocation(
    candidates: list[dict],
) -> dict | None:
    """
    求解固定分配到同一个边缘节点的任务资源分配问题。

    优先：
        使用闭式解。

    只有闭式解违反 deadline 时：
        才调用 SLSQP 求带 deadline 约束的凸子问题。

    若该任务集合在节点资源限制下不可行：
        返回 None。
    """
    n = len(candidates)

    if n == 0:
        return {
            "cost": 0.0,
            "alloc": {},
            "used_slsqp": False,
        }

    w0, b0 = closed_form_allocation(candidates)

    if allocation_feasible(
        candidates,
        w0,
        b0,
    ):
        w = w0
        b = b0
        used_slsqp = False

    else:
        A = np.array(
            [c["A"] for c in candidates],
            dtype=float,
        )

        Bcoef = np.array(
            [c["Bcoef"] for c in candidates],
            dtype=float,
        )

        alpha = np.array(
            [c["alpha"] for c in candidates],
            dtype=float,
        )

        beta = np.array(
            [c["beta"] for c in candidates],
            dtype=float,
        )

        Tmax = np.array(
            [c["Tmax"] for c in candidates],
            dtype=float,
        )

        def objective(x):
            wv = x[:n]
            bv = x[n:]
            return float(
                np.sum(A / wv + Bcoef / bv)
            )

        def resource_constraints(x):
            wv = x[:n]
            bv = x[n:]

            return np.array(
                [
                    1.0 - np.sum(wv),
                    1.0 - np.sum(bv),
                ]
            )

        def deadline_constraints(x):
            wv = x[:n]
            bv = x[n:]

            return (
                Tmax
                - alpha / wv
                - beta / bv
            )

        x0 = np.concatenate([w0, b0])

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=[
                (config.RESOURCE_EPS, 1.0)
            ] * (2 * n),
            constraints=[
                {
                    "type": "ineq",
                    "fun": resource_constraints,
                },
                {
                    "type": "ineq",
                    "fun": deadline_constraints,
                },
            ],
            options={
                "ftol": config.SLSQP_FTOL,
                "maxiter": config.SLSQP_MAXITER,
                "disp": False,
            },
        )

        if not result.success:
            return None

        w = result.x[:n]
        b = result.x[n:]

        if not allocation_feasible(
            candidates,
            w,
            b,
        ):
            return None

        used_slsqp = True

    allocation = {}
    total_cost = 0.0

    for i, c in enumerate(candidates):
        delay = (
            c["alpha"] / w[i]
            + c["beta"] / b[i]
        )

        energy_j = (
            config.TX_POWER_W
            * c["beta"]
            / b[i]
        )

        cost = (
            c["A"] / w[i]
            + c["Bcoef"] / b[i]
        )

        total_cost += cost

        allocation[c["task_id"]] = {
            "mode": c["node_type"],
            "node_id": c["node_id"],
            "w": float(w[i]),
            "b": float(b[i]),
            "delay": float(delay),
            "energy_J": float(energy_j),
            "cost": float(cost),
            "distance_m": c["distance_m"],
            "rbar_bps": c["rbar_bps"],
        }

    return {
        "cost": float(total_cost),
        "alloc": allocation,
        "used_slsqp": used_slsqp,
    }


# =============================================================================
# 单时隙 P3
# =============================================================================

def solve_slot_p3(
    slot_tasks: list[dict],
    edge_nodes: list[dict],
    positions: dict,
    cache_state: dict[tuple[str, int], bool],
) -> tuple[dict, dict]:
    """
    精确枚举当前时隙任务的可行卸载节点。

    当前开发规模很小，所以参考求解器采用枚举：
        Cloud / 每个合法边缘节点

    对每一个离散分配：
        各边缘节点独立求 CPU / 带宽凸子问题。

    该实现用于：
    1. 作为 P3 的正确性参考求解器；
    2. 后续 P1/P2/Pjoint 评价某个配置时调用。
    """

    cloud_map = {}
    edge_map = {}
    option_lists = []

    eligible_task_count = 0

    # ---------- 为每个任务构造候选集合 ----------
    for task in slot_tasks:
        cloud = cloud_metrics(task)

        if (
            cloud["delay"]
            > task["Tmax"]
            + config.FEASIBILITY_TOL
        ):
            raise RuntimeError(
                f"任务 {task['task_id']} 的 Cloud fallback "
                "不满足 deadline。"
            )

        cloud_map[task["task_id"]] = cloud

        candidates = {}

        for node in edge_nodes:
            c = make_edge_candidate(
                task,
                node,
                positions,
                cache_state,
            )

            if c is not None:
                candidates[node["node_id"]] = c

        edge_map[task["task_id"]] = candidates

        if candidates:
            eligible_task_count += 1

        # Cloud 始终是兜底选项。
        option_lists.append(
            ["cloud"] + list(candidates.keys())
        )

    best_cost = math.inf
    best_result = None

    assignments_evaluated = 0
    assignments_feasible = 0
    best_slsqp_nodes = 0

    # ---------- 枚举离散卸载分配 ----------
    for choices in itertools.product(*option_lists):
        assignments_evaluated += 1

        total_cost = 0.0
        task_result = {}

        # 按边缘节点收集任务。
        node_groups = defaultdict(list)

        feasible = True
        slsqp_nodes = 0

        for task, choice in zip(
            slot_tasks,
            choices,
        ):
            task_id = task["task_id"]

            if choice == "cloud":
                cloud = cloud_map[task_id]

                total_cost += cloud["cost"]

                task_result[task_id] = {
                    "mode": "CLOUD",
                    "node_id": "cloud",
                    "w": None,
                    "b": None,
                    "delay": cloud["delay"],
                    "energy_J": cloud["energy_J"],
                    "cost": cloud["cost"],
                    "distance_m": None,
                    "rbar_bps": config.CLOUD_RATE_BPS,
                }

            else:
                node_groups[choice].append(
                    edge_map[task_id][choice]
                )

        # ---------- 连续资源分配 ----------
        for node_id, candidates in node_groups.items():
            solved = solve_node_allocation(candidates)

            if solved is None:
                feasible = False
                break

            total_cost += solved["cost"]

            slsqp_nodes += int(
                solved["used_slsqp"]
            )

            task_result.update(
                solved["alloc"]
            )

        if not feasible:
            continue

        assignments_feasible += 1

        if (
            total_cost
            < best_cost - 1e-12
        ):
            best_cost = total_cost
            best_result = task_result
            best_slsqp_nodes = slsqp_nodes

    if best_result is None:
        raise RuntimeError(
            f"slot={slot_tasks[0]['slot']} "
            "没有找到可行 P3 解。"
        )

    return best_result, {
        "eligible_tasks": eligible_task_count,
        "assignments_evaluated": assignments_evaluated,
        "assignments_feasible": assignments_feasible,
        "best_slsqp_nodes": best_slsqp_nodes,
    }


# =============================================================================
# 完整 episode P3
# =============================================================================

def solve_episode_p3(
    tasks: list[dict],
    edge_nodes: list[dict],
    positions: dict,
    cache_state: dict[tuple[str, int], bool],
) -> tuple[dict, dict]:
    """
    对整个 400-slot episode 逐时隙求 P3。

    注意：
    P3 本身只处理“给定当前配置”的任务执行。
    缓存更新、UAV 更新和异步触发属于外层 P1/P2/Pjoint，
    不放进这里。
    """

    tasks_by_slot = defaultdict(list)

    for task in tasks:
        tasks_by_slot[task["slot"]].append(task)

    all_results = {}

    stats = {
        "eligible_tasks": 0,
        "assignments_evaluated": 0,
        "assignments_feasible": 0,
        "best_slsqp_nodes": 0,
    }

    for slot in sorted(tasks_by_slot):
        result, slot_stats = solve_slot_p3(
            tasks_by_slot[slot],
            edge_nodes,
            positions,
            cache_state,
        )

        all_results.update(result)

        for key in stats:
            stats[key] += slot_stats[key]

    return all_results, stats


# =============================================================================
# 结果输出
# =============================================================================

def write_results(
    tasks: list[dict],
    results: dict,
    output_file: Path,
) -> dict:
    """将 P3 任务级结果写入 CSV，并返回摘要统计。"""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "task_id",
        "slot",
        "vehicle_id",
        "service_id",
        "mode",
        "node_id",
        "delay_s",
        "deadline_s",
        "energy_J",
        "cost",
        "w_cpu",
        "b_bw",
        "distance_m",
        "full_rate_Mbps",
    ]

    mode_counts = Counter()
    node_counts = Counter()

    total_cost = 0.0
    total_delay = 0.0
    total_energy = 0.0
    violations = 0

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for task in sorted(
            tasks,
            key=lambda x: x["task_id"],
        ):
            r = results[task["task_id"]]

            mode_counts[r["mode"]] += 1
            node_counts[r["node_id"]] += 1

            total_cost += r["cost"]
            total_delay += r["delay"]
            total_energy += r["energy_J"]

            if (
                r["delay"]
                > task["Tmax"]
                + config.FEASIBILITY_TOL
            ):
                violations += 1

            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "slot": task["slot"],
                    "vehicle_id": task["vehicle_id"],
                    "service_id": task["service_id"],
                    "mode": r["mode"],
                    "node_id": r["node_id"],
                    "delay_s": f"{r['delay']:.9f}",
                    "deadline_s": f"{task['Tmax']:.9f}",
                    "energy_J": f"{r['energy_J']:.12f}",
                    "cost": f"{r['cost']:.12f}",
                    "w_cpu": (
                        ""
                        if r["w"] is None
                        else f"{r['w']:.9f}"
                    ),
                    "b_bw": (
                        ""
                        if r["b"] is None
                        else f"{r['b']:.9f}"
                    ),
                    "distance_m": (
                        ""
                        if r["distance_m"] is None
                        else f"{r['distance_m']:.6f}"
                    ),
                    "full_rate_Mbps": (
                        f"{r['rbar_bps']/1e6:.6f}"
                    ),
                }
            )

    n = len(tasks)

    return {
        "tasks": n,
        "mode_counts": mode_counts,
        "node_counts": node_counts,
        "deadline_violations": violations,
        "mean_cost": total_cost / n,
        "mean_delay": total_delay / n,
        "mean_energy_mJ": 1000.0 * total_energy / n,
    }


# =============================================================================
# 当前阶段的直接运行入口
# =============================================================================

def main():
    """
    当前直接运行时：
    1. 使用 config.NUM_SV 个 SV；
    2. 临时令所有 SV 缓存全部服务；
    3. Cloud 作为兜底；
    4. 不包含 Local；
    5. RSU/UAV 接口已预留，但尚未加入未冻结参数。
    """

    required_files = [
        config.MOBILITY_FILE,
        config.ROLE_POOL_FILE,
        config.TASK_FILE,
        config.RSU_DEPLOYMENT_FILE,
    ]

    for file in required_files:
        if not file.is_file():
            raise FileNotFoundError(file)

    positions = load_positions()
    tasks = load_tasks()

    edge_nodes = build_edge_nodes()

    cache_state = cache_model.build_initial_cache_state(
        edge_nodes
    )

    # ???????????
    cache_model.save_cache_state(
        cache_state,
        edge_nodes,
    )

    results, solver_stats = solve_episode_p3(
        tasks,
        edge_nodes,
        positions,
        cache_state,
    )

    output_file = (
        config.RESULTS_DIR
        / "p3_current.csv"
    )

    summary = write_results(
        tasks,
        results,
        output_file,
    )

    print("=" * 72)
    print("正式 P3 核心模块运行完成")
    print("=" * 72)
    print("执行模式            : SV + RSU + Cloud")
    print("本地执行            : 已删除")
    print(f"任务数              : {summary['tasks']}")
    print(f"SV 数量             : {config.NUM_SV}")
    print(
        f"RSU 数量            : "
        f"{sum(1 for n in edge_nodes if n['node_type'] == 'RSU')}"
    )
    print(
        "缓存状态            : "
        "当前临时全服务可用，P1 接入后替换"
    )
    print()

    print(
        f"存在至少一个边缘候选的任务 : "
        f"{solver_stats['eligible_tasks']}"
    )
    print(
        f"枚举的离散分配数量         : "
        f"{solver_stats['assignments_evaluated']}"
    )
    print(
        f"可行离散分配数量           : "
        f"{solver_stats['assignments_feasible']}"
    )
    print(
        f"最优解中 SLSQP 节点调用数  : "
        f"{solver_stats['best_slsqp_nodes']}"
    )
    print()

    for mode, count in sorted(
        summary["mode_counts"].items()
    ):
        print(
            f"{mode:<10}: "
            f"{count}/{summary['tasks']} "
            f"({100.0*count/summary['tasks']:.2f}%)"
        )

    print()

    print(
        f"deadline 违反数      : "
        f"{summary['deadline_violations']}"
    )
    print(
        f"平均任务时延         : "
        f"{summary['mean_delay']:.6f} s"
    )
    print(
        f"平均 TV 发射能耗     : "
        f"{summary['mean_energy_mJ']:.6f} mJ"
    )
    print(
        f"平均归一化代价 J     : "
        f"{summary['mean_cost']:.6f}"
    )
    print()

    print("节点选择次数：")
    for node_id, count in sorted(
        summary["node_counts"].items()
    ):
        print(f"  {node_id}: {count}")

    print()
    print(f"结果文件：{output_file}")
    print()
    print(
        "下一步：不再修改 P3 主体，"
        "开始接入真实缓存状态 c_{m,f}。"
    )


if __name__ == "__main__":
    main()
