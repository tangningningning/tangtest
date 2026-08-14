# -*- coding: utf-8 -*-
"""
异步自适应配置更新辅助模块。

核心触发条件：

    gross_gain
    = V_hat(old) - V_hat(candidate)

    net_gain
    = gross_gain - switching_cost

只有：

    net_gain > CFG_IMPROVEMENT_EPS

才允许真实物理配置更新。

重要：
P1 / P2 / Pjoint 内部已经按这个准则做严格改善接受，
因此本模块不再增加第二层决策门槛，避免重复扣除切换成本。

本模块负责：
1. 把已有严格改善条件显式写成 TriggerMetrics；
2. 检查“算法报告 changed=True”时确实满足触发条件；
3. 记录每个缓存节点自己的真实更新时间；
4. 记录每个 UAV 自己的真实移动时间；
5. 计算相邻真实更新之间的 slot 间隔。

由此得到不规则的：
    T_m^ca
    T_u^q
即异步更新时刻集合。
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass
class TriggerMetrics:
    """一次配置候选的触发量。"""

    old_p3_cost: float
    new_p3_cost: float
    gross_gain: float
    switching_cost: float
    net_gain: float
    triggered: bool


def compute_trigger_metrics(
    old_p3_cost: float,
    new_p3_cost: float,
    switching_cost: float,
) -> TriggerMetrics:
    """
    计算自适应更新触发量。

    对代价最小化模型：

        gross_gain
        = J_old - J_new

        net_gain
        = gross_gain - C_switch

    当 net_gain > epsilon_cfg 时更新。
    """
    old_p3_cost = float(
        old_p3_cost
    )
    new_p3_cost = float(
        new_p3_cost
    )
    switching_cost = float(
        switching_cost
    )

    gross_gain = (
        old_p3_cost
        - new_p3_cost
    )

    net_gain = (
        gross_gain
        - switching_cost
    )

    triggered = (
        net_gain
        > config.CFG_IMPROVEMENT_EPS
    )

    return TriggerMetrics(
        old_p3_cost=old_p3_cost,
        new_p3_cost=new_p3_cost,
        gross_gain=gross_gain,
        switching_cost=switching_cost,
        net_gain=net_gain,
        triggered=triggered,
    )


def require_trigger_if_changed(
    changed: bool,
    metrics: TriggerMetrics,
    label: str,
) -> None:
    """
    仅做运行期一致性保护。

    若优化器报告真实候选已发生改变，
    但净收益没有超过 epsilon，则说明实现逻辑发生冲突。
    """
    if (
        changed
        and not metrics.triggered
    ):
        raise RuntimeError(
            f"{label} 报告 changed=True，"
            "但净收益没有超过 "
            "CFG_IMPROVEMENT_EPS。"
        )


class AsyncUpdateTracker:
    """
    记录各节点各自独立的真实更新时间。

    initial_slot=0 表示：
        t=0 为外生初始配置时刻，
        不是一次动态更新事件。

    若某节点第一次在 t=17 更新，
    则其第一次 update_interval_slots=17。
    """

    def __init__(
        self,
        cache_node_ids: list[str],
        uav_ids: list[str],
        initial_slot: int = 0,
    ):
        self.last_cache_update_slot = {
            str(node_id):
                int(initial_slot)
            for node_id
            in cache_node_ids
        }

        self.last_uav_update_slot = {
            str(uav_id):
                int(initial_slot)
            for uav_id
            in uav_ids
        }

        self.cache_events: list[dict] = []
        self.uav_events: list[dict] = []

    def record_cache_changes(
        self,
        slot: int,
        changes: list[dict],
        source: str,
        decision_net_gain: float,
    ) -> None:
        """
        记录最终真实缓存变化。

        一个节点只有最终：
            C_m(t) != C_m(t-1)
        才产生一次异步更新事件。
        """
        for change in changes:
            node_id = str(
                change["node_id"]
            )

            previous_slot = (
                self.last_cache_update_slot
                .get(
                    node_id,
                    0,
                )
            )

            interval = (
                int(slot)
                - int(previous_slot)
            )

            self.cache_events.append(
                {
                    "slot":
                        int(slot),
                    "node_id":
                        node_id,
                    "node_type":
                        change["node_type"],
                    "update_source":
                        source,
                    "update_interval_slots":
                        interval,
                    "decision_net_gain":
                        float(
                            decision_net_gain
                        ),
                    "removed":
                        list(
                            change["removed"]
                        ),
                    "added":
                        list(
                            change["added"]
                        ),
                }
            )

            self.last_cache_update_slot[
                node_id
            ] = int(slot)

    def record_uav_changes(
        self,
        slot: int,
        changes: list[dict],
        source: str,
        decision_net_gain: float,
    ) -> None:
        """
        记录最终真实 UAV 移动事件。

        一个 UAV 只有最终：
            q_u(t) != q_u(t-1)
        才产生一次异步位置更新事件。
        """
        for change in changes:
            uav_id = str(
                change["uav_id"]
            )

            previous_slot = (
                self.last_uav_update_slot
                .get(
                    uav_id,
                    0,
                )
            )

            interval = (
                int(slot)
                - int(previous_slot)
            )

            event = dict(change)

            event.update(
                {
                    "slot":
                        int(slot),
                    "update_source":
                        source,
                    "update_interval_slots":
                        interval,
                    "decision_net_gain":
                        float(
                            decision_net_gain
                        ),
                }
            )

            self.uav_events.append(
                event
            )

            self.last_uav_update_slot[
                uav_id
            ] = int(slot)
