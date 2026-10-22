"""把候选条件化 DQN 接到正式的一 Slot 系统执行接口。"""

from __future__ import annotations

import hashlib
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decision import slot_environment, task_allocation, uav_configuration
from online_joint_optimizer.candidate_evaluator import (
    SearchBudget,
    SlotInput,
    SystemModelContext,
)
from online_joint_optimizer.cache_optimizer import CacheOptimizer
from online_joint_optimizer.uav_scheduler import UavScheduler

from .candidate_features import (
    ActionCandidate,
    CandidateFeatureEncoder,
    CandidateOption,
    DecisionForecast,
    _state_fingerprint,
)


@dataclass(frozen=True)
class EnvironmentTransition:
    """正式执行器返回的真实单 Slot 奖励及推进后的状态。"""

    reward: float
    next_state_features: np.ndarray | None
    next_state_fingerprint: str | None
    next_legal_candidates: tuple[CandidateOption, ...]
    terminated: bool
    duration_slots: int
    operation: str
    candidate_id: str
    final_cache_state: np.ndarray
    final_uav_positions: Mapping[str, np.ndarray]
    executor_calls: int
    matching_calls: int
    kkt_solver_calls: int
    state_isolation_pass: bool
    mutual_exclusion_pass: bool
    diagnostics: Mapping[str, Any]


def _input_fingerprint(slot_input: SlotInput) -> str:
    payload = (
        slot_input.slot_id,
        slot_input.time_s,
        slot_input.scenario_slice,
        slot_input.tasks,
        slot_input.vehicle_positions,
        slot_input.predicted_vehicle_positions,
        slot_input.prediction_horizon_mask,
        slot_input.prediction_metadata,
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


class ExclusiveDQNEnvironment:
    """按真实 Episode 顺序推进，每次只执行一个合法互斥配置。"""

    def __init__(
        self,
        *,
        context: Any,
        slot_inputs: Sequence[SlotInput],
        decision_steps: Sequence[Any],
        search_budget: SearchBudget | None = None,
        decision_count: int | None = None,
        terminal_at_end: bool = True,
    ) -> None:
        if not slot_inputs:
            raise ValueError("环境至少需要一个已对齐的正式 Slot 输入")
        if any(item.source_kind != "current_actual" for item in slot_inputs):
            raise ValueError("训练环境只接受当前实际 Slot，不接受未来真实输入冒充预测")
        self.context = context
        self.slot_inputs = tuple(slot_inputs)
        self.decision_forecasts = tuple(
            DecisionForecast.from_aligned_step(step) for step in decision_steps
        )
        if len(self.decision_forecasts) != len(self.slot_inputs):
            raise ValueError("正式对齐预测步骤与 Slot 输入数量不一致")
        if any(
            forecast.slot != slot_input.slot_id
            for forecast, slot_input in zip(
                self.decision_forecasts, self.slot_inputs, strict=True
            )
        ):
            raise ValueError("正式预测步骤与 Slot 输入未按同一时隙对齐")
        self.model = self.slot_inputs[0].model
        self.search_budget = search_budget or self._candidate_capacity_budget()
        self.decision_count = len(self.slot_inputs) if decision_count is None else int(decision_count)
        self.terminal_at_end = bool(terminal_at_end)
        if self.decision_count <= 0 or self.decision_count > len(self.slot_inputs):
            raise ValueError("决策 Slot 数量与对齐输入长度不匹配")
        if not self.terminal_at_end and self.decision_count >= len(self.slot_inputs):
            raise ValueError("截断窗口需要额外的下一 Slot 观测用于 Double DQN bootstrap")
        self.evaluator = None
        self.uav_scheduler = UavScheduler()
        self.cache_optimizer = CacheOptimizer(self.uav_scheduler)
        self.encoder = CandidateFeatureEncoder(
            self.slot_inputs[0],
            observation_spec=context.observation_spec,
            train_origin_start=int(context.train_origin_start),
            train_origin_end=int(context.train_origin_end),
        )
        self._initial_cache = np.asarray(context.initial_cache, dtype=np.int8).copy()
        self._initial_uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in context.initial_uav.items()
        }
        self.reset()

    def _candidate_capacity_budget(self) -> SearchBudget:
        """将 DQN 适配层上限设为正式候选生成器容量的自然上界。"""

        cache_node_count = len(tuple(self.model.cache_nodes["node_ids"]))
        cache_capacity = int(self.model.cache_config.target_candidate_capacity)
        uav_capacity = int(
            uav_configuration.uav_search_config_from_project(
                dict(self.model.project_cfg)
            ).target_candidate_capacity
        )
        # 缓存目标包含 KEEP、首提议组合及每节点提议；UAV 上限还包含 STAY。
        return SearchBudget(
            max_cache_candidates=2 + cache_node_count * cache_capacity,
            max_uav_candidates_per_uav=uav_capacity + 1,
        )

    @classmethod
    def from_formal_training_episode(
        cls,
        project_root: str | Path,
        *,
        episode_seed: int,
        origin_start: int = 0,
        max_steps: int = 2,
        search_budget: SearchBudget | None = None,
    ) -> "ExclusiveDQNEnvironment":
        """读取并验证当前正式 Training Pool 的一个短窗口。"""

        if int(max_steps) <= 0:
            raise ValueError("max_steps 必须大于 0")
        root = Path(project_root).resolve()
        training_scripts = root / "scripts" / "training"
        if str(training_scripts) not in sys.path:
            sys.path.insert(0, str(training_scripts))
        import training_pool_contract as pool_contract
        from ppo_simple.interface import aligned_train_steps

        pool, pool_info = pool_contract._validate_pool(
            root,
            pool_contract.TRAIN_POOL_RELATIVE,
            expected_split="train",
            expected_seed_start=pool_contract.DEFAULT_TRAINING_SEED_START,
            expected_episode_count=pool_contract.FORMAL_TRAINING_EPISODE_COUNT,
        )
        context = pool_contract._episode_context(root, pool, int(episode_seed), pool_info)
        steps = aligned_train_steps(
            context,
            origin_start=int(origin_start),
            max_steps=int(max_steps) + 1,
        )
        decision_count = min(int(max_steps), len(steps))
        terminal_at_end = (
            int(origin_start) + decision_count - 1 >= int(context.train_origin_end)
        )
        model = SystemModelContext(
            service_catalog=context.service_catalog,
            project_cfg=context.cfg,
            cache_nodes=context.cache_nodes,
            cache_config=context.cache_cfg,
            flight_boundary=context.flight_boundary,
        )
        slot_inputs = tuple(
            SlotInput(
                slot_id=int(step.slot),
                time_s=float(step.time_s),
                scenario_slice=step.scenario_slice,
                tasks=tuple(step.actual_tasks),
                vehicle_positions=step.actual_vehicle_positions,
                predicted_vehicle_positions=step.predicted_vehicle_positions,
                model=model,
                prediction_horizon_mask=step.prediction_horizon_mask,
                scenario_id=f"formal_train:{int(episode_seed)}:{int(step.slot)}",
                source_kind="current_actual",
                prediction_metadata={
                    "source": "formal_aligned_train_steps",
                    "episode_seed": int(episode_seed),
                    "origin_slot": int(step.slot),
                    "future_ground_truth_used_for_candidate_features": False,
                },
            )
            for step in steps
        )
        return cls(
            context=context,
            slot_inputs=slot_inputs,
            decision_steps=steps,
            search_budget=search_budget,
            decision_count=decision_count,
            terminal_at_end=terminal_at_end,
        )

    def reset(self) -> None:
        """回到该 realization 的初始配置和窗口起点。"""

        self.index = 0
        self.cache_state = self._initial_cache.copy()
        self.uav_positions = {
            key: value.copy() for key, value in self._initial_uav.items()
        }
        self._cached_state_fingerprint: str | None = None
        self._cached_state_features: np.ndarray | None = None
        self._cached_candidates: tuple[CandidateOption, ...] | None = None
        self._candidate_generation_diagnostics: dict[str, Any] = {}
        self.executor_calls = 0
        self.matching_calls = 0
        self.kkt_solver_calls = 0

    @property
    def terminated(self) -> bool:
        return self.index >= self.decision_count and self.terminal_at_end

    @property
    def can_act(self) -> bool:
        """窗口内是否还有需要执行动作的 Slot。"""

        return self.index < self.decision_count

    def observe(self) -> tuple[np.ndarray, tuple[CandidateOption, ...]]:
        """返回当前观测及与该状态匹配的全部有界合法候选。"""

        if not self.can_act:
            raise StopIteration("当前窗口没有更多待执行决策 Slot")
        return self._prepare_observation()

    def _prepare_observation(self) -> tuple[np.ndarray, tuple[CandidateOption, ...]]:
        if self.index >= len(self.slot_inputs):
            raise StopIteration("没有可用于当前状态或 bootstrap 的下一 Slot 输入")
        slot_input = self.slot_inputs[self.index]
        forecast = self.decision_forecasts[self.index]
        formal_observation = self._formal_observation(forecast)
        state_features = self.encoder.encode_state(formal_observation, forecast)
        fingerprint = self._decision_state_fingerprint(
            slot_input, forecast, state_features
        )
        if fingerprint != self._cached_state_fingerprint:
            input_before = _input_fingerprint(slot_input)
            cache_before = self.cache_state.copy()
            uav_before = {key: value.copy() for key, value in self.uav_positions.items()}
            self._cached_state_features = state_features
            self._cached_candidates = self._generate_legal_candidates(
                slot_input, forecast, fingerprint
            )
            generation_isolation = (
                _input_fingerprint(slot_input) == input_before
                and np.array_equal(cache_before, self.cache_state)
                and all(
                    np.array_equal(uav_before[key], self.uav_positions[key])
                    for key in uav_before
                )
            )
            if not generation_isolation:
                raise RuntimeError("候选生成意外修改了环境输入或当前配置状态")
            self._candidate_generation_diagnostics[
                "candidate_generation_state_isolation_pass"
            ] = True
            self._cached_state_fingerprint = fingerprint
        assert self._cached_state_features is not None
        assert self._cached_candidates is not None
        return self._cached_state_features.copy(), self._cached_candidates

    def _generate_legal_candidates(
        self,
        slot_input: SlotInput,
        forecast: DecisionForecast,
        source_state_fingerprint: str,
    ) -> tuple[CandidateOption, ...]:
        # 候选器只接收预测场景及配置，不接触当前真实任务和实际车辆轨迹。
        candidate_input = replace(slot_input, tasks=(), vehicle_positions={})
        if candidate_input.tasks or candidate_input.vehicle_positions:
            raise RuntimeError("候选生成输入未能隔离当前真实任务或车辆位置")
        candidates: list[CandidateOption] = []
        keep = ActionCandidate.from_states(
            previous_cache_state=self.cache_state,
            previous_uav_positions=self.uav_positions,
            target_cache_state=self.cache_state,
            target_uav_positions=self.uav_positions,
            slot_input=candidate_input,
            source_state_fingerprint=source_state_fingerprint,
            proposal_source="KEEP_STAY",
        )
        candidates.append(
            CandidateOption.create(
                keep,
                self.encoder.encode_candidate(
                    keep,
                    previous_cache_state=self.cache_state,
                    previous_uav_positions=self.uav_positions,
                    slot_input=candidate_input,
                    decision_forecast=forecast,
                ),
            )
        )
        cache_generation_diagnostics: dict[str, Any] = {}
        cache_targets, cache_generation_complete = self.cache_optimizer.generate_candidates(
            current_cache_state=self.cache_state,
            slot_input=candidate_input,
            search_budget=self.search_budget,
            generation_diagnostics=cache_generation_diagnostics,
        )
        for target in cache_targets:
            if not target.changed_node_ids:
                continue
            action = ActionCandidate.from_states(
                previous_cache_state=self.cache_state,
                previous_uav_positions=self.uav_positions,
                target_cache_state=target.target_state,
                target_uav_positions=self.uav_positions,
                slot_input=candidate_input,
                source_state_fingerprint=source_state_fingerprint,
                proposal_source=target.proposal_source,
            )
            candidates.append(
                CandidateOption.create(
                    action,
                    self.encoder.encode_candidate(
                        action,
                        previous_cache_state=self.cache_state,
                        previous_uav_positions=self.uav_positions,
                        slot_input=candidate_input,
                        decision_forecast=forecast,
                    ),
                )
            )
        uav_generation_diagnostics: dict[str, Any] = {}
        moves, uav_generation_complete = self.uav_scheduler.enumerate_move_candidates(
            previous_cache_state=self.cache_state,
            current_uav_positions=self.uav_positions,
            previous_uav_positions=self.uav_positions,
            slot_input=candidate_input,
            search_budget=self.search_budget,
            generation_diagnostics=uav_generation_diagnostics,
        )
        for move in moves:
            action = ActionCandidate.from_states(
                previous_cache_state=self.cache_state,
                previous_uav_positions=self.uav_positions,
                target_cache_state=self.cache_state,
                target_uav_positions=move.target_uav_positions,
                slot_input=candidate_input,
                source_state_fingerprint=source_state_fingerprint,
                proposal_source=move.proposal_source,
            )
            candidates.append(
                CandidateOption.create(
                    action,
                    self.encoder.encode_candidate(
                        action,
                        previous_cache_state=self.cache_state,
                        previous_uav_positions=self.uav_positions,
                        slot_input=candidate_input,
                        decision_forecast=forecast,
                    ),
                )
            )
        unique: dict[str, CandidateOption] = {}
        for option in candidates:
            if option.action.source_state_fingerprint != source_state_fingerprint:
                raise RuntimeError("候选动作来源状态与当前环境状态不一致")
            if option.action.candidate_id in unique:
                continue
            unique[option.action.candidate_id] = option
        if not unique or next(iter(unique.values())).action.operation != "KEEP_STAY":
            raise RuntimeError("合法候选集合必须始终包含 KEEP_STAY")
        self._candidate_generation_diagnostics = {
            "candidate_count": len(unique),
            "candidate_count_by_operation": {
                operation: sum(item.action.operation == operation for item in unique.values())
                for operation in ("KEEP_STAY", "CACHE_UPDATE", "UAV_MOVE")
            },
            "cache_candidate_generation_complete": bool(cache_generation_complete),
            "uav_candidate_generation_complete": bool(uav_generation_complete),
            "candidate_generation_complete_within_bounded_proposal_policy": bool(
                cache_generation_complete and uav_generation_complete
            ),
            "candidate_legal_action_space_exhaustive": False,
            "cache_candidate_generation_diagnostics": cache_generation_diagnostics,
            "uav_candidate_generation_diagnostics": uav_generation_diagnostics,
            "dqn_adapter_additional_candidate_truncation": False,
            "dqn_adapter_cache_candidate_limit": int(
                self.search_budget.max_cache_candidates
            ),
            "dqn_adapter_uav_candidates_per_uav_limit_including_stay": int(
                self.search_budget.max_uav_candidates_per_uav
            ),
            "candidate_generation_used_matching_or_kkt": False,
            "candidate_generation_received_actual_tasks": False,
            "candidate_generation_received_actual_vehicle_positions": False,
            "state_features_used_current_actual_task_data": False,
            "state_features_used_current_actual_vehicle_positions": False,
            "candidate_features_used_current_actual_vehicle_positions": False,
            "decision_state_feature_contract_version": self.encoder.state_feature_contract_version,
            "decision_candidate_feature_contract_version": self.encoder.candidate_feature_contract_version,
            "prediction_horizon_count": self.encoder.horizon_count,
            "prediction_horizon_mask": self.encoder._forecast_mask(forecast).tolist(),
        }
        return tuple(unique.values())

    def _formal_observation(self, forecast: DecisionForecast) -> np.ndarray:
        """复用正式 MAPPO 观测构造器，输入对象仅含当前决策可见的预测。"""

        from ppo_simple.interface import build_step_observation

        single_origin_override = None
        if (
            self.context.observation_spec.time_to_go_slice is not None
            and int(self.context.train_origin_end) == int(self.context.train_origin_start)
        ):
            single_origin_override = 0.0
        return build_step_observation(
            self.cache_state,
            self.uav_positions,
            forecast,
            self.context.observation_spec,
            tuple(self.model.flight_boundary),
            int(self.context.train_origin_start),
            int(self.context.train_origin_end),
            time_to_go_override=single_origin_override,
        )

    def _decision_state_fingerprint(
        self,
        slot_input: SlotInput,
        forecast: DecisionForecast,
        state_features: np.ndarray,
    ) -> str:
        prediction_context = {
            "origin_index": int(forecast.origin_index),
            "slot": int(forecast.slot),
            "prediction_horizon_mask": (
                None
                if forecast.prediction_horizon_mask is None
                else np.asarray(forecast.prediction_horizon_mask, dtype=np.int8)
            ),
            # 场景字段由正式预测池提供，候选生成器据此构造有限合法动作。
            "prediction_scenario": slot_input.scenario_slice,
            "predicted_vehicle_positions_horizon": forecast.predicted_vehicle_positions_horizon,
            "predicted_service_popularity_horizon": forecast.predicted_service_popularity_horizon,
        }
        return _state_fingerprint(
            self.cache_state,
            self.uav_positions,
            state_features,
            prediction_context,
        )

    def step(self, option: CandidateOption) -> EnvironmentTransition:
        """只把所选候选送入正式执行器，并推进一个真实 Slot。"""

        state, legal_candidates = self.observe()
        if not isinstance(option, CandidateOption):
            raise TypeError("step 只接受当前状态的 CandidateOption")
        match = next(
            (
                item
                for item in legal_candidates
                if item.action.candidate_id == option.action.candidate_id
            ),
            None,
        )
        if match is None or match.action.to_dict() != option.action.to_dict():
            raise ValueError("所选动作不是当前状态下生成的合法候选")
        candidate_generation_diagnostics = dict(self._candidate_generation_diagnostics)
        action = match.action
        if action.source_state_fingerprint != self._current_state_hash():
            raise ValueError("候选动作属于其他状态，拒绝执行")
        previous_cache = self.cache_state.copy()
        previous_uav = {key: value.copy() for key, value in self.uav_positions.items()}
        target_cache = action.cache_array()
        target_uav = action.uav_dict()
        changed_cache = bool(action.changed_node_ids)
        moved_uav = bool(action.moved_uav_ids)
        if changed_cache and moved_uav:
            raise ValueError("候选动作违反缓存与 UAV 互斥约束")
        slot_input = self.slot_inputs[self.index]
        input_hash = _input_fingerprint(slot_input)
        context_cache_before = np.asarray(self.context.initial_cache, dtype=np.int8).copy()
        context_uav_before = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in self.context.initial_uav.items()
        }
        cache_triggers = np.any(target_cache != previous_cache, axis=1).astype(np.int8)
        tolerance = float(
            self.model.project_cfg.get("algorithm", {})
            .get("uav", {})
            .get("position_tolerance_m", 1.0e-6)
        )
        uav_triggers = {
            uav_id: int(float(np.linalg.norm(target_uav[uav_id] - previous_uav[uav_id])) > tolerance)
            for uav_id in previous_uav
        }
        kkt_cache = task_allocation.KKTResourceAllocationCache()
        performance_profile: dict[str, float] = {}
        result = slot_environment.execute_configured_slot(
            previous_cache,
            previous_uav,
            target_cache,
            target_uav,
            dict(slot_input.scenario_slice),
            list(slot_input.tasks),
            dict(slot_input.vehicle_positions),
            dict(self.model.service_catalog),
            dict(self.model.project_cfg),
            dict(self.model.cache_nodes),
            self.model.cache_config,
            tuple(self.model.flight_boundary),
            slot_id=int(slot_input.slot_id),
            time_s=slot_input.time_s,
            performance_profile=performance_profile,
            matching_kkt_cache=kkt_cache,
            node_evaluation_cache=task_allocation.NodeEvaluationCache(),
        )
        if not result.step_feasible or result.system_utility is None:
            raise RuntimeError(
                "正式 Slot 执行未产生可用转移："
                f"slot={slot_input.slot_id}, reason={result.infeasible_reason}"
            )
        if result.next_cache_state is None or result.next_uav_positions is None:
            raise RuntimeError("正式 Slot 执行器未返回下一缓存/UAV 状态")
        actual_matching = getattr(result, "matching", None)
        actual_execution_node_data_available = isinstance(actual_matching, Mapping)
        actual_execution_nodes = (
            [
                {
                    "task_id": str(task_id),
                    "execution_node": str(execution_node),
                }
                for task_id, execution_node in sorted(
                    actual_matching.items(), key=lambda item: str(item[0])
                )
            ]
            if actual_execution_node_data_available
            else None
        )
        actual_execution_node_counts = (
            dict(
                sorted(
                    Counter(
                        str(execution_node)
                        for execution_node in actual_matching.values()
                    ).items()
                )
            )
            if actual_execution_node_data_available
            else None
        )
        final_cache = np.asarray(result.next_cache_state, dtype=np.int8).copy()
        final_uav = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in result.next_uav_positions.items()
        }
        if not np.array_equal(final_cache, target_cache) or any(
            not np.array_equal(final_uav[key], target_uav[key]) for key in target_uav
        ):
            raise RuntimeError("执行器推进后的配置与已选候选不一致")
        actual_cache_changed = not np.array_equal(previous_cache, final_cache)
        actual_uav_moved = any(
            float(np.linalg.norm(final_uav[key] - previous_uav[key])) > tolerance
            for key in previous_uav
        )
        mutual_exclusion_pass = not (actual_cache_changed and actual_uav_moved)
        if not mutual_exclusion_pass:
            raise RuntimeError("正式执行结果同时改变缓存与 UAV 位置")
        isolation_pass = (
            _input_fingerprint(slot_input) == input_hash
            and np.array_equal(context_cache_before, self.context.initial_cache)
            and all(
                np.array_equal(context_uav_before[key], self.context.initial_uav[key])
                for key in context_uav_before
            )
            and np.array_equal(previous_cache, self.cache_state)
            and all(np.array_equal(previous_uav[key], self.uav_positions[key]) for key in previous_uav)
        )
        if not isolation_pass:
            raise RuntimeError("候选执行意外修改了输入 realization 或执行前状态")
        executor_calls = 1
        matching_calls = int(bool(result.matching_summary))
        kkt_calls = int(kkt_cache.solver_calls)
        self.executor_calls += executor_calls
        self.matching_calls += matching_calls
        self.kkt_solver_calls += kkt_calls
        self.cache_state = final_cache
        self.uav_positions = final_uav
        self.index += 1
        self._cached_state_fingerprint = None
        self._cached_state_features = None
        self._cached_candidates = None
        done = self.terminated
        next_state: np.ndarray | None = None
        next_state_fingerprint: str | None = None
        next_candidates: tuple[CandidateOption, ...] = ()
        if not done:
            next_state, next_candidates = self._prepare_observation()
            next_state_fingerprint = self._current_state_hash()
        return EnvironmentTransition(
            reward=float(result.system_utility),
            next_state_features=next_state,
            next_state_fingerprint=next_state_fingerprint,
            next_legal_candidates=next_candidates,
            terminated=done,
            duration_slots=1,
            operation=action.operation,
            candidate_id=action.candidate_id,
            final_cache_state=final_cache.copy(),
            final_uav_positions={key: value.copy() for key, value in final_uav.items()},
            executor_calls=executor_calls,
            matching_calls=matching_calls,
            kkt_solver_calls=kkt_calls,
            state_isolation_pass=isolation_pass,
            mutual_exclusion_pass=mutual_exclusion_pass,
            diagnostics={
                "slot_id": int(slot_input.slot_id),
                "actual_task_utility": float(result.actual_task_utility),
                "cache_cost": (
                    None if result.cache_cost is None else float(result.cache_cost)
                ),
                "cache_update_cost_available": result.cache_cost is not None,
                "movement_cost": (
                    None
                    if result.movement_cost is None
                    else float(result.movement_cost)
                ),
                "uav_movement_cost_available": result.movement_cost is not None,
                "beta_ca": float(result.beta_ca),
                "beta_q": float(result.beta_q),
                "matching_summary": result.matching_summary,
                "actual_execution_node_counts": actual_execution_node_counts,
                "actual_execution_nodes": actual_execution_nodes,
                "actual_execution_node_data_available": (
                    actual_execution_node_data_available
                ),
                "executor_performance_profile": performance_profile,
                "duration_semantics": "每个动作恰好执行一个正式 Slot",
                **candidate_generation_diagnostics,
                "state_features_used_future_actual_data": False,
                "state_features_used_current_actual_task_data": False,
                "candidate_generation_received_actual_tasks": False,
                "candidate_generation_received_actual_vehicle_positions": False,
                "candidate_features_used_current_actual_vehicle_positions": False,
                "decision_state_feature_contract_version": self.encoder.state_feature_contract_version,
                "decision_candidate_feature_contract_version": self.encoder.candidate_feature_contract_version,
                "prediction_horizon_count": self.encoder.horizon_count,
                "prediction_horizon_mask": self.encoder._forecast_mask(
                    self.decision_forecasts[self.index - 1]
                ).tolist(),
            },
        )

    def _current_state_hash(self) -> str:
        if self.index >= len(self.slot_inputs):
            return _state_fingerprint(self.cache_state, self.uav_positions)
        slot_input = self.slot_inputs[self.index]
        forecast = self.decision_forecasts[self.index]
        formal_observation = self._formal_observation(forecast)
        state_features = self.encoder.encode_state(formal_observation, forecast)
        return self._decision_state_fingerprint(
            slot_input, forecast, state_features
        )


__all__ = ["EnvironmentTransition", "ExclusiveDQNEnvironment"]
