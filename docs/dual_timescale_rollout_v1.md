# 双时间尺度缓存—UAV 联合滚动优化（研究原型 v1）

## 目标与不可变合同

任务车辆 TV 不本地执行。SV/RSU/UAV/Cloud 执行与现有 Edge 带宽依赖噪声通信模型、Cloud 每任务 20 Mbps 上传、13 倍时延、Cloud 共享 40 GHz CPU 一致。保留正式奖励 `G(t)=Q(t)-C_cache(t)-C_UAV(t)`、正式 Cache/UAV 成本函数与时隙独立任务完成语义。不修改 PPO 主线、模型参数、正式训练池。

该研究分支基于远程 `research/joint-resource-oracle-v1`，**不包含用户在本地 worktree 尚未提交的 `assignment_optimizer.py` 搜索改进**。正式比较前必须通过本地保存 diff / 在另一独立 worktree 应用 diff，验证没有冲突与覆盖；不要擅自丢弃已有未提交修改。

## 本阶段新增或扩展的接口

1. `src/decision/dual_timescale_controller.py`：
   - `initialize_controller(cache, uav_positions)` 建立显式状态；
   - `decide_dual_timescale_action(state, origin_scenario, ...)` 只接收**预测场景**而不接收实际未来任务；
   - Macro Cache 在固定 `cache_hold_slots=1..5` 帧边界可进行预测驱动 KEEP/UPDATE，帧内冻结；
   - Micro UAV 每时隙重新生成 STAY/MOVE 候选、校验区域边界/位移/安全间距；
   - 复用正式 `enumerate_all_cache_target_configurations` / `enumerate_uav_target_candidates` 获取候选，通过资源响应预测值进行联合评分；
   - 首动作执行、保存缓存/UAV 状态和持有计数，下一 slot 重新规划；
   - 保留若干联合缓存更改和多 UAV 联动候选，默认最大候选笛卡尔积 64。

2. `src/decision/joint_held_optimizer.py`：新增 `horizon_mask` 并传入预测任务资源评价，控制计算预算。每个候选在预测时仍假定“本次 UAV 目标位置在预测窗口内保持不变”，**不预知下一时隙实际 UAV 决策**。实际 Micro 决策下一时隙会重新规划。

3. `scripts/research/run_dual_timescale_short_rollout.py`：独立最多 5 个 Training 连续时隙的命令行入口。严格先预测决策，再读取当前真实任务与位置。每个时隙都以正式 `execute_configured_slot` 得到 Matching+KKT 实际净效用，又在**相同当前配置、相同真实任务**上评价独立任务分配器的净效用。

## 四种最小可比控制模式

- `--mode keep_stay`：缓存 KEEP，UAV STAY；
- `--mode cache_only`：允许 Macro Cache UPDATE，UAV STAY；
- `--mode uav_only`：缓存 KEEP，允许 Micro UAV MOVE；
- `--mode joint`：允许 Macro Cache 和 Micro UAV 联合选择。

所有模式必须固定同一 Episode、相同初始状态、相同预测源、时隙数、算法预算、CPU/带宽物理合同。用于验证控制链路与发现效用/计算开销差异，**不能以 2 个 slot 宣称长期效果**。

## 在真实 Conda 环境执行（由 Codex 进行）

```powershell
conda activate adaptive_cache_mec
# 在确保不会覆盖当前未提交变更的条件下，将远程研究分支检出至独立 worktree。
# 先把上一轮本地 assignment_optimizer.py 的修改安全地带入该 worktree；
# 若无法取得源 worktree 的修改，请停止并报告，不能把旧版算法标注为新改进版。

python scripts/research/run_dual_timescale_short_rollout.py --realization-dir environment_data/proposed_joint_training/episode_20001 --mode keep_stay --slots 2 --cache-hold-slots 2 --horizon 2
python scripts/research/run_dual_timescale_short_rollout.py --realization-dir environment_data/proposed_joint_training/episode_20001 --mode cache_only --slots 2 --cache-hold-slots 2 --horizon 2
python scripts/research/run_dual_timescale_short_rollout.py --realization-dir environment_data/proposed_joint_training/episode_20001 --mode uav_only --slots 2 --cache-hold-slots 2 --horizon 2
python scripts/research/run_dual_timescale_short_rollout.py --realization-dir environment_data/proposed_joint_training/episode_20001 --mode joint --slots 2 --cache-hold-slots 2 --horizon 2
```

若 worktree 位于正式项目根目录之外，且依赖 `environment_data/` 未被 Git 跟踪，`--realization-dir` 使用现有实际 Episode 的**绝对路径**；正式配置引用的 service catalog、SUMO 地图等依赖也必须来自现有正确的项目文件，不能让 Codex 为了跑通虚构或重建数据。可以在受控工作树中正确映射已有数据路径，禁止改写正式 artifacts。

先只运行一个模式且最多 2 slot，核对 CLI、依赖与配置成本后，再继续其余 3 种；如发生错误，定位并修复明确的接口问题，不启动多 Episode 训练或无关审计。

## 重要的理论/工程边界

1. 这是**预测驱动、固定 Macro 帧长 + 每时隙 Micro UAV 再优化**的双时间尺度滚动启发式，尚没有实现由预测收益主动选择最优帧长或 PPO 的 SMDP termination 语义，也没有未来 UAV 决策的场景树非预知追索。
2. 最终缓存变更成本和当前 UAV 移动成本各扣一次，未来可能发生的重新配置成本没有纳入此 held-configuration 价值函数；其预测净效用不是完整多阶段 MPC 目标的精确价值函数。
3. 候选搜索具有计算预算和工程性排序，会遗漏可能优于当前解的组合。KEEP+STAY 保留为共同基线，但真实结果不保证单调。
4. Cloud CPU 用满 40 GHz 不独立证明资源瓶颈。报告实际 Oracle 调用数及底层 KKT 求解次数，而不是只报告候选数量。
5. F=10 的正式服务数允许使用现有穷举 Cache 目标生成器；若以后扩大 F，该候选生成器具有指数增长风险，需改为增量邻域/启发式候选。
6. 当前新分支不能直接与上一轮本地优化器结果混称同一个版本；先解决代码版本对齐。
7. 仅使用必要的短时隙验证，不跑 PPO，不改 main，不合并研究 PR；任务结束后删除 Codex 创建的临时补丁/临时脚本和产物，保留研究模块和已有用户工作区内容。
