# adaptive_cache_mec · Codex 项目协作规则

## 项目定位与证据优先级
- 项目研究车辆边缘计算与服务缓存：车辆/任务/预测数据 → DQN Cache/UAV 配置决策 → Transfer–Exchange Matching → KKT 资源分配 → 时延/能耗/成本。
- 当前源码和当前任务指定的配置是事实依据；历史阶段报告、Checkpoint 和聊天中的版本号只作为线索。先确认实际文件与版本，不能将历史 Architecture/Schema 当作现行正式状态。
- 同一文件或资料存在不同上传/落地版本时，以最新版本为准；除非任务要求比较历史版本。区分代码事实、实验结果、文献依据和推断；不编造结论。

## 环境
- Conda 环境名：`adaptive_cache_mec`。
- 优先 `conda activate adaptive_cache_mec`；若 Windows 终端无法执行 Conda 入口，使用已确认的解释器 `C:\software\Anaconda_envs\envs\adaptive_cache_mec\python.exe`，可通过 `sys.prefix` 复核。不要切换至 base，也不要擅自安装新依赖。
- 当前 Training Pool：`environment_data/proposed_joint_training/`。按当前 manifest、配置及任务要求确定数据协议；默认不访问 Validation/Test。

## 修改边界
- 开始修改前明确目标、允许修改的文件、禁止修改的合同；优先最小实现，避免顺带重构、清理无关历史代码或扩大审计。
- 以 `configs/dqn/online_joint_dqn.toml` 作为 DQN 的共享物理系统配置；动作候选合同、网络架构和 Checkpoint 版本必须从当前 DQN 代码核实。
- 未经明确要求，不修改 DQN 配置、训练入口、Training Pool、Episode manifest、Reward、Cache/UAV 惩罚权重或已有 Run/Checkpoint。
- 新通信合同和新策略实验优先用独立版本化配置与输出目录；不要用不同通信配置的旧 Checkpoint 伪装为严格 Resume，不绕过版本或配置哈希检查。
- 不主动运行耗时 DQN 训练、大规模数据生成、完整 Episode 重放或 Matching/KKT 对照；只有任务明确授权时才执行。

## Python 编码风格与数值合同
- 新增和修改Python代码遵守 `docs/CODING_STANDARDS.md`，必要时调用 `$adaptive-python-coding-standards`；该规范不授权对旧代码做无关格式化或全局重构。
- 尤其注意物理量单位、Tensor shape、KKT导数/资源合同、DQN Q 值与 Replay 合同、报错可见性、配置及Checkpoint兼容性。

## 编码与验证
- 修改 Python 代码时遵守 `docs/CODING_STANDARDS.md`；整合配置时保持 DQN 入口与共享配置一致；研究通信公式与物理依据时调用 `$adaptive-model-evidence`。
- **验证按风险和改动范围最小化**：优先已有加载/编译或窄范围检查。只有确有行为变更风险、任务要求或既有回归保护时才运行对应的小测试；不要自动创建大量测试脚本或全量测试。
- 工作结束删除**本次创建**的临时脚本、临时配置、缓存及中间文件；不要误删既有用户数据、报告、Run 或项目文件。
- 默认不 `git commit`、不 `git push`、不覆盖已有输出目录。新文件要清楚标明阶段、来源和实验合同。
- 最终只汇报实际改动、验证结果、尚存风险和下一步；不将“未验证”报告为“通过”。
