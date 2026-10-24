# continuous_v2 交通数据生成与质量验收

## 结果

- 数据集：environment_data/continuous_v2/；状态：complete，131/131 Episode 完成。
- Train：单条连续轨迹 [0,90400)，400 秒预热，正式 [400,90400)，90,000 秒、300 个 300 秒窗口。
- Validation：30 个独立 Episode，每条 [0,700)，正式 [400,700)。
- Test：100 个独立 Episode，每条 [0,700)，正式 [400,700)。
- 时间步长 1 秒；路线计划发车间隔 6 秒。
- 实际生成耗时 398.752 秒 (6.65 分钟)，SUMO 合计 181,400 秒。
- 实际目录占用 372,337,069 字节 (0.372 GB)；FCD 337,399,947 字节，route 12,645,025 字节，检查点 1,344,367 字节。

## 配置、种子与哈希

- 路网和车辆物理/信号参数沿用已核实的 DQN/SUMO 网络；未修改正式 DQN 配置。
- 独立生成配置 environment_data/continuous_v2_generation_config.toml 只覆盖 warmup_time_s=400 和 random_trips_period_s=6；其他 TOML 值与正式配置相同。
- 生成配置 SHA-256：71b02c2f1570f9c134fca74ed21bb0ba6202e3d9043ecd8e535e644ebf763776；正式 DQN 配置 SHA-256：3800487e13a5052d218ffb597158c5fdc6ceef3661560014d8a8f179fc2f8833。
- 网络 SHA-256：30379989e0f94505e30ed5ec27449367615847ab4baf1bfe2340d8c405388c7d；randomTrips.py SHA-256：e6d794ef1bce059b46ef65121d5e47a2df35603afa113015483dce74ae2ec01c；生成器 SHA-256：4c3e51b117d9ac68d2b96b14cdd7181b019be3b8062bb873b0cf3bafc5e12886。
- 种子合同文件 SHA-256：70e6aea73efb23c929cecd2372d9c155b4aebea25c3bcfb59750c742f686f26d；Train root seed=20001，Validation=30001–30030，Test=40001–40100；root 与 SUMO/traffic 子种子互不重叠。
- Dataset Manifest SHA-256：39fc420df6bd2d40a390ba7f945abf6289ef154bcf0183db10fecf77f9b2e06d；131 条 route 哈希唯一。逐 Episode route/trips/FCD/summary/activity/episode manifest SHA-256 见各自 Manifest。
- 离线质量摘要 SHA-256：4a536a51d931a8f00a4665698e5616fcc014564a73a4aa5c9c9ca86ea7527e42。
- Train route 实有 15,067 辆，depart=0–90,396 秒，每个间隔恰为 6 秒；300 个正式窗口各计划 50 辆，含最后一窗。
- Validation/Test 每条 route 实有 117 辆，depart=0–696 秒；每条正式窗口计划 50 辆。

## 交通质量

- Train 全程计划 15,067 辆，最终插入 15,067，未插入 0；到达 14,952，结束时在路网内 115；终点 waiting=0，累计 teleport=0。
- Train 正式期平均速度 9.572 m/s；最大 waiting=3，halting 峰值=93。
- 50 个 30 分钟 Train 窗口：前半/后半平均速度 9.546/9.598 m/s（变化 0.54%）；到达 7,448/7,504，插入 7,500/7,500（计划各 7,500）；最大 waiting=3，teleport=0。完整逐窗数值在 quality_validation.json。
- Train 300 个正式窗口完整车辆数（每个正式秒步有有效 FCD 记录且至少一条预热记录）：min/median/max=51/62.0/81；≥25 辆 300/300；首窗=51，末窗=65。
- Validation：≥25 条完整正式轨迹 30/30；min/median/max=45/51.0/55。
- Test：≥25 条完整正式轨迹 100/100；min/median/max=44/50.5/57。
- Validation/Test 正式期插入增量/本期计划发车比（可能计入预热期排队车辆） 100.00%–102.00%；waiting 峰值最大 2；平均速度 9.671–10.779 m/s；halting 峰值最大 73；teleport 总数 0。
- FCD 时间戳逐秒连续、无缺失/重复/乱序；逐 timestep 活动车辆行数与 SUMO summary running 数相等。每个 route ID 都有 activity 记录；仿真结束仍活动的车辆保留 active 状态，没有伪造到达记录。
- SUMO FCD 实际提供 lane ID（由 lane ID 可恢复 edge ID）和位置/速度/加速度等字段；没有单独的 edge 属性，route_edges 来自对应的 sumo_demand.rou.xml。

## 断点恢复

- 生成器自带 60 秒恢复烟测在 30 秒受控中断后按同 route/seed 重建；不中断与恢复运行的 FCD 语义 SHA-256 相同，烟测临时文件已清理。
- 正式 131 个 Episode 本次未中断；生成器按 Episode 原子发布，并从固定 seed 和 route 重建中断段。这不是 SUMO 指令级精确状态恢复。

## MEC 接入审计与阻塞

- continuous_v2 是动态 ID 的 FCD XML gzip 与 vehicle activity，不是当前 DQN realization。现有入口要求 trajectory.csv、vehicle_roles.json、tasks.csv 和预测 CSV/NPZ，并按固定车辆 ID 列表和逐 Slot ID join 构建 observation。
- 当前 Environment reset 重置 rollout index、缓存与 UAV 初态；训练 Episode 在 300 Slot 后终止，下一 Episode 使用新 realization。不存在跨 300 秒边界的活动任务状态容器。
- 当前任务行/Task 只含 Slot 到达、服务、输入量、CPU cycles 等值；没有剩余上传量、剩余计算量或跨 Slot 活跃任务账本。
- 正式车辆选择只接受完整覆盖整个 formal Episode 且有预热记录的 ID；车辆离场时活动任务的继续、迁移、清算或失败规则没有定义。因此本轮没有修改 MEC 接口，也没有自创离场/重置语义。
- 后续接入最小工作：先确定离场与活动任务处置合同，再新增绝对时间 FCD/activity reader、ID-keyed 动态状态映射、任务账本和跨窗口 terminal/reset 生命周期适配。

## 运行边界与清理

- 未读取 continuous_v1 Train FCD；continuous_v1 未覆盖。Train route 与 v1 的原始输入路线不同，且本次生成只写入 continuous_v2 和独立生成配置。
- 生成器 Manifest 中的资源估算基于 period=1 秒；其 7.06 GB 估算不是本次实测。本报告以实际字节数和已验证 route 发车数为准。
- 恢复烟测临时目录已由生成器删除；没有保留临时脚本、临时路线或中间 FCD。
- 本次没有启动 DQN/MAPPO，没有调用 Matching/KKT，也没有启动其他实验。

逐 Episode 计数与每个 300 秒 Train 窗口结果见 environment_data/continuous_v2/quality_validation.json；route、FCD、episode manifest 及其他 artifact SHA-256 见各 Episode Manifest。
