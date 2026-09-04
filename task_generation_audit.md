# Task generation audit

审计日期：2026-09-04  
审计范围：只读取当前代码和正式配置，静态分析
`src/scenario/task_scenario.py` 与 `src/scenario/scenario_generator.py`。

本审计没有运行任务生成器、场景生成器或实验，也没有读取并重新统计
`tasks.csv` 等生成数据。因此，下面的 input size min/max 是由代码和当前配置
推导出的理论范围，不是某一次随机运行的经验 min/max。

## 结论摘要

| 路径 | service_id 的来源 | 是否 prediction-driven |
|---|---|---|
| 正式真实任务：`task_scenario.generate_tasks` | 当前时隙 `rho_star` 的 Categorical 抽样；`rho_star` 来自动态 Zipf 排名 | 否 |
| 预测场景：`scenario_generator.sample_tasks` | 预测分布 `rho_hat` 的 Categorical 抽样 | 是 |

当前正式配置下，真实任务不是 uniform service sampling，而是 `zeta=0.8`
的 popularity/Zipf sampling。输入数据量是 service catalog 的分层均匀基准值乘以
独立的 ±20% 相对扰动；cycles 是输入数据量乘以 service 的
`cycles_per_bit` 再乘以独立的 ±10% 负载扰动。

因此：

- 如果 `computation_density` 定义为每个任务的有效密度
  `cycles_per_bit * load_factor`，则 `cycles = input_size * computation_density`
  对有效任务严格成立。
- 如果 `computation_density` 只指 catalog 中固定的 `cycles_per_bit`，则不严格
  成立，因为代码额外乘了每任务的 `load_factor`。

## 当前正式任务参数

`task_config_from_project_config` 将 `system_config.toml` 映射为任务生成配置；当前
正式值为：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| service 数量 | 20 | service catalog 和概率向量维度 |
| `task.arrival_probability` | 0.7 | 每个 task vehicle、每个 slot 的 Bernoulli 到达概率 |
| `task.popularity_zeta` | 0.8 | Zipf 指数 |
| popularity epoch length | 20 slot | rank 演化边界 |
| 每 epoch rank 变化次数 | 1 | 局部 rank swap/shift 次数 |
| `input_relative_jitter` | 0.20 | input size 的相对扰动 |
| `load_relative_jitter` | 0.10 | cycles 的负载相对扰动 |
| `mean_input_size_min_mbit` / max | 0.5 / 1.5 Mbit | service 基准 input size 范围 |
| `cycles_per_bit_min` / max | 200 / 1000 cycle/bit | service 基准计算密度范围 |

注：`TaskConfig` 中的 `arrival_probability=0.5` 等数值是无 TOML 时的回退默认值；
正式入口读取当前 `system_config.toml`，所以本审计对正式配置采用 0.7。

## 1. service_id 生成方式

### 1.1 正式真实任务

`src/scenario/task_scenario.py` 的流程是：

1. `make_popularity_trace_with_history` 先将 service ID 做一次随机初始排名。
2. `zipf_probabilities` 对排名第 `r` 的 service 使用：

   ```text
   weight(r) = r^(-zeta)
   rho_star[service_id] = weight(rank) / sum_j weight(j)
   ```

3. 每个 popularity epoch（当前为 20 slot）只做少量局部相邻交换或移动，形成
   动态排名。
4. `generate_tasks` 对每个 task vehicle、每个 slot 先做 arrival Bernoulli 抽样；
   只有 arrival=1 时才执行：

   ```text
   service_id ~ Categorical(probabilities_by_slot[slot])
   ```

   这里的 `probabilities_by_slot` 就是该 slot 的 `rho_star`。

结论：

- 不是 uniform service sampling（当前 `zeta=0.8 > 0`，各 rank 概率不同）。
- 是 popularity-based / Zipf-based sampling。
- 初始排名本身使用随机 permutation，但这只随机化“哪个 service 占据哪个 rank”，
  不会把最终请求分布变成 uniform。
- 真实任务生成不读取 prediction，也不使用 `rho_obs` 反向生成当前任务；观察到的
  request count/`rho_obs` 是任务请求发生后再计算或附加的结果。

### 1.2 预测场景任务

`src/scenario/scenario_generator.py` 的 `sample_tasks` 接收
`rho_hat[origin, horizon, service]`，先将该向量归一化，再执行：

```text
service_id ~ Categorical(rho_hat[origin, horizon])
```

`rho_hat` 由 `predict_rho_hat` 提供，来自 popularity prediction/GRU；代码元数据
明确记录预测场景的 task 是 sampled from `rho_hat`, not `rho_star`。因此预测场景
的 service selection 是 prediction-driven，但它不是正式真实 `tasks.csv` 的生成
路径。

### 1.3 分类

```text
正式真实任务：popularity_based + Zipf
预测场景任务：prediction_driven（以预测 popularity rho_hat 为概率）
uniform：否
```

## 2. input data size 生成

### 2.1 service catalog 基准值

`make_service_catalog` 对 `mean_input_size` 使用
`stratified_uniform_samples`：在 `[0.5, 1.5]` Mbit 的 20 个等宽 strata 中每个
抽样一次，然后用独立 permutation 分配给 service ID，最后转换为 bit：

```text
mean_input_size_f = sampled_mbit_f * 1,000,000
```

这不是 20 个完全独立的普通 uniform 样本，而是覆盖整个区间的分层均匀样本；
service ID 与 input size 没有固定的大小排序语义。

### 2.2 每个有效任务的 input size

两条任务生成路径都使用同一个公式：

```text
input_factor ~ Uniform(-0.20, +0.20)
input_size = mean_input_size_f * (1 + input_factor)
```

所以：

| 层次 | min | max | 分布 |
|---|---:|---:|---|
| catalog `mean_input_size` | 0.5 Mbit（500,000 bit） | 1.5 Mbit（1,500,000 bit） | 分层均匀，每个 service 一个基准值 |
| 有效任务 `input_size` | 约 0.4 Mbit（400,000 bit） | 小于约 1.8 Mbit（1,800,000 bit） | 条件于 service 的乘法 Uniform ±20%，跨 service 为混合分布 |

NumPy 的 `uniform(low, high)` 使用下界包含、上界不包含；因此按当前实现写成
`400,000 <= input_size < 1,800,000 bit` 更精确。实际一次运行的精确 min/max
还会受 catalog 抽样、请求 service 分布和 jitter 随机值影响。

边界内的边缘情况：arrival=0 时不构造任务，正式任务行写入
`service_id=""`、`input_size=0`；预测场景数组写入 `service_id=-1`、
`input_size=0`。上面的 input size 范围只针对 arrival=1 的有效任务。

## 3. computation cycles 生成

对 arrival=1 的任务，代码公式为：

```text
load_factor ~ Uniform(-0.10, +0.10) 后加 1
cpu_cycles = service.cycles_per_bit * input_size * load_factor
```

展开 input size 后：

```text
cycles
  = cycles_per_bit_f
    * mean_input_size_f
    * (1 + input_factor)
    * (1 + load_factor_noise)
```

其中：

- `cycles_per_bit_f` 是 catalog 中按 service 固定的基准值，范围为
  200–1000 cycle/bit，并采用分层均匀生成后独立分配到 service ID。
- `load_factor` 的实际乘数范围为 `[0.9, 1.1)`。
- `input_size` 单位是 bit，`cpu_cycles` 单位是 cycle。

等价地，代码隐含的每任务有效 computation density 为：

```text
computation_density_effective
  = cpu_cycles / input_size
  = cycles_per_bit_f * load_factor
```

其理论范围约为 180–1100 cycle/bit（上端按 NumPy 半开区间理解）。

## 4. task arrival probability 的作用

在 `task_scenario.generate_tasks` 中，每个 slot 对每个 task vehicle 独立执行：

```text
arrival_i,t ~ Bernoulli(task.arrival_probability)
```

当前正式值为 `p=0.7`。因此，在 30 个 task vehicles 的正式配置下，每个 slot 的
期望到达数为：

```text
E[arrivals per slot] = 30 * 0.7 = 21
```

它的作用是门控后续生成流程：

- arrival=0：不抽 service_id，不抽 input jitter/load jitter；input size 和 cycles
  为 0。
- arrival=1：才按照当前 `rho_star`（真实任务）或 `rho_hat`（预测场景）抽 service，
  然后生成 input size 和 cycles。

所以 arrival probability 不决定 service popularity，也不改变 Zipf 指数；它只决定
每个 vehicle-slot 是否产生任务，并间接改变单位 slot 的请求数量。

预测场景中同样在每个 scenario、horizon、TV 维度执行该 Bernoulli 抽样，并把实际
到达率记录为诊断统计；这不会改变 service probability 向量的来源。

## 5. `cycles = input_size * computation_density` 检查

| 对 `computation_density` 的定义 | 是否成立 | 原因 |
|---|---|---|
| 有效的每任务密度 `cycles_per_bit * load_factor` | 是，严格成立 | 代码直接计算 `cycles = input_size * (cycles_per_bit * load_factor)` |
| catalog 固定密度 `cycles_per_bit` | 否，不严格成立 | 每个任务还有独立的 `load_factor`，所以 `cycles / input_size` 会变化 |
| arrival=0 的占位行/数组元素 | 仅形式上为 0 | 没有有效 task；代码写入空/`-1` service、0 input、0 cycles |

最终判断：当前模型是“输入大小 × 有效计算密度”的比例模型；它不是“输入大小 ×
固定 catalog 计算密度”的严格模型。`load_relative_jitter` 只改变有效密度，不会
破坏 cycles 对 input size 的线性结构。

## 6. 源码定位

- `C:\Users\tangning\Desktop\code\adaptive_cache_mec\src\scenario\task_scenario.py`
  - `stratified_uniform_samples`
  - `make_service_catalog`
  - `zipf_probabilities`
  - `make_popularity_trace_with_history`
  - `generate_tasks`
  - `task_config_from_project_config`
- `C:\Users\tangning\Desktop\code\adaptive_cache_mec\src\scenario\scenario_generator.py`
  - `predict_rho_hat`
  - `task_config_from_catalog`
  - `sample_tasks`
  - `online_information_boundary` 元数据

