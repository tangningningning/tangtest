# 执行层参数校准说明

本文件对应：

```text
data/audit_execution_balance.py
```

## 目的

当前已经完成：

```text
SV + RSU + UAV + Cloud
    +
P1 + P2 + Pjoint + Async + P3
```

完整结构接入。

下一阶段不再增加模型模块，而是校准当前 pilot 参数，首先检查 Cloud 是否相对边缘过强。

## `audit_execution_balance.py` 做什么

它不会修改任何配置，只读取现有：

```text
src/config.py
task_trace
mobility trace
service catalog
RSU deployment
UAV 初始位置
```

然后对每个任务计算：

### 1. 当前 Cloud 代价

使用当前 P3 的 Cloud 公式。

### 2. 边缘满资源代价下界

对每个 SV / RSU / UAV：

```text
w = 1
b = 1
```

因此：

\[
J_{edge}^{LB}=A+B.
\]

真实共享资源只可能使边缘代价更高，所以这是判断“边缘是否具备基本竞争力”的保守下界。

### 3. 两种边缘情况

`all-service`：

假设服务全部可用，只检查通信和计算能力。

`initial-cache`：

使用当前真实初始缓存，进一步检查缓存门的影响。

### 4. Cloud psi break-even

保持当前：

```text
CLOUD_RATE_BPS
CLOUD_CPU_HZ
```

不变，计算每个任务对应的临界：

\[
\psi_c^\star
\]

使：

\[
J_{cloud}(\psi_c^\star)
=
\min_m J_{edge,m}^{LB}.
\]

这个数只用于诊断，不是自动推荐参数。

## 输出

```text
results/execution_balance_audit.csv
```

以及控制台摘要。

## 当前阶段的正确使用方式

先运行：

```powershell
python data\audit_execution_balance.py
```

根据结果再决定 Cloud 参数调整方向。

不要在运行本诊断前直接改：

```text
CLOUD_PSI
CLOUD_RATE_BPS
CLOUD_CPU_HZ
```

避免无依据调参。
