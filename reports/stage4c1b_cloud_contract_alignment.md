# 阶段4C-1B：Cloud固定速率合同一致性修正

## 修改与合同

本次只修改了以下两处语义文字：

- `configs/proposed/stage3_cloud_finite_60ghz.toml`：明确每个Cloud任务车辆使用固定等效`R_c=200 Mbps`，不按同槽Cloud任务数分摊；说明`total_bandwidth_bps`是兼容现有接口的重复字段，不是共享总吞吐池；注明200 Mbps是仿真假设，不是物理保障速率。同步明确`resource_sharing=false`表示不建模Cloud接入用户间带宽竞争，Cloud CPU共享由`resource_mode`单独控制。
- `src/common/project_config.py`：将重复上行字段不一致时的报错文字改为“兼容字段须匹配每任务固定等效上行速率，且不代表共享Cloud总带宽池”。只改提示文字，没有改校验条件。

合同与现有代码及`11.pdf`的公式抽象一致：每个Cloud任务独立使用等效速率`R_c`；Cloud CPU仍受60 GHz有限共享池约束。`ψ_c=11`只乘Cloud传输时延`ψ_c D/R_c`，车辆上传能耗仍为`pD/R_c`。

## 静态核对

- 使用`adaptive_cache_mec`环境完成一次现有配置加载与解析：通过。加载值为`uplink_rate_bps=200000000`、`total_bandwidth_bps=200000000`、`resource_sharing=false`、`resource_mode=finite_pool`、`cpu_capacity_cycles_per_s=60000000000`、`cloud_delay_factor=11.0`。
- 修改前后TOML非注释内容SHA-256相同：`5df475736ced9a2571cd3d09e961ef195b134ee7ed540ce983048b30c860c65b`，因此配置数值及键值未变。
- TOML原始字节SHA-256由`1b14a49f430aca3d4c795e576010d9aa0c5e71405701440b3aebaec348578527`变为`dd3f0ae104109f264758ce47a2f478d5df13655638497fe576c1874424f02ba0`。变化来自注释文字；既有Stage3B实验仍对应其原有配置哈希，不能说成使用了新哈希。
- `src/common/system_model.py::cloud_execution()`中的速率、时延和能耗公式未改；Cloud CPU KKT、Matching、PPO和Reward未改，也没有新增Cloud共享带宽约束。

## 后续状态

Cloud合同说明现在与执行语义一致，可冻结为当前仿真的固定等效速率假设。该冻结不构成对现实网络200 Mbps物理保障的标定。未运行仿真、训练、Validation或Test；未修改历史结果、Checkpoint或其他既有未提交文件。

```makefile
STAGE4C1B_DONE = YES
CLOUD_CONTRACT_ALIGNED = YES
FORMULAS_UNCHANGED = YES
NUMERIC_PARAMETERS_UNCHANGED = YES
MATCHING_KKT_UNCHANGED = YES
PPO_REWARD_UNCHANGED = YES
CONFIG_PARSE_PASS = YES
READY_FOR_STAGE4C2 = YES (freeze the fixed-equivalent-rate simulation contract; not a physical guarantee)
```
