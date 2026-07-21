# Decode 架构趋势指标定义与计算边界

版本：v0.1

## 1. 目的与范围

本文定义用于研究 LLM Decode 架构演进趋势的统一指标合同。目标是把每个模型发布转换为关于上下文长度 `C` 和并发数 `B` 的三个可比较函数：

```text
FLOPs/token = F(C, B)
Bytes/token = G(C, B)
Capacity    = H(C, B)
```

本文只规定：

- 需要保存的模型级原始指标；
- Decode 计算量、数据搬运量和持久容量的定义；
- 指标的归一化方式、单位和统计边界。

本文暂不规定：

- 数据从何处获取；
- `C` 和 `B` 的离散扫描点；
- 历史样本选择和趋势拟合方法；
- Peak FLOPS、Peak HBM Bandwidth、TTFT、TPOT 或目标 Token/s；
- 片间通信、Kernel 利用率和真实硬件性能。

这是趋势研究的目标指标合同。部分字段尚未由当前引擎直接输出，不能把本文误解为现有代码的完整功能清单。

## 2. 统一计算边界

### 2.1 逻辑设备边界

统计对象是一颗能够承载完整模型的“逻辑超大芯片”：

- 全模型权重只保存一份；
- 统计整模型在该边界内的总计算、逻辑 HBM 流量和持久容量；
- 不统计边界内部的 TP、EP、PP、CP 等通信；
- 不统计因设备切分产生的权重复制和通信缓冲。

因此，结果是忽略通信的架构级下界，不等价于真实多芯片节点的物理需求。

### 2.2 Decode step

定义：

```text
C = Decode step 开始前，每个请求已有的上下文 Token 数
B = 本 step 中同时推进的请求数
```

v0.1 使用等长请求：

```text
context_tokens = [C] * B
```

每个请求在本 step 生成一个 Token，因此：

```text
output_tokens_per_step = B
```

所有每 Token 工作量都必须先计算完整 step，再除以 `B`：

```text
metric_per_output_token = metric_per_step / B
```

v0.1 同时把 `B` 视为需要保存持久请求状态的并发数。若后续需要表达“驻留请求数大于本 step 调度 batch”，应新增独立变量 `N_resident`，不能继续复用 `B`。

### 2.3 上下文上限

模型历史属性使用：

```text
advertised_max_context_tokens_at_release
```

它表示该模型在对应 release 时公开宣称支持的最大上下文，不等同于典型部署上下文。若数据可得，trained、evaluated、effective 和 deployed context 应保存为其他字段，不能覆盖该值。

计算点满足：

```text
C <= advertised_max_context_tokens_at_release
```

时属于原生范围；超过该范围的理论计算必须设置：

```text
is_extrapolated = true
```

未知上限记为 `null`，不能解释为无限上下文。

### 2.4 部署精度

每条计算记录只对应一个明确的 deployment profile。以下位宽必须按参数组分别保存，不能只记录一个笼统的“模型精度”：

```text
dense_weight_bits
expert_weight_bits
output_head_weight_bits
kv_bits
index_bits
state_bits
```

架构元素数与部署 bytes 分开保存。同一个模型的统一理论精度和真实部署精度是不同 profile，不是不同模型样本。

## 3. 单位与缺失值

- 参数和状态规模使用 element count；
- 计算量使用 FLOPs；
- 容量和流量使用 bytes；
- `Byte/FLOP` 使用 base SI bytes/FLOP；
- 1 MAC 统一记为 2 FLOPs；
- 未知值使用 `null` 并记录原因，不能用 0 替代；
- 估算值必须与官方报告值、实测值分开标记。

## 4. 参数指标

### 4.1 总参数量

```text
total_parameter_elements
```

定义为推理时需要保存的全部唯一参数元素，包括：

- Dense Attention/FFN 等参数；
- 全部 Routed Experts，而非单 Token 激活的 Experts；
- Shared Experts 和 Router；
- Embedding 和 LM Head；
- Norm、Bias 及其他持久参数。

共享权重只计一次。例如 Embedding 与 LM Head 权重绑定时，容量中只保留一份。

### 4.2 每 Token 激活矩阵参数量

```text
active_matrix_parameter_elements_per_token
```

定义为一个 Decode Token 实际参与参数化矩阵计算的参数元素：

```text
always_active_matrix_parameters
+ selected_routed_expert_parameters
+ output_head_parameters
```

它包括 Attention projection、Dense FFN、Router、Shared Experts、选中的 Routed Experts、LM Head 等矩阵路径。

Embedding lookup、Norm、Bias 和其他非矩阵运算不进入该指标。

### 4.3 激活参数比例

```text
active_parameter_ratio =
    active_matrix_parameter_elements_per_token
    / total_parameter_elements
```

该比率用于观察总模型容量与每 Token 参数计算是否逐渐解耦，尤其适用于 MoE。

## 5. 权重容量

### 5.1 权重 payload

```text
weight_payload_bytes =
    Σ(parameter_group_elements * group_storage_bits / 8)
```

必须计算全部参数组和全部 Experts，不受 `C` 或 `B` 影响。

### 5.2 总权重容量

```text
total_weight_capacity_bytes =
    weight_payload_bytes
  + quantization_metadata_bytes
```

量化 metadata 包括已实际存储的 scale、zero point、packing metadata 等：

- 已知时计入；
- 明确的理论 profile 可以标记为 `metadata_excluded`；
- 未知时不能静默按 0 处理，应分别报告 payload 和 metadata 状态。

权重绑定在容量中只计一份，但若同一权重在一次 step 中参与多个独立操作，其流量应遵循对应操作的读取边界。

## 6. 持久状态容量

容量统一表示 Decode step 开始前已经存在的持久状态，不包含本 step 即将写入的新 Token。

### 6.1 每请求容量

```text
kv_cache_bytes_per_request(C)
index_cache_bytes_per_request(C)
state_cache_bytes_per_request(C)
```

总和为：

```text
cache_capacity_per_request(C) =
    kv_cache_bytes_per_request(C)
  + index_cache_bytes_per_request(C)
  + state_cache_bytes_per_request(C)
```

语义：

- KV capacity 保存已有 `C` 个上下文产生的有效 KV；
- Index capacity 保存已有上下文对应的持久索引；
- State capacity 保存 Linear Attention、SSM、Mamba 等请求私有状态；
- 本 step 新 Token 的 KV/index/state 更新计入写流量，不计入 start-of-step capacity；
- 如需执行后容量，应显式计算 `capacity(C + 1)`。

### 6.2 总持久容量

```text
persistent_memory_capacity(C, B) =
    total_weight_capacity_bytes
  + B * cache_capacity_per_request(C)
```

该指标是 v0.1 的逻辑最小持久容量，不包含：

- 临时 Activation；
- Kernel workspace；
- Runtime buffer；
- 内存分页碎片；
- 权重复制；
- 通信 buffer；
- 容量安全余量。

因此不能直接把它当作最终 HBM 配置值。

## 7. Decode 计算量

### 7.1 分项

```text
parameter_flops_per_token(C, B)
attention_flops_per_token(C, B)
index_flops_per_token(C, B)
state_flops_per_token(C, B)
extra_flops_per_token(C, B)
```

定义如下：

| 分项 | 定义 |
|---|---|
| Parameter | Projection、FFN、Router、Shared/Routed Experts、LM Head 等参数矩阵 MAC |
| Attention | 主 Attention 的 QK 与 AV；Projection 不在此重复计数 |
| Index | 稀疏 Attention 的 indexer score 及明确约定的 selection 成本 |
| State | Linear Attention、SSM、Mamba 等 recurrence/state update |
| Extra | 明确配置并能审计的其他 FLOPs |

统一规则：

- 1 MAC = 2 FLOPs；
- LM Head 在每个 Decode Token 上执行；
- Embedding lookup 只产生读取流量，不计参数矩阵 FLOPs；
- 固定 Top-k MoE 的参数 FLOPs/token 通常不随 `B` 改变；
- `B` 主要影响权重读取复用和一次 step 触达的 Expert 并集；
- Softmax、Norm、激活函数、采样、量化/反量化等不能无依据折算成 FLOPs。

无法统一折算的非矩阵操作应保留独立操作计数或进入显式 `extra_flops`，不能暗含在 Parameter/Attention 中。

### 7.2 总计算量

先计算：

```text
total_flops_per_step(C, B)
```

再归一化：

```text
total_flops_per_token(C, B) =
    total_flops_per_step(C, B) / B
```

并满足：

```text
total_flops_per_token =
    parameter_flops_per_token
  + attention_flops_per_token
  + index_flops_per_token
  + state_flops_per_token
  + extra_flops_per_token
```

## 8. Decode 数据搬运量

### 8.1 Logical-HBM 边界

v0.1 使用整模型 logical-HBM 口径：

- 统计一次 Decode step 逻辑上需要从 HBM 读取或写入的数据；
- 默认不假设权重、KV、index 或 state 跨 step 常驻片上；
- 不统计 Kernel tiling 或容量不足造成的重复加载；
- 不统计片间通信；
- 不统计 Cache capacity 本身；
- 不统计 Activation、Workspace、Logits 输出和 Host I/O；
- 不声称等于 profiler 实测 HBM traffic。

若未来增加 measured-HBM 或 estimated-HBM，必须作为另一套 view 保存，不能覆盖 logical-HBM。

### 8.2 分项

```text
weight_read_bytes_per_token(C, B)
kv_read_bytes_per_token(C, B)
kv_write_bytes_per_token(C, B)
index_read_bytes_per_token(C, B)
index_write_bytes_per_token(C, B)
state_read_bytes_per_token(C, B)
state_write_bytes_per_token(C, B)
other_read_bytes_per_token(C, B)
```

其中：

- Weight read：本 step 执行参数化操作需要读取的权重；
- KV read：当前 query 实际访问的已有历史 KV；
- KV write：本 step 新 Token 产生并持久化的 KV；
- Index read/write：稀疏 Attention 的持久索引访问；
- State read/write：请求私有 recurrent state 的读取和更新；
- Other read：Embedding lookup 等不属于上述类别的明确固定读取。

### 8.3 权重读取

Dense 权重在完整 step 中逻辑读取一次，再由 `B` 个输出 Token 摊薄：

```text
dense_weight_read_bytes_per_token =
    dense_weight_read_bytes_per_step / B
```

Routed Expert 权重按本 step 的 `B` 个 Token 实际或预计触达的 Expert 并集计算：

```text
expert_weight_read_bytes_per_step =
    Σ(touched_expert_weight_bytes)
```

它既不等于全部 Expert 权重容量，也不能简单按 `B * selected_experts` 计算，除非明确采用“无 batch 复用”的流量上界。

### 8.4 总搬运量

先计算：

```text
total_bytes_per_step(C, B)
```

再归一化：

```text
total_bytes_per_token(C, B) =
    total_bytes_per_step(C, B) / B
```

并满足：

```text
total_bytes_per_token =
    weight_read_bytes_per_token
  + kv_read_bytes_per_token
  + kv_write_bytes_per_token
  + index_read_bytes_per_token
  + index_write_bytes_per_token
  + state_read_bytes_per_token
  + state_write_bytes_per_token
  + other_read_bytes_per_token
```

## 9. 机制的上下文语义

以下表格定义不同机制如何影响主读取量和持久容量。实际 bytes 还需乘对应 layout 的 entry 大小和层数。

| 机制 | 主读取 entry 数/Token | 主持久 entry 数/请求 | 额外路径 |
|---|---:|---:|---|
| Full Attention | `C` | `C` | 无 |
| GQA/MQA | `C` | `C` | 每 entry 的 KV 更小 |
| MLA | `C` | `C` | 每 entry 为 latent + RoPE |
| SWA | `min(C, W)` | `min(C, W)` | 无 |
| HCA/Compressed Full | `floor(C/m)` | `floor(C/m)` | 压缩路径 |
| Fixed Top-k | `min(k, candidates)` | `candidates` | Selection 成本在外部或已固定 |
| DSA | `min(k, C)` | KV 通常为 `C` | Indexer 扫描 `C` 个候选 |
| CSA | `min(k, floor(C/m))` | 约 `floor(C/m)` | Indexer 扫描压缩候选 |
| Linear Attention/SSM | 与 `C` 无关 | 固定 State | State read/update |

该拆分用于区分四种不同的优化：

1. 只减小每个 KV entry；
2. 减少每 Token 读取的 entry 数；
3. 减少持久保存的 entry 数；
4. 用固定 State 替代随上下文增长的 Cache。

不能只用一个“稀疏率”同时缩放计算、流量和容量。

## 10. 派生指标

### 10.1 Byte/FLOP

```text
bytes_per_flop(C, B) =
    total_bytes_per_token(C, B)
    / total_flops_per_token(C, B)
```

它描述 logical-HBM 工作负载的带宽算力平衡点，不能代替绝对 FLOPs、Bytes 或容量。

### 10.2 Cache/Weight 容量比

```text
cache_to_weight_capacity_ratio(C, B) =
    B * cache_capacity_per_request(C)
    / total_weight_capacity_bytes
```

该指标用于观察瓶颈是否从权重容量转向请求私有状态容量。

### 10.3 原始结构计数

为避免相同 FLOPs 掩盖不同的数据通路，建议同时保留下列可审计计数：

```text
attention_entries_read_per_token
index_candidates_scanned_per_token
state_elements_read_per_token
state_elements_written_per_token
expected_unique_experts_touched_per_step
```

它们不属于 Peak 性能指标，但可支持后续分析 Dense、Gather/Top-k 和 State 路径的变化。

## 11. 数据记录粒度

一条计算记录唯一对应：

```text
model_release_id
deployment_profile_id
context_C
concurrency_B
```

### 11.1 标识和输入字段

```text
model_family
model_release_id
release_date
architecture_mechanisms
mechanism_layer_counts
deployment_profile_id
advertised_max_context_tokens_at_release
context_C
concurrency_B
is_extrapolated
```

### 11.2 模型级指标

```text
total_parameter_elements
active_matrix_parameter_elements_per_token
active_parameter_ratio
weight_payload_bytes
quantization_metadata_bytes
total_weight_capacity_bytes
```

### 11.3 工作量与容量指标

```text
parameter_flops_per_token
attention_flops_per_token
index_flops_per_token
state_flops_per_token
extra_flops_per_token
total_flops_per_token

weight_read_bytes_per_token
kv_read_bytes_per_token
kv_write_bytes_per_token
index_read_bytes_per_token
index_write_bytes_per_token
state_read_bytes_per_token
state_write_bytes_per_token
other_read_bytes_per_token
total_bytes_per_token

kv_cache_bytes_per_request
index_cache_bytes_per_request
state_cache_bytes_per_request
cache_capacity_per_request
persistent_memory_capacity

bytes_per_flop
cache_to_weight_capacity_ratio
```

所有总量都必须保留对应分项，不能只保存 `total_flops_per_token`、`total_bytes_per_token` 或比率。

## 12. 必须保持的边界

1. 先计算完整 Decode step，再除以 `B`。
2. 总参数容量包含全部 Experts；激活参数只包含每 Token 实际执行的 Experts。
3. 权重容量、权重读取量和参数 FLOPs 是三个不同概念。
4. Cache capacity 不能加入 HBM traffic。
5. Start-of-step capacity 使用 `C`；本 step 的新 Token 只进入 write traffic。
6. `C_max` 是模型历史属性，`C` 是可配置计算自变量。
7. 超过官方上下文范围的点必须标记为理论外推。
8. Logical-HBM 不能冒充 measured-HBM。
9. 不同部署精度是同一模型的不同 profile，不是独立模型样本。
10. Peak Compute、Peak Bandwidth 和芯片配置必须等目标 Token/s 确定后再推导。

## 13. 后续待讨论项

在本指标合同确认后，再分别讨论：

1. `C` 和 `B` 的扫描点；
2. 模型样本和原始字段的获取方式；
3. 历史数据质量与来源等级；
4. 趋势统计和拟合方法；
5. 从预测 workload 到芯片 Peak Compute、Peak Bandwidth 和 HBM Capacity 的换算。
