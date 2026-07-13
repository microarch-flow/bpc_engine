# Prefill 工作量指标与三组标准实验

本文说明如何为本项目定义 prefill 阶段的计算量和数据搬运量，以及为什么需要三组互补的实验。它是 [README](../README.md) 中 decode 口径的扩展，默认仍统计整模型在所有设备上的**逻辑 HBM 流量**，不把卡间通信、调度开销或 kernel 实现造成的重复读取隐式算入结果。

本文首先讨论普通 prefill，即一次处理每个请求的完整 prompt；随后给出 prefix cache 和 chunked prefill 的统一表示。

## 1. Prefill 到底在测什么

Decode 的典型工作单元是：一个 batch 中的每个请求各生成一个新 token。Prefill 的工作单元则是：一次执行中处理一个 batch 的若干输入 token。

对于静态架构分析，prompt 的**文本内容**通常不是主要自变量，prompt 被 tokenizer 转换后的**token 数量和批次形状**才是。文本内容只有在以下情况中会改变工作量：

- MoE 的实际专家路由不同；
- 动态稀疏注意力选中的位置不同；
- prefix cache 的命中长度不同；
- 模型或 serving 系统存在其他输入相关的动态路径。

因此，基础接口应接收 token 长度，而不是一组自然语言字符串；有真实 routing trace 或 cache trace 时，再用 trace 覆盖分析近似。

## 2. B、L、T 和 shape

设一次 prefill 中有 `B` 个请求，第 `i` 个请求的 prompt 长度为 `L_i`：

```text
lengths = [L_1, L_2, ..., L_B]

B       = len(lengths)                 # 请求数
T       = sum(L_i)                     # 有效输入 token 总数
L_max   = max(L_i)                     # batch 内最长 prompt
shape   = 这组长度的组成方式
```

当所有请求等长时，可以简写为：

```text
lengths = [L] * B
T       = B * L
shape   = B × L
```

例如：

```text
lengths = [1000, 2000, 500]
B       = 3
T       = 3500
L_max   = 2000
```

这里的 `T` 不是某个请求的上下文长度，而是**这一次 prefill batch 总共处理的有效输入 token 数**。许多 serving 调度器会给一次迭代设置 token budget，因此 `T` 也是调度和容量分析中的自然单位。

仅记录 `B` 和平均长度是不够的。注意力成本含有 `L_i²` 项，下面两个 batch 虽然 `B` 和 `T` 相同，注意力工作量仍可能不同：

```text
[512, 512, 512, 512]
[128, 128, 128, 1664]
```

因此，通用输入应保留完整的 `lengths` 向量。

## 3. 一次执行的总量与归一化指标

Prefill 应先计算完整的一次执行，再按需要归一化：

```text
prefill_work = weight_work_for_execution
             + sum(sequence_work(L_i) for each request i)

flops_per_input_token = total_flops / valid_tokens
bytes_per_input_token = total_bytes / valid_tokens
```

其中：

```text
valid_tokens = T = sum(L_i)
```

如果使用 padding，还应另记实际执行位置数：

```text
executed_tokens = B * L_max             # 朴素 padded batch
token_efficiency = valid_tokens / executed_tokens
```

建议同时保留以下三类结果，不能只保留每 token 数值：

1. **一次 prefill 的总量**：`total_flops`、`total_bytes`。默认 `total_bytes` 来自 compulsory/logical-HBM 的 `batch_work`；attention/index 展开的 `batch_operand_work.total_bytes` 作为另一套边界单列。它们描述一次请求批次带来的总工作量，是分析 TTFT 压力的基础。
2. **每有效输入 token 的归一化量**：`flops_per_input_token`、`bytes_per_input_token`。它们便于比较吞吐和不同 batch shape。
3. **工作量比例**：`bytes_per_flop` 和 `tbps_per_pflops = 1000 * bytes_per_flop`。它们与当前 decode 输出保持一致。

这些指标是工作量，不是延迟预测。没有硬件带宽、峰值算力、利用率、kernel 和通信模型时，不能从 FLOPs/Bytes 直接声称 TTFT 是多少。

Python API 直接接收完整长度向量：

```python
from decode_engine import calculate_prefill, load_engine_config

config = load_engine_config("configs/16bit/qwen3_8b_16bit.json")
result = calculate_prefill(
    config.model,
    config.deployment,
    prompt_tokens=[128, 512, 2048, 64],
    execution_mode="varlen",
    logits_mode="last",
)
print(result.batch_work.total_flops)
print(result.per_input_work.total_bytes)
```

## 4. Full-causal reference pair slots

普通 full causal attention 中，第 `t` 个输入 token 可以关注本序列的第 `1..t` 个 token。因此单个长度为 `L_i` 的请求包含：

```text
attention_pairs_i = 1 + 2 + ... + L_i
                  = L_i * (L_i + 1) / 2
```

整个 ragged batch 的有效 pair 数为：

```text
valid_causal_pair_slots = sum(L_i * (L_i + 1) / 2)
```

对于等长 batch：

```text
valid_causal_pair_slots = B * L * (L + 1) / 2
```

如果 `entry_flops` 表示一个 query 与一个 KV entry 完成 QK 和 AV 所需的 FLOPs，则：

```text
attention_flops = entry_flops * valid_causal_pair_slots
```

这里的 causal pair slots 是输入 shape 的 full-causal 参考量。它只在 Full
attention 中等于实际访问 pair；SWA、DSA、CSA 等机制的真实访问量应查看
`attention_flops` 与 operand read 分项，Mamba/SSM 则根本不执行这些 pair。

本项目现有 decode 公式中的 context 表示已经存在的历史 entry，所以通常只扫描这 `C` 个历史位置。Prefill 的标准 causal mask 则包含 query 自身的对角项。两者在边界上相差每个输入 token 一个 pair。实现和输出必须显式说明 `include_self_attention` 或等价语义，不能悄悄混用 `L(L+1)/2` 与 `L(L-1)/2`。

### 4.1 Prefix cache 与 chunked prefill 的统一公式

令：

```text
C_i = 第 i 个请求在本次执行前已有的 cached prefix 长度
Q_i = 本次新处理的 token 数
```

本次新 token 对 full causal attention 产生的 pair 数为：

```text
pairs_i = Q_i * C_i + Q_i * (Q_i + 1) / 2
```

它统一描述了：

- 普通 prefill：`C_i = 0`，`Q_i = L_i`；
- prefix-cache prefill：`C_i > 0`；
- chunked prefill：`Q_i` 是本 chunk 长度，下一 chunk 的 `C_i` 随之增加；
- decode 的近似特例：`Q_i = 1`，但是否计入自身对角项应遵循 decode 的既有口径。

这个公式统一的是 attention/cache/sequence-mixer 部分。最终 prompt 调用使用
`logits_mode=last`；不生成 logits 的非最终 chunk 使用 `logits_mode=none`。
第一版还不能表达同一个 batch 中“只有部分请求在本 chunk 结束”的任意
logit-position 子集，这种 continuous batching 仍需后续增加显式位置数或 trace。

对 SWA、DSA、CSA 等机制，不应先计算 full pair 数再乘一个统一“稀疏率”。更可靠的方式是逐位置聚合访问函数：

```text
read_entries_i = sum(accessed_entries(C_i + t) for t in 1..Q_i)
```

主 KV 路径、indexer candidate 路径和最终 cache 保存量应分别计算。例如 DSA 的主路径可能每个 query 只读 top-k KV，但 indexer 仍需要扫描更多候选项。

## 5. 参数矩阵计算与 LM head

### 5.1 Backbone

对于每个输入位置都执行的 dense backbone 参数：

```text
backbone_flops = mac_flops
               * backbone_active_parameter_elements
               * executed_backbone_tokens
```

理想 varlen/packing 下，`executed_backbone_tokens = T`；朴素 padding 下可能是 `B * L_max`。默认 `mac_flops = 2`。

因此，在固定 `T` 的实验中，dense backbone FLOPs 大致不变；attention FLOPs 却会随 `lengths` 的组成而变化。

### 5.2 LM head 的 `last`、`all` 与 `none`

当前 decode 配置把 LM head 包含在 `always_active_parameters` 中，因为 decode 每生成一个 token 都需要一次 logits。Prefill 不能无条件把整个 `always_active_parameters` 乘以 `T`，否则普通 serving 场景会明显高估 LM head。

当前实现支持以下两种模式：

```text
last:
  logit_positions = B
  # 每个请求只对最后一个有效 prompt 位置产生 next-token logits

all:
  logit_positions = T
  # 为全部输入位置产生 logits，例如 prompt logprobs 或训练式分析

none:
  logit_positions = 0
  # 非最终 chunk 不执行 LM head
```

上面的 `T` 指 useful/varlen/packed 路径。当前 padded 模式还会给补齐位置执行参数计算，因此 `all` 的 batch executed positions 为 `B * L_max`；`last` 始终只取每个请求最后一个有效位置，所以仍为 `B`；`none` 始终为 0。输出同时保留 useful 与 batch work，避免把 padding 开销混入有效语义。

对应计算量为：

```text
lm_head_flops = mac_flops
                * output_head_parameters
                * logit_positions
```

`output_head_parameters` 是 always-active 参数总数中的 LM-head 子集，不是新增的一份权重；`output_head_weight_bits` 用于在 `none` 模式中从 weight read 中排除整张 head。仓库自带的 27 个真实配置均已填写。兼容旧配置时参数字段默认是 0，此时引擎会把全部 always-active 参数视为 backbone；结果中的 `output_head_parameters_configured` 会标出配置是否显式提供了拆分，不能把值为 `false` 的 `last/none` 结果当作精确 LM-head 口径。

LM-head 权重在一次 batched GEMM 中的逻辑读取通常仍是一份，而不是乘以 `logit_positions`；如果 logits 张量写入 HBM，则其大小属于 activation/output traffic，应由明确的 materialization 或 spill 配置控制。

### 5.3 MoE routed experts

MoE 的参数 FLOPs 按 routed token 数计算，而不是按请求数 `B` 计算。若某层有 `E` 个专家、每 token 选择 `k` 个，且一层中有 `N` 个 routed token，均匀独立路由近似下，该层触达的专家并集为：

```text
E_unique(N) = E * [1 - (1 - k/E)^N]
```

Varlen/packed prefill 中这里的 `N = T`，而不是 `B`。当前 padded 分析把补齐位置视为真实执行位置，因此 routed-expert FLOPs 和专家并集使用 `N = B * L_max`；同时另保留只基于有效 token 的 useful 结果。若实际框架会在进入 MoE 前 compact 掉 padding，应选择 `varlen`/`packed` 模式或提供校准 profile，不能继续使用 padded 模式却假定 padding 不参与路由。真实路由具有相关性时，应使用 trace 或显式专家并集：decode trace 写入 `expected_unique_experts_by_batch`，prefill trace 写入 `expected_unique_experts_by_active_tokens`。

## 6. 数据搬运的三个不同概念

Prefill 最容易出错的地方，是把“算法需要某个 K/V 参与多少次乘法”直接当作“HBM 将这个 K/V 读取多少次”。两者在 decode 中常常接近，在 prefill 中却可能相差很大，因为同一个 K/V tile 可以被许多 query 复用。

### 6.1 Compulsory/logical HBM：本项目的默认主口径

为与 README 保持一致，默认 `work`（也称 compulsory/logical-HBM view）统计整模型在所有设备上，按 fused-prefill 假设必须从 HBM 读取或写入的逻辑 tensor 数据量，但不统计 kernel tiling、调度或实现细节造成的重复读取。一个 fused invocation 对所需的旧 prefix entry 只读一次，并假定本次产生的新 entry 可以直接在 invocation 内复用，不把它们再次算作 HBM read。

典型组成包括：

```text
logical_hbm_bytes = dense_weight_reads
                  + routed_expert_weight_reads
                  + persistent_kv_or_index_writes
                  + required_cached_prefix_reads
                  + recurrent_state_reads_and_writes
                  + configured_activation_spills
                  + fixed_other_traffic
```

需要遵守以下规则：

- Dense 权重在一次 prefill kernel invocation/microbatch 中逻辑读取一次，而不是每输入 token 读取一次。若 prompt 被拆成多个独立 chunk 执行，则每个 chunk 可能各有一次权重读取。
- MoE 权重读取按本次执行实际或预计触达的专家并集计算，而不是简单乘 routed token 数。
- 普通 full-attention prefill 结束后，需要为未来 decode 持久化的新 KV cache 写入量通常与 `T` 成正比。
- 命中的旧 prefix cache 是 HBM 输入时，按指定访问机制统计一次必要的逻辑读取；当前 compulsory view 不把本 invocation 刚产生的 K/V 再次计作 HBM read。
- 对 content-selected top-k 且 `C_i > 0`，没有 selection trace 时无法知道多个
  query 的真实 KV 并集；当前 compulsory main-KV read 使用可能触达的
  distinct-prefix 保守上界，而不是伪装成精确的实测并集。
- `weight_hbm_fraction`、`kv_hbm_fraction`、`index_hbm_fraction`、`state_hbm_fraction` 等 deployment 假设只缩放相应流量，不改变 FLOPs 或逻辑 cache 容量。
- 默认不包含 tensor/expert parallel 通信，也不包含 kernel 对同一 tile 的重复加载。

对于普通 full attention，持久化 KV cache 的最终有效容量为：

```text
kv_cache_capacity = sum_i(L_i * kv_entry_bytes_per_all_layers)
```

Padding buffer 的临时分配容量可能是 `B * L_max` 个 entry，但当前实现的最终持久 KV、index 和 recurrent state 只按有效请求/token 保存；padding 位置不会成为未来 decode 的 cache。临时分配容量与最终有效 cache 容量应分开报告。

### 6.2 Operand view：展开 attention/index 操作数

当前实现还输出一个可替代 compulsory view 的 `operand_work`。它保留相同的 FLOPs、权重流量、持久化写入和 recurrent-state 流量，只把 attention 与 index read 替换为按所有 query 展开的 pair/candidate-stream 访问量。两套 view 是不同边界，**必须二选一比较，不能相加**。

如果把每个 attention pair 使用的 K 和 V 都展开计数，则：

```text
attention_kv_operand_bytes
    = kv_entry_bytes * attention_pairs
```

对于长度为 `L` 的 full attention，它是 `O(L²)`。这个指标适合：

- 表示没有跨 query 复用时的展开访问量；
- 比较不同 attention 机制的算法数据复用机会；
- 给 tile/kernel 模型提供输入。

但它**不是默认逻辑 HBM 流量，也不是实测 HBM 流量**。FlashAttention、tiling、融合和片上 SRAM 可以让同一 K/V tile 服务多个 query；反过来，片上容量不足也可能让某些 tile 被重复加载。Indexer read 按全部候选 entry 同样展开。第一版 operand view 只对 attention/index 进行这种展开，不展开 recurrent state。

因此，不应执行下面这种隐式替换：

```text
错误做法：total_bytes = work.total_bytes + operand_work.total_bytes
```

否则既会重复加入两套 view 的共同部分，也会把“每个 query 都从 HBM 重新读取所有 K/V”这一特殊边界当成通用事实。

### 6.3 Estimated/Measured HBM：带 kernel 假设的估算或实测

只有在提供以下信息之一时，才应输出 `estimated_hbm_bytes` 或 `measured_hbm_bytes`：

- 明确的 query/KV tile 大小和片上容量模型；
- 每类 tensor 的 read-amplification profile；
- 指定 kernel 的 profiler/硬件计数器结果；
- 经实测校准的经验模型。

输出中必须保留估算所用的 kernel/profile 名称和参数。建议三套结果并列，而不是互相覆盖或相加：

```text
work                         # compulsory/logical-HBM 主口径
operand_work                 # 仅展开 attention/index read 的替代边界
estimated_or_measured_work   # 有 kernel/profile 时才存在
```

### 6.4 SSM、Mamba 和 recurrent state

SSM/Mamba prefill 常使用 fused scan。把 decode 中“每个 token 从 HBM 读取并写回完整状态”的流量直接乘以 `T`，通常只是一个很松的实现上界，不应默认当作 prefill HBM。

当前第一版在 compulsory 与 operand 两套 view 中都采用同一个 fused-scan 口径：

- recurrence/scan 算术量仍按本次 executed token 数累计；
- 普通新 prompt 从隐式零状态开始，不读取初始 state；
- 当 `C_i > 0` 且本次有新 token 时，每个请求只读取一次已有 persistent state；
- 本次有新 token 且启用 state write 时，每个请求只写一次最终 state；
- 中间 recurrence 值不算 HBM，`work` 与 `operand_work` 的 state 分项完全相同。

第一版不输出“逐 token decode 展开”的 `state_operand_bytes`。如果未来确实需要这一实现上界，应增加名称明确的第三种独立 view，并避免与当前 attention/index operand view 混用。`state_hbm_fraction` 仍用于表达 persistent state 的片上驻留假设。

## 7. 三组标准实验

三组实验回答不同问题，不能互相替代。

### 7.1 实验一：固定 B，扫描等长 prompt 的 L

构造：

```text
B 固定
lengths = [L] * B
L = 128, 512, 2048, 8192, ...
```

例如固定 `B = 4`：

| lengths | B | L | T |
|---|---:|---:|---:|
| `[128, 128, 128, 128]` | 4 | 128 | 512 |
| `[512, 512, 512, 512]` | 4 | 512 | 2048 |
| `[2048, 2048, 2048, 2048]` | 4 | 2048 | 8192 |

它回答：

> 同时处理固定数量的请求时，prompt 越长，prefill 的总工作量和每输入 token 工作量怎样变化？

对 full attention，主要趋势是：

```text
dense parameter FLOPs total  = O(BL)
attention FLOPs total        = O(BL²)
attention FLOPs/input-token  = O(L)
persistent KV write total    = O(BL)
```

Dense 权重的逻辑读取量在一次执行内大致固定，所以 `weight_bytes/input-token` 会随着 `B * L` 增大而降低。推荐画多条固定 batch 曲线，例如 `B = 1` 和 `B = 32`。

这组实验最直观，适合展示模型架构随上下文长度的趋势，但不能代表 token-budget 调度，也不能代表真实的长短请求混合。

### 7.2 实验二：固定 T，改变 batch shape

构造相同的总 token budget `T`，但让 token 来自不同数量的请求：

```text
T = 4096

B=1:   [4096]
B=4:   [1024, 1024, 1024, 1024]
B=16:  [256, 256, ..., 256]
B=32:  [128, 128, ..., 128]
```

对应 full causal attention pair 数为：

| shape | T | attention pairs |
|---|---:|---:|
| `1 × 4096` | 4096 | 8,390,656 |
| `4 × 1024` | 4096 | 2,099,200 |
| `16 × 256` | 4096 | 526,336 |
| `32 × 128` | 4096 | 264,192 |

等长 shape 在固定 `T = B * L` 时满足：

```text
attention_pairs = T * (L + 1) / 2
```

实现不会要求 `T` 必须能被 `B` 整除。令：

```text
q, r = divmod(T, B)
lengths = [q + 1] * r + [q] * (B - r)
```

这样各请求长度最多相差 1，并严格保持 `sum(lengths) = T`。例如 `T=10, B=3` 得到 `[4, 3, 3]`，而不是截断为 9 或向上补成 12；此时不再使用单一 `L` 的等长公式，attention pairs 应按完整向量求和。因为每个请求至少要有一个 token，当前实现要求 `B <= T`。

所以同样处理 4096 个 token，一个 4096-token 长请求的 attention 工作量约是 16 个 256-token 请求之和的 16 倍。与此同时：

- 在 varlen/packed 下，dense backbone 参数 FLOPs 大致相同，因为都处理 `T` 个位置；
- full-attention 的最终有效 KV 容量和新 KV 写入量大致相同，因为都保存 `T` 个 entry；
- attention FLOPs 和 attention operand bytes 明显不同；
- `last` 模式的 LM-head positions 等于 `B`，因此会随 shape 变化；`all` 模式则等于固定的 `T`；
- 在 varlen/packed 的理想独立 MoE 路由近似中，每层 routed token 数相同，专家并集大致相同；真实相关路由可能不同。

若对不能整除的均衡向量选择 padded 执行，短请求还会补到 `q+1`，此时 batch executed tokens 可能大于 `T`；该额外开销会如实进入 batch work 和 padded MoE 路由，而 useful work 仍严格对应原 token budget `T`。

它对应的典型场景是 serving 调度器设置：

```text
max_prefill_tokens_per_batch = T
```

同一 token budget 可能被一个长请求占满，也可能装入很多短请求。该实验用来判断“token 数相同”是否真的意味着成本相同，并为 admission control、chunked prefill 和 batch 组形提供依据。

### 7.3 实验三：真实 ragged 长度向量

`ragged` 只表示同一个 batch 中的请求长度不相等。例如：

```text
lengths = [128, 512, 2048, 64]
B       = 4
T       = 2752
L_max   = 2048
```

这个 batch 的有效 full-attention pair 数为：

```text
valid_causal_pair_slots
  = 128*129/2 + 512*513/2 + 2048*2049/2 + 64*65/2
  = 2,239,840
```

它回答：

> 面对真实在线服务中的长短请求混合，不同 batching 执行方式会造成多少 padding 或调度浪费？

真实向量最好来自请求 trace；没有 trace 时，也可以从明确记录的长度分布和随机种子生成代表性 batch。不要只用平均长度代替向量，因为 attention 的平方项使长尾请求具有不成比例的影响。

## 8. Padding、varlen 与 packing

这三个词描述的是如何执行 ragged batch，而不是三种不同的模型。

### 8.1 Padding

把所有请求补齐到 `L_max`：

```text
executed_tokens = B * L_max
executed_causal_pair_slots = B * L_max * (L_max + 1) / 2
```

对于 `[128, 512, 2048, 64]`：

```text
valid_tokens              = 2752
padded_executed_tokens    = 8192
valid_causal_pair_slots   = 2,239,840
padded_causal_pair_slots  = 8,392,704
```

以这个分析模型计算，token 位置膨胀约 `2.98×`，causal pair slots 膨胀约 `3.75×`。当前 padded 模式把 executed 参数/attention/index/state FLOPs，以及 attention/index operand read，按 `B × L_max` 的 padded shape 计算；这是一种明确的 dense-shape 分析边界。实际 kernel 若会跳过更多 masked tile，应另用 profiler 校准。

Padding 位置只影响 executed work，不改变请求语义。当前实现会用原始 `lengths` 重新保留 KV write、index write、state write 和最终 cache capacity，因此持久化 KV/index/state 全都只按有效请求/token 计算；padding 位置不会成为未来 decode 的 cache。Padded MoE 则按 executed tokens 路由，若实际框架先 compact 再进入 MoE，应改用 `varlen`/`packed` 或 profile。

当前 padded 模式用于普通 one-shot prefill，不接受非零 cached prefix；带 prefix cache 的分析应使用 `varlen` 或 `packed`。

### 8.2 Varlen

Varlen kernel 通过每个请求的长度或 cumulative sequence lengths，只处理有效 token，并让各请求保持独立的 causal attention 边界。理想分析口径为：

```text
executed_tokens          = sum(L_i)
executed_causal_pair_slots = sum(L_i * (L_i + 1) / 2)
```

它消除了 padding 的逻辑浪费，但实际效率仍受 kernel tile、长度分布和 launch 策略影响。

### 8.3 Packing

Packing 把多个请求的有效 token 紧凑放进一个或少数几个 buffer，以减少空洞和 padding。正确 packing 必须保留 segment boundary 或 block-diagonal causal mask，禁止不同请求相互 attention。

理想 packing 的有效 token 和 pair 数与 varlen 相同：

```text
tokens = sum(L_i)
pairs  = sum(L_i * (L_i + 1) / 2)
```

不能把所有 token 拼成一个普通长序列后使用单一 causal mask。对上述例子，错误的普通拼接会得到：

```text
T * (T + 1) / 2 = 3,788,128 pairs
```

其中包含跨请求 attention，既改变模型语义，又高估有效工作量。某些 block-mask kernel 虽然语义正确，底层仍可能执行部分被 mask 的 tile；这属于 `executed_*` 或 kernel estimate，而不是 `valid_*`。

## 9. 推荐输出字段与分项

每个数据点至少应携带足够信息，使结果可以复算和审计。

### 9.1 输入与执行元数据

```text
experiment
prompt_tokens
cached_context_tokens
batch_size
valid_input_tokens
executed_input_tokens
average_prompt_tokens
max_prompt_tokens
execution_mode              # varlen, packed, padded
include_self_attention
logits_mode                 # last, all, none
valid_logit_positions
executed_logit_positions
output_head_parameters
output_head_parameters_configured
output_head_weight_bits
topk_cached_prefix_union_policy  # cached top-k 无 trace 时为保守 distinct 上界
valid_causal_pair_slots
executed_causal_pair_slots
token_efficiency
causal_pair_efficiency
```

以上名称对应 JSON/API。CSV 为便于表格使用，会作以下扁平化：

- `batch_size` 记为 `batch`；
- `prompt_tokens`、`cached_context_tokens` 分别成为 JSON 字符串列
  `prompt_lengths`、`cached_lengths`；
- work 分项使用 `batch_`、`useful_`、`per_input_`、`batch_operand_`、
  `useful_operand_`、`per_input_operand_` 前缀；
- cache 同时提供 total 与 per-request 列；
- `per_executed_token_work` 目前只在 JSON/API 中保留，没有重复展开进 CSV。

Table 只展示其中最常用的摘要列。

### 9.2 两套工作量 view

```text
useful_work                 # 仅有效 token，compulsory/logical-HBM
batch_work                  # 选定执行布局，padded 时包含补齐开销
per_input_work              # batch_work / valid_input_tokens
per_executed_token_work     # batch_work / executed_input_tokens

useful_operand_work         # 仅有效 token，attention/index read 展开
batch_operand_work          # 选定执行布局的 operand view
per_input_operand_work      # batch_operand_work / valid_input_tokens
```

每个 `WorkCost` 都包含同一组 FLOPs 与 bytes 字段：

```text
parameter_flops
attention_flops
index_flops
state_flops
extra_flops

weight_read_bytes
kv_read_bytes / kv_write_bytes
index_read_bytes / index_write_bytes
state_read_bytes / state_write_bytes
activation_bytes
other_read_bytes
total_flops / total_bytes
```

`work` 与 `operand_work` 的 FLOPs、write、weight、activation、other 和 state 字段相同；只有 attention/index read 边界不同。第一版没有单独的逐 token `state_operand_bytes`。

### 9.3 Cache 容量与路由审计

```text
cache_capacity_total             # kv_bytes, index_bytes, state_bytes
cache_capacity_per_request_average
expert_weight_sets_read          # 选定布局；padded 时按 executed tokens
useful_expert_weight_sets_read   # 只按有效 token
```

Cache capacity 始终是本次 prefill 结束后的 persistent capacity，不是逐位置容量之和；padded 结果也只保存有效请求/token 的 KV、index 和 state。

### 9.4 派生指标

```text
bytes_per_flop          = batch_work.total_bytes / batch_work.total_flops
tbps_per_pflops         = 1000 * bytes_per_flop
operand_bytes_per_flop  = batch_operand_work.total_bytes
                          / batch_operand_work.total_flops
operand_tbps_per_pflops = 1000 * operand_bytes_per_flop
token_efficiency        = valid_input_tokens / executed_input_tokens
causal_pair_efficiency  = valid_causal_pair_slots
                          / executed_causal_pair_slots
```

所有 `per_input_token` 指标默认除以有效 token 数。若还输出 `per_executed_token`，名称必须显式区分。

## 10. 三组实验的建议图表

实验一建议：

- 横轴：等长 prompt 的 `L`；
- 曲线：不同固定 `B`；
- 纵轴：总 FLOPs、FLOPs/input-token、logical Bytes/input-token、Byte/FLOP；
- attention、权重、KV/index/state 使用分项曲线或 CSV 字段保留。

实验二建议：

- 固定一个或多个 `T`；
- 横轴使用 `L` 或标注为 `B × L` 的 shape；
- 同时展示 backbone FLOPs、attention FLOPs、LM-head FLOPs 和总量；
- 不要只画 per-token 指标，否则会隐藏一次 batch 的总压力。

实验三建议：

- 对相同的真实 `lengths` 分别计算 padding、varlen 和 packing；
- 展示 token efficiency、causal pair-slot efficiency 和总 FLOPs/Bytes；
- 对一段 trace 汇总 p50/p90/p95/p99，而不是只报均值；
- 保留原始长度向量或 trace 标识，确保结果可重现。

### 10.1 仓库中的直接生成脚本

安装 `requirements-plot.txt` 后，可以一次生成三组实验的完整 JSON、明细
CSV、图表用汇总 CSV、PNG 和 SVG：

```bash
python3 scripts/generate_prefill_plots.py --precision 16
```

默认输出到 `outputs/prefill/16bit/`。工作量图的四个面板分别是每有效输入
token 的计算量、logical-HBM 搬运量、operand-stream 流量边界以及
logical Byte/FLOP（图中使用等价的 `TB/s per PFLOPS`）；ragged 另有
token/pair-slot efficiency 图。总 batch 工作量
及全部 FLOPs/bytes 分项仍保存在 detail CSV 和 JSON 中。实验二固定 `T`，因此
按 input token 归一化只相差同一个常数 `T`，不会改变 batch shape 对比曲线。

使用 `--experiments`、`--models`、`--prompt-lengths`、`--equal-batches`、
`--token-budgets`、`--token-budget-batches`、可重复的 `--ragged-lengths` 和
`--ragged-execution-modes` 可以缩小或替换默认扫描点；完整示例见 README。

## 11. 实现时的最小假设集

为了让计算结果有明确含义，每次 prefill 分析至少需要确定：

1. 每个请求的新 token 长度；使用 prefix cache/chunk 时还需要 cached length。
2. 实验输入是等长 shape 还是 ragged；执行模式选择 `varlen`、`packed` 或 `padded`。
3. Causal attention 是否包含自身对角项。
4. LM head 使用 `last`、`all` 还是非最终 chunk 的 `none`。
5. LM-head 参数如何从 always-active 参数中拆分。
6. KV/index/state 在 prefill 中的 materialization、持久化和 HBM residency 假设。
7. MoE 使用分析路由模式还是真实 trace。
8. 是否存在 kernel profile；若没有，只输出 compulsory/logical-HBM 与 attention/index operand 两套 view，不输出伪精确的实际 HBM。

在缺少真实 serving trace 和 kernel profile 时，一套可用的第一版默认值是：

```text
ordinary prefill: C_i = 0, Q_i = L_i
execution:        ideal varlen（等长实验与其等价）
causal diagonal:  included
LM head:          last
HBM:              README 定义的逻辑 HBM
operand view:     只展开 attention/index read，与 work 二选一比较
recurrent state:  两套 view 都使用 fused scan
MoE routing:      uniform_independent，并标记为分析近似
```

## 12. 数值自检清单

实现完成后，可用以下恒等式做基础测试：

- `B=1, L=1` 且包含对角项时，full attention pairs 等于 1。
- 等长 batch 满足 `T = B * L`。
- 等长 full attention 满足 `pairs = B * L * (L+1) / 2`。
- 固定 `T` 的等长 shape 满足 `pairs = T * (L+1) / 2`。
- 固定 `T` 且不能整除 `B` 时，均衡长度向量仍严格满足 `sum(lengths) = T`，且最大、最小长度之差不超过 1。
- Ragged varlen 满足 `pairs = sum(L_i*(L_i+1)/2)`。
- Padding 下 `executed_tokens >= valid_tokens`，`executed_pairs >= valid_pairs`。
- `last` 模式下 `logit_positions = B`；`all` 的 useful/varlen/packed positions 为 `T`，padded batch executed positions 为 `B * L_max`；`none` 为 0。
- Varlen/packed 的 dense backbone 参数 FLOPs 对有效 token 数呈线性关系；padded batch work 对 executed token 数呈线性关系。
- 普通 full-attention prefill 的最终有效 KV entry 数等于 `T`，padded 位置不增加最终 KV/index/state capacity。
- 一次未切 chunk 的 dense 权重逻辑读取不应因为 `L` 增加而按 token 重复计数。
- `operand_work` 不应与 `work` 相加；两者的 recurrent-state 分项应完全相同。

## 13. 如何理解三组实验的关系

可以用三个问题概括：

1. **固定 B、扫描 L**：模型面对越来越长的 prompt 时，工作量如何增长？
2. **固定 T、改变 shape**：相同 token budget 被一个长请求或许多短请求占用时，成本是否相同？
3. **真实 ragged**：现实中的长短请求混合，在 padding、varlen、packing 下各浪费多少？

第一组给出清晰的架构趋势，第二组揭示调度 shape 的影响，第三组把公式带回真实 serving 分布。三者共同使用同一个长度向量接口和同一套工作量分项，才能让模型配置、调度策略和 kernel 实现之间的差异被正确定位。
