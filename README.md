# LLM Decode 每 Token 工作量计算引擎

这个目录提供一个配置驱动的 Python 计算引擎，用于计算 LLM 在 decode 阶段每生成一个 token 所需的：

- 参数矩阵计算量、注意力扫描计算量、索引计算量和状态更新计算量；
- 权重读取、KV Cache 读写、索引读写、固定状态读写和激活溢出流量；
- `Byte/FLOP`，以及等价的 `TB/s per PFLOPS`；
- 每个请求的 KV、索引及固定状态容量。

项目只依赖 Python 标准库，建议使用 Python 3.10 或更高版本。

## 1. 最重要的计算口径

引擎先计算完整的 decode step，再除以本 step 产生的 token 数：

```text
step_work = weight_work_for_batch
          + sum(sequence_work(context_i) for each request i)

per_output_token_work = step_work / batch_size
```

这样有三个直接结果：

1. Dense 权重在 batch 内只读取一次，然后按 batch 摊薄；
2. KV Cache 和 recurrent state 是请求私有数据，不会被错误地当成共享权重；
3. 计算结果同时保留 step 总量和每输出 token 总量，不再混淆两种单位。

默认统计整模型在所有设备上的逻辑 HBM 流量，不包含卡间通信流量、调度开销和 kernel 实现造成的重复读取。

## 2. 快速使用

在本目录执行：

```bash
python3 -m decode_engine \
  --config configs/gqa_8b_document_example.json \
  --contexts 128 512 2048 8192 32768 131072 \
  --batches 1 32
```

输出 CSV，供后续 matplotlib 或其他绘图程序使用：

```bash
python3 -m decode_engine \
  --config configs/deepseek_r1_mla_bf16.json \
  --format csv \
  --output deepseek_r1_decode.csv
```

`configs/` 中的真实模型配置如下：

| 配置 | 主要序列机制 | 默认部署精度 |
|---|---|---|
| `deepseek_r1_mla_bf16.json` | MLA | BF16 |
| `deepseek_v4_pro.json` | shared-KV + HCA/CSA | 官方 FP8/FP4 混合精度 |
| `glm_5_2_dsa_bf16.json` | MLA + DSA + IndexShare | BF16，router/index head 为 FP32 |
| `qwen3_235b_a22b_bf16.json` | GQA + MoE | BF16 |
| `llama_3_3_70b_bf16.json` | GQA | BF16 |
| `qwen3_8b_bf16.json` | GQA | BF16 |
| `qwen3_4b_bf16.json` | GQA | BF16 |
| `qwen3_next_80b_a3b_bf16.json` | Gated DeltaNet + GQA + MoE | BF16 |
| `mamba_2_8b_bf16.json` | Mamba-1 SSM | BF16，A_log/D 为 FP32 |

V4-Pro 配置把 31 个 HCA 层和 30 个 CSA 层拆成 window、compressed 和 learned-top-k 分支，并保留 FP8 shared expert、FP4 routed expert、FP32 mHC 等权重组的不同位宽。GLM-5.2 则把 21 个实际执行 DSA indexer 的层与 57 个复用 top-k 结果的层分开，避免把 IndexShare 错算成每层都扫描索引。

每个真实模型配置的 `metadata` 都保存了官方来源、原始结构字段、活跃参数拆分以及未计入项。JSON 不支持注释，因此这些元数据就是配置的可审计说明，不参与引擎公式。

`examples/mechanism_catalog.json` 集中展示 MHA、MQA、GQA、MLA、SWA、DSA、CSA、HCA、线性注意力、SSM、Mamba-1 和 Mamba-2 的配置语法。它只是机制目录，不是一个真实 checkpoint。

如果不传 `--contexts` 和 `--batches`，CLI 使用配置文件中 `analysis` 下的默认值。

### 2.1 生成统一数据与曲线

绘图是可选功能，计算引擎本身仍只依赖 Python 标准库。先在隔离环境中安装绘图依赖：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-plot.txt
```

精度对比配置位于 `configs/4bit`、`configs/8bit` 和 `configs/16bit`。三者都强制相应的权重与 KV 位宽，index/state 仍保留独立配置。基础模型结构变化后，可以重新生成这些 profile：

```bash
.venv/bin/python scripts/generate_precision_configs.py --bits 4 8 16
```

分别生成三种精度的统一数据和两张 batch 曲线：

```bash
.venv/bin/python scripts/generate_decode_ratio_plots-4bit.py
.venv/bin/python scripts/generate_decode_ratio_plots_8bit.py
.venv/bin/python scripts/generate_decode_ratio_plots_16bit.py
```

默认扫描以下 14 个上下文长度：

```text
128, 256, 512, 1K, 2K, 4K, 8K, 16K,
32K, 64K, 128K, 256K, 512K, 1024K
```

输出分别位于 `outputs/decode_ratio/{4bit,8bit,16bit}/`：

- `decode_ratio_all.csv`：9 个模型、2 个 batch 的 252 个完整数据点；
- `decode_ratio_batch_1.csv`、`decode_ratio_batch_32.csv`：按图拆分的数据；
- `decode_ratio_batch_1.png`、`decode_ratio_batch_32.png`：位图；
- 同名 `.svg`：适合报告排版和无损缩放的矢量图。

图的纵轴是 `TB/s per PFLOPS = 1000 * Byte/FLOP`。CSV 同时保存 `total_bytes_per_token`、`total_flops_per_token`、`bytes_per_flop`、`tbps_per_pflops` 以及权重、KV、索引和状态的各个分量。

为了让所有模型都有完整的 14 个横轴点，脚本会对超过配置中 `max_context_tokens` 的位置作理论外推。外推点在 CSV 中标记为 `is_extrapolated=true`，图中使用虚线；它们表示架构公式的延伸，不代表 checkpoint 官方支持该上下文长度。

## 3. 配置结构

一个配置文件描述“模型结构 + 部署选择 + 默认扫描范围”：

```json
{
  "schema_version": 1,
  "model": {
    "name": "example",
    "max_context_tokens": 131072,
    "weights": {},
    "layer_groups": [],
    "metadata": {}
  },
  "deployment": {},
  "analysis": {
    "contexts": [128, 512, 2048],
    "batches": [1, 32]
  }
}
```

### 3.1 参数权重

```json
"weights": {
  "always_active_parameters": 8000000000,
  "routed_expert_groups": []
}
```

`always_active_parameters` 是一个 token 实际执行的所有非 routed-expert 参数化矩阵元素数，包括：

- Q/K/V/O 等 attention projection；
- Dense FFN；
- MoE router 和 shared experts；
- LM head；
- 其他每 token 必经的参数化 projection。

这些参数只在模型级别统计一次。层组中的 attention mixer 只统计 QK、AV、indexer scan、状态更新等非参数化计算，不能再次加入 projection 参数，否则会重复计数。

如果 always-active 权重采用不同精度，可以改用分组形式：

```json
"always_active_parameter_groups": [
  {"name": "attention_router_head", "parameters": 12000000000, "weight_bits": 8},
  {"name": "shared_experts", "parameters": 2500000000, "weight_bits": 8},
  {"name": "special_connection", "parameters": 100000000, "weight_bits": 32}
]
```

`always_active_parameters` 是上述写法只有一个统一精度参数组时的简写，两种形式不能同时出现。

参数化计算量为：

```text
parameter_flops_per_token = mac_flops * active_parameter_elements
```

默认 `mac_flops=2`。

### 3.2 MoE routed experts

```json
"routed_expert_groups": [
  {
    "name": "routed_ffn_experts",
    "layers": 58,
    "expert_count": 256,
    "selected_per_token": 8,
    "parameters_per_expert": 44040192,
    "routing_mode": "uniform_independent"
  }
]
```

`parameters_per_expert` 表示单层中一个 expert 的参数量。Shared expert 必须计入 `always_active_parameters`，不能放在 routed group 中。

默认采用均匀独立路由近似。Batch 内预计触达的 routed expert 并集为：

```text
E_unique(B) = E * [1 - (1 - k/E)^B]
```

可选路由模式：

- `uniform_independent`：默认分析近似；
- `same_experts`：所有请求选择相同专家，代表最佳复用边界；
- `no_batch_reuse`：每个 token 的 expert 权重均重新读取，代表流量上界；
- `explicit_unique`：从实测路由 trace 填写不同 batch 对应的专家并集。

实测配置示例：

```json
"routing_mode": "explicit_unique",
"expected_unique_experts_by_batch": {
  "1": 8,
  "32": 147.5
}
```

### 3.3 Layer group

模型通过若干层组描述异构结构：

```json
{
  "name": "local GQA layers",
  "layers": 30,
  "mixers": [
    {
      "kind": "softmax_attention",
      "kv_layout": {},
      "access": {}
    }
  ]
}
```

一个层组可以包含多个 mixer。例如 V4 风格的每层 `window + compressed` 双分支，可以在同一层组的 `mixers` 数组中放入两个 attention mixer；不同层机制则拆成多个 layer group。

无法归入通用机制的小项可以使用 `fixed_cost`。例如 FP8 input embedding lookup 可写为：

```json
{
  "name": "input embedding lookup",
  "layers": 1,
  "mixers": [
    {
      "kind": "fixed_cost",
      "work": {"other_read_bytes": 7168},
      "cache": {}
    }
  ]
}
```

## 4. 支持的注意力表示

注意力由两个相互独立的部分组成：

1. `kv_layout` 决定每个历史 entry 的字节数和 QK/AV FLOPs；
2. `access` 决定一个 query 实际读取、写入和保存多少个 entry。

因此 GQA+SWA、MLA+DSA、shared-KV+CSA 等组合不需要新增硬编码公式。

### MHA、MQA、GQA

配置可以直接使用机制名称：

```json
"kv_layout": {
  "kind": "gqa",
  "query_heads": 32,
  "kv_heads": 8,
  "head_dim": 128
}
```

- `mha`：自动令 `kv_heads = query_heads`；
- `mqa`：自动令 `kv_heads = 1`；
- `gqa`：要求显式提供 `1 < kv_heads < query_heads`；
- `grouped`：兼容旧配置的通用写法。

每个历史 token 的 KV 字节数和注意力计算量分别为：

```text
entry_bytes = H_KV * d_head * (key_bits + value_bits) / 8
entry_flops = 4 * H_Q * d_head
```

GQA/MQA 只降低 KV 容量和搬运量，不降低按 Query head 执行的 QK/AV FLOPs。

### MLA

```json
"kv_layout": {
  "kind": "mla",
  "query_heads": 128,
  "latent_dim": 512,
  "rope_dim": 64
}
```

`mla` 与旧名称 `latent` 等价。每个历史 token 保存 `latent_dim + rope_dim` 个元素；absorbed MLA 的扫描计算量为：

```text
2 * H_Q * (2 * latent_dim + rope_dim) FLOPs/history-entry
```

可以分别配置 `latent_bits` 和 `rope_bits`。

### Shared KV

`shared_kv`（旧名称 `shared`）表示一个向量同时作为 key 和 value，例如 V4 的 shared-KV MQA。传统 MQA 仍保存独立的 K、V 两个向量，不能与 shared KV 混用。

## 5. 稀疏和压缩访问

`access.kind` 支持机制名称和旧通用名称：

| 机制 | `access.kind` | 主 attention 读取数 | 主 KV 保存数 | 额外索引 |
|---|---|---:|---:|---:|
| Full | `full` | `C` | `C` | 无 |
| SWA | `swa` / `sliding_window` | `min(C,W)` | `min(C,W)` | 无 |
| DSA | `dsa` | `min(k,C)` | `C` | 扫描 `C` 个 index entry |
| CSA | `csa` | `min(k,floor(C/m))` | `floor(C/m)` | 扫描 `floor(C/m)` 个 index entry |
| HCA | `hca` | `floor(C/m)` | `floor(C/m)` | 无 |

DSA/CSA 会自动同时计算主 KV 路径和 indexer 路径，不能再对总流量简单乘一个稀疏率。示例 CSA 分支：

```json
"access": {
  "kind": "csa",
  "compression_ratio": 4,
  "top_k": 1024,
  "index_entry_elements": 128,
  "index_query_heads": 64,
  "index_head_dim": 128,
  "index_bits": 4
}
```

这里 indexer score 的默认 FLOPs 为：

```text
2 * index_query_heads * index_head_dim * candidate_entries
```

`selection_flops_per_candidate` 可以补充 ReLU、head weighting 或选择算法的项目约定。`fixed_topk` 仍可用于不需要动态 indexer 的结构化稀疏机制。

重要：V4 风格 CSA/HCA 每层还包含一个独立 SWA 分支。`csa`/`hca` 只描述压缩分支，配置时应在同一 layer group 中再放一个 `swa` mixer；`examples/mechanism_catalog.json` 给出了完整组合。

## 6. 线性注意力、SSM 与 Mamba

这些机制不随上下文长度扫描 KV，而是读写固定大小的请求私有状态。参数 projection、卷积权重和输出 projection 仍计入 `always_active_parameters`；下面的 mixer 只补充非参数化 recurrence 与状态流量。

### 线性注意力

```json
{
  "kind": "linear_attention",
  "query_heads": 32,
  "key_dim": 128,
  "value_dim": 128,
  "normalizer_state": true,
  "state_bits": 16
}
```

默认推导：

```text
matrix_state = H * key_dim * value_dim
normalizer_state = H * key_dim                         # 可关闭
state_flops = 4 * matrix_state
            + 3 * normalizer_state
            + H * value_dim                            # normalization divide
```

其中 `4 * matrix_state` 包括 KV outer-product 更新与 query/state contraction。机制特有的 feature map、gate 等可用 `extra_flops_per_token` 补充。

### 通用对角 SSM

```json
{
  "kind": "ssm",
  "channels": 8192,
  "state_dim": 16,
  "conv_state_length": 0,
  "recurrence_flops_per_state_element": 5
}
```

```text
state_elements = channels * (state_dim + conv_state_length)
state_flops = channels * state_dim * recurrence_flops_per_state_element
```

默认的 5 FLOPs 表示已生成离散系数后的 `A*x + B*u` 更新和 `C*x` 输出收缩。不同 SSM 的离散化、指数函数与 gate 不统一，可以调整系数并用 `extra_flops_per_token` 加项。

### Mamba-1 与 Mamba-2

```json
{
  "kind": "mamba2",
  "inner_dim": 8192,
  "ssm_dim": 8192,
  "state_dim": 128,
  "conv_kernel": 4,
  "groups": 1
}
```

状态尺寸按官方 inference cache 结构推导：

```text
Mamba-1:
  ssm_state  = inner_dim * state_dim
  conv_state = inner_dim * conv_kernel

Mamba-2:
  ssm_state  = ssm_dim * state_dim
  conv_state = (ssm_dim + 2 * groups * state_dim) * conv_kernel
```

`ssm_dim` 仅用于 Mamba-2，默认等于 `inner_dim`；当 block 的一部分是 gated MLP 时，应填写实际参与 SSM 的维度。状态形状对应 [Mamba-1](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) 和 [Mamba-2](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py) 官方实现。

无法归入上述推导的 Gated DeltaNet、KDA 或定制 SSM，可以继续使用完全显式的 `recurrent_state`：

```json
{
  "kind": "recurrent_state",
  "state_elements": 1048576,
  "read_elements_per_token": 1048576,
  "write_elements_per_token": 1048576,
  "flops_per_token": 4194304,
  "state_bits": 16
}
```

默认假设每个 decode step 从 HBM 读取并写回整个状态。如果状态跨 token 常驻片上，将 deployment 中的 `state_hbm_fraction` 设为 `0`；这只消除片外流量，不会删除状态容量和 recurrence FLOPs。

## 7. Deployment 参数

常用字段：

```json
"deployment": {
  "weight_bits": 8,
  "expert_weight_bits": 4,
  "kv_bits": 8,
  "index_bits": 4,
  "state_bits": 16,
  "include_kv_write": true,
  "weight_hbm_fraction": 1.0,
  "kv_hbm_fraction": 1.0,
  "index_hbm_fraction": 1.0,
  "state_hbm_fraction": 1.0
}
```

`*_hbm_fraction` 只影响数据搬运，不影响逻辑缓存容量。当前版本不自动求解片上驻留分配；它只接受明确的部署假设。

## 8. 输出口径

CLI 默认表格包含：

- `GFLOP/token`；
- `GB/token`；
- `Byte/FLOP`；
- `TB/s/PFLOPS = 1000 * Byte/FLOP`；
- 权重和 KV 读取分项；
- 每请求缓存容量。

JSON 输出还保留完整的 decode-step 总量、每 token 分项和每层预计读取的 expert 权重集合数。CSV 使用长表格式，适合直接绘制 batch=1 和 batch=32 的两组曲线。

## 9. 验证

运行全部单元测试：

```bash
python3 -m unittest discover -s tests -v
```

测试包含：

- EBpC 文档中的 GQA 8B 数值锚点；
- Dense batch 权重摊薄；
- MoE batch 专家并集；
- MHA、MQA、GQA、MLA 命名布局及扫描公式；
- SWA、DSA、CSA、HCA 的读取、存储与 indexer 双路径；
- 显式 recurrent state 的上下文无关性；
- 线性注意力、SSM、Mamba-1、Mamba-2 的状态推导；
- DeepSeek-V4-Pro 1M context 架构锚点和 shared-expert 精度；
- continuous batching 的不同上下文求和。

## 10. 当前边界

- 只计算 decode，不计算 prefill；
- 默认忽略 softmax、RMSNorm、激活函数等小项，可通过配置加入；
- SSM/Mamba 默认 recurrence 系数不把指数、softplus 和 gate 强行折算成普通 FLOPs，精确项目需通过可配置系数和额外 FLOPs 补充；
- 不统计 tensor/expert parallel 的卡间通信；
- 不根据峰值算力推断利用率；
- MoE 均匀路由只是分析近似，严谨结果应由真实 routing trace 校准；
- 配置中的参数和位宽必须附带来源，模型名称本身不会触发任何隐藏公式。
