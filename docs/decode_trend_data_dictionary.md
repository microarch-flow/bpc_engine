# Decode 趋势研究字段数据字典

版本：v1.0
适用数据：`decode-trend-v1.0.0`
范围：模型配置、研究事实、P3审计和统一Decode结果

## 1. 通用规则

| 规则 | 定义 |
|---|---|
| `C` | Decode step开始前，每个请求已经存在的上下文Token数 |
| `B` | 本step同时推进、各生成一个Token的请求数 |
| element | 参数或状态元素个数，不隐含数据类型 |
| FLOPs | 浮点等价运算量；项目统一`1 MAC = 2 FLOPs` |
| bytes | base-SI字节，不使用GiB作为存储字段 |
| `null` | 未知或不适用但无法由字段类型表达；禁止用0代替未知 |
| `0` | 已确认该项不存在或计算结果确实为0 |
| step | 完整scheduler step的总量 |
| per token | 完整step总量除以`B` |

所有结果描述逻辑超大芯片边界：完整模型权重只保存一份，不统计TP/PP/EP/CP
通信、复制、Workspace、分页碎片、临时Buffer和实际Kernel利用率。

## 2. 模型计算配置

配置文件位于`configs/<year>/*.json`，解析规则以
[`decode_engine/config.py`](../decode_engine/config.py)为准。

### 2.1 顶层字段

| 字段 | 类型 | 单位 | 含义 |
|---|---|---:|---|
| `schema_version` | integer | — | 配置语法版本，当前必须为1 |
| `model` | object | — | 模型结构与权重计算路径 |
| `deployment` | object | — | 部署精度和logical-HBM假设 |
| `analysis` | object | — | CLI默认扫描点，不改变模型结构 |

### 2.2 `model`

| 字段 | 类型 | 单位 | 含义 |
|---|---|---:|---|
| `name` | string | — | 展示名称，不能作为稳定主键 |
| `max_context_tokens` | integer/null | token | 配置原生最大上下文；研究外推需显式允许 |
| `weights` | object | — | 参数矩阵和权重读取定义 |
| `layer_groups` | array | — | 具有相同序列机制的层组 |
| `metadata` | object | — | 来源、拆分和边界说明；引擎不读取其中公式 |

`metadata`允许模型特有字段，因此不作为统一计算接口。进入跨模型分析的事实必须
回填到`studies/decode_trend/models.json`。

### 2.3 `model.weights`

| 字段 | 类型 | 单位 | 含义 |
|---|---|---:|---|
| `always_active_parameters` | number | element | 单一精度时每Token必经的参数矩阵元素数 |
| `always_active_parameter_groups` | array | — | 多精度必经参数组；与上一字段二选一 |
| `always_active_parameter_groups[].name` | string | — | 参数组稳定名称 |
| `always_active_parameter_groups[].parameters` | number | element | 该组矩阵参数元素数 |
| `always_active_parameter_groups[].weight_bits` | number/null | bit/element | 局部存储位宽；空时继承权重默认值 |
| `weight_bits` | number/null | bit/element | 模型权重局部默认值 |
| `output_head_parameters` | number | element | Always-active中的LM Head子集，不是额外参数 |
| `output_head_weight_bits` | number/null | bit/element | LM Head位宽 |
| `routed_expert_groups` | array | — | Routed Expert参数组 |

Always-active组包括Attention Projection、Dense FFN、Router、Shared Expert、
LM Head等参数矩阵。Embedding lookup只读取一行，不把整张Embedding作为每Token流量。

### 2.4 `routed_expert_groups[]`

| 字段 | 类型 | 单位 | 含义 |
|---|---|---:|---|
| `name` | string | — | Expert组名称 |
| `layers` | integer | layer | 使用该Expert集合的层数 |
| `expert_count` | integer | expert/layer | 每层Routed Expert总数`E` |
| `selected_per_token` | integer | expert/token/layer | 每Token选择数`k` |
| `parameters_per_expert` | number | element/expert/layer | 单层单Expert矩阵参数 |
| `weight_bits` | number/null | bit/element | Expert局部位宽 |
| `routing_mode` | enum | — | Batch内Expert权重集合触达模型 |
| `expected_unique_experts_by_batch` | object | expert | Decode实测/指定Expert并集 |
| `expected_unique_experts_by_active_tokens` | object | expert | Prefill实测/指定Expert并集 |

`routing_mode`：

| 值 | 含义 |
|---|---|
| `uniform_independent` | `E × [1-(1-k/E)^N]`统计近似 |
| `same_experts` | 所有Token选中相同Expert，最佳复用边界 |
| `no_batch_reuse` | 每Token重新读取Expert，流量上界 |
| `explicit_unique` | 使用配置中的实测或外部给定并集 |

Decode中`N=B`。Shared Expert不能放在Routed组，必须属于Always-active组。

### 2.5 `layer_groups[]`

| 字段 | 类型 | 单位 | 含义 |
|---|---|---:|---|
| `name` | string | — | 层组说明 |
| `layers` | integer | layer | 该结构重复层数 |
| `mixers` | array | — | 每层包含的序列机制；同层可有多个分支 |

每个Mixer返回单层、单请求的非参数序列工作量与持久Cache，再乘`layers`。
Projection等参数矩阵已在`weights`中，不能在Mixer重复计算。

### 2.6 Attention Mixer

公共结构：

```text
kind = softmax_attention
kv_layout = 每个历史entry的字节和QK/AV计算
access = 读取、写入和保存多少个entry
softmax_flops_per_score = 可选项目约定，默认0
```

KV Layout字段：

| `kv_layout.kind` | 必需字段 | 每历史Token逻辑含义 |
|---|---|---|
| `mha` | `query_heads, head_dim` | `kv_heads=query_heads`，独立K/V |
| `mqa` | `query_heads, head_dim` | `kv_heads=1`，独立K/V |
| `gqa` | `query_heads, kv_heads, head_dim` | 分组K/V |
| `mla` | `query_heads, latent_dim, rope_dim` | absorbed latent与RoPE key |
| `shared_kv` | `query_heads, head_dim[, rope_dim]` | 同一向量同时作为K/V |
| `explicit` | `query_heads, bytes_per_entry, flops_per_entry` | 已发表机制的显式系数 |

局部精度字段`key_bits/value_bits`、`latent_bits/rope_bits`、
`non_rope_bits/rope_bits`优先于`deployment`默认值。

Access字段：

| `access.kind` | 主要字段 | Decode读取数 | Cache保存数 |
|---|---|---:|---:|
| `full` | — | `C` | `C` |
| `swa` | `window_tokens` | `min(C,W)` | `min(C,W)` |
| `chunked_block` | `chunk_tokens, retain_full_history` | `C mod W` | `C`或`C mod W` |
| `compressed_full`/`hca` | `compression_ratio` | `floor(C/m)` | `floor(C/m)` |
| `fixed_topk` | `top_k[, compression_ratio]` | `min(k,floor(C/m))` | `floor(C/m)` |
| `learned_topk`/`dsa`/`csa` | Top-k与Indexer字段 | `min(k,floor(C/m))` | 主Cache与Index Cache |

Learned Top-k字段：

| 字段 | 单位 | 含义 |
|---|---:|---|
| `top_k` | entry | 主Attention选中的entry数 |
| `compression_ratio` | token/entry | DSA固定为1；CSA必须大于1 |
| `index_entry_elements` | element/entry | 每个持久Index key大小 |
| `index_query_heads` | head | Index query heads |
| `index_head_dim` | element/head | Index dot-product维度 |
| `index_bits` | bit/element | Index存储位宽 |
| `selection_flops_per_candidate` | FLOPs/candidate | 可审计的ReLU、加权等额外成本 |

### 2.7 Recurrent Mixer

| `kind` | 主要字段 | 含义 |
|---|---|---|
| `recurrent_state` | `state_elements, read_elements_per_token, write_elements_per_token, flops_per_token` | KDA等显式状态模型 |
| `linear_attention` | `query_heads, key_dim, value_dim, normalizer_state` | 矩阵状态线性注意力 |
| `ssm` | `channels, state_dim, conv_state_length` | 对角SSM |
| `mamba`/`mamba2` | `inner_dim, state_dim, conv_kernel[, ssm_dim, groups]` | Mamba状态与卷积历史 |

公共可选字段：

| 字段 | 单位 | 含义 |
|---|---:|---|
| `state_bits` | bit/element | 局部状态精度 |
| `read_hbm_fraction` | ratio | 状态读取到达HBM的比例 |
| `write_hbm_fraction` | ratio | 状态写入到达HBM的比例 |
| `extra_flops_per_token` | FLOPs/token/layer | 已审计的额外计算 |

`fixed_cost`仅用于Embedding一行读取等小型显式加项：

| 字段 | 含义 |
|---|---|
| `work` | 与WorkCost同名的固定分项 |
| `cache` | 与CacheCapacity同名的固定容量 |
| `prefill_scope` | `per_token`或`per_request`；Decode始终每请求每Token一次 |

### 2.8 `deployment`

| 字段 | 类型/单位 | 默认 | 含义 |
|---|---|---:|---|
| `weight_bits` | bit/element | 必填 | Always-active默认位宽 |
| `expert_weight_bits` | bit/element | `weight_bits` | Routed Expert默认位宽 |
| `kv_bits` | bit/element | 必填 | KV默认位宽 |
| `index_bits` | bit/element | `kv_bits` | Index默认位宽 |
| `state_bits` | bit/element | 16 | State默认位宽 |
| `mac_flops` | FLOPs/MAC | 2 | MAC换算 |
| `include_kv_write` | boolean | true | 是否统计KV写流量 |
| `include_index_write` | boolean | true | 是否统计Index写流量 |
| `include_state_write` | boolean | true | 是否统计State写流量 |
| `*_hbm_fraction` | ratio `[0,1]` | 1 | 对应逻辑读写到达HBM的比例 |
| `weight_read_multiplier` | ratio `>0` | 1 | 权重重复读取系数 |
| `activation_bytes_per_output_token` | bytes/token | 0 | Decode诊断Activation流量 |
| `extra_flops_per_output_token` | FLOPs/token | 0 | Decode全局显式加项 |

趋势研究的历史实际部署profile统一使用`*_hbm_fraction=1`，表示保守logical-HBM
边界，不代表真实芯片缓存命中率。

## 3. 统一模型研究事实

文件：`studies/decode_trend/models.json`。

### 3.1 顶层

| 字段 | 含义 |
|---|---|
| `schema_version` | 事实清单语法版本 |
| `study_id` | 研究稳定ID |
| `ytd_cutoff` | 2026YTD资料截止日期 |
| `mechanism_audit_path` | P3机器审计相对路径 |
| `common_contexts` | 所有模型公共`C`扫描点 |
| `main_batches` | 正式`B`扫描点 |
| `stress_batches` | 可选压力`B`点，不进入本次正式主网格 |
| `models` | 精确release与deployment profile列表 |

### 3.2 `models[]`

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `model_release_id` | string | 精确release稳定主键 |
| `deployment_profile_id` | string | 历史实际部署profile稳定ID |
| `year` | integer | 样本年份 |
| `organization` | string | 发布组织 |
| `release_date` | ISO date | 精确发布日期 |
| `config_path` | path | 引擎配置，相对清单文件解析 |
| `checkpoint` | string | checkpoint或论文profile标识 |
| `checkpoint_revision` | string/null | 固定revision；不可获得时为null |
| `sample_roles` | array | 旗舰、采用、效率、机制等样本用途 |
| `parameters.decode_resident_parameter_elements` | element | Decode profile需要常驻的原始参数元素 |
| `parameters.active_matrix_parameter_elements_per_token` | element/token | 单Token执行的参数矩阵元素 |
| `capacity.decode_profile_weight_capacity_bytes` | bytes | 标准Decode路径权重与量化metadata容量 |
| `capacity.full_checkpoint_capacity_bytes` | bytes/null | 完整checkpoint tensor payload |
| `capacity.status` | enum | exact、derived或estimated状态 |
| `capacity.note` | string | 容量边界与差异解释 |
| `context.advertised_max_context_tokens_at_release` | token | 发布时宣称上限 |
| `context.trained_max_context_tokens` | token/null | 可审计训练上限 |
| `context.evaluated_max_context_tokens` | token/null | 可审计评测上限 |
| `context.deployed_max_context_tokens` | token/null | 可审计部署上限 |
| `context.effective_context_observations` | array | 任务、阈值、评测和长度绑定的观察 |
| `context_anchors` | array | 模型上限与机制边界点 |
| `calculation_status` | object | FLOPs、Traffic、Capacity基础审计状态 |
| `source_refs` | object | 指向配置`metadata.sources`的来源ID |
| `known_gaps` | array | 缺失、估算和不可审计项 |

两种容量不能相互替代。完整checkpoint不可核验时必须为`null`，此时所有由它派生
的完整常驻容量也为`null`。

## 4. P3机制审计字段

文件：`studies/decode_trend/mechanism_audit.json`。

| 字段 | 含义 |
|---|---|
| `audit_id` | 审计稳定版本 |
| `dimensions` | `flops/traffic/cache/weight`四个审计维度 |
| `status_definitions` | 状态正式定义 |
| `profiles` | 可复用机制审计模板 |
| `profiles.<id>.dimensions` | 每维支持状态 |
| `profiles.<id>.boundary` | 公式与部署边界 |
| `profiles.<id>.anchors` | 必须验证的数值锚点 |
| `models[].overall` | 模型级最弱状态 |
| `models[].mechanisms` | 配置机制或Routed Expert组到审计模板的映射 |
| `models[].mechanisms[].overrides` | 有部署证据时的模型级状态覆盖 |
| `models[].known_gaps` | P3缺口 |

状态为：

- `supported`：可进入对应指标的高置信分析；
- `partially_supported`：可计算，但必须保留假设并做敏感性分析；
- `unsupported`：不得进入中心拟合；
- `not_applicable`：机制不产生该维度。

## 5. 统一结果

正式数据同时保存JSONL和CSV。JSONL是完整规范结构，CSV是分析便利视图。

### 5.1 记录标识与输入

| 字段 | 类型/单位 | 含义 |
|---|---|---|
| `result_schema_version` | integer | 结果语法版本 |
| `run_id` | string | 单次生成运行ID |
| `study_id` | string | 研究ID |
| `model_release_id` | string | 模型release外键 |
| `deployment_profile_id` | string | 部署profile外键 |
| `config_sha256` | hex string | 实际计算配置SHA-256 |
| `context_C` | token | Start-of-step上下文 |
| `concurrency_B` | request | Active Decode请求数 |
| `batch_class` | enum | `main/stress/custom` |

CSV额外展开`year/organization/release_date/checkpoint/config_path`等模型事实，
避免简单统计时必须Join。

### 5.2 `scope`

| 字段 | 类型 | 含义 |
|---|---|---|
| `is_extrapolated` | boolean | `C > advertised_max_context` |
| `within_advertised_context` | boolean | 是否位于发布宣称范围 |
| `within_trained_context` | boolean/null | 训练上限未知时为null |
| `within_evaluated_context` | boolean/null | 评测上限未知时为null |
| `within_deployed_context` | boolean/null | 部署上限未知时为null |
| `context_anchor_tags` | array | 公共点、机制边界和上下文事实标签 |

### 5.3 WorkCost

以下字段同时存在于`step_work`和`per_output_work`。CSV分别使用`step_`和
`per_token_`前缀。

| 基础字段 | 单位 | 含义 |
|---|---:|---|
| `parameter_flops` | FLOPs | Projection、FFN、Router、Shared/Routed Expert、LM Head |
| `attention_flops` | FLOPs | 主Attention QK与AV |
| `index_flops` | FLOPs | 稀疏Attention Indexer与显式selection成本 |
| `state_flops` | FLOPs | Linear/SSM/Mamba recurrence |
| `extra_flops` | FLOPs | 其他显式审计加项 |
| `weight_read_bytes` | bytes | 本step触达的矩阵权重逻辑读取 |
| `kv_read_bytes` | bytes | 历史KV读取 |
| `kv_write_bytes` | bytes | 新Token KV写入 |
| `index_read_bytes` | bytes | 历史Index读取 |
| `index_write_bytes` | bytes | 新Token Index写入 |
| `state_read_bytes` | bytes | 请求私有固定状态读取 |
| `state_write_bytes` | bytes | 请求私有固定状态写入 |
| `activation_bytes` | bytes | 诊断Activation流量；趋势标准值必须为0 |
| `other_read_bytes` | bytes | Embedding一行读取等显式其他读取 |
| `total_flops` | FLOPs | 五类FLOPs之和 |
| `total_bytes` | bytes | 引擎兼容总量，包含Activation |
| `logical_hbm_bytes` | bytes | 趋势总量，等于`total_bytes-activation_bytes` |

关系：

```text
per_output_work = step_work / B
```

Dense权重通常每step读取一次，因此其per-token值随`B`摊薄；KV/Index/State是
请求私有流量，不按请求间共享。

### 5.4 `capacity`

| 字段 | 单位 | 含义 |
|---|---:|---|
| `decode_profile_weight_capacity_bytes` | bytes | 标准Decode权重常驻容量 |
| `full_checkpoint_capacity_bytes` | bytes/null | 完整checkpoint tensor容量 |
| `kv_cache_bytes_per_request` | bytes/request | 单请求start-of-step KV |
| `index_cache_bytes_per_request` | bytes/request | 单请求Index |
| `state_cache_bytes_per_request` | bytes/request | 单请求固定State |
| `cache_bytes_per_request` | bytes/request | 上述三项之和 |
| `kv_cache_bytes_total` | bytes | `B`个请求KV总容量 |
| `index_cache_bytes_total` | bytes | `B`个请求Index总容量 |
| `state_cache_bytes_total` | bytes | `B`个请求State总容量 |
| `batch_cache_bytes` | bytes | 三类Batch Cache总和 |
| `persistent_decode_profile_bytes` | bytes | Decode权重容量加Batch Cache |
| `persistent_full_checkpoint_bytes` | bytes/null | 完整checkpoint容量加Batch Cache |

Cache capacity是持久容量，不是流量，禁止与`logical_hbm_bytes`相加。

### 5.5 `derived`

| 字段 | 单位 | 公式 |
|---|---:|---|
| `logical_hbm_bytes_per_flop` | bytes/FLOP | `per_token_logical_hbm_bytes / per_token_total_flops` |
| `flops_per_logical_hbm_byte` | FLOPs/byte | 上式倒数 |
| `tbps_per_pflops` | TB/s per PFLOP/s | `1000 × bytes/FLOP` |
| `cache_to_decode_weight_capacity_ratio` | ratio | `batch_cache_bytes / decode_profile_weight_capacity_bytes` |

`tbps_per_pflops`是芯片带宽/算力需求比，不是实际吞吐或性能预测。

### 5.6 Expert与审计字段

| 字段 | 含义 |
|---|---|
| `expert_weight_sets_read` | 本step每个Routed组预计读取的Expert集合数 |
| `routing_assumptions` | 每组使用的路由模式 |
| `audit.flops/traffic/capacity` | 模型事实层计算状态 |
| `audit.assumptions` | 公式与部署假设 |
| `audit.p3_overall` | 模型四维P3状态 |
| `audit.p3_mechanisms` | 展开的逐机制状态、边界和锚点 |
| `audit.p3_known_gaps` | P3缺失项 |
| `audit.warnings` | 外推、部分支持和数值范围警告 |

CSV将P3状态展开为：

```text
p3_flops_support
p3_logical_hbm_traffic_support
p3_cache_capacity_support
p3_weight_capacity_support
```

JSON字符串形式的CSV字段（如`context_anchor_tags`、`expert_weight_sets_read`、
`routing_assumptions`和`p3_known_gaps`）需要再次JSON解析，不能按普通字符串分析。

## 6. 运行与冻结产物

| 文件 | 角色 |
|---|---|
| `run_manifest.json` | 运行时间、Git状态、输入哈希、扫描点和选中模型 |
| `model_profiles.jsonl` | 20个模型事实、配置摘要、机制层数和P3审计 |
| `decode_results.jsonl` | 完整规范结果，一行一个`模型×C×B` |
| `decode_results.csv` | 与JSONL同一批数值的扁平分析视图 |
| `validation_report.json` | 行数、外推数、告警和恒等式验证状态 |
| `release_manifest.json` | 正式数据版本、来源快照和规范数据入口 |
| `SHA256SUMS` | 冻结目录所有其他文件的SHA-256 |

正式版本目录中的`source_snapshot/`保留生成时使用的模型配置、研究事实、P3审计、
引擎源码和运行器。后续配置发生变化不能覆盖已有冻结版本，只能发布新版本。
