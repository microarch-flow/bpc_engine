# Decode 趋势结果字段能力矩阵

版本：v0.1
阶段：P2 实施后复核
范围：2022–2026YTD 的 20 个已确认模型，只计算实际部署精度

## 1. 目的

本文回答两个问题：

1. 统一结果 Schema 中的字段，当前计算引擎能否产生；
2. 哪些字段需要从模型事实补充，哪些必须修改代码。

状态定义：

| 状态 | 含义 |
|---|---|
| `engine_direct` | `DecodeResult` 已直接提供 |
| `derived` | 可由引擎结果和标准模型事实无损推导 |
| `config_extractable` | 配置中已有结构，但当前结果未导出 |
| `metadata_unstandardized` | 多数配置已记录，但字段路径或口径不统一 |
| `missing` | 当前没有可靠字段或实现 |
| `contract_gap` | 代码语义与趋势指标合同存在差异，必须先解决 |

## 2. 总体结论

当前引擎已经能够计算核心的：

- 完整 Decode step 和每输出 Token 的 FLOPs 分项；
- Weight、KV、Index、State 等 logical-HBM 流量分项；
- 单请求及 Batch 总 KV/Index/State 容量；
- MoE 在给定 `B` 下的预计 Expert 权重集合触达量。

本轮实施已经关闭以下执行缺口：

1. `studies/decode_trend/models.json` 已统一 20 个模型的 release、profile、参数、上下文和双容量事实；
2. 引擎支持显式 `allow_extrapolation`，默认仍拒绝越界；
3. 研究运行器已输出完整 Step/Token、稳定 ID、范围标记和运行追踪；
4. 已增加排除 Activation 的 `logical_hbm_bytes`，同时保留原 `total_bytes` 兼容语义；
5. 已保存 FLOPs、流量和容量的审计状态与假设。

尚未关闭的是字段级置信度/估算范围，以及正式的模型 × 机制 × 引擎支持矩阵。

## 3. 标识、输入与证据字段

| 目标字段 | 当前来源 | 状态 | 处理建议 |
|---|---|---|---|
| `model_family` | 各年度 README/模型名 | `metadata_unstandardized` | 写入统一研究元数据 |
| `model_release_id` | checkpoint、variant、revision 分散保存 | `metadata_unstandardized` | 定义稳定 ID，禁止使用展示名称作主键 |
| `deployment_profile_id` | 文件名和精度说明隐式表达 | `missing` | 为实际部署 profile 定义稳定 ID |
| `year`、`organization`、`release_date` | 年度目录及 metadata；19/20 配置有统一 `release_date` | `metadata_unstandardized` | GLM-5.2 等特殊字段统一迁移 |
| `sample_roles` | `selection_role`、`role` 或 README | `metadata_unstandardized` | 统一为字符串数组 |
| `config_path`、`config_sha256` | 运行时可获得 | `derived` | 由研究运行器写入 |
| `architecture_mechanisms` | `layer_groups` 与 metadata | `config_extractable` | 从机制配置提取标准标签 |
| `mechanism_layer_counts` | `layer_groups[].layers` | `config_extractable` | 由运行器提取，不手工重复维护 |
| `context_C` | `DecodeResult.context_tokens` | `engine_direct` | v0.1 只生成等长 Batch |
| `concurrency_B` | `DecodeResult.batch_size` | `engine_direct` | 当前语义是 active Decode requests |
| `advertised_context` | `model.max_context_tokens` 和不同 metadata 路径 | `metadata_unstandardized` | 保留 release 事实，不能被扫描上限覆盖 |
| `trained_context`、`evaluated_context`、`deployed_context` | 仅部分模型有 `context_profile` | `metadata_unstandardized` | 允许 `null`，每个值绑定来源 |
| `effective_context_observations` | 目前只对少数模型记录 | `metadata_unstandardized` | 保存为“任务/评测/阈值/长度”的观察数组，禁止压成单值 |
| `is_extrapolated` | 当前没有 | `missing` | 定义为 `C > advertised_context` |
| `context_anchor_tags` | 当前没有 | `missing` | 标记 power-of-two、机制边界、训练/部署/宣称上限 |
| `capability_gate`、`mechanism_effect_gate`、`deployment_gate` | Scout 等少数模型有记录 | `metadata_unstandardized` | 统一状态、证据和解释边界 |

## 4. 参数、精度与权重容量

| 目标字段 | 当前来源 | 状态 | 处理建议 |
|---|---|---|---|
| `total_parameter_elements` | 20 个配置均有参数拆分，但精确总量字段名不同 | `metadata_unstandardized` | 建立唯一标准字段，保留 reported/exact 口径 |
| `active_matrix_parameter_elements_per_token` | 权重组和 Expert Top-k 可重构；metadata 字段名不同 | `derived` | 由配置结构计算并与审计锚点比较 |
| `active_parameter_ratio` | 上述两项 | `derived` | 分母必须使用同一参数边界 |
| `dense_weight_bits`、`expert_weight_bits`、`output_head_weight_bits` | deployment 与参数组 | `config_extractable` | 输出有效的组级位宽，不用单个“模型精度”替代 |
| `kv_bits`、`index_bits`、`state_bits` | deployment 与机制 layout | `config_extractable` | 机制局部覆盖优先于全局默认值 |
| `weight_payload_bytes` | 仅 4/20 使用同名字段，其余字段各异 | `metadata_unstandardized` | 标准化为推理边界内全部唯一参数 payload |
| `quantization_metadata_bytes` | 15/20 有 weight-capacity 结构，10 个使用同名字段 | `metadata_unstandardized` | 未知使用 `null`，不能静默写 0 |
| `total_weight_capacity_bytes` | 10/20 使用同名字段，其余有等价但不同口径字段 | `metadata_unstandardized` | 明确 text-only/full-checkpoint/运行时边界 |

注意：`B=1` 的 `weight_read_bytes` 只读取本 Token 触达的 Expert，不等于包含全部 Expert 的权重容量，不能用它补 `total_weight_capacity_bytes`。

## 5. 计算量字段

下列字段在 `DecodeResult.step_work` 和 `per_output_work` 中均已存在：

```text
parameter_flops
attention_flops
index_flops
state_flops
extra_flops
total_flops
```

| 目标结果 | 状态 | 当前问题 |
|---|---|---|
| 完整 step 的全部 FLOPs 分项 | `engine_direct` | JSON 保留，Decode CSV 未展开 |
| 每输出 Token 的全部 FLOPs 分项 | `engine_direct` | Decode CSV 已展开 |
| `active_matrix_parameter_elements_per_token` | `derived` | 可用 `parameter_flops / mac_flops` 交叉检查，但应优先从配置结构重构 |
| `flops_status` | `missing` | 需要区分精确公式、显式假设和不支持 |

## 6. logical-HBM 流量字段

下列 Step 和每输出 Token 分项均由 `WorkCost` 直接提供：

```text
weight_read_bytes
kv_read_bytes
kv_write_bytes
index_read_bytes
index_write_bytes
state_read_bytes
state_write_bytes
activation_bytes
other_read_bytes
```

| 目标结果 | 状态 | 当前问题 |
|---|---|---|
| Step 流量分项 | `engine_direct` | JSON 保留，Decode CSV 未展开 |
| 每 Token 流量分项 | `engine_direct` | Decode CSV 已展开 |
| `logical_hbm_bytes_per_token` | `contract_gap` | 当前 `total_bytes` 包含 Activation；趋势合同明确排除 Activation |
| `traffic_status` | `missing` | MoE routing、量化 scale 等流量假设没有标准状态字段 |
| `expected_unique_experts_touched_per_step` | `engine_direct` | 已输出映射，但没有同时输出 routing mode |

处理原则：

```text
logical_hbm_bytes =
    weight_read_bytes
  + kv_read_bytes + kv_write_bytes
  + index_read_bytes + index_write_bytes
  + state_read_bytes + state_write_bytes
  + other_read_bytes
```

`activation_bytes` 保留为诊断字段，但不进入本研究的 `logical_hbm_bytes`。实际部署配置应在生成时断言它为 0，避免静默改变历史口径。

## 7. 容量字段

| 目标字段 | 当前来源 | 状态 | 处理建议 |
|---|---|---|---|
| `kv_cache_bytes_per_request` | `cache_capacity_per_request_average.kv_bytes` | `engine_direct` | 等长 Batch 下即单请求值 |
| `index_cache_bytes_per_request` | 同上 | `engine_direct` | 保持与 KV 分项独立 |
| `state_cache_bytes_per_request` | 同上 | `engine_direct` | 固定 State 仍是请求私有容量 |
| `cache_bytes_per_request` | 三项之和 | `engine_direct` | Start-of-step，使用已有 `C` |
| `kv/index/state_cache_bytes_total` | `cache_capacity_total` | `engine_direct` | 当前 JSON 已提供 |
| `batch_cache_bytes` | `cache_capacity_total.total_bytes` | `engine_direct` | 等价于 `B × cache_bytes_per_request` |
| `persistent_memory_bytes` | 权重容量 + Batch Cache | `derived` | 依赖标准化 `total_weight_capacity_bytes` |
| `cache_to_weight_capacity_ratio` | Batch Cache / 权重容量 | `derived` | 依赖相同模型边界 |
| `capacity_status` | 当前没有 | `missing` | 需区分精确、估算和不支持 |

容量不包含 Activation、Workspace、分页碎片、通信 Buffer 和安全余量。

## 8. 派生与审计字段

| 目标字段 | 状态 | 说明 |
|---|---|---|
| `bytes_per_flop` | `contract_gap` | 引擎已算，但分子沿用包含 Activation 的 `total_bytes` |
| `flops_per_byte` | `derived` | `1 / bytes_per_flop` |
| `tbps_per_pflops` | `contract_gap` | 与 `bytes_per_flop` 使用相同分子 |
| `routing_assumption` | `config_extractable` | 来自每个 routed expert group |
| `calculation_warnings` | `missing` | 由运行器汇总外推、缺失 metadata 和部分支持 |
| `engine_version`、`config_hash`、`run_id` | `missing` | 由研究运行器和 run manifest 提供 |
| 字段级来源、置信度、缺失原因 | `missing` | 模型事实层必须补充 |

以下建议结构计数目前没有进入 `DecodeResult`，不阻塞第一版结果生成，但在机制贡献分析前需要评估是否补充：

```text
attention_entries_read_per_token
index_candidates_scanned_per_token
state_elements_read_per_token
state_elements_written_per_token
```

## 9. 数据载体能力

| 载体 | 当前能力 | 结论 |
|---|---|---|
| 年度模型 JSON | 能承载公式配置和任意 metadata | 可继续作为模型级事实来源，但必须标准化研究字段 |
| CLI JSON | 保留完整 Step/Token/Cache 嵌套结果 | 可作为引擎原始结果，不包含研究标识和范围证据 |
| CLI CSV | 适合简单绘图 | 当前丢失 Step 分项、稳定 ID、权重容量、范围标记和运行追踪 |
| `outputs/` | 已跟踪基准 | 不能用于未确认实验 |
| `/tmp` | 可保存实验结果 | 第一版研究运行器默认输出位置 |

当前数据规模约为：

```text
20 models × 18 common C points × 9 main B points = 3240 rows
```

加上机制和上下文锚点后仍只是数千行。第一阶段使用 JSONL + CSV 足够，不需要数据库或 Parquet 依赖。

## 10. 最小实施清单

按以下顺序实施：

1. [x] 定义统一的模型研究事实结构，并回填 20 个模型；
2. [x] 明确 `logical_hbm_bytes` 排除 Activation 的代码接口；
3. [x] 增加显式 `allow_extrapolation`，默认仍拒绝越界；
4. [x] 增加研究运行器，输出：

```text
run_manifest.json
model_profiles.jsonl
decode_results.jsonl
decode_results.csv
validation_report.json
```

5. [x] 用 Llama 2 70B 和 GLM-5.2 完成简单/复杂端到端试算；
6. [x] 在 `/tmp/bpc_engine_decode_trend_full` 运行全部 20 个模型，共 3357 行。

本阶段不修改 FLOPs、Attention、KV、Index 或 State 的既有公式。
