# Decode 架构趋势研究交接

更新时间：2026-07-21（Asia/Shanghai）

写给完全没有上下文的新会话：本项目正在做 2022–2026YTD 的 LLM Decode workload 趋势研究。当前不是普通模型配置整理，而是为了以后从历史模型的计算量、访存量、权重容量、KV/State 容量趋势中反推下一代推理芯片指标。

## 1. 新会话先读什么

按顺序读：

1. [AGENTS.md](AGENTS.md)：项目约束，已被用户要求精简，不能重新扩写。
2. [docs/decode_trend_metrics.md](docs/decode_trend_metrics.md)：Decode 指标和计算边界，是指标合同。
3. [docs/decode_trend_research_todo.md](docs/decode_trend_research_todo.md)：P0–P9 总路线。
4. [P8行业需求包络报告](docs/decode_trend_p8_envelope_report.md)：已完成的第一类趋势、
   结果边界和复现入口。
5. [P9A技术轨迹函数报告](docs/decode_trend_p9a_technology_trends_report.md)：
   已完成的边际趋势函数、证据等级、否定结论和P9B边界。
6. 各年度 README：
   - [2022](configs/2022/README.md)
   - [2023](configs/2023/README.md)
   - [2024](configs/2024/README.md)
   - [2025](configs/2025/README.md)
   - [2026](configs/2026/README.md)

研究窗口固定为 **2022–2026YTD**；本轮 2026 YTD 截止日固定为 **2026-07-17**。

## 2. 我们在做什么

目标：统计过去几年代表性 LLM 在 Decode 阶段的资源需求，观察是否存在可外推趋势，再把趋势换算成芯片指标。

已确认关注三条趋势：

1. **行业需求包络**：代表模型在证据化profile与统一情景下的绝对资源是否增长。
2. **算法效率趋势**：固定能力、精度和工作负载时，资源需求是否下降。
3. **部署采用趋势**：不同架构实际占多少部署量或 token 份额。

最终希望反推的芯片指标包括：

- peak compute；
- peak HBM bandwidth；
- HBM capacity；
- bandwidth/compute；
- 在不同上下文、并发、模型采用率情景下的 P10/P50/P90。

数据准备、P8行业需求包络v0.2和P9A技术轨迹函数已经完成；尚未实施P9B联合未来
配置、P9C引擎/芯片换算、能力归一化效率拟合和部署采用统计。跨年度样本清单见
[docs/decode_trend_sample_manifest.md](docs/decode_trend_sample_manifest.md)。

## 3. 已冻结边界

- 只研究 **推理 Decode**。
- 不研究训练，不把 Prefill 混进本轮趋势。
- 单芯片或整个节点抽象为“逻辑超大芯片”；第一阶段不统计 TP/PP/EP/CP 通信。
- 当前引擎是静态 workload 计算器，不预测真实 latency、TPOT、利用率或功耗。
- `C` 是 Decode step 开始前已有上下文长度，是趋势变量，不能固定成“标准上下文”。
- `B` 是同时 Decode 的请求数，是可扫描参数，不能替用户固定。
- 核心函数是：

```text
FLOPs/token = F(C, B)
Bytes/token = G(C, B)
Capacity    = H(C, B)
```

- Decode 必须先计算完整 scheduler step，再除以 `B`。
- Dense 权重可在一个 step 内被 batch 摊薄；KV/Index/State 是请求私有，不能按 batch 共享。
- Cache capacity 不是 traffic，绝不能与 HBM bytes moved 相加。
- 1 MAC = 2 FLOPs。
- 当前 bytes 是 logical-HBM 口径，不是 profiler 实测 HBM。
- 超过官方 release 支持上下文的点必须标记外推。
- 未知值用 `null` 并说明原因，不能用 0 或猜测代替。

## 4. 样本选择规则

每年三个角色：

1. 旗舰资源需求；
2. 主流部署；
3. 架构演进对照。

用户明确要求：

- 用户参与每个模型选择；
- 一次只讨论一个模型；
- 用户确认后立刻查清 release / 架构 / 实际部署精度；
- 立刻写对应 JSON 配置；
- 立刻更新该年度 README；
- 立刻跑测试；
- 再讨论下一个模型。

不要再回到“先做大清单、以后再写配置”的旧流程。

年度目录 `configs/2022` 至 `configs/2026` 只保存当年实际部署 profile。不要在年度目录里生成假设性的统一 4/8/16-bit 对照配置；那是后续单独 profile。

## 5. 已完成内容

### 5.1 文档和目录

- [AGENTS.md](AGENTS.md) 已精简。
- [docs/decode_trend_metrics.md](docs/decode_trend_metrics.md) 已固化指标定义。
- [docs/decode_trend_research_todo.md](docs/decode_trend_research_todo.md) 已建立推进路线。
- 已建立 `configs/2022/` 至 `configs/2026/`。
- 每个年度 README 都写明模型选择理由、架构说明、精度边界和证据来源。
- 2022–2026YTD 的 15 个核心角色样本均已确认并落盘。
- 为补齐上下文和机制演进，另增加 5 个样本，共 20 个：
  - 2023：Mistral 7B Instruct v0.1、Yi-34B-200K；
  - 2024：DeepSeek-V3、Jamba 1.5 Large；
  - 2025：Llama 4 Scout。
- 已生成并通过用户验收的跨年度 [sample manifest](docs/decode_trend_sample_manifest.md)。

### 5.2 已确认模型清单

| 年份 | 旗舰资源需求 | 主流部署 | 架构演进对照 |
|---|---|---|---|
| 2022 | PaLM 540B INT8W/BF16KV | BLOOM 176B BF16 | GLM-130B INT4W/FP16KV |
| 2023 | Falcon 180B BF16 | Llama 2 70B Chat FP16 | Mixtral 8x7B Instruct BF16 |
| 2024 | Llama 3.1 405B Instruct mixed FP8/BF16KV | Llama 3.1 70B Instruct NVIDIA FP8/BF16KV | DeepSeek-V2-Chat BF16 MLA |
| 2025 | Kimi-K2-Thinking native INT4/BF16KV | Qwen3-32B-AWQ INT4W/FP16KV | Kimi-Linear-48B-A3B BF16 KDA+MLA |
| 2026YTD | Kimi-K2.6 native INT4/BF16KV | Qwen3.6-35B-A3B-FP8 | GLM-5.2-FP8 DSA+IndexShare |

增补样本不能被误当成新的年度核心角色：

| 年份 | 增补样本 | 用途 |
|---|---|---|
| 2023 | Mistral 7B Instruct v0.1 | SWA/滚动 KV 效率对照 |
| 2023 | Yi-34B-200K | 全注意力超长上下文资源压力 |
| 2024 | DeepSeek-V3 | DeepSeek-V2 同族 MLA/MoE/FP8 演进 |
| 2024 | Jamba 1.5 Large | Attention/Mamba 混合固定状态 |
| 2025 | Llama 4 Scout | 有条件的 MoE 与固定分块/全局混合注意力对照 |

配置文件：

- 2022：
  - [palm_540b_int8w_bf16kv.json](configs/2022/palm_540b_int8w_bf16kv.json)
  - [bloom_176b_bf16.json](configs/2022/bloom_176b_bf16.json)
  - [glm_130b_int4w_fp16kv.json](configs/2022/glm_130b_int4w_fp16kv.json)
- 2023：
  - [falcon_180b_bf16.json](configs/2023/falcon_180b_bf16.json)
  - [llama_2_70b_chat_fp16.json](configs/2023/llama_2_70b_chat_fp16.json)
  - [mixtral_8x7b_instruct_v0_1_bf16.json](configs/2023/mixtral_8x7b_instruct_v0_1_bf16.json)
- 2024：
  - [llama_3_1_405b_instruct_fp8mixed_bf16kv.json](configs/2024/llama_3_1_405b_instruct_fp8mixed_bf16kv.json)
  - [llama_3_1_70b_instruct_nvidia_fp8_bf16kv.json](configs/2024/llama_3_1_70b_instruct_nvidia_fp8_bf16kv.json)
  - [deepseek_v2_chat_bf16.json](configs/2024/deepseek_v2_chat_bf16.json)
- 2025：
  - [kimi_k2_thinking_int4w_bf16kv.json](configs/2025/kimi_k2_thinking_int4w_bf16kv.json)
  - [qwen3_32b_awq_int4w_fp16kv.json](configs/2025/qwen3_32b_awq_int4w_fp16kv.json)
  - [kimi_linear_48b_a3b_bf16.json](configs/2025/kimi_linear_48b_a3b_bf16.json)
- 2026：
  - [kimi_k2_6_int4w_bf16kv.json](configs/2026/kimi_k2_6_int4w_bf16kv.json)
  - [qwen3_6_35b_a3b_fp8_bf16kv.json](configs/2026/qwen3_6_35b_a3b_fp8_bf16kv.json)
  - [glm_5_2_fp8_dsa_fp8kv.json](configs/2026/glm_5_2_fp8_dsa_fp8kv.json)

### 5.3 最近一次新增：2026 GLM-5.2-FP8

2026“架构演进对照”样本已确认并完成配置。

关键口径：

- checkpoint：`zai-org/GLM-5.2-FP8`
- fixed revision：`ba978f7d347eaf65d22f1a86833408afdb953541`
- release 在 2026YTD 截止日前，可用。
- 角色：长上下文 sparse attention / IndexShare 对照。
- 主干 baseline：78 层普通 one-token Decode。
- MTP/next-token-prediction 层：排除，不混入标准 Decode。
- 架构：MLA + DSA + IndexShare。
- 上下文：1,048,576。
- 精度：FP8 主矩阵，FP8 MLA KV，BF16 DSA index key。
- active matrix 参数：`40,297,758,720 / token`。
- 1M context、batch=1 默认点：
  - parameter FLOPs：`80,595,517,440`
  - attention FLOPs：`22,246,588,416`
  - index FLOPs：`181,819,932,672`
  - total FLOPs：`284,662,038,528`
  - total bytes：`47,100,654,720`
  - persistent cache：`52,747,567,104 bytes/request`

注意：GLM-5.2-FP8 不是全模型 FP8。`lm_head`、Embedding、router、indexer `weights_proj`、norm 为 BF16；FP8 scale 和 correction bias 记录在 metadata 中，但当前 schema 不把这类零 FLOP 元数据流量放进 `weight_read_bytes`。

### 5.4 统一事实与批量计算链（本轮新增）

- 新增 [studies/decode_trend/models.json](studies/decode_trend/models.json)，已回填全部 20 个模型。
- 权重容量同时保存：
  - `decode_profile_weight_capacity_bytes`：标准 Decode 主口径；
  - `full_checkpoint_capacity_bytes`：完整 checkpoint 补充口径，不能核验时为 `null`。
- PaLM 540B 和 GLM-130B 的完整 checkpoint 容量不可审计，已明确保留 `null`，没有猜数。
- 新增 [scripts/run_decode_trend.py](scripts/run_decode_trend.py)，统一生成运行清单、模型事实、JSONL、CSV 和验证报告。
- 引擎新增显式外推开关；默认仍拒绝超过 release 上下文。
- 新增 `logical_hbm_bytes`，明确排除 Activation；旧 `total_bytes` 保持兼容。
- Llama 2 70B 与 GLM-5.2 两个端到端试点已通过。
- 全部 20 个模型的主网格已在 `/tmp/bpc_engine_decode_trend_full` 试运行，共 3357 行。

### 5.5 P3 机制支持审计

- 新增 [P3 人工审计矩阵](docs/decode_trend_p3_mechanism_audit.md)。
- 新增 [P3 机器审计](studies/decode_trend/mechanism_audit.json)。
- 每个模型分别标记 FLOPs、logical-HBM traffic、Cache capacity 和 Weight capacity。
- 研究运行器会严格校验所有 layer-group 和 routed expert group 都被 P3 覆盖，并把状态写入 JSONL/CSV。
- 当前 20 个模型没有整体 `unsupported` 指标，但现代 MoE/量化模型多数存在 `partially_supported` traffic。
- P3 全量结果在 `/tmp/bpc_engine_decode_trend_p3`，共 3357 行。

### 5.6 字段字典与正式冻结数据

- 新增 [字段数据字典](docs/decode_trend_data_dictionary.md)，覆盖配置、研究事实、P3审计、JSONL/CSV结果、单位、公式和空值规则。
- 新增 [冻结工具](scripts/freeze_decode_trend_release.py)。
- [数据生成说明](studies/decode_trend/releases/README.md)记录了20模型完整生成命令。
- 正式冻结版本`studies/decode_trend/releases/v1.0.0/`已提交Git，新clone可直接运行
  P8/P9A；其他未发布release仍默认被Git忽略。
- 生成版本包含20个模型、3357行结果、完整`source_snapshot/`和`SHA256SUMS`。
- 已于2026-07-20从干净的Git提交`4ca035f`生成正式版本`v1.0.0`；验证状态为`pass`，3357行均通过校验，其中1440行为显式理论外推。
- `v1.0.0`有两条精度警告：BLOOM 176B与GLM-130B在`C=16M、B=256`的外推batch cache超过IEEE-754精确整数范围；它们不影响验证通过状态，但极端点统计应保留警告。
- 后续统计应读取`v1.0.0/data/decode_results.csv`，不要回退到临时运行目录。

### 5.7 P8行业需求包络v0.2

- 实施计划见[docs/decode_trend_p8_envelope_plan.md](docs/decode_trend_p8_envelope_plan.md)。
- 可视化修订见
  [docs/decode_trend_p8_visualization_revision_plan.md](docs/decode_trend_p8_visualization_revision_plan.md)。
- 正式结果解释见[docs/decode_trend_p8_envelope_report.md](docs/decode_trend_p8_envelope_report.md)。
- 分析入口为[scripts/analyze_decode_trend_envelope.py](scripts/analyze_decode_trend_envelope.py)，
  默认从冻结`v1.0.0`读取并写到`/tmp/decode_trend_p8_envelope/`。
- 产物包括质量摘要、模型概览、全样本与指定前沿年度包络、固定`C=2048`的
  60行模型对照、advertised max、canonical context boundaries、分项占比、
  首次/稳定交叉点和29组PNG/SVG图。
- 动态年度图是逐点最大值，不做平均或加权；eligible模型ID集合改变时断线。
  跨年度主比较使用固定`C=2048`点图，各模型自身上限图只表示条件压力。
- P3 partial进入inclusive包络但逐指标标记；年度表同时保存supported-only最大值。
- canonical边界以profile标量事实为准；Llama 4的evaluated 131K、trained 262K、
  deployed 3.6M和advertised 10M已分开输出。
- KDA通过显式recurrent state建模；State Capacity支持完整，非线性/归一化FLOPs
  因公开事实不足仍为partial。
- 结果是静态logical workload，不承诺相对真实profiler误差小于10%，也不预测延迟。

### 5.8 P9A技术轨迹函数

- 实施计划见
  [docs/decode_trend_p9a_technology_trends_plan.md](docs/decode_trend_p9a_technology_trends_plan.md)。
- 正式结果解释见
  [docs/decode_trend_p9a_technology_trends_report.md](docs/decode_trend_p9a_technology_trends_report.md)。
- 分析入口为
  [scripts/analyze_decode_technology_trends.py](scripts/analyze_decode_technology_trends.py)，
  默认读取冻结`v1.0.0`并写到`/tmp/decode_trend_p9a_technology_trends/`。
- 20个release生成43项函数记录和3个组成轴；按记录行计为11项`emerging`、
  17项`unstable`、15项`insufficient`，没有`established`。行数不等于独立发现数。
- 当前支持激活参数占比下降、四种语义各异的上下文边界增长、精选样本中MoE出现
  倾向上升，以及KV-layout组成轴向MLA份额移动的组级信号。
- 当前不支持总参数稳定增长、绝对激活参数稳定、MHA被线性注意力稳定替代、
  MLA/State-like模型出现率形成可靠扩散曲线、Sparse访问或低比特行业采用率
  稳定增长。
- Kimi Linear、Qwen3.6和Jamba按物理层canonical分类，避免把同层多个成本组件
  重复统计。
- 组成轴以归一化后的整组预测选模并保持份额闭合；参数基和MoE两部分模型施加
  声明内恒等式。跨技术轴联合约束仍留给P9B，不能直接把P9A的2028边际值拼成
  一台模型。

## 6. 验证状态

最近一次完整测试已通过：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
```

结果：

```text
Ran 80 tests
OK
```

运行测试时必须保留 `PYTHONDONTWRITEBYTECODE=1` 和 `-B`，避免继续刷新仓库里被错误追踪的 `__pycache__`。

## 7. 当前卡在哪里

没有代码阻塞。

当前状态是：**20个模型的数据准备、P3审计、字段字典、全量计算链、P7自动质量
验证、P8行业需求包络v0.2和P9A技术轨迹函数均已完成**。核心Step/Token FLOPs、
  logical-HBM流量、Cache和Weight Capacity公式已经具备；正式`v1.0.0`随仓库提交，
  临时候选release和分析输出不提交Git。

尚未完成：

- P2中的模型能力/质量控制字段，以及更细的字段级置信度和估算范围；
- P8算法效率和部署采用两类统计；
- P9B满足代数与组成约束的联合未来模型配置；
- P9C未来workload计算与芯片指标换算。

另外，部署采用趋势目前只有下载量、传播度、生态接入等代理证据，尚没有真实 token 份额。

## 8. 下一步计划

新会话不要继续盲目加模型，也不要重做批量运行器。

建议推进顺序：

1. 先读P9A报告；只有冻结输入、配置或分析脚本变化时才重跑该阶段。
2. 先编写P9B实施计划，明确哪些P9A函数可作为primitive、哪些只能作为diagnostic，
   以及MoE、Attention、精度和上下文之间的条件分支。
3. P9B输出低/中/高机器可读情景、假设清单、约束验证和解释报告；每个情景必须能被
   当前引擎解析，并保持参数/MoE恒等式和三个组成轴闭合。
4. P9B验收后再进入P9C，将联合情景送入引擎计算workload，换算Peak Compute、
   Peak HBM Bandwidth、HBM Capacity和Bandwidth/Compute，并输出敏感性报告。
5. P2能力字段、P8算法效率和部署采用数据作为并行补强；真实采用率仍需另收部署量
   或Token份额。

不要直接把P9A的独立函数值拼成未来模型，也不要把精选样本presence称为行业采用率。

## 9. 当前工作区状态提醒

每次接手先运行`git status --short --branch`，不要依赖旧交接记录猜测工作区状态。
2026-07-20已完成P8和P9A的计划、分析器、测试与报告。2026-07-21已将正式冻结
`v1.0.0`纳入版本控制，修复新clone缺少默认分析输入的问题。后续代码提交完成后应
保持工作区干净；提交状态仍以实际Git输出为准。

`studies/decode_trend/releases/v1.0.0/`是受版本控制的正式研究输入；其他候选release
默认由`.gitignore`排除。不要修改已发布版本内容；数据变化必须生成新版本。不要运行
`git reset --hard`、`git checkout --`或清理用户改动，除非用户明确要求。

## 10. 绝对不要再踩的坑

### 10.1 与用户协作

- 用户不喜欢大而全、泛泛而谈的回答。必须直接回答当前问题。
- 不要一次性展开所有后续阶段。
- 不要重新争论三个角色是否合理；已经确认。
- 不要重新选模型，除非用户要求。
- 不要把上下文长度固定成“标准服务场景”；上下文长度本身就是研究趋势。
- 不要把 `B` 固定成单一值；它是扫描参数。

### 10.2 指标边界

- 总参数、活跃矩阵参数、权重容量、每 step 权重读取、参数 FLOPs 是不同概念。
- Decode 先算完整 step，再除以 `B`。
- Dense 权重可摊薄；KV/Index/State 不可按 batch 当作共享。
- Cache capacity 绝不能加到 traffic。
- Input Embedding lookup 只读一行，不要把整张 Embedding 当作每 token 读取。
- tied Embedding 容量只计一份，但 LM Head 仍参与 Decode Token 的矩阵计算。
- 训练 dtype、checkpoint dtype、推理 dtype 不能自动等同。
- logical-HBM 不能冒充真实 HBM 流量或 latency。
- 没有真实路由 trace 时，MoE batch expert union 是分析近似，不是实测。

### 10.3 年度样本特殊坑

- PaLM 540B：
  - `head_dim=256`，不是 `18432/48=384`。
  - TPU 物理 padding 到 64 query heads 是实现开销，不是逻辑架构。
  - 低延迟 profile 是 INT8 权重、BF16 KV，不是 KV8。
- Mixtral：
  - 是 32K 全注意力，不是 sliding window。
  - `46.7B total`、`12.9B active capacity`、`12.7486B engine active matrix` 三个口径不能混用。
- Llama 3.1 405B FP8：
  - FP8 只覆盖中间 124 层 FFN，不是全模型 FP8。
  - 必须使用修正后的 8 KV heads，不能按首发错误索引算成 16 KV heads。
- Llama 3.1 70B NVIDIA FP8：
  - 使用固定 2024 revision `811ca36…`。
  - 当前 main 已升级并可能启用 FP8 KV，不能倒灌进 2024 样本。
- DeepSeek-V2：
  - MLA 按逻辑吸收式 latent KV 建模；公开 vLLM 展开 K/V 的物理 cache 不能直接比较。
- Kimi-K2 / Kimi-K2.6：
  - native INT4 只覆盖 routed expert Linear，不是全模型 INT4。
  - scale 和 packed metadata 计入容量，但当前不进 `weight_read_bytes`。
  - 多模态 Kimi-K2.6 的 vision tower / projector 不参与 text Decode 每 token 工作量。
- Qwen3-32B-AWQ：
  - AWQ 只覆盖 Transformer Linear，不覆盖 Embedding/LM Head/Norm。
  - 128K 需要官方 YaRN override，不是原生上下文。
- Kimi-Linear：
  - KDA recurrent state 是 FP32；short-conv state 是 BF16 且使用 `kernel_size - 1`。
  - 不要把 HF Transformers 展开 K/V 路径与官方 vLLM KDA profile 混用。
- Qwen3.6-35B-A3B-FP8：
  - 标准 baseline 不启用 MTP speculative serving。
  - Gated DeltaNet state、conv state、GQA KV 要分清。
- GLM-5.2-FP8：
  - 架构演进重点是 DSA + IndexShare，不是“又一个 FP8 MoE”。
  - 21 层 full indexer + 57 层 shared index，分组保持总量精确但不表示显示顺序。
  - MTP speculative draft/verify 不进入普通 Decode baseline。
  - FP8 scale_inv / correction bias 不能硬塞进参数组制造假 FLOPs。

## 11. 最重要的接手原则

如果用户继续当前任务，优先说清：

> 20个样本、正式冻结数据、P3/P7验证、P8行业需求包络v0.2和P9A技术轨迹函数均已
> 完成；若继续未来预测，下一步是P9B联合配置约束，不能把独立边际函数直接拼成
> 一台未来模型。

不要再从 2022 或 2026 模型选择重新开始。
