# P3 模型 × 机制 × 引擎支持审计

版本：v0.1
范围：20 个 Decode 趋势样本及其历史实际部署 profile

## 1. 审计口径

P3 不判断模型“能不能运行”，而是分别判断以下结果能否代表选定部署边界：

- `F`：FLOPs；
- `T`：logical-HBM traffic；
- `C`：KV/Index/State Cache capacity；
- `W`：Decode profile weight capacity。

状态：

- `S` / `supported`：公式、模型事实和数值锚点闭合；
- `P` / `partially_supported`：可计算，但存在可能显著影响结果的部署或统计假设；
- `U` / `unsupported`：核心行为缺失，不得进入中心趋势；
- `N` / `not_applicable`：该机制不产生此类指标。

机器可读审计见
[`studies/decode_trend/mechanism_audit.json`](../studies/decode_trend/mechanism_audit.json)。

## 2. 机制级结论

| 机制 profile | F | T | C | W | 主要边界 |
|---|---:|---:|---:|---:|---|
| MHA/MQA/GQA Full | S | S | S | N | QK+AV 与逻辑 KV；不计 Softmax 临时量 |
| GQA Sliding Window | S | S | S | N | 读取和滚动容量均为 `min(C, W)` |
| GQA Chunked Block | S | S | P | N | 读取按 `C mod 8192`；容量取决于是否淘汰已完成块 |
| Full MLA | S | S | P | N | 逻辑 absorbed MLA；需确认部署没有展开 K/V |
| MLA + DSA | P | S | S | N | 未计 Top-k 排序/选择 FLOPs |
| MLA + Shared Top-k | S | S | S | N | Index 由 DSA 层生成，本层不重复计容量 |
| Mamba-1 State | P | S | S | N | 只计核心 recurrence，不折算非线性算子 |
| KDA State | P | S | S | N | 状态形状闭合；非线性与归一化 FLOPs 未计 |
| Gated DeltaNet State | P | S | S | N | 核心状态计算闭合，特征映射/门控未完整折算 |
| Short-conv State | N | S | S | N | 卷积 MAC 已计入参数 FLOPs |
| MoE Uniform Routing | S | P | N | S | `B>1` Expert 并集来自均匀独立路由假设 |

## 3. 模型级矩阵

模型级状态取该模型所有机制、精度与容量证据的最弱状态。

| 年份 | 模型 | 主要机制 | F | T | C | W |
|---:|---|---|---:|---:|---:|---:|
| 2022 | PaLM 540B | MQA、INT8W | S | P | S | P |
| 2022 | BLOOM 176B | MHA、ALiBi | S | S | S | S |
| 2022 | GLM-130B INT4 | MHA、INT4W | S | P | S | P |
| 2023 | Llama 2 70B Chat | GQA | S | S | S | S |
| 2023 | Falcon 180B | Multigroup GQA | S | S | S | S |
| 2023 | Mistral 7B | GQA、SWA | S | S | S | S |
| 2023 | Yi-34B-200K | GQA、Full Attention | S | S | S | S |
| 2023 | Mixtral 8x7B | GQA、MoE Top-2 | S | P | S | S |
| 2024 | DeepSeek-V2 | MLA、MoE Top-6 | S | P | P | S |
| 2024 | Llama 3.1 405B FP8 | GQA、Mixed FP8 | S | P | S | S |
| 2024 | Llama 3.1 70B FP8 | GQA、FP8 | S | P | S | S |
| 2024 | Jamba 1.5 Large | GQA、Mamba、MoE | P | P | S | S |
| 2024 | DeepSeek-V3 | MLA、MoE、FP8 | S | P | S | S |
| 2025 | Llama 4 Scout | Full/Chunked GQA、MoE | S | P | P | S |
| 2025 | Qwen3-32B-AWQ | GQA、AWQ | S | P | S | S |
| 2025 | Kimi-Linear | KDA、MLA、MoE | P | P | S | S |
| 2025 | Kimi-K2-Thinking | MLA、MoE、INT4 | S | P | S | S |
| 2026 | Kimi-K2.6 | MLA、MoE、INT4 | S | P | S | S |
| 2026 | Qwen3.6-35B-A3B | DeltaNet、GQA、MoE | P | P | S | S |
| 2026 | GLM-5.2 | MLA、DSA、IndexShare、MoE | P | P | S | S |

## 4. 数值锚点

已由引擎测试或研究运行器验证：

- SWA：`C=4095/4096/4097` 的读取和容量饱和；
- Chunked Block：`C=8191/8192/8193` 的读取重置；
- DSA：`C=2047/2048/2049` 的主 Attention Top-k 饱和与 Index 全扫描；
- Mamba/KDA/DeltaNet：State capacity 不随 `C` 增长；
- MoE：`B=1` 触达 Expert 数等于 Top-k，`B>1` 使用显式路由假设；
- 所有模型：完整 Step 等于每 Token 结果乘 `B`；
- Cache capacity 与 traffic 分开，未相加。

## 5. P3 结论

- 20 个模型均无整体 `unsupported` 指标，因此都能生成研究结果。
- Dense Full-Attention 历史基线的四类核心指标最完整。
- 新机制的主要不确定性不是矩阵参数 FLOPs，而是实际路由、量化 metadata
  traffic、Recurrent 非线性 FLOPs和缓存实现策略。
- `partially_supported` 数据是否进入中心拟合属于 P8 统计策略，P3 只保留状态和假设，
  不在本阶段替用户决定。

## 6. 当前样本之外的机制

以下机制没有进入20个标准 Decode profile，不能因为引擎存在 `fixed_cost` 就视为已支持：

| 机制 | 当前状态 | 原因 |
|---|---|---|
| MTP speculative draft/verify | unsupported | 标准单 Token Decode 明确排除，尚无统一 draft/acceptance 模型 |
| YOCO | unsupported | 当前没有入选 release 和对应 Cache/Attention 公式审计 |
| Mixture of Depths | unsupported | 动态跳层率与 Batch 执行行为尚未建模 |
| Diffusion LLM Decode | unsupported | 不满足每请求每 Step 生成一个 Token的当前合同 |
| BLT/Byte Latent | unsupported | 动态 patch 边界和输出单位尚未映射到 Token 合同 |
| Multimodal encoder/prompt projector | not_applicable | 属于输入处理或 Prefill，不进入稳态文本 Decode |

若未来把这些机制加入样本，必须新增明确公式和部署 profile，不能用常量
`fixed_cost` 隐藏核心的动态行为。
