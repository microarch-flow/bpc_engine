# Decode 趋势研究样本清单

版本：v0.1
研究窗口：2022–2026YTD（截止 2026-07-17）
状态：P1 待人工验收

本清单只定义“研究哪些精确模型 release，以及它们用于回答什么问题”。部署精度、参数拆分和计算边界以对应 JSON 与年度 README 为准；这里不重复维护完整数值。

## 纳入规则

一个样本必须：

1. 对应精确的模型、变体和 release；
2. 有可审计的一手架构来源；
3. 能闭合 Decode 所需的主要结构与实际部署精度；
4. 明确属于资源包络、效率对照、采用观察或机制锚点；
5. 不把宣称上下文、训练上下文、实际部署上限和任务有效长度混为一谈。

此后新增机制样本还必须通过三项审查：

- `capability_gate`：通用知识/推理能力足以作为趋势证据；
- `mechanism_effect_gate`：有证据表明新机制产生了实际收益；
- `deployment_gate`：存在可部署 checkpoint、runtime 和精度 profile。

审查可按机制分别通过。模型不能因为某一项宣传指标失败而整体删除，也不能把失败项作为正向趋势数据。

## 样本清单

角色缩写：

- `FE`：`frontier_envelope`，年度旗舰资源包络；
- `AO`：`adoption_observation`，主流部署或采用观察；
- `EC`：`efficiency_comparison`，效率或同族演进对照；
- `MA`：`mechanism_anchor`，重要机制锚点。

| 年份 | 精确 release | 机构 | 发布日期 | 参数规模口径 | 角色 | 主要技术与上下文边界 | 配置 |
|---|---|---|---|---|---|---|---|
| 2022 | PaLM 540B | Google | 2022-04-05 | 540B dense | FE | MQA、SwiGLU、RoPE；2K | [JSON](../configs/2022/palm_540b_int8w_bf16kv.json) |
| 2022 | BLOOM 176B | BigScience | 2022-07-12 | 176B dense | AO | MHA、ALiBi、BF16；2K | [JSON](../configs/2022/bloom_176b_bf16.json) |
| 2022 | GLM-130B official INT4 | Tsinghua/智谱 | 2022-08-24 | 130B dense | MA | GLM Prefix、GeGLU、DeepNorm、INT4 Linear；2K | [JSON](../configs/2022/glm_130b_int4w_fp16kv.json) |
| 2023 | Falcon 180B base | TII | 2023-09-06 | 180B dense | FE | Multigroup GQA、BF16；2K | [JSON](../configs/2023/falcon_180b_bf16.json) |
| 2023 | Llama 2 70B Chat HF | Meta | 2023-07-18 | 69.0B dense | AO | GQA、SwiGLU、FP16；4K | [JSON](../configs/2023/llama_2_70b_chat_fp16.json) |
| 2023 | Mistral 7B Instruct v0.1 | Mistral AI | 2023-09-27 | 7.24B dense | EC, MA | GQA、4K SWA 与滚动 KV；发布上下文 8K | [JSON](../configs/2023/mistral_7b_instruct_v0_1_bf16.json) |
| 2023 | Yi-34B-200K original | 01.AI | 2023-11-05 | 34.4B dense | MA | 全上下文 GQA；200K 是 release 上限与压力点，不代表复杂推理有效长度 | [JSON](../configs/2023/yi_34b_200k_bf16.json) |
| 2023 | Mixtral 8x7B Instruct v0.1 | Mistral AI | 2023-12-11 | 46.7B total / 12.75B active matrix | EC, MA | Top-2 MoE、全上下文 GQA；32K | [JSON](../configs/2023/mixtral_8x7b_instruct_v0_1_bf16.json) |
| 2024 | Llama 3.1 405B Instruct FP8 | Meta | 2024-07-23 | 405.9B dense | FE | GQA、混合 FP8/BF16；128K | [JSON](../configs/2024/llama_3_1_405b_instruct_fp8mixed_bf16kv.json) |
| 2024 | Llama 3.1 70B Instruct NVIDIA FP8 | Meta/NVIDIA | 2024-07-23 | 70.6B dense | AO, EC | 同族缩放、W8A8 FP8、BF16 KV；128K | [JSON](../configs/2024/llama_3_1_70b_instruct_nvidia_fp8_bf16kv.json) |
| 2024 | DeepSeek-V2-Chat RL | DeepSeek | 2024-05-06 | 236B total / 约 21B active | EC, MA | DeepSeekMoE、吸收式 MLA；128K | [JSON](../configs/2024/deepseek_v2_chat_bf16.json) |
| 2024 | DeepSeek-V3 initial release | DeepSeek | 2024-12-26 | 671B total / 36.6B active matrix | EC, MA | MLA、256 Expert Top-8、native FP8；128K；普通 Decode 排除 MTP | [JSON](../configs/2024/deepseek_v3_native_fp8_bf16kv.json) |
| 2024 | AI21 Jamba 1.5 Large | AI21 | 2024-08-22 | 398B total / 94B active（官方取整） | EC, MA | 1:7 Attention/Mamba、Top-2 MoE、ExpertsInt8；有效上下文 256K | [JSON](../configs/2024/jamba_1_5_large_experts_int8_bf16kv.json) |
| 2025 | Kimi-K2-Thinking | Moonshot AI | 2025-11-06 | 1.026T total / 31.69B active matrix | FE | MLA、384 Expert Top-8、Routed Expert native INT4；256K | [JSON](../configs/2025/kimi_k2_thinking_int4w_bf16kv.json) |
| 2025 | Qwen3-32B-AWQ | Qwen | 2025-04-29 | 32.76B dense | AO | AWQ INT4 Linear、FP16 KV；原生 32K，官方 YaRN 部署 128K | [JSON](../configs/2025/qwen3_32b_awq_int4w_fp16kv.json) |
| 2025 | Kimi-Linear-48B-A3B-Instruct | Moonshot AI | 2025-10-30 | 49.12B total / 3.11B active matrix | EC, MA | 20 层 KDA + 7 层 MLA、固定状态；原生 1M | [JSON](../configs/2025/kimi_linear_48b_a3b_bf16.json) |
| 2025 | Llama 4 Scout 17B-16E Instruct | Meta | 2025-04-05 | 108.64B checkpoint / 16.14B active matrix | EC, MA（有条件） | Top-1 MoE、12 层全局 + 36 层固定 8K 分块；训练 256K、部署 3.5M、宣称 10M；10M 不进入有效上下文拟合 | [JSON](../configs/2025/llama_4_scout_17b_16e_instruct_bf16.json) |
| 2026YTD | Kimi-K2.6 | Moonshot AI | 2026-04-14 | 1.026T text / 31.69B active matrix | FE | MLA、384 Expert Top-8、Routed Expert native INT4；256K | [JSON](../configs/2026/kimi_k2_6_int4w_bf16kv.json) |
| 2026YTD | Qwen3.6-35B-A3B-FP8 | Qwen | 2026-04-15 | 34.66B text / 2.95B active matrix | AO, MA | 30 层 Gated DeltaNet + 10 层全注意力、FP8；256K | [JSON](../configs/2026/qwen3_6_35b_a3b_fp8_bf16kv.json) |
| 2026YTD | GLM-5.2-FP8 | Z.ai | 2026-06-16 | 753.38B checkpoint / 40.30B active matrix | EC, MA | MLA、DSA Top-2048、IndexShare、FP8 KV；1M；普通 Decode 排除 MTP | [JSON](../configs/2026/glm_5_2_fp8_dsa_fp8kv.json) |

## 覆盖结论

20 个样本已经覆盖：

- 参数组织：稠密、稀疏 MoE、Shared Expert；
- Attention/KV：MHA、MQA、GQA、SWA、固定分块、MLA、DSA；
- 固定状态路线：Mamba、KDA、Gated DeltaNet；
- 部署精度：BF16/FP16、INT8、FP8、AWQ INT4、native INT4；
- 上下文变化：2K、4K、8K、32K、128K、200K、256K、1M，以及单独保存但不直接作为有效能力的更长宣称/部署上限；
- 纵向对照：Llama 3.1 70B/405B、DeepSeek-V2/V3、Kimi-K2-Thinking/K2.6。

现阶段不建议仅为增加数量继续纳入模型。新增样本必须补上当前未覆盖且通过三项审查的机制，否则会稀释统计口径。

## 已知边界

- `AO` 目前主要依赖下载量、生态接入和服务可用性等代理，不能解释为历史 Token 份额。
- 闭源旗舰没有足够结构资料时不进入计算清单，只能在趋势解释中作为外部能力参照。
- 参数规模栏用于快速识别样本，精确参数与容量必须读取对应 JSON。
- Scout 的 10M、Yi 的 200K 等上限必须按各自说明使用，不能直接拼成“有效上下文逐年最大值”。
- Kimi K3 保持 P0 观察候选；在 2026YTD 截止点缺少可审计完整权重和技术资料，不进入本清单。

年度选择理由、架构细节和一手来源见：
[2022](../configs/2022/README.md)、
[2023](../configs/2023/README.md)、
[2024](../configs/2024/README.md)、
[2025](../configs/2025/README.md)、
[2026YTD](../configs/2026/README.md)。
