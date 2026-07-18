# 2025 年代表模型配置

本目录保存 2025 年发布的精确模型版本及其当年代表性实际部署精度配置。

- 配置必须遵循计算引擎的 `schema_version: 1`。
- 一个文件对应一个精确模型版本和一个实际部署精度。
- `metadata` 中记录发布日期、架构与精度来源、推导过程及不确定项。
- `analysis.contexts` 和 `analysis.batches` 仅为默认值，统一研究扫描可由 CLI 覆盖。
- 不在本目录保存假设性的统一 4/8/16-bit 对照配置。

## 代表模型与选择状态

| 角色 | 精确模型版本 | 状态与实际部署精度 | 配置 |
|---|---|---|---|
| 旗舰资源需求 | Kimi-K2-Thinking，revision `6e3cdad…` | 已确认；Routed Expert 为 native INT4，其他权重及逻辑吸收式 MLA KV 为 BF16 | [kimi_k2_thinking_int4w_bf16kv.json](kimi_k2_thinking_int4w_bf16kv.json) |
| 主流部署 | Qwen3-32B-AWQ，revision `0499c3a…` | 已确认；Transformer Linear 为 AWQ INT4，其他权重运行时及 KV 为 FP16；采用官方 YaRN 128K profile | [qwen3_32b_awq_int4w_fp16kv.json](qwen3_32b_awq_int4w_fp16kv.json) |
| 架构演进对照 | Kimi-Linear-48B-A3B-Instruct，revision `e1df551…` | 已确认；BF16 权重与 MLA KV，KDA recurrent state 为 FP32，KDA short-conv state 为 BF16；原生 1M context | [kimi_linear_48b_a3b_bf16.json](kimi_linear_48b_a3b_bf16.json) |
| 有条件的效率/机制对照 | Llama 4 Scout 17B-16E Instruct，revision `63bc3b6…` | 已确认；BF16 权重与 KV，48 层全 MoE，12 层全局 NoPE 加 36 层固定 8K 分块 RoPE；10M 不作为有效上下文正向点 | [llama_4_scout_17b_16e_instruct_bf16.json](llama_4_scout_17b_16e_instruct_bf16.json) |

## 选择理由

| 角色与模型 | 为什么选择 | 解释边界 |
|---|---|---|
| 旗舰资源需求：Kimi-K2-Thinking native INT4 | 它在 2025 年公开模型中同时具备第一梯队知识/推理与 Agent 能力、约 1T 总参数、约 32B 激活参数、256K 上下文和官方可部署的 QAT INT4 checkpoint。公开权重、配置、评测和部署路径使旗舰能力与绝对资源需求都可审计，而不是只凭模型名称或参数规模判断。 | 它不能代表所有闭源旗舰；官方 1T/32B 是取整值。“native INT4”只覆盖 Routed Expert 权重，不是全模型 INT4，也不是 INT4 KV。Thinking 会改变输出长度和上下文轨迹，不改变单个 Decode Token 的结构公式。 |
| 主流部署：Qwen3-32B-AWQ | Qwen3-32B 在稠密 32B 档位兼顾推理与通用对话，官方同时发布 AWQ checkpoint、vLLM/SGLang 服务方法、128K YaRN 路径及 AWQ 能力结果。它当前约 193 万次 Hugging Face 近期下载可作为传播度旁证，并与 1T MoE 旗舰样本构成“常用稠密部署—旗舰资源上界”对照。 | 下载量是 2026-07-18 获取的当前传播代理，不是 2025 历史快照、Token 份额或所有 Qwen3 部署量。AWQ 只覆盖 64 层内的 448 个 Linear；Embedding、LM Head 和 Norm 不是 INT4。131,072 需要官方 YaRN override，不是原生上下文。 |
| 架构演进对照：Kimi-Linear-48B-A3B-Instruct | 它把 27 层中的 20 层替换为 KDA recurrent mixer，只保留 7 层全局 MLA，并公开 1M 原生上下文与官方 vLLM 服务命令。这个样本能直接观察“随上下文线性增长的 KV Cache”向“固定 recurrent state + 少量全局 MLA KV”迁移时，对 Decode 访存和状态容量的影响。 | 它是架构趋势锚点，不是主流部署份额样本；当前传播度弱于 Qwen3-32B-AWQ。配置绑定官方 vLLM 原生 KDA+MLA profile；HF Transformers 展开 K/V 路径、社区量化版本和 TP4 物理通信不纳入本样本。 |
| 有条件的效率/机制对照：Llama 4 Scout | 它以 109B 总参数、约 17B 激活参数达到上一代 Llama 3.3 70B 附近的通用能力，并在 2025 年获得 vLLM、AWS 等真实部署。其“全层 MoE + 1/4 全局注意力 + 3/4 固定分块注意力”能观察容量、参数计算和长上下文注意力斜率如何分离。 | 它不是 2025 旗舰能力样本。训练长度只有 256K，独立 NoLiMa 在语义检索上的有效长度为 1K；10M 只保存为宣称/压力上限，不进入有效上下文或芯片容量主拟合。AWS 3.5M 是服务上限，也不等价于该长度上的任务质量。 |

四个样本的互补关系是：Kimi-K2-Thinking 给出公开旗舰资源包络，Qwen3-32B-AWQ 给出高传播稠密 INT4 部署点，Kimi-Linear 给出混合线性注意力的固定状态方向，Llama 4 Scout 则给出“全层 MoE + 固定分块/全局混合注意力”的已部署效率对照及宣称上下文失效边界。

## 架构说明

### Llama 4 Scout 17B-16E Instruct BF16

Scout 的文本骨干为 48 层、隐藏维度 5,120 的全层稀疏 MoE decoder。每层有 16 个中间维度 8,192 的 Routed SwiGLU Expert，每 Token 选择 1 个，同时执行一个同尺寸 Shared Expert。词表为 202,048，输入 Embedding 与 LM Head 不共享。完整多模态 checkpoint 精确包含 108,641,793,536 个 BF16 参数；其中文本骨干为 107,769,861,120，视觉编码器与多模态投影器为 871,932,416。视觉部分只在图像 Prompt 处理时执行，保留在容量元数据中，但不进入稳态文本 Decode 工作量。

Attention 使用 40 个 Query Head、8 个 KV Head、Head Dim 128 的 GQA。第 4、8、12……层共 12 层使用全上下文 NoPE 和 attention-temperature tuning；其余 36 层使用 RoPE、无参数 QK Norm 和固定不重叠的 8,192 Token 分块。固定分块不是 SWA：在上下文 `C` 处，每个局部层读取 `C mod 8192` 个历史 KV，因此工作量在块边界出现锯齿。配置默认同时扫描 8,191 和 8,192，避免把边界归零误拟合为平滑下降。

本配置绑定 2025 年可复核的 BF16 服务路径。vLLM 0.8.3 给出 8×H100 运行 1.28M、8×H200 运行 3.6M 的命令，AWS Bedrock 当年提供 3.5M 上限。该时期的选定 profile 对 36 个分块层也保留完整历史 KV，只在读取时应用分块 mask；因此单请求 KV 容量仍为 `48 × C × 4,096 bytes`。在 3.5M 处为 688,128,000,000 bytes，在宣称 10M 处为 2,061,584,302,080 bytes。若未来运行时能淘汰完成的局部块，应建立 `retain_full_history=false` 的独立 profile，不能覆盖此历史部署结果。

能力审查采用“机制级准入”：Scout 可用于 MoE 容量—计算解耦及混合注意力系数比较，但不能用于 2025 旗舰能力包络。Meta 的所有公开质量表均为 BF16；官方虽然提供 on-the-fly INT4 单 H100 路径，却未公开对应的精确质量差，因此这里不把 INT4 宣传替换成主配置。上下文分别保存训练 262,144、AWS 部署 3.5M、宣称 10,485,760 和任务相关有效长度；NoLiMa 的 1K 结论只针对其去字面匹配语义检索阈值，也不能外推成所有任务都只支持 1K。

来源：[Meta 发布](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)、[官方模型卡](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct-Original/blob/main/README.md)、[固定配置](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/63bc3b67bc6e8f6857e7819c1218c8f18474f04a/config.json)、[Transformers 4.51 实现](https://github.com/huggingface/transformers/blob/v4.51.0/src/transformers/models/llama4/modeling_llama4.py)、[vLLM 0.8.3 支持](https://github.com/vllm-project/vllm/pull/16104)、[AWS 2025 部署](https://aws.amazon.com/blogs/aws/llama-4-models-from-meta-now-available-in-amazon-bedrock-serverless/)、[NoLiMa](https://github.com/adobe-research/NoLiMa)。

### Kimi-K2-Thinking native INT4

该模型是 61 层稀疏 MoE causal decoder-only Transformer，隐藏维度为 7,168。第 0 层是中间维度 18,432 的稠密 SwiGLU；后续 60 层各有 384 个中间维度 2,048 的 Routed Expert，每个 Token 选择 8 个，另有 1 个始终执行的 Shared Expert。输入 Embedding 与 LM Head 不共享。

Attention 使用 MLA：64 个 Query Head，经 1,536 维 Query LoRA 和 512 维 KV LoRA 压缩，NoPE、RoPE 和 Value Head 维度分别为 128、64 和 128。配置固定官方 256K 上下文的精确值 262,144，并采用优化服务路径中的逻辑吸收式 MLA：每层、每历史 Token 保存 512 个压缩 KV 元素和 64 个 RoPE Key 元素。BF16 下 61 层合计为 70,272 bytes，在最大上下文处单请求 KV 容量为 18,421,383,168 bytes。

这里的 native INT4 是 QAT 后的 weight-only profile。固定 checkpoint 的 `compressed-tensors` 配置使用对称 INT4、group size 32，并排除全部 Attention、Shared Expert、稠密 MLP 和 LM Head；Router 也不是 `Linear` 模块。因此只有 Routed Expert 的三组投影是 INT4，其他原模型参数为 BF16，Router correction bias 为 FP32。`kv_cache_scheme: null` 且官方部署命令没有 KV 精度覆盖，本样本据此固定 BF16 KV；这是“配置与框架默认值推导”的 B 级证据，不是模型卡直接声明。

精确原模型参数量为 1,026,408,232,448；引擎每 Decode Token 执行 31,686,066,176 个矩阵参数，与官方取整的 32B 相符。总权重容量为 594,205,904,896 bytes，其中 INT4 Expert payload 为 507,343,011,840 bytes，BF16 group scale 与 shape metadata 为 63,418,429,440 bytes。checkpoint 另存 15,616 bytes 非参数 RoPE buffer，故完整 safetensors tensor 容量恰好闭合到 594,205,920,512 bytes。Hugging Face API 显示的 `I32` 是对 INT4 packed container 的逻辑展开统计，不能解释为 32-bit 模型精度或直接乘 4 计算容量。

当前 schema 能准确表示矩阵 FLOPs、INT4/BF16 packed weight payload、逻辑 MLA KV，以及无真实路由 Trace 时的 Batch Expert 并集近似；但不能表示“零 FLOPs、随同一 Expert 并集读取”的 group scale 流量。Scale 已计入容量，暂不混入引擎的 `weight_read_bytes`。在 Batch=1 时，引擎矩阵 payload 为 31,663,194,112 bytes，另有 1,321,205,760 bytes Expert scale 流量未进入结果。不要把 Scale 塞进参数组、`fixed_cost` 或全局 multiplier，否则会分别制造假 FLOPs、错误的 Batch 归一化或放大 BF16 权重。

本配置绑定逻辑吸收式 MLA 服务 profile。固定仓库的通用 Transformers remote-code 路径会先展开逐头 K/V 再交给 DynamicCache，属于另一种物理部署，不能与本结果直接对比。官方部署指南给出的 8×H200 TP8 是 native INT4+256K 的可部署证据；本研究仍按逻辑单芯片/超大芯片边界排除片间通信、Expert dispatch 和临时 workspace。

来源：[官方发布](https://moonshotai.github.io/Kimi-K2/thinking.html)、[固定模型卡](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/6e3cdad87f3e39a24d887ed53b494ba91bfadced/README.md)、[固定架构与量化配置](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/6e3cdad87f3e39a24d887ed53b494ba91bfadced/config.json)、[固定权重索引](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/6e3cdad87f3e39a24d887ed53b494ba91bfadced/model.safetensors.index.json)、[固定部署指南](https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/6e3cdad87f3e39a24d887ed53b494ba91bfadced/docs/deploy_guidance.md)、[Kimi K2 技术报告](https://arxiv.org/abs/2507.20534)。

### Qwen3-32B-AWQ

该模型是 64 层稠密 causal decoder-only Transformer，隐藏维度为 5,120，SwiGLU FFN 中间维度为 25,600。Attention 使用 64 个 Query Head、8 个 KV Head、Head Dim 128 的全上下文 GQA，并在 Q、K 上增加 RMSNorm。词表为 151,936，输入 Embedding 与 LM Head 不共享。同一 checkpoint 可通过模板开关运行 Thinking 或 Non-Thinking；这会改变生成长度和上下文轨迹，但不会改变固定 `C`、`B` 下一个 Decode Token 的结构公式。

“AWQ 4-bit”不是全模型 INT4。固定 checkpoint 包含 448 组 `qweight/qzeros/scales`，恰好对应 `64 × (Q/K/V/O + Gate/Up/Down)`，因此 31,205,621,760 个 Transformer Linear 权重为 group-size 128 的非对称 INT4。输入 Embedding、LM Head、全部 RMSNorm 和 Q/K Norm 仍是普通 BF16 checkpoint tensor；官方服务配置的 `torch_dtype=float16` 使这些张量、Activation 与默认 KV 路径按 FP16 运行。KV 精度属于“配置与框架默认值推导”的 B 级证据，不能说成模型卡直接声明。

精确原模型参数量为 32,762,123,264；引擎每 Decode Token 执行 31,983,534,080 个矩阵参数。原参数 payload 为 18,715,813,888 bytes；另有 487,587,840 bytes BF16 group scale 和 121,896,960 bytes packed zero point，因此总权重容量为 19,325,298,688 bytes。固定 index 的 `metadata.total_size` 比四个 shard header 闭合出的实际 tensor payload 多 13,107,200 bytes，本配置保留该异常但不使用错误的 index 数值覆盖容量。

当前 schema 能准确表示 INT4/FP16 矩阵 FLOPs、packed weight payload、全上下文 GQA 和 FP16 KV，但不能表达“无 FLOPs、每 Decode step 读取一次并由 Batch 共享”的 AWQ scale/zero-point 流量。它们已计入权重容量，暂不混入引擎的 `weight_read_bytes`。Batch=1 时，引擎矩阵 payload 为 17,158,635,520 bytes；若元数据也随权重流式读取，还应另加 609,484,800 bytes。把元数据塞进参数组会制造假 FLOPs，塞进 `fixed_cost` 则会按请求错误重复。

上下文保留三个不同边界：原生为 32,768；checkpoint 默认 `max_position_embeddings=40,960`，对应常见 8,192 Token Prompt 加 32,768 Token Output 的预算；官方用静态 YaRN factor 4 验证并提供了 131,072 的 vLLM/SGLang 部署方法。本样本为观察上下文演进采用这个公开支持的 128K profile，不能把它误写成“原生 128K”。FP16 KV 下，全模型每个历史 Token 占 262,144 bytes，因此单请求在 32K、40K 和 128K 处分别为 8、10 和 32 GiB。

来源：[Qwen3 官方发布](https://qwenlm.github.io/blog/qwen3/)、[固定模型卡](https://huggingface.co/Qwen/Qwen3-32B-AWQ/blob/0499c3ac83fdef8810b907a23894ba91e95eddd8/README.md)、[固定架构与 AWQ 配置](https://huggingface.co/Qwen/Qwen3-32B-AWQ/blob/0499c3ac83fdef8810b907a23894ba91e95eddd8/config.json)、[固定权重索引](https://huggingface.co/Qwen/Qwen3-32B-AWQ/blob/0499c3ac83fdef8810b907a23894ba91e95eddd8/model.safetensors.index.json)、[AWQ 部署文档](https://qwen.readthedocs.io/en/latest/quantization/awq.html)、[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)。

### Kimi-Linear-48B-A3B-Instruct

该模型是 27 层稀疏 MoE causal decoder-only 模型，隐藏维度为 2,304，词表为 163,840，输入 Embedding 与 LM Head 不共享。第 0 层是中间维度 9,216 的稠密 SwiGLU；后续 26 层各有 256 个中间维度 1,024 的 Routed Expert，每个 Token 选择 8 个，另有 1 个始终执行的 Shared Expert。官方 48B/3B 是取整标签；固定 checkpoint 的精确总参数为 49,122,681,728，引擎 active matrix/conv 参数为 3,106,750,464/token。

Attention 是 20 层 KDA 加 7 层全局 MLA 的混合结构。KDA 使用 32 heads、head dim 128 和 kernel width 4 的 q/k/v short-conv；MLA 使用 32 个 Query Head、512 维 KV latent、64 维 RoPE Key、128 维 NoPE 和 128 维 Value。配置固定原生 1,048,576 上下文，并绑定官方 vLLM 原生服务 profile：7 层 MLA 每历史 Token 保存 512 个 latent 元素和 64 个 RoPE Key 元素，BF16 下合计 8,064 bytes/token/request；20 层 KDA 保存固定状态，其中 recurrent matrix 为 FP32，共 41,943,040 bytes，short-conv state 为 BF16 且 vLLM 使用 `kernel_size - 1`，共 1,474,560 bytes。

官方 checkpoint 基本是 BF16，但不是“全 BF16”：49,122,599,168 个参数为 BF16，20 层 KDA 的 `A_log` 与 `dt_bias` 合计 82,560 个参数为 FP32。精确权重容量为 98,245,528,576 bytes。没有官方量化；不要套用社区 AWQ/FP8/GGUF 版本。KDA recurrent FLOPs 采用与本项目 Qwen3-Next DeltaNet 相同的显式状态口径：每 KDA 层 `7*state_matrix + 2*heads*value_dim = 3,678,208` FLOPs/token；gate 非线性、指数、归一化、路由和采样不计入。

当前 schema 能准确表示 BF16 矩阵/卷积权重、MoE batch expert 并集近似、逻辑吸收式 MLA KV，以及 KDA FP32/BF16 固定状态读写；但不能表示 vLLM 可能把部分 state 保留在片上的实现优化。本配置使用保守 logical-HBM 边界，假设 state 每 Decode token 读写 HBM；后续若研究片上驻留，应通过 `state_hbm_fraction` 做情景参数，而不是改架构参数。

本配置不能与 HF Transformers remote-code 路径直接混用：通用 Transformers 实现可能缓存展开后的 K/V，而官方 vLLM 原生路径使用 KDA 固定状态和吸收式 MLA latent KV。TP4 物理切分、通信、分页 metadata、workspace 和 speculative decoding 也都在本研究单芯片/超大芯片边界之外。

来源：[固定模型卡](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/README.md)、[固定架构配置](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json)、[固定权重索引](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/model.safetensors.index.json)、[固定配置代码](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/configuration_kimi.py)、[Kimi-Linear 论文](https://arxiv.org/abs/2510.26692)、[vLLM Kimi-Linear 支持](https://github.com/vllm-project/vllm/pull/27809)。
