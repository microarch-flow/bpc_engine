# 2024 年代表模型配置

本目录保存 2024 年发布的精确模型版本及其当年代表性实际部署精度配置。

- 配置必须遵循计算引擎的 `schema_version: 1`。
- 一个文件对应一个精确模型版本和一个实际部署精度。
- `metadata` 中记录发布日期、架构与精度来源、推导过程及不确定项。
- `analysis.contexts` 和 `analysis.batches` 仅为默认值，统一研究扫描可由 CLI 覆盖。
- 不在本目录保存假设性的统一 4/8/16-bit 对照配置。

## 已确认代表模型

| 角色 | 精确模型版本 | 实际部署精度 | 配置 |
|---|---|---|---|
| 旗舰资源需求 | Llama 3.1 405B Instruct FP8 | 中间 124 层 FFN 为 FP8，其余权重及 KV 为 BF16 | [llama_3_1_405b_instruct_fp8mixed_bf16kv.json](llama_3_1_405b_instruct_fp8mixed_bf16kv.json) |
| 主流部署 | Llama 3.1 70B Instruct NVIDIA ModelOpt FP8 | Transformer block Linear 为 FP8，Embedding/LM Head/Norm 及 KV 为 BF16 | [llama_3_1_70b_instruct_nvidia_fp8_bf16kv.json](llama_3_1_70b_instruct_nvidia_fp8_bf16kv.json) |
| 架构演进对照 | DeepSeek-V2-Chat RL | BF16 权重、逻辑吸收式 MLA KV 为 BF16 | [deepseek_v2_chat_bf16.json](deepseek_v2_chat_bf16.json) |
| MLA/MoE 扩展与原生 FP8 对照 | DeepSeek-V3 首发版 | 大矩阵与 Expert 为原生 FP8 W8A8，Embedding/LM Head/Router 及逻辑吸收式 MLA KV 为 BF16；普通 Decode 排除 MTP | [deepseek_v3_native_fp8_bf16kv.json](deepseek_v3_native_fp8_bf16kv.json) |
| 混合 SSM 与超长上下文对照 | AI21 Jamba 1.5 Large | ExpertsInt8：MoE 与稠密 MLP 权重 INT8，其余主要权重、激活、KV 和 Mamba State 为 BF16 | [jamba_1_5_large_experts_int8_bf16kv.json](jamba_1_5_large_experts_int8_bf16kv.json) |

## 选择理由

这些样本不是“年度榜单”，而是分别回答资源上界、实际部署基线和不同架构变化问题。

| 角色与模型 | 为什么选择 | 解释边界 |
|---|---|---|
| 旗舰资源需求：Llama 3.1 405B Instruct FP8 | 405B 具有当年第一梯队能力、128K 上下文和官方生产混合 FP8 profile，能同时锚定旗舰模型的绝对资源需求与大模型低精度部署方式。 | 它代表可审计的旗舰资源包络，不代表所有闭源旗舰；FP8 仅覆盖中间 124 层 FFN，不能按全模型 8 bit 处理。 |
| 主流部署：Llama 3.1 70B Instruct NVIDIA ModelOpt FP8 | 70B 兼顾较强能力、广泛 Llama 部署生态和公开可复现的 NVIDIA W8A8 FP8 服务路径；与 405B 同族还能在基本架构一致时比较规模与量化策略的影响。 | 本配置固定公开 ModelOpt/vLLM profile，不把 NIM 内部实现当作已知事实；采用证据主要是家族和平台级，不能解释为该 revision 的 Token 份额。 |
| 架构演进对照：DeepSeek-V2-Chat BF16 | 160 Expert Top-6 DeepSeekMoE 将约 236B 常驻参数压缩到约 21B 活跃子集，MLA 又把每层每历史 Token 的 KV 压缩为 512 维 latent 加 64 维 RoPE key；它能同时观察权重容量、活跃计算和长上下文 KV 三者的解耦。 | 官方 236B/21B 是取整值；Batch Expert 流量使用均匀独立近似，MLA 按吸收投影后的逻辑 KV 建模，不包含专家并行通信、路由分发或参考实现可能物化的临时 K/V。 |
| MLA/MoE 扩展与原生 FP8 对照：DeepSeek-V3 | 与 DeepSeek-V2 构成同族纵向对照：保留 128K MLA 路线，同时把稀疏模型扩展到 671B、256 Expert Top-8 和 36.6B 活跃矩阵参数，并引入原生 FP8、无辅助损失负载均衡与 MTP 训练目标，可分辨规模、稀疏度和精度变化对硬件需求的影响。 | 固定 2024-12-26 首发版而非 V3-0324；标准逐 Token Decode 不执行 MTP。FP8 scale、真实分组路由、专家通信和算子计算 dtype 不在当前引擎结果中。 |
| 混合 SSM 与超长上下文对照：Jamba 1.5 Large | 398B 常驻、94B 活跃参数结合 Top-2 MoE，并以 1:7 的 Attention/Mamba 比例把大部分序列层替换为固定状态递归；它在 256K 有效上下文下可观察混合 SSM 如何同时改变 KV 容量、状态容量、活跃计算和权重带宽。 | 398B/94B 是官方取整口径。ExpertsInt8 不是全模型 INT8 或 W8A8：仅 MoE/MLP 权重为 INT8，计算激活、KV 与 Mamba State 保持 BF16；引擎不复现融合反量化、并行通信或实测延迟。 |

两个 Llama 3.1 样本构成同族规模与精度的受控对照；DeepSeek-V2/V3 提供 MLA、DeepSeekMoE、稀疏规模和原生 FP8 的同族演进轨迹；Jamba 1.5 Large 再补入 Attention/Mamba 混合固定状态路线。

## 架构说明

### Llama 3.1 405B Instruct FP8

该模型是 126 层稠密 causal decoder-only Transformer。隐藏维度为 16,384，SwiGLU FFN 中间维度为 53,248；Attention 使用 128 个 Query Head、8 个 KV Head、Head Dim 128 的全上下文 GQA。Llama 3 RoPE scaling 将原生上下文扩展到 131,072。输入 Embedding 与 LM Head 不共享。

这里的“FP8”不是全模型 FP8。官方生产 checkpoint 只将中间 124 层 FFN 的 `gate_proj`、`up_proj` 和 `down_proj` 做 W8A8 dynamic row-wise FP8；Attention、首尾两层 FFN、Embedding、LM Head 和 RMSNorm 保持 BF16。持久 KV 也按 BF16 统计。对应的精确可训练参数为 405,853,388,800，其中 324,538,466,304 个 FFN 权重元素为 FP8，81,314,922,496 个参数元素为 BF16。

完整可训练权重 payload 为 487,168,311,296 bytes。官方 checkpoint 另有 15,237,120 个 FP32 FFN weight scale，共 60,948,480 bytes；因此本配置记录的总权重容量为 487,229,259,776 bytes。当前引擎的 `weight_read_bytes` 只统计参与矩阵计算的权重 payload，不统计这些 scale：现有 schema 没有“每 Decode step 读取一次、无 FLOPs、再按 Batch 摊销”的元数据字段，强行塞入参数组或 `fixed_cost` 都会改变计算语义。

本配置固定到 2024 年 8 月 16 日的修正 revision。首发 checkpoint 曾把 Tensor Parallel shard 中重复的 KV 权重误记成 16 个 KV Head；修正后的真实架构是 8 个。按首发索引计算会把 KV 投影与 KV Cache 错算为两倍。

来源：[Meta 发布说明](https://ai.meta.com/blog/meta-llama-3-1/)、[技术报告](https://arxiv.org/abs/2407.21783)、[官方 checkpoint](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct-FP8)、[Hugging Face 发布说明](https://huggingface.co/blog/llama31)、[KV Head 修正记录](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct-FP8/discussions/14)。

### Llama 3.1 70B Instruct NVIDIA ModelOpt FP8

该模型与 405B 样本同属 Llama 3.1 稠密架构，但规模下降到 80 层：隐藏维度为 8,192，SwiGLU FFN 中间维度为 28,672；Attention 使用 64 个 Query Head、8 个 KV Head、Head Dim 128 的全上下文 GQA。词表为 128,256，输入 Embedding 与 LM Head 不共享，原生上下文同样为 131,072。

本样本固定到 NVIDIA 于 2024 年 8 月 29 日公开的 ModelOpt FP8 checkpoint，而不是 Meta 的 BF16 原始 checkpoint。80 层内的 Q/K/V/O、`gate_proj`、`up_proj` 和 `down_proj` 共 560 个 Linear 全部使用 W8A8 FP8；Embedding、LM Head 和 RMSNorm 保持 BF16。固定 revision 的量化配置明确写明 `kv_cache_quant_algo: null`；其 2024 vLLM 示例没有覆盖 dtype 或 KV dtype，而 vLLM 的 `auto` 规则会让 BF16 模型使用 BF16 KV，因此本配置固定这条 BF16 KV 部署路径。

精确可训练参数为 70,553,706,496。Transformer block 中 68,451,041,280 个权重元素为 FP8；其余 2,102,665,216 个 BF16 参数恰好由两份独立的 Embedding/LM Head 和 RMSNorm 构成。checkpoint 还为每个 FP8 Linear 保存一个 FP32 `weight_scale` 和一个 FP32 `input_scale`，共 1,120 个标量、4,480 bytes。总 tensor 容量为 72,656,376,192 bytes。

NVIDIA 2024 模型卡给出了 TensorRT-LLM 和 vLLM 部署方式及 H100 实测，NIM 1.1 也列出了该模型的 FP8 latency/throughput profile。因此它是可追溯的实际部署精度，而不是假设性的统一 8-bit 对照。NIM 内部的 KV 精度和序列上限并未公开，本配置不假设其 engine 与公开 checkpoint 完全相同。当前引擎的 `weight_read_bytes` 仍只统计矩阵 payload；4,480 bytes scale 计入容量并单独记录，但不伪装成带 FLOPs 的参数组。

仓库当前 `main` 已升级为 ModelOpt 0.23.0 并启用 FP8 KV，与 2024 profile 不同。后续复算必须始终使用 `811ca36…`，不能从 `main` 刷新精度字段。

来源：[Meta 发布说明](https://ai.meta.com/blog/meta-llama-3-1/)、[固定 NVIDIA FP8 checkpoint](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8/tree/811ca36d86c5e5d63aa07b7c4b7f738c8af0c63e)、[2024 部署说明](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8/blob/57d2a4b129544a41f766589825dd2e70089bf6b0/README.md)、[固定量化配置](https://huggingface.co/nvidia/Llama-3.1-70B-Instruct-FP8/blob/811ca36d86c5e5d63aa07b7c4b7f738c8af0c63e/hf_quant_config.json)、[NIM 1.1 支持矩阵](https://docs.nvidia.com/nim/large-language-models/1.1.0/support-matrix.html)。

### DeepSeek-V2-Chat BF16

该模型是 60 层稀疏 MoE causal decoder-only Transformer，隐藏维度为 5,120。第 0 层使用中间维度 12,288 的稠密 SwiGLU；后续 59 层各有 160 个中间维度 1,536 的 Routed Expert，每个 Token 选择 6 个，另有 2 个始终执行的 Shared Expert。路由先从 8 个 Expert Group 中选择 3 组，再选 Top-6。

Attention 使用 MLA：128 个 Query Head，经 1,536 维 Query LoRA 和 512 维 KV LoRA 压缩；每个历史 Token、每层只保留 512 个压缩 KV 元素和 64 个解耦 RoPE Key 元素。按 BF16 计算，每层每历史 Token 为 1,152 bytes，60 层合计 69,120 bytes；在公开的 131,072 Token 上下文上，单请求逻辑 MLA KV 容量为 9,059,696,640 bytes。这里采用论文中的吸收投影逻辑 MLA 口径。2024 年公开的 vLLM 参考 PR 会展开并 Padding K/V 后使用通用 Attention Cache，因此只能证明公开服务路径，不能证明其物理 Cache 与本配置相同；两者的实测物理流量禁止直接比较。

官方将模型概括为 236B 总参数、21B 激活参数。固定 checkpoint 的精确参数量是 235,741,434,880，全部为 BF16；其中 222,717,542,400 个参数属于 Routed Expert。当前引擎将全部 MLA 投影、稠密 FFN、Router、Shared Expert 和 LM Head 计为 12,498,862,080 个 Always-active Matrix Parameter，再加每层 Top-6 Routed Expert，得到每 Token 20,850,769,920 个活跃矩阵参数。该精确值与官方取整口径兼容，但不能直接用“21B”替换。

官方发布的服务上下文是 128K，因此配置采用 131,072；checkpoint 中的 `max_position_embeddings=163840` 是 YaRN 扩展后的内部位置上限，不视作另一个已验证服务长度。官方 checkpoint 和推理说明均采用 BF16，配置据此将权重与压缩 MLA KV 都记为 16 bit。Batch 大于 1 时没有公开真实路由 Trace，Expert Weight Traffic 使用均匀独立近似；它不包含 Group-limited 路由相关性、专家并行通信和路由分发流量。

来源：[官方发布与模型表](https://github.com/deepseek-ai/DeepSeek-V2)、[技术报告](https://arxiv.org/abs/2405.04434)、[固定 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat/tree/8e3f5f6c2226787e41ba3e9283a06389d178c926)、[固定架构配置](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat/blob/8e3f5f6c2226787e41ba3e9283a06389d178c926/config.json)、[固定权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat/blob/8e3f5f6c2226787e41ba3e9283a06389d178c926/model.safetensors.index.json)、[2024 vLLM 集成](https://github.com/vllm-project/vllm/pull/4650)。

### DeepSeek-V3 native FP8

DeepSeek-V3 是 61 层稀疏 MoE causal decoder。前三层使用中间维度 18,432 的稠密 SwiGLU；后续 58 层各有 256 个中间维度 2,048 的 Routed Expert，每个 Token 选择 8 个，另执行 1 个 Shared Expert。它沿用 512 维 KV latent、64 维解耦 RoPE Key 的 MLA，但把主模型扩展到精确 671,026,419,200 个参数。当前引擎计得每 Token 36,624,596,992 个活跃矩阵参数；首发权重说明中的 36.7B 后来被官方更正为 36.6B，与该值取整一致。

首发 checkpoint 只发布原生 FP8 权重：量化矩阵采用 E4M3、128×128 block scale 和动态 W8A8 activation；Embedding、LM Head、Router 与 Norm 保持 BF16，scale 为 FP32。按 dtype 逐项闭合，普通 Decode 加载的 61 层主模型 tensor 容量为 673,150,611,808 bytes。权重索引中的 `total_size=1,369,062,772,000` 是把所有 tensor 元素按两字节计算的 BF16 等价值，不是 FP8 物理容量，禁止直接用于芯片容量趋势。

KV 固定为 BF16：首发官方 demo 先设置 BF16 默认 dtype，再用未指定 dtype 的张量物化 512 维 latent cache 与 64 维 RoPE cache。每层每历史 Token 因此是 1,152 bytes，61 层合计 70,272 bytes；在公开 131,072 Token 上下文时为 9,210,691,584 bytes（8.578125 GiB）/请求。模型卡提到 SGLang 支持 FP8 KV，但这是可选能力，不代表该首发基线默认启用。

完整 Hugging Face artifact 还包含一个 MTP Module，因物理重复共享表并包含 FP32 scale，共有 684,531,386,000 个 tensor 元素、688,574,839,360 bytes。普通逐 Token Decode 不执行它：论文允许推理时丢弃 MTP，首发模型卡也说明社区支持仍在开发，官方 demo 只运行 61 层主模型。若研究 speculative decoding，必须另建包含 Draft、Verify、接受率和输出 Token 归一化的 profile，不能简单给本配置多加一层。

当前引擎准确表达混合 FP8/BF16 权重、吸收式 MLA KV、Top-8 活跃 Expert 和 Batch Expert Union 近似；它不统计 FP8 scale 读取、动态量化/反量化、Sigmoid 与修正 Bias、真实 8 组选 4 组路由、专家并行通信或 dtype 分项 Peak FLOPs。

来源：[官方发布](https://api-docs.deepseek.com/news/news1226)、[技术报告](https://arxiv.org/abs/2412.19437)、[2024-12-26 首发 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/dd31960ee457249502f7c6a652a30ff78e9fc792)、[固定架构配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/dd31960ee457249502f7c6a652a30ff78e9fc792/config.json)、[固定权重索引](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/dd31960ee457249502f7c6a652a30ff78e9fc792/model.safetensors.index.json)、[固定官方 demo](https://github.com/deepseek-ai/DeepSeek-V3/tree/4c2fdb8f55e049553b9f4f1a3241f86d739c8cf8/inference)、[活跃参数勘误](https://huggingface.co/deepseek-ai/DeepSeek-V3/commit/0cf17482555fbf7bc49273e499647cc71c2bd7a7)。

### Jamba 1.5 Large ExpertsInt8

Jamba 1.5 Large 是 72 层混合 causal decoder，由 9 个八层 Jamba Block 组成。全模型共有 9 个全上下文 GQA 层和 63 个 Mamba-1 层，隐藏维度为 8,192；Attention 使用 64 个 Query Head、8 个 KV Head、Head Dim 128。36 层使用 16 Expert、Top-2 MoE，另外 36 层使用中间维度 24,576 的稠密 SwiGLU。官方口径为约 398B 总参数、94B 活跃参数；按公开结构逐项闭合的精确 checkpoint 参数量为 398,555,145,696。

Mamba 展开维度为 16,384，State Dim 为 16，卷积窗口为 4。按 BF16 逻辑口径，63 层固定 State 为 41,287,680 bytes（39.375 MiB）/请求；只有 9 个 Attention 层保存随上下文增长的 KV，在 262,144 Token 时恰为 9 GiB/请求。这与 Llama 3.1 的逐层 GQA、DeepSeek-V2 的逐层 MLA 压缩形成三条不同的长上下文资源路线。

本配置采用官方推荐的 vLLM ExpertsInt8 部署。vLLM 将 16-Expert MoE 和以单 Expert 实现的稠密 MLP 权重保存为 INT8，并使用 FP32 scale，在融合 kernel 内反量化后以 BF16 激活计算；Attention、Mamba 投影、Embedding 和 LM Head 不量化，KV 与 Mamba Cache 为 BF16，Mamba A 运行时参数为 FP32。因此不能按“全模型 8 bit”计算容量或流量。当前引擎可表达 GQA KV、Mamba 固定状态和 Top-2 MoE 活跃矩阵，但 Mamba recurrence FLOPs 仍是核心算术近似，且不包含 scale 读取、融合反量化、真实路由、并行通信或延迟。

来源：[AI21 发布说明](https://www.ai21.com/blog/announcing-jamba-model-family/)、[技术报告](https://arxiv.org/abs/2408.12570)、[官方模型卡](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.5)、[2024 首发 checkpoint](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.5/tree/bdf8d19ae6e1b5b062b0cbc31d932b532515ef02)、[精确参数 API](https://huggingface.co/api/models/ai21labs/AI21-Jamba-Large-1.5/revision/bdf8d19ae6e1b5b062b0cbc31d932b532515ef02)、[vLLM 0.5.5 Jamba 实现](https://github.com/vllm-project/vllm/blob/v0.5.5/vllm/model_executor/models/jamba.py)、[ExpertsInt8 实现](https://github.com/vllm-project/vllm/blob/v0.5.5/vllm/model_executor/layers/quantization/experts_int8.py)。
