# 2023 年代表模型配置

本目录保存 2023 年发布的精确模型版本及其当年代表性实际部署精度配置。

- 配置必须遵循计算引擎的 `schema_version: 1`。
- 一个文件对应一个精确模型版本和一个实际部署精度。
- `metadata` 中记录发布日期、架构与精度来源、推导过程及不确定项。
- `analysis.contexts` 和 `analysis.batches` 仅为默认值，统一研究扫描可由 CLI 覆盖。
- 不在本目录保存假设性的统一 4/8/16-bit 对照配置。

## 已确认代表模型

| 角色 | 精确模型版本 | 实际部署精度 | 配置 |
|---|---|---|---|
| 旗舰资源需求 | Falcon 180B base | BF16 权重、BF16 KV | [falcon_180b_bf16.json](falcon_180b_bf16.json) |
| 主流部署 | Llama 2 70B Chat HF | FP16 权重、FP16 KV | [llama_2_70b_chat_fp16.json](llama_2_70b_chat_fp16.json) |
| 效率与局部注意力对照 | Mistral 7B Instruct v0.1 | BF16 权重、BF16 KV | [mistral_7b_instruct_v0_1_bf16.json](mistral_7b_instruct_v0_1_bf16.json) |
| 超长上下文资源压力对照 | Yi-34B-200K 原始版本 | BF16 权重、BF16 KV | [yi_34b_200k_bf16.json](yi_34b_200k_bf16.json) |
| 架构演进对照 | Mixtral 8x7B Instruct v0.1 | BF16 权重、BF16 KV | [mixtral_8x7b_instruct_v0_1_bf16.json](mixtral_8x7b_instruct_v0_1_bf16.json) |

## 选择理由

这些样本不是“年度榜单”，而是分别回答资源上界、实际部署基线和架构变化等不同问题。

| 角色与模型 | 为什么选择 | 解释边界 |
|---|---|---|
| 旗舰资源需求：Falcon 180B | 它是 2023 年公开可审计的超大稠密模型，可作为 BF16 权重、矩阵计算和 KV 容量的高资源锚点；8-KV-head multigroup GQA 还能与 2022 PaLM 的单 KV Head MQA 连续比较。 | 它是公开旗舰的资源代理，不等于证明其综合能力或部署量为年度第一；原生上下文只有 2K。 |
| 主流部署：Llama 2 70B Chat | 70B Chat 兼顾较强通用知识/推理能力与明确的云端、模型和推理生态接入，适合作为 2023 年面向服务的 FP16/GQA/4K 基线。 | 采用证据主要属于 Llama 2 家族和平台可用性，不是该 checkpoint 的实测 Token 份额，不能直接换算行业占比。 |
| 效率与局部注意力对照：Mistral 7B Instruct v0.1 | 它以约 7.24B 稠密参数同时引入 GQA、4K Sliding-Window Attention 和滚动 KV Cache，使 Attention 扫描量和 KV 容量在 4K 后封顶；可与 Llama 2 70B 的全注意力及 Mixtral 的 MoE/32K 全注意力直接比较。 | 它是效率和机制锚点，不代表年度能力上界或部署份额；8K 是发布上下文，checkpoint 的 `max_position_embeddings=32768` 不能解释为原生 32K。 |
| 超长上下文资源压力对照：Yi-34B-200K | 它在约 34B 稠密 GQA 架构上把公开上下文推进到 200K，却仍采用全上下文 Attention，适合观察上下文增长快于 Attention/KV 优化时，对 Decode 计算、带宽和容量形成的压力。它与 Mistral 7B 的 4K SWA/滚动 Cache 形成直接对照。 | 它是 Base 模型和资源压力样本，不代表 Chat 部署量或 200K 下的复杂推理质量。配置固定 2023 原始权重；2024 年继续训练后的更新权重和评测不能倒灌。 |
| 架构演进对照：Mixtral 8x7B Instruct v0.1 | 每层 8 Expert、Top-2 路由将约 46.7B 常驻参数与约 12.9B 活跃参数子集分离，并把上下文扩展到 32K，适合研究 MoE 如何重新分配容量、权重带宽和计算量。 | 它在 2023 年 12 月才发布，不代表全年采用；Batch Expert 流量使用均匀独立近似，且 12.9B 容量口径不能与 12.7486B 引擎矩阵口径混用。 |

五个样本形成“180B 稠密旗舰—70B 主流服务—7B 稠密 SWA—34B/200K 全注意力—稀疏 MoE”的互补切片，而不是一组高度相似的模型。

## 架构说明

### Falcon 180B

Falcon 180B 是稠密 causal decoder-only Transformer，共 80 层。隐藏维度为 14,848，GeLU FFN 维度为 59,392，使用 RoPE 和并行 Attention/FFN block；词表为 65,024，输入与输出 Embedding 共享，原生上下文长度为 2,048。

Attention 有 232 个 Query Head，Head Dim 为 64，但只保留 8 个独立 KV Head。Falcon 将其称为 multigroup multiquery：训练时每个 Tensor Parallel rank 对应一组 KV；在本研究的逻辑超大芯片边界下，它等价于全上下文 GQA。相较 2022 年 PaLM 的单 KV Head MQA，它在并行实现友好性和 KV 容量之间取了折中。

本配置采用 2023 年发布时官方演示的原生 BF16 推理路径，权重与 KV 均按 16 bit 统计。同期发布资料也实际测试了 8-bit 和 4-bit/GPTQ 推理，但它们属于同一 release 的其他部署 profile；旗舰资源需求样本保留 BF16 参考路径，不额外虚增模型样本。

参数矩阵口径为 178,552,307,712：包含 Attention 投影、两层 GeLU FFN 矩阵及只计一份的共享 Embedding/LM Head；不包含 LayerNorm、逐元素操作和实现相关物理开销。FlashAttention 和 Tensor Parallel 布局不改变当前 logical-HBM 的完整上下文 Decode 口径。

来源：[TII 发布说明](https://www.tii.ae/news/technology-innovation-institute-introduces-worlds-most-powerful-open-llm-falcon-180b)、[Hugging Face 发布与推理说明](https://huggingface.co/blog/falcon-180b)、[模型卡](https://huggingface.co/tiiuae/falcon-180B)、[架构论文](https://arxiv.org/abs/2311.16867)。

### Llama 2 70B Chat

Llama 2 70B Chat 是面向对话部署的稠密 causal decoder-only Transformer，共 80 层。隐藏维度为 8,192，SwiGLU FFN 维度为 28,672，使用 RoPE、RMSNorm 和全上下文 GQA；Attention 有 64 个 Query Head、8 个 KV Head，Head Dim 为 128，原生上下文从上一代常见的 2,048 扩展到 4,096。

输入 Embedding 与 LM Head 不共享，词表为 32,000。引擎矩阵口径为 68,713,185,280 个参数：包含 Attention、FFN 和完整 LM Head；独立输入 Embedding 的 262,144,000 个参数只进入完整权重容量，Decode step 仅读取其中一行。连同 RMSNorm，可训练参数总数为 68,976,648,192，FP16 可训练权重容量为 137,953,296,384 bytes。官方元数据另有 5,120 个非参数 FP32 RoPE `inv_freq` buffer，不计入该权重容量。

本配置对应官方 Hugging Face FP16 checkpoint。Transformers 4.31 将投影后的 8 组 K/V 直接存入 cache，不做单独精度转换，因此 KV 同样按 FP16 统计。Attention softmax 的 FP32 临时计算不改变持久 KV 精度，也不进入当前 logical-HBM 基线。

Llama 2 家族在 2023 年进入 AWS、Azure、Hugging Face 等部署生态；采用证据主要是家族级而非 70B 单尺寸 Token 份额。选择 70B Chat 是为了在主流生态中保留较强的通用知识与推理能力代表性。

来源：[Meta 发布说明](https://about.fb.com/news/2023/07/llama-2/)、[官方模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama2/MODEL_CARD.md)、[Hugging Face checkpoint](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)、[论文](https://arxiv.org/abs/2307.09288)。

### Mistral 7B Instruct v0.1

Mistral 7B Instruct v0.1 是 32 层稠密 causal decoder-only Transformer，隐藏维度为 4,096，SwiGLU FFN 维度为 14,336，使用 RMSNorm 和 RoPE。Attention 包含 32 个 Query Head、8 个 KV Head，Head Dim 为 128，并采用窗口为 4,096 Token 的 Sliding-Window GQA；词表为 32,000，输入 Embedding 与 LM Head 不共享。

官方发布口径中的上下文为 8,192，4,096 是每层直接可见的历史窗口；checkpoint 中的 `max_position_embeddings=32768` 是另一个位置配置边界，不能覆盖前两者。本配置因此把顶层计算上限设为 8,192，并按 `min(C,4096)` 计算每层 Attention 扫描和滚动 KV 容量。BF16 下全模型每个历史 Token 的 KV 为 131,072 bytes，单请求容量在窗口饱和后固定为 536,870,912 bytes（512 MiB）。跨层有效感受野可能超过 4K，但不会增加单层 Decode 的直接 KV 读取量。

固定的 2023 revision 含 7,241,732,096 个 BF16 参数元素，权重 payload 为 14,483,464,192 bytes。本引擎每 Token 执行的矩阵口径为 7,110,393,856：包含 Attention、FFN 和完整 LM Head；独立 Input Embedding 只以单行读取进入 Decode 流量。配置绑定官方描述的 rotating-buffer 优化路径；如果某个 runtime 只施加局部 Attention mask、却仍保存完整历史 KV，应另建物理部署 profile。

来源：[官方发布说明](https://mistral.ai/news/announcing-mistral-7b/)、[官方模型文档](https://docs.mistral.ai/models/model-cards/mistral-7b-0-1)、[技术报告](https://arxiv.org/abs/2310.06825)、[固定 checkpoint 配置](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/464c09acb438a06c3a5eaafa25b90069df87efca/config.json)、[固定 checkpoint index](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/464c09acb438a06c3a5eaafa25b90069df87efca/model.safetensors.index.json)。

### Yi-34B-200K

Yi-34B-200K 是 60 层稠密 Llama-style causal decoder-only Transformer。隐藏维度为 7,168，SwiGLU FFN 维度为 20,480，使用 RMSNorm 和 RoPE；Attention 包含 56 个 Query Head、8 个 KV Head，Head Dim 为 128，采用全上下文 GQA，不使用 Sliding-Window Attention。词表为 64,000，输入 Embedding 与 LM Head 不共享。

本研究固定官方指定的 2023-11-05 原始权重版本 `069cd341…cf4`：其配置声明 `max_position_embeddings=200000`、`rope_theta=5000000`、无 RoPE scaling，全部 safetensors 权重为 BF16。2024 年 3 月更新后的当前权重把 `rope_theta` 改为 10,000,000，并继续进行了长上下文训练；这些更新及后续 Needle-in-a-Haystack 结果不属于本样本。

固定 index 的 BF16 payload 为 68,777,834,496 bytes，对应 34,388,917,248 个实际 tensor 元素；本引擎每 Token 执行的矩阵口径为 33,929,297,920。全 60 层每个历史 Token 的 BF16 GQA KV 为 245,760 bytes，因此在 200,000 上下文时，单请求逻辑 KV 容量为 49,152,000,000 bytes（约 45.78 GiB），Attention 为 344,064,000,000 FLOPs/Token。三者都随 `C` 线性增长，没有 SWA 饱和点。

“200K”在本研究中表示该 release 的官方声明、模型配置和 tokenizer 上限，不直接等同于 200K 下的复杂长文本推理质量。固定 revision 的发布期证据没有分别给出训练与评测最大长度，因此这两个字段保留为 `null`；后来的技术报告仅用于解释全注意力和长上下文训练路线。

来源：[官方发布记录](https://github.com/01-ai/Yi)、[固定 checkpoint 配置](https://huggingface.co/01-ai/Yi-34B-200K/blob/069cd341d60f4ce4b07ec394e82b79e94f656cf4/config.json)、[固定 tokenizer 配置](https://huggingface.co/01-ai/Yi-34B-200K/blob/069cd341d60f4ce4b07ec394e82b79e94f656cf4/tokenizer_config.json)、[固定 checkpoint index](https://huggingface.co/01-ai/Yi-34B-200K/blob/069cd341d60f4ce4b07ec394e82b79e94f656cf4/model.safetensors.index.json)、[后续技术报告](https://arxiv.org/abs/2403.04652)。

### Mixtral 8x7B Instruct v0.1

Mixtral 8x7B Instruct v0.1 是稀疏 MoE causal decoder-only Transformer，共 32 层。隐藏维度为 4,096，每层包含 8 个中间维度为 14,336 的 SwiGLU Expert，Router 为每个 Token 选择其中 2 个。Attention 使用 32 个 Query Head、8 个 KV Head、Head Dim 128 的全上下文 GQA，原生上下文长度为 32,768；输入 Embedding 与 LM Head 不共享。

“8x7B”不等于 56B。官方 checkpoint 的精确参数量为 46,702,792,704，其中全部 Expert 占 45,097,156,608。本项目按“全部共享参数加每层两个 Expert”的容量口径重构出 12,879,925,248，与官方四舍五入的 12.9B 吻合；官方资料没有公布该精确计数公式。本引擎只统计实际执行的参数化矩阵，因此每 Token 的活跃矩阵参数为 12,748,587,008。两种口径不能混用。

本配置采用官方 checkpoint 的 BF16 执行路径：权重和持久 KV 均按 16 bit 统计。精确 BF16 权重容量为 93,405,585,408 bytes；Transformers 4.36 将 BF16 投影 K/V 写入 Cache 时不转换精度。每请求每个历史 Token 的全层 KV 为 131,072 bytes，因此在 32K 上下文时单请求 KV 容量为 4 GiB。官方也提供 FP16 执行方式，但那属于另一个部署 profile。

Decode 的计算始终只激活每层 2 个 Expert，但 Batch 内不同 Token 可能触达更多 Expert。没有真实路由 Trace 时，配置采用均匀独立路由近似：`B=1` 精确触达 2 个 Expert，`B=32` 每层期望触达约 7.999 个。这是权重流量分析假设，不是部署采用或路由分布实测值。

Mixtral 使用 32K 全注意力，而不是 Mistral 7B 的滑动窗口。首发后的 Hugging Face 集成配置曾短暂包含 `sliding_window`，官方于 2023 年 12 月 15 日修正为 `null`，并明确说明该模型从未设计为滑动窗口注意力。

来源：[官方发布博客](https://mistral.ai/news/mixtral-of-experts/)、[官方发布生命周期](https://legal.mistral.ai/ai-governance/models/mixtral-8-7b)、[官方模型文档](https://docs.mistral.ai/models/model-cards/mixtral-8x7b-0-1)、[checkpoint](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)、[全注意力修正](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/commit/125c431e2ff41a156b9f9076f744d2f35dd6e67a)、[技术报告](https://arxiv.org/abs/2401.04088)。
