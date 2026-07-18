# 2026 年代表模型配置

本目录保存 2026 年发布的精确模型版本及其当年代表性实际部署精度配置。本轮研究的 YTD 截止日期为 2026-07-17。

- 配置必须遵循计算引擎的 `schema_version: 1`。
- 一个文件对应一个精确模型版本和一个实际部署精度。
- `metadata` 中记录发布日期、架构与精度来源、推导过程及不确定项。
- `analysis.contexts` 和 `analysis.batches` 仅为默认值，统一研究扫描可由 CLI 覆盖。
- 不在本目录保存假设性的统一 4/8/16-bit 对照配置。

## 三个名额与选择状态

只允许使用 2026-07-17 截止日前已发布的信息，不能把后续 release 倒灌进 YTD 样本。每个模型经用户确认后，都必须补充精确 release、模型级选择理由和证据边界。

| 角色 | 当前状态 | 配置 |
|---|---|---|
| 旗舰资源需求 | 已选择 Kimi-K2.6，revision `7eb5002…`；语言模型 routed expert 为 native INT4，其他文本权重与逻辑吸收式 MLA KV 为 BF16 | [kimi_k2_6_int4w_bf16kv.json](kimi_k2_6_int4w_bf16kv.json) |
| 主流部署 | 已选择 Qwen3.6-35B-A3B-FP8，revision `95a723d…`；标准 text-only profile 使用 FP8 大矩阵、BF16 KV/State/LM Head/Embedding | [qwen3_6_35b_a3b_fp8_bf16kv.json](qwen3_6_35b_a3b_fp8_bf16kv.json) |
| 架构演进对照 | 已选择 GLM-5.2-FP8，revision `ba978f7…`；标准长上下文 profile 使用 FP8 主权重、FP8 MLA KV、BF16 DSA index key，并排除 MTP speculative draft/verify | [glm_5_2_fp8_dsa_fp8kv.json](glm_5_2_fp8_dsa_fp8kv.json) |

三者互补关系：Kimi-K2.6 覆盖 1T 级稀疏 MoE + MLA + native INT4 的旗舰资源包络；Qwen3.6-35B-A3B-FP8 覆盖更接近主流部署成本点的 35B/3B active hybrid Gated DeltaNet 路线；GLM-5.2-FP8 覆盖 1M context 下 DSA sparse attention + IndexShare 的长上下文 Decode 路线。它们都不能单独代表 2026 全年：Kimi-K2.6 不代表后续 K3 级资源边界，Qwen3.6 不代表超大旗舰，GLM-5.2 不代表常规短上下文低成本服务。

## P0 观察候选

Kimi K3 记录为 2026 旗舰 P0 观察候选。它的公开信息显示为 2.8T 总参数、1M context、KDA + AttnRes、Stable LatentMoE 16/896 experts、MXFP4/MXFP8 训练与推理 profile，是强烈的 2026 旗舰替换候选。但在本轮 2026YTD 截止点，full model weights 与技术报告尚未进入可审计状态，因此不能替代当前已能闭合配置和计算的 Kimi-K2.6。

## 选择理由

| 角色与模型 | 为什么选择 | 解释边界 |
|---|---|---|
| 旗舰资源需求：Kimi-K2.6 native INT4 | Kimi-K2.6 在 2026YTD 已公开 checkpoint、config、safetensors index、模型卡与部署指南；模型卡给出 1T 总参数、32B 激活参数、61 层、384 experts、top-8、256K context 和 MLA，且 native INT4 方法沿用 Kimi-K2-Thinking。它能作为 2026YTD 可审计旗舰资源包络，而不是只有新闻口径。 | Kimi-K2.6 是多模态 checkpoint；本配置只计算语言模型 text Decode。vision tower 与 mm_projector 不参与每个生成 token 的 Decode FLOPs/权重流量，但在 metadata 中记录 checkpoint 容量。它也不能代表 2026 下半年可能被 Kimi K3 等模型推高后的最终旗舰边界。 |
| 主流部署：Qwen3.6-35B-A3B-FP8 | 它是 Qwen 官方 2026YTD FP8 checkpoint，当前 Hugging Face 近期下载约 765 万，明显高于同窗口多数公开模型，是强传播度代理。35B 总参数、3B 激活参数、FP8 大矩阵和官方 vLLM/SGLang/KTransformers 服务路径，使它比 Kimi-K2.6 更像可广泛部署的成本点。 | 下载量是 2026-07-18 获取的当前传播代理，不是 2026 历史 token 份额。该 checkpoint 是多模态模型并包含 MTP；本配置固定官方标准 text-only Decode profile，不启用 speculative MTP，也不把 vision encoder 计入每 token Decode。 |
| 架构演进对照：GLM-5.2-FP8 | 它是 Z.ai 官方 2026YTD FP8 checkpoint，公开 1M context，并明确提出 DSA sparse attention 的 IndexShare：每 4 层复用一次 indexer，官方称 1M context 下 per-token FLOPs 降低 2.9×。它补上了 Kimi-K2.6/Qwen3.6 未覆盖的“长上下文稀疏索引复用”路线。 | vLLM recipes 会在部分命令中启用 MTP speculative serving；本配置为了和其他年度样本可比，只计算普通 one-token Decode baseline，不计 MTP draft/verify。FP8 scale_inv 与 correction bias 计入 checkpoint payload，但当前 schema 不把这类零 FLOP 元数据流量混入 `weight_read_bytes`。 |

## 架构说明

### Kimi-K2.6 native INT4

该 checkpoint 是 image/video-text-to-text 模型，但 text decoder 是 Kimi-K2 系列 61 层稀疏 MoE MLA 语言模型。隐藏维度为 7,168，词表为 163,840；第 0 层是中间维度 18,432 的稠密 SwiGLU，后续 60 层各有 384 个中间维度 2,048 的 Routed Expert，每个 token 选择 8 个，另有 1 个始终执行的 Shared Expert。输入 Embedding 与 LM Head 不共享。

Attention 使用 MLA：64 个 Query Head，经 1,536 维 Query LoRA 和 512 维 KV LoRA 压缩，NoPE、RoPE 和 Value Head 维度分别为 128、64 和 128。配置固定官方 256K 上下文的精确值 262,144，并采用优化服务路径中的逻辑吸收式 MLA：每层、每历史 token 保存 512 个压缩 KV 元素和 64 个 RoPE Key 元素。BF16 下 61 层合计为 70,272 bytes/token/request。

native INT4 不是全模型 INT4。固定 config 的 `compressed-tensors` 规则只量化 routed expert Linear 权重，并排除 self-attention、shared experts、dense MLP、LM Head、vision tower 和 multimodal projector。语言模型精确 original 参数量为 1,026,408,232,448；引擎每 Decode token 执行 31,686,066,176 个矩阵参数。text Decode 权重容量为 594,205,904,896 bytes；完整多模态 checkpoint tensor payload 为 595,148,192,736 bytes，其中 vision tower 为 833,732,064 bytes，mm projector 为 108,555,776 bytes。

当前 schema 能准确表示 INT4/BF16 矩阵 FLOPs、packed routed expert payload、逻辑 MLA KV，以及无真实路由 trace 时的 batch expert 并集近似；但不能表达“零 FLOPs、随同一 Expert 并集读取”的 group scale 流量。Scale 已计入容量，暂不混入引擎的 `weight_read_bytes`。视觉编码器和 multimodal projector 属于多模态输入处理，不属于 text Decode 每 token 工作量。

来源：[固定模型卡](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/README.md)、[固定配置](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json)、[固定权重索引](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/model.safetensors.index.json)、[固定部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/docs/deploy_guidance.md)。

### Qwen3.6-35B-A3B-FP8

该 checkpoint 是 Causal Language Model with Vision Encoder，但本样本绑定官方 vLLM/SGLang 标准 text-only Decode profile。Text decoder 有 40 层，隐藏维度 2,048，词表 padding 到 248,320，输入 Embedding 与 LM Head 不共享。层布局为 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，即 30 层 Gated DeltaNet linear attention 和 10 层 full-context gated GQA。

每层都有 MoE：256 个 Routed Expert，每 token 选择 8 个，另有 1 个 Shared Expert；单个 expert 的 SwiGLU 参数为 3,145,728。精确 text original 参数量为 34,660,610,688；引擎每 Decode token 执行 2,946,252,800 个 matrix/conv 参数，与官方 35B/3B 取整标签一致。当前 Hugging Face 近期下载量约 7,651,080，是主流部署传播度旁证。

FP8 不是全模型 FP8。固定 safetensors headers 显示 text 主模型有 33,617,346,560 个 FP8 E4M3 matrix elements、1,043,264,128 个 BF16 original text elements，以及 2,051,840 个 BF16 `weight_scale_inv` elements。LM Head、Embedding、router、shared expert gate、norm、Gated DeltaNet 的小投影和 conv 权重仍是 BF16。Text Decode 权重容量为 35,707,978,496 bytes；vision encoder 为 893,142,496 bytes，MTP 为 853,668,480 bytes，二者只记录容量，不进入标准 Decode 工作量。

Sequence mixer 分两类：10 层 full-context GQA 每历史 token 合计保存 20,480 bytes BF16 KV，在 262,144 native context 下为 5,368,709,120 bytes/request；30 层 Gated DeltaNet 使用固定 BF16 state。状态口径采用官方 vLLM-style 标准 profile：不启用 speculative MTP，conv state 为 `kernel_size - 1`，因此全模型 Gated DeltaNet state 为 32,931,840 bytes/request。Gated DeltaNet recurrent FLOPs 沿用本项目 Qwen3-Next 口径：每层 `7*state_matrix + 2*heads*value_dim = 3,678,208` FLOPs/token。

当前 schema 能准确表示 FP8/BF16 matrix FLOPs、MoE batch expert 并集近似、full GQA KV 和 Gated DeltaNet fixed state；但不能表达“无 FLOPs、每 step 读取一次并按 batch 共享”的 FP8 `weight_scale_inv` 流量。Scale 已计入容量，暂不混入 `weight_read_bytes`。若未来启用 MTP speculative serving，需要单独建立 profile，不能把 MTP draft/verify 开销混入本标准 Decode baseline。

来源：[固定模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/95a723d08a9490559dae23d0cff1d9466213d989/README.md)、[固定配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/95a723d08a9490559dae23d0cff1d9466213d989/config.json)、[固定权重索引](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/95a723d08a9490559dae23d0cff1d9466213d989/model.safetensors.index.json)、[Qwen3.6 官方博客](https://qwen.ai/blog?id=qwen3.6-35b-a3b)。

### GLM-5.2-FP8

GLM-5.2-FP8 是 `glm_moe_dsa` 架构的 1M context 长上下文 MoE 模型。主干 baseline 固定为 78 层普通 one-token Decode：前 3 层为 dense FFN，后 75 层为 sparse MoE；隐藏维度 6,144，词表 154,880，输入 Embedding 与 LM Head 不共享。每个 sparse MoE 层有 256 个 Routed Expert，每 token 选择 8 个，另有 1 个 Shared Expert。官方 checkpoint 还包含 1 个 MTP/next-token-prediction 层；本配置不把 MTP draft/verify 混入标准 Decode baseline。

Attention 使用 MLA + DSA。每层保存 512 维 latent KV 与 64 维 RoPE key；在本 FP8 服务 profile 下，MLA KV 按 FP8 计算。DSA 路径每个 full-indexer 层扫描历史 token 的 128 维 BF16 index key，并选择 top-2048 进入主 attention。IndexShare 是该样本的核心：78 个主干层里只有 21 层实际执行 full indexer，另外 57 层复用最近一次 top-k 结果。配置按“21 层 full indexer + 57 层 fixed top-k”分组，保持 Decode 总量精确，但不保留显示顺序。

精度不是“全模型 FP8”。safetensors 显示主体矩阵为 FP8 E4M3，`lm_head`、Embedding、MoE router、indexer `weights_proj` 和 norm 为 BF16，FP8 `weight_scale_inv` 与 MoE `e_score_correction_bias` 为 F32。引擎每 Decode token 执行 40,297,758,720 个 matrix 参数，其中 Routed Expert 激活参数为 22,649,241,600；完整 checkpoint tensor payload 为 755,617,140,416 bytes。FP8 scale 与 correction bias 记录在 metadata 中，但当前 schema 不能表达其零 FLOP 元数据流量，因此不计入 `weight_read_bytes`。

在 1,048,576 context、batch=1 的默认点上，本配置的 per-output 结果为：parameter FLOPs 80,595,517,440，attention FLOPs 22,246,588,416，index FLOPs 181,819,932,672；persistent cache 为 47,110,422,528 bytes MLA KV + 5,637,144,576 bytes DSA index。这个样本主要用于观察超长上下文下 IndexShare 如何改变 index FLOPs、index read bytes 与 KV capacity 的趋势。

来源：[固定模型卡](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/README.md)、[固定配置](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/config.json)、[固定权重索引](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/model.safetensors.index.json)、[vLLM GLM-5.2 recipe](https://recipes.vllm.ai/zai-org/GLM-5.2)、[GLM-5.2 官方博客](https://huggingface.co/blog/zai-org/glm-52-blog)。
