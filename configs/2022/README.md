# 2022 年代表模型配置

本目录保存 2022 年发布的精确模型版本及其当年代表性实际部署精度配置。

- 配置必须遵循计算引擎的 `schema_version: 1`。
- 一个文件对应一个精确模型版本和一个实际部署精度。
- `metadata` 中记录发布日期、架构与精度来源、推导过程及不确定项。
- `analysis.contexts` 和 `analysis.batches` 仅为默认值，统一研究扫描可由 CLI 覆盖。
- 不在本目录保存假设性的统一 4/8/16-bit 对照配置。

## 已确认代表模型

| 角色 | 精确模型版本 | 实际部署精度 | 配置 |
|---|---|---|---|
| 旗舰资源需求 | PaLM 540B | INT8 权重、BF16 KV | [palm_540b_int8w_bf16kv.json](palm_540b_int8w_bf16kv.json) |
| 主流部署 | BLOOM 176B | BF16 权重、BF16 KV | [bloom_176b_bf16.json](bloom_176b_bf16.json) |
| 架构演进对照 | GLM-130B INT4 | INT4 线性层、FP16 Embedding/Head/KV | [glm_130b_int4w_fp16kv.json](glm_130b_int4w_fp16kv.json) |

## 选择理由

三个名额不是“年度榜单前三”，而是分别回答资源上界、实际部署基线和架构变化三个不同问题。

| 角色与模型 | 为什么选择 | 解释边界 |
|---|---|---|
| 旗舰资源需求：PaLM 540B | 540B 稠密模型适合锚定 2022 年公开可审计的权重容量、矩阵计算和权重流量高位需求；官方低延迟推理论文还给出了实际 INT8 权重部署路径。 | 它是资源包络样本，不是开放 checkpoint 或采用率样本；2K 上下文、TPU Padding、片间通信和实现流量不能外推成行业统一结论。 |
| 主流部署：BLOOM 176B | BF16 checkpoint、模型卡、论文和推理资料均公开，BigScience/Hugging Face 生态传播广，适合作为当年公开可部署的传统稠密 MHA/BF16 基线。 | “主流”依据可获得性和传播度，不代表掌握了 BLOOM 的部署量或 Token 份额；176B 也不是典型部署尺寸的统计中位数。 |
| 架构演进对照：GLM-130B INT4 | 同时提供 GLM 双向 Prefix/自回归生成、GeGLU、DeepNorm 等架构差异，以及官方大模型 INT4 部署 profile，可观察架构变化和低比特权重对资源指标的影响。 | INT4 表示持久权重存储，实际矩阵乘前还原为 FP16；量化元数据与反量化物理流量未计，当前引擎只适合其 Decode 口径。 |

三者组合覆盖了超大稠密 MQA/INT8、公开稠密 MHA/BF16 和 GLM/INT4，避免用单个模型同时代表资源上界、行业采用和技术演进。

## 架构说明

### PaLM 540B

PaLM 540B 是稠密 decoder-only Transformer，共 118 个并行 Attention/FFN block。隐藏维度为 18,432，FFN 维度为 73,728，使用 SwiGLU、RoPE 和全上下文 MQA。每层有 48 个 Query Head、1 个共享 KV Head，Head Dim 为 256；输入与输出 Embedding 共享，词表按 256,000 计算，公开训练和验证上下文长度为 2,048。

本配置采用 Google 2022 年实际运行的低延迟推理方案：INT8 权重、未量化的 16-bit BF16 KV。推理论文为 TPU 分片将 48 个 Query Head 物理填充到 64 个的实现开销不属于逻辑模型架构，因此不计入。

来源：[PaLM 论文](https://arxiv.org/abs/2204.02311)、[PaLM 推理论文](https://arxiv.org/abs/2211.05102)。

### BLOOM 176B

BLOOM 176B 是稠密 GPT 风格 decoder-only Transformer，共 70 层。隐藏维度为 14,336，FFN 维度为 57,344，使用 GeLU、全上下文 MHA 和 ALiBi。每层有 112 个 Query/KV Head，Head Dim 为 128；输入与输出 Embedding 共享。Tokenizer 有 250,680 个 token，实际权重矩阵填充为 250,880 行，训练上下文长度为 2,048。

本配置采用原生 BF16 checkpoint 和 2022 年 BF16 推理方案，权重与 KV 均按 16 bit 统计。官方总参数为 176,247,271,424；引擎按矩阵运算口径统计其中 176,234,168,320 个参数，其余 Bias 和 LayerNorm 参数保留在元数据中。

来源：[发布说明](https://huggingface.co/blog/bloom)、[模型卡](https://huggingface.co/bigscience/bloom)、[论文](https://arxiv.org/abs/2211.05100)。

### GLM-130B INT4

GLM-130B 是稠密双向 GLM Transformer，共 70 层。隐藏维度为 12,288，GeGLU FFN 维度为 32,768，使用 RoPE、DeepNorm Post-LN 和全上下文 MHA。每层有 96 个 Query/KV Head，Head Dim 为 128；输入与输出 Embedding 共享，实际矩阵填充为 150,528 行，最大上下文长度为 2,048。其 `[gMASK]` 生成采用双向 Prefix 加自回归生成段；对 Decode 而言仍需保存并扫描完整 Prefix KV。

本配置对应 2022 年 8 月 24 日发布的官方 INT4 版本。INT4 只覆盖 Attention 和 FFN 线性层；Embedding/LM Head、LayerNorm 与 Bias 保持 FP16，激活和 KV 也按 FP16 统计。官方实现在矩阵乘前将打包的 INT4 权重动态还原为 FP16，因此本配置中的 4 bit 表示持久权重存储和逻辑 HBM 读取，不表示 INT4 计算。量化 Scale 和实现相关的反量化物理流量暂不计入。该配置仅用于 Decode；当前引擎的因果 Prefill 不能表达 GLM 的双向 Prefix 和空白填充 Mask。

来源：[官方仓库与发布记录](https://github.com/zai-org/GLM-130B)、[INT4 配置](https://github.com/zai-org/GLM-130B/blob/main/configs/model_glm_130b_int4.sh)、[量化说明](https://github.com/zai-org/GLM-130B/blob/main/docs/quantization.md)、[论文](https://arxiv.org/abs/2210.02414)。
