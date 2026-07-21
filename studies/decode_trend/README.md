# Decode 趋势研究数据结构

`models.json` 是 20 个研究样本的机器可读事实清单，`scripts/run_decode_trend.py`
负责校验事实、调用引擎并生成统一结果。

`mechanism_audit.json` 是 P3 机器审计：分别记录 FLOPs、logical-HBM
traffic、Cache capacity 和 Weight capacity 的支持状态。人工矩阵见
[`docs/decode_trend_p3_mechanism_audit.md`](../../docs/decode_trend_p3_mechanism_audit.md)。

## 容量口径

- `decode_profile_weight_capacity_bytes`：标准单 Token Decode 路径需要常驻的权重与量化元数据，是芯片容量趋势的主口径。
- `full_checkpoint_capacity_bytes`：完整 checkpoint 的 tensor payload，包含 MTP、Vision 等未进入标准 Decode 的模块；无法核验时必须为 `null`。
- `persistent_decode_profile_bytes`：Decode profile 权重容量加当前 Batch 的 KV/Index/State Cache。
- `persistent_full_checkpoint_bytes`：完整 checkpoint 容量加 Cache；完整容量未知时仍为 `null`。

文件尺寸、序列化 header、workspace、分页碎片、通信 buffer 和安全余量不属于上述容量。

## 运行

最小检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/run_decode_trend.py \
  --contexts 128 --batches 1 \
  --output-dir /tmp/bpc_engine_decode_trend_check
```

20 模型主网格：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/run_decode_trend.py \
  --output-dir /tmp/bpc_engine_decode_trend_full
```

输出包含 `run_manifest.json`、`model_profiles.jsonl`、
`decode_results.jsonl`、`decode_results.csv` 和 `validation_report.json`。
超出 release 上下文的点会显式标记 `is_extrapolated=true`。

## 使用和生成20模型正式数据

正式冻结版本`v1.0.0`已随仓库提交，新clone可直接读取：

```text
studies/decode_trend/releases/v1.0.0/
```

验证冻结文件：

```bash
cd studies/decode_trend/releases/v1.0.0
sha256sum -c SHA256SUMS
```

需要从当前源码生成新的候选版本时，在项目根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  scripts/freeze_decode_trend_release.py \
  --version v1.0.1-dev
```

命令读取`models.json`中的全部20个模型，生成：

```text
studies/decode_trend/releases/v1.0.1-dev/
```

版本目录包含完整CSV/JSONL、模型profile、验证报告、配置/源码/文档快照和
`SHA256SUMS`。未发布的`releases/*/`默认被Git忽略；正式版本需要在`.gitignore`中
显式放行并提交。版本目录已存在时脚本会拒绝覆盖；重新生成前请使用新的版本号。
