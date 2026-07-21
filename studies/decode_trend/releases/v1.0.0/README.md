# Decode Trend Dataset v1.0.0

这是2022–2026YTD代表模型的正式冻结数据版本。

- 模型数：20
- 数据行数：3357
- 阶段：Decode
- 数据网格：公共 C 点、模型/机制锚点和主 B 点
- `data/`：正式结果
- `source_snapshot/`：生成该结果的配置、事实、P3审计、引擎和运行器快照
- `source_snapshot/docs/`：字段字典、指标合同和P3人工审计快照
- `release_manifest.json`：版本、来源与数据摘要
- `SHA256SUMS`：除自身之外所有文件的 SHA-256

复算：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  source_snapshot/scripts/run_decode_trend.py \
  --manifest source_snapshot/studies/decode_trend/models.json \
  --output-dir /tmp/decode-trend-v1.0.0-reproduced
```

复算结果中的时间、run_id 和 Git 状态可能不同；数值字段应一致。
