# Decode 趋势数据生成

正式冻结版本`v1.0.0`已经提交Git。Clone项目后可以直接运行P8/P9A分析，无需先
生成数据。使用前可在根目录校验冻结内容：

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

该命令会：

1. 读取`studies/decode_trend/models.json`中的全部20个模型；
2. 校验P3机制审计与模型配置；
3. 执行公共C点、模型/机制锚点和主B点；
4. 生成CSV、JSONL、模型profile和验证报告；
5. 快照配置、事实、审计、引擎、运行器和字段文档；
6. 生成`release_manifest.json`和`SHA256SUMS`。

上述示例输出：

```text
studies/decode_trend/releases/v1.0.1-dev/
```

分析入口是：

```text
studies/decode_trend/releases/v1.0.0/data/decode_results.csv
```

版本目录已存在时脚本会拒绝覆盖。模型事实、公式、网格或字段发生变化时，应使用新的
版本号。除已正式发布并在`.gitignore`中显式放行的版本外，新生成的release默认保持
忽略；完成冻结校验和文档更新后再将其提升为受版本控制的正式数据版本。
