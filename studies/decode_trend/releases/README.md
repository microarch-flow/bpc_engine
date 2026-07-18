# Decode 趋势数据生成

版本目录是可复现生成物，不提交Git。Clone项目后，在根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  scripts/freeze_decode_trend_release.py \
  --version v1.0.0
```

该命令会：

1. 读取`studies/decode_trend/models.json`中的全部20个模型；
2. 校验P3机制审计与模型配置；
3. 执行公共C点、模型/机制锚点和主B点；
4. 生成CSV、JSONL、模型profile和验证报告；
5. 快照配置、事实、审计、引擎、运行器和字段文档；
6. 生成`release_manifest.json`和`SHA256SUMS`。

默认输出：

```text
studies/decode_trend/releases/v1.0.0/
```

分析入口是：

```text
studies/decode_trend/releases/v1.0.0/data/decode_results.csv
```

版本目录已存在时脚本会拒绝覆盖。模型事实、公式、网格或字段发生变化时，应使用新的
版本号。
