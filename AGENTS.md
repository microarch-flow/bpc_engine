# 项目指南

本项目是 Python 3.10+ 的配置驱动 LLM decode/prefill 静态工作量计算器；不执行模型推理，也不预测延迟。功能、命令和边界见 [README](README.md)，Decode 趋势研究见 [定义规范](docs/decode_trend_metrics.md)和[推进待办](docs/decode_trend_research_todo.md)，Prefill 公式见 [指标规范](docs/prefill_metrics.md)，配置语法见 [机制目录](examples/mechanism_catalog.json)。

修改时：

- 以 [schema](decode_engine/schema.py)、[mechanisms](decode_engine/mechanisms.py)、[engine](decode_engine/engine.py) 和 [tests](tests/) 为行为依据。
- Decode 先算完整 step 再按 batch 归一化；cache capacity 不计入流量；Prefill logical-HBM 与 operand-stream 是替代视图，禁止相加。
- 核心保持标准库依赖。验证运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
```

`outputs/` 是已跟踪基准，生成实验请输出到 `/tmp`。`generate_precision_configs.py` 依赖的基础 JSON 当前缺失，修复生成链前不要运行。
