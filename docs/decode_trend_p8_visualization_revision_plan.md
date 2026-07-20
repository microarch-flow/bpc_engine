# P8 年度包络可视化修订计划

版本：v0.2

输入：`studies/decode_trend/releases/v1.0.0/`

输出：`/tmp/decode_trend_p8_envelope/`

## 1. 修订原因

现有`annual_envelope_*`图虽然数值可追溯，但单图没有充分回答：

- 每年具体包含哪些模型；
- “包络”是最大值、平均值还是加权值；
- 每个点由哪个模型贡献；
- 为什么曲线会在相邻上下文点之间突然下降；
- 哪些下降来自模型集合变化，而不是同一模型的工作量变化。

尤其当`C`超过某个模型的advertised max后，该模型会退出eligible集合。若仍把退出
前后的年度最大值直接连线，视觉上会形成不存在的连续下降趋势。

## 2. 修订目标

修订后的呈现必须把三类问题分开：

1. **动态原生集合条件上包络**：在每个`year × C × B`中，对仍处于原生范围的
   模型取最大值，用于回答样本内条件资源上界。
2. **固定公共工作负载年度比较**：固定所有20个模型均支持的`C=2048`，展示每个
   模型原始点和年度最大值，用于避免eligible集合变化。
3. **各模型自身advertised ceiling压力**：每个模型在自己的发布宣称上限取值，
   用于展示条件压力情景；明确不同模型的`C`不同，不能当作同工作负载横比。

三类图均不做部署量加权、年度平均、回归、平滑或插值。

## 3. 精确定义

对年份`y`、上下文`C`和Batch`B`，定义：

```text
M(y, C) = {
    model |
    model.year == y
    and C <= model.advertised_max_context
    and row.batch_class == main
    and row.is_extrapolated == false
    and C is a common power-of-two grid point
}
```

动态条件上包络：

```text
E_metric(y, C, B) =
    max(metric(model, C, B) for model in M(y, C))
```

这里的“年度”只表示按release year分组，“包络”表示逐点最大值；没有选择固定年度
模型，也没有对模型做平均或加权。不同`C/B/metric`的最大值贡献模型可以不同。

## 4. 实施内容

### 4.1 重做现有动态包络图

保留文件名：

```text
annual_envelope_flops_B{1,32,256}
annual_envelope_logical_hbm_B{1,32,256}
```

每张图改为上下两个面板：

- 上面板只画`all_representative_models`的inclusive最大值；
- 当`eligible_model_release_ids`变化时断开曲线，禁止跨不同候选集合连线；
- 在初始点、winner变化点和eligible集合变化点标注
  `winner short name + n=<eligible count>`；
- 实心点表示最大值贡献者P3 supported，空心点表示partially supported；
- 图内明确写出“maximum across eligible models; no averaging or weighting”。
- 下面板画`eligible_model_count(C)`，让模型退出范围的时间点直接可见。

`designated_frontier`数据继续保存在`annual_envelope.csv`，但不再与动态全样本上包络
混画。它是预先指定的样本角色，不是从数据中计算出的Pareto frontier。

### 4.2 新增固定`C=2048`明细表

新增：

```text
fixed_context_comparison.csv
```

每个模型、每个`B=1/32/256`一行，保存：

- 年份、模型、角色和deployment profile；
- 固定`context_C=2048`与`concurrency_B`；
- FLOPs/token、logical-HBM bytes/token、Cache/request和persistent容量；
- 四维P3状态。

脚本必须验证20个模型均存在该上下文和三档Batch，因此正式版本应输出60行。

### 4.3 新增固定公共工作负载图

新增：

```text
fixed_context_C2048_flops_by_year
fixed_context_C2048_logical_hbm_B{1,32,256}_by_year
```

绘图规则：

- 每个模型显示为独立点，不隐藏年度内部差异；
- 使用2×3年度分面；模型名位于y轴，x轴为对应指标；
- 年度最大值使用黑边菱形突出；
- 指标x轴使用对数刻度；
- 图注明确`C=2048`对全部20个样本均为advertised-native；
- 不画连接线，避免暗示连续年度拟合。

FLOPs/token在当前公式下不随`B`变化，因此只生成一张固定上下文FLOPs图；
logical-HBM分别生成`B=1/32/256`三张。

### 4.4 新增各模型advertised ceiling图

复用`native_max_points.csv`，新增：

```text
advertised_ceiling_flops_by_model
advertised_ceiling_logical_hbm_B{1,32,256}_by_model
```

绘图规则：

- 使用2×3年度分面；每个模型一个独立点；
- x轴为该模型自己的advertised `C`，y轴为对应指标，模型名直接标在点旁；
- supported使用实心点，partially supported使用空心点；
- 不连接不同模型；
- 标题和图注明确“每个模型的C不同”，只表示条件压力，不表示同工作负载趋势。

## 5. 文档修订

更新`docs/decode_trend_p8_envelope_report.md`：

- 列出20个模型并标记每年designated frontier；
- 用公式和小例子解释“包络”；
- 逐点解释2023、2025和2026曲线下降—上升的贡献模型变化；
- 明确动态包络、固定公共上下文和own-ceiling三类图分别回答什么问题；
- 把固定`C=2048`图作为跨年度比较主图；
- 明确动态包络不能单独用于算法效率或行业平均趋势。

同步更新实施计划、README/交接中受影响的产物数量和说明。

## 6. 测试

扩展`tests/test_decode_trend_envelope.py`：

1. `fixed_context_comparison.csv`行数、唯一键和20模型覆盖；
2. 固定上下文必须为2048，Batch必须为1/32/256；
3. 动态包络分段函数在eligible集合变化时产生新segment；
4. winner变化但eligible集合不变时不错误断段；
5. 新增图表存在、PNG/SVG有效且SVG无行尾空白；
6. 正式输出应为29组PNG与29组SVG。

完成后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest \
  tests.test_decode_trend_envelope -v

PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  scripts/analyze_decode_trend_envelope.py \
  --release-dir studies/decode_trend/releases/v1.0.0 \
  --output-dir /tmp/decode_trend_p8_envelope
```

## 7. 验收标准

1. 打开任一`annual_envelope_*`图即可看到指标是逐点最大值、没有加权。
2. 图中可以识别winner、eligible模型数及候选集合变化位置。
3. 不再跨eligible集合变化点连接曲线。
4. 固定`C=2048`图显示全部20个模型，而非只显示年度最大值。
5. own-ceiling图明确每个模型使用不同`C`。
6. 所有图中数值均可追溯到CSV。
7. 冻结release不被修改，完整测试通过。
