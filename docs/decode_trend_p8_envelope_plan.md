# P8 行业需求包络分析实施计划

版本：v0.2

输入数据：`studies/decode_trend/releases/v1.0.0/`

实验输出：`/tmp/decode_trend_p8_envelope/`

本计划的可视化解释性修订见
[P8年度包络可视化修订计划](decode_trend_p8_visualization_revision_plan.md)。

## 1. 目标

本轮只实施P8的第一部分“行业需求包络”，回答2022–2026YTD代表模型在
release-specific证据化部署精度profile与统一`C/B`情景扫描下，Decode阶段以下
资源需求如何变化：

- Decode profile权重容量与每Token激活参数；
- `FLOPs/token = F(C,B)`及Parameter、Attention、Index、State分项；
- `logical-HBM Bytes/token = G(C,B)`及Weight、KV、Index、State分项；
- 每请求Cache与`persistent_decode_profile_bytes = H(C,B)`；
- `logical_hbm_bytes_per_flop`和`TB/s per PFLOPS`；
- Parameter/Weight主导转向Sequence/Cache主导的离散扫描交叉点。

本轮不进行算法能力归一化、部署采用率估计、未来预测或芯片指标换算。

## 2. 数据与边界

规范计算结果读取：

```text
studies/decode_trend/releases/v1.0.0/data/decode_results.csv
```

模型角色、机制和P3审计读取：

```text
studies/decode_trend/releases/v1.0.0/data/model_profiles.jsonl
```

分析遵循以下边界：

1. 只使用`batch_class == "main"`。
2. 历史实际包络只使用`is_extrapolated == false`。
3. 不删除`partially_supported`数据；保留四维P3状态，并在表格、图例和报告中显式标记。
4. `logical-HBM`只解释为统一算法边界，不解释为profiler实测流量。
5. Cache capacity不与Traffic相加。
6. 主容量口径使用`decode_profile_weight_capacity_bytes`和
   `persistent_decode_profile_bytes`；完整checkpoint容量只作补充。
7. `C`和`B`均保留为扫描变量，不固定成单一“标准场景”。
8. 年度样本量较小，年度统计使用模型明细、最大值和范围，不报告P90或统计显著性。

## 3. 实现

新增一个标准库负责解析、校验和CSV/JSON导出，并在可用时使用matplotlib绘图的
可复用脚本：

```text
scripts/analyze_decode_trend_envelope.py
```

脚本必须：

- 正确解析CSV中的boolean、nullable number和JSON字符串字段；
- 以`model_release_id + deployment_profile_id + context_C + concurrency_B`
  检查结果唯一性；
- 校验`step_* = per_token_* × B`以及容量恒等式；
- 将模型profile与结果记录关联；
- 校验CSV中的run/study、年份、静态参数、上下文范围和P3状态与manifest/profile一致；
- 拒绝未知schema版本、缺失模型profile或重复结果；
- 默认读取`v1.0.0`，默认写入`/tmp/decode_trend_p8_envelope/`；
- 不修改冻结版本目录和仓库内已跟踪基准。

## 4. 输出产物

### 4.1 数据质量和模型概览

```text
analysis_manifest.json
quality_summary.json
model_summary.csv
```

`model_summary.csv`每个模型一行，保存年份、角色、机制、上下文上限、权重容量、
激活参数、激活比例和四维P3状态。

`analysis_manifest.json`同时保存冻结输入哈希和本分析脚本哈希，避免脚本更新后
误把旧产物当作同一版本结果。

### 4.2 年度包络

```text
annual_envelope.csv
fixed_context_comparison.csv
native_max_points.csv
context_boundary_points.csv
```

- `annual_envelope.csv`：对每个`year × C × B`，在当年仍处于原生上下文范围的
  样本中计算各核心指标最大值，并保存贡献最大值的模型。年度横向包络只使用
  带`common_power_of_two`标签的公共上下文点；模型专属机制锚点只用于模型内分析。
  同时输出`all_representative_models`全样本包络和P1预先指定
  `frontier_resource_envelope`模型的`designated_frontier`轨迹。各指标分别保存
  inclusive最大值、贡献模型、贡献模型角色和P3状态，并补充supported-only最大值。
- `fixed_context_comparison.csv`：全部模型在公共`C=2048`和
  `B=1/32/256`下的模型级明细，用于不改变eligible集合的年度横向比较。
- `native_max_points.csv`：每个模型在其advertised max context、主Batch点上的
  完整指标与分项。
- `context_boundary_points.csv`：每个模型在advertised、trained、evaluated和
  deployed等已落盘边界上的`B=1/32/256`指标。规范边界标签以profile中的上下文
  标量事实为准，不依赖可选anchor标签是否重复写入，用于区分“发布宣称范围”和
  “有部署/训练/评测证据的范围”。

年度包络表示“本研究样本集合中的可审计上界”，不外推为全行业统计分位数。
配置中的精度是release-specific证据化部署profile，但统一扫描的`C/B`不是观测到的
真实服务负载；报告中必须使用“advertised-native范围”，不能称为实际部署上下文。

### 4.3 分项和交叉点

```text
component_shares.csv
crossover_points.csv
```

- `component_shares.csv`：模型在原生最大上下文、`B=1/32/256`时的FLOPs、
  logical-HBM，以及Weight、KV Cache、Index Cache、State Cache容量分项占比。
- `crossover_points.csv`：逐模型、逐Batch记录以下第一个原生扫描点：
  - Non-parameter FLOPs不小于Parameter FLOPs；
  - Non-weight logical-HBM不小于Weight read；
  - 每请求Cache不小于Decode profile权重容量，作为请求级补充指标；
  - Batch Cache不小于Decode profile权重容量，作为`H(C,B)`的主交叉指标；
  若原生范围内没有交叉则显式留空并标记`not_reached`。

交叉点同时保存首次达到区间、首次达到后是否发生反转、反转次数和此后持续主导的
稳定交叉区间。这样分块或窗口机制在边界处产生的锯齿不会被误写成永久转向。

这里的Non-parameter FLOPs为Attention、Index、State和Extra之和；Non-weight
logical-HBM为KV、Index、State和Other read/write之和，明确排除Activation。

### 4.4 图表

同时输出PNG和SVG：

```text
weight_capacity_by_release
active_parameter_ratio_by_release
flops_per_token_by_context_B{1,32,256}
logical_hbm_bytes_per_token_by_context_B{1,32,256}
cache_capacity_per_request_by_context
persistent_memory_by_context_B{1,32,256}
tbps_per_pflops_by_context_B{1,32,256}
annual_envelope_flops_B{1,32,256}
annual_envelope_logical_hbm_B{1,32,256}
fixed_context_C2048_flops_by_year
fixed_context_C2048_logical_hbm_B{1,32,256}_by_year
advertised_ceiling_flops_by_model
advertised_ceiling_logical_hbm_B{1,32,256}_by_model
```

上下文、FLOPs、Bytes和Capacity使用适合跨数量级比较的对数轴。原生曲线按模型
区分，包含部分支持指标的曲线使用虚线或显式标记；图内说明实线/虚线、
实心/空心点和年度颜色的含义，共输出29组PNG和29组SVG。

动态年度包络只画`all_representative_models`，使用逐点最大值，不做平均或加权。
当`eligible_model_release_ids`变化时断开曲线，并用下方面板显示eligible模型数。
`designated_frontier`仍保存在CSV中，但不与全样本包络混画。固定`C=2048`图展示
每个模型的原始点；advertised-ceiling图则明确每个模型使用不同`C`。

## 5. 结果解释文档

新增：

```text
docs/decode_trend_p8_envelope_report.md
```

报告必须包含：

- 输入版本、复现命令和输出目录；
- 数据质量与P3支持状态；
- 年度权重、计算、流量、Cache和持久容量包络；
- Batch摊薄、上下文增长和架构机制对分项的影响；
- 交叉点及其芯片资源含义；
- `partially_supported`、logical-HBM和小样本边界；
- 所有关键结论对应的CSV字段或结果表，避免只凭图形描述。

## 6. 验证与验收

实施完成后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/analyze_decode_trend_envelope.py \
  --release-dir studies/decode_trend/releases/v1.0.0 \
  --output-dir /tmp/decode_trend_p8_envelope
```

验收条件：

1. 冻结输入未被修改，Git只显示本轮脚本和文档等预期变化。
2. 20个模型均进入模型概览；只在原生范围内生成包络和交叉点。
3. 所有派生比例分母非零，分项和在浮点容差内闭合。
4. CSV行数、唯一键、筛选数量和警告写入`quality_summary.json`。
5. 图表数据能够追溯到CSV；报告中的数值与生成结果一致。
6. 完整单元测试通过。
