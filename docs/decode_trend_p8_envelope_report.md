# P8 Decode 行业需求包络分析报告

版本：v0.2

研究窗口：2022–2026YTD

冻结输入：`studies/decode_trend/releases/v1.0.0/`

分析输出：`/tmp/decode_trend_p8_envelope/`

## 1. 结论摘要

本报告完成P8的第一部分“行业需求包络”。结论建立在20个代表模型、
3357条冻结Decode记录和1917条advertised-native主网格记录上。

主要结论如下：

1. 样本中的资源上界不是一条单调年度曲线。Decode profile权重年度最大值依次为
   503.244、332.584、626.920、553.397和694.380 GiB；样本组成、量化精度、
   MoE激活比例和原生上下文共同改变年度上界。
2. Resident weight与每Token计算正在解耦。年度最低active parameter ratio从
   2022年的99.990%下降到2025和2026YTD的3.087%，但这不表示所有模型都按同一路径
   演进，也不表示总权重容量同步下降。
3. Batch同时产生两个相反效应：共享权重流量按`B`摊薄，但请求私有
   KV/Index/State容量按`B`复制。`B=32`时有11/20个模型在原生扫描范围内达到
   `batch cache >= decode weight`；`B=256`时为18/20。
4. 长上下文会使Attention、Index、State等非参数计算和非Weight流量成为主导项。
   12/20个模型在原生范围内达到“Non-parameter FLOPs不小于Parameter FLOPs”；
   非Weight logical-HBM达到Weight read的模型数从`B=1`的4/20增加到
   `B=256`的18/20。
5. Llama 4 Scout的10,485,760 Token full-retention点是强离群压力情景，不是典型
   部署负载。在`B=32`下，其evaluated、trained、deployed和advertised四个边界的
   persistent Decode容量分别为0.946、1.696、20.795和60.196 TiB。
6. `annual_envelope_*`是随`C`改变eligible集合后的逐点最大值，不是年度平均值或
   加权模型。旧图中的V形主要来自模型退出原生范围；修订图在候选集合变化处断线，
   跨年度比较应优先使用固定`C=2048`的全模型点图。

这些结果证明的是公式、分项和聚合逻辑闭合，不是与真实profiler结果相比
“误差小于10%”的保证。引擎仍是静态workload计算器，不执行模型推理，也不预测
latency、TPOT、利用率、通信或物理多卡显存。

## 2. 数据口径

### 2.1 纳入范围

分析使用以下筛选：

```text
phase == decode
batch_class == main
is_extrapolated == false
within_advertised_context == true
```

`C`是Decode step开始前已有上下文，`B`是本step同时推进的等长请求数。每个step
先计算完整工作量，再除以`B`：

```text
per_token_work = step_work / B
```

因此当前公式下FLOPs/token不随`B`变化；Batch效应主要体现在Weight read摊薄和
Batch Cache复制。

配置中的位宽是release-specific证据化部署精度profile。统一扫描的`C/B`是研究
情景，不是观测到的历史服务负载；本报告使用“advertised-native范围”，不使用
“真实部署上下文”这一表述。

### 2.2 年度模型名单与两种scope

`annual_envelope.csv`同时保存：

- `all_representative_models`：当年全部研究样本；
- `designated_frontier`：P1阶段预先标记
  `frontier_resource_envelope`角色的子样本。

`designated_frontier`不是从结果中计算出的Pareto frontier。年度横向包络只使用
`common_power_of_two`上下文点，模型专属机制锚点不参与年度横比。

全部20个模型如下。`★`表示P1阶段预先指定的
`frontier_resource_envelope`角色，不表示本分析重新计算出的Pareto前沿。

| 年份 | 当年全部`all_representative_models`样本 |
|---|---|
| 2022 | PaLM 540B★、BLOOM 176B、GLM-130B |
| 2023 | Llama 2 70B、Falcon 180B★、Mistral 7B、Yi-34B、Mixtral 8x7B |
| 2024 | DeepSeek-V2、Llama 3.1 405B★、Llama 3.1 70B、Jamba 1.5、DeepSeek-V3 |
| 2025 | Llama 4 Scout、Qwen3-32B、Kimi-Linear、Kimi K2 Thinking★ |
| 2026YTD | Kimi K2.6★、Qwen3.6-35B、GLM-5.2 |

随着`C`增大，原生范围较短的模型会退出eligible cohort。因此年度图中的下跳或V形
可能来自`eligible_model_count`和最大值贡献模型变化，不表示同一模型的需求突然
下降。修订图只连接候选模型ID集合完全相同的相邻点；集合变化处留出断点。
完整候选名单分别保存在`eligible_model_release_ids`和
`eligible_model_short_names`。

### 2.3 “包络”的精确定义

对年份`y`和公共上下文点`C`，先定义仍处于发布宣称原生范围的模型集合：

```text
R_y(C) = {
    m |
    release_year(m) = y
    and C <= advertised_max_context(m)
    and row.batch_class = main
    and row.is_extrapolated = false
    and C is a common_power_of_two point
}
```

对指标`q`和Batch`B`，年度条件上包络及贡献模型集合为：

```text
U_q(y, C, B) = max(q(m, C, B) for m in R_y(C))
A_q(y, C, B) = argmax(q(m, C, B) for m in R_y(C))
```

也就是说，一个点取当年eligible模型的最大值；不同`C/B/q`可以由不同模型贡献。
这里没有年度模型加权、平均、回归、平滑或插值，也没有先选一个固定模型代表全年。
“包络”只是样本内逐点上边界，不是行业均值。

### 2.4 三种呈现分别回答什么

1. `annual_envelope_*`：动态原生集合条件上包络，回答“在该`C`仍然eligible的
   当年样本中，最大逻辑工作量是多少”。它适合看条件上界，不适合直接拟合趋势。
2. `fixed_context_C2048_*_by_year`：固定所有20个样本均支持的公共工作负载，
   每个点是一台模型，黑边菱形是年度最大值。这是跨年度横比的主图。
3. `advertised_ceiling_*_by_model`：每个模型使用自己的advertised max `C`，
   回答“各模型在自身宣称边界的条件压力”。各点`C`不同，不能当作同工作负载比较。

### 2.5 三种资源视图

- `FLOPs/token`：Parameter、Attention、Index、State、Extra之和；
- `logical-HBM bytes/token`：Weight、KV、Index、State和Other流量之和，
  明确排除Activation；
- `persistent_decode_profile_bytes`：Decode profile weight加
  `B × cache_bytes_per_request`。

Cache capacity不进入Traffic求和。`TB/s per PFLOPS`等于
`1000 × logical_hbm_bytes_per_flop`，只描述算法逻辑带宽/算力平衡点，不是实测
硬件效率。

## 3. 数据质量

| 项目 | 结果 |
|---|---:|
| 冻结CSV记录 | 3357 |
| 模型profile | 20 |
| 显式外推记录 | 1440 |
| advertised-native主网格记录 | 1917 |
| 原生模型-上下文组合 | 213 |
| 每个原生组合的Batch覆盖 | 9档：1–256的2次幂 |
| 年度包络记录 | 936 |
| 固定`C=2048`模型对照记录 | 60 |
| advertised max记录 | 180 |
| canonical context boundary记录 | 72 |
| 分项占比记录 | 60 |
| 交叉点记录 | 180 |

分析器逐行复核：

- 四字段结果键唯一，profile join和config SHA-256一致；
- CSV中的run/study、年份、静态参数、权重容量、上下文scope和P3状态与
  manifest/profile一致；
- `step_* = per_token_* × B`；
- FLOPs、engine traffic、logical-HBM和Cache分项闭合；
- `persistent = decode weight + batch cache`；
- 每请求Cache不随`B`变化；
- Byte/FLOP与TB/s per PFLOPS恒等式。

冻结验证的两条IEEE-754精确整数范围警告只出现在BLOOM和GLM-130B的
`C=16M、B=256`外推点，不进入本报告的native统计。原生范围中的warning全部是
P3支持状态标记，不是数值越界。

P3模型级支持情况：

| 维度 | supported | partially supported |
|---|---:|---:|
| FLOPs | 16 | 4 |
| logical-HBM Traffic | 5 | 15 |
| Cache Capacity | 18 | 2 |
| Weight Capacity | 18 | 2 |

最终完整测试共65项，全部通过；冻结`SHA256SUMS`中的全部文件也已逐项校验通过。

本报告的inclusive包络保留`partially_supported`，同时在年度表中提供
`supported_only_max_*`。模型曲线虚线表示对应维度partial；年度图空心点表示最大值
贡献者partial，实心点表示supported。Weight/active ratio柱图使用斜线标出partial。

## 4. 模型级年度概览

下表直接来自`model_summary.csv`，每个deployment profile只计一次。

| 年份 | 样本数 | Decode weight范围（GiB） | 最低active ratio | 最大advertised `C` |
|---|---:|---:|---:|---:|
| 2022 | 3 | 62.531–503.244 | 99.990%（GLM-130B） | 2,048（三样本并列） |
| 2023 | 5 | 13.489–332.584 | 27.297%（Mixtral 8x7B） | 200,000（Yi-34B） |
| 2024 | 5 | 67.667–626.920 | 5.458%（DeepSeek-V3） | 262,144（Jamba 1.5） |
| 2025 | 4 | 17.998–553.397 | 3.087%（Kimi K2 Thinking） | 10,485,760（Llama 4） |
| 2026YTD | 3 | 33.256–694.380 | 3.087%（Kimi K2.6） | 1,048,576（GLM-5.2） |

`active_parameter_ratio`是
`active_matrix_parameter_elements_per_token / decode_resident_parameter_elements`
的元素比例，不是Activation内存比例，也不是权重字节比例。Qwen3-32B等Dense模型
仍保持高比例，因此不能把年度最低值解释为全行业单调下降。

## 5. 为什么动态年度包络会先下降再上升

以`annual_envelope_flops_B1`为例。当前公式下FLOPs/token不随`B`变化，因此
`B=1/32/256`的FLOPs包络数值相同。下降有两种完全不同的来源：

- winner改变但候选集合不变：仍是同一批模型之间的最大值切换，可以连接；
- eligible集合改变：某些模型超过自己的advertised max后退出，此时最大值的定义域
  已改变，修订图断线，不能把两点解释为同一连续趋势。

### 5.1 2023：Falcon、Llama退出后再由Yi随上下文增长

| `C` | eligible数 | FLOPs上包络（TF/token） | winner | 解释 |
|---:|---:|---:|---|---|
| 2,048 | 5 | 0.366835 | Falcon 180B | 当年五个样本全部eligible |
| 4,096 | 4 | 0.148164 | Llama 2 70B | Falcon超过2K上限后退出 |
| 8,192 | 3 | 0.081951 | Yi-34B | Llama 2超过4K上限后退出 |
| 16,384 | 2 | 0.096044 | Yi-34B | 候选集合再次缩小 |
| 32,768 | 2 | 0.124230 | Yi-34B | 同一候选集合内随`C`上升 |
| 65,536 | 1 | 0.180601 | Yi-34B | 只剩Yi |
| 131,072 | 1 | 0.293344 | Yi-34B | 同一模型随`C`继续上升 |

从2K到8K看似下降77.66%，核心原因是高计算量的Falcon和Llama 2先后退出候选集合；
8K以后Yi的Attention工作量随`C`增加，曲线重新上升。这不是“一台年度模型先变快
再变慢”。

### 5.2 2025：Kimi K2退出造成88.57%的断层

| `C` | eligible数 | FLOPs上包络（TF/token） | winner |
|---:|---:|---:|---|
| 131,072 | 4 | 1.176842 | Kimi K2 Thinking |
| 262,144 | 3 | 2.290313 | Kimi K2 Thinking |
| 524,288 | 2 | 0.261838 | Kimi-Linear |
| 1,048,576 | 2 | 0.517388 | Kimi-Linear |
| 2,097,152 | 1 | 0.547672 | Llama 4 Scout |
| 4,194,304 | 1 | 1.063068 | Llama 4 Scout |
| 8,388,608 | 1 | 2.093860 | Llama 4 Scout |

Kimi K2 Thinking的advertised max为262K，超过该点后退出；524K处改由
Kimi-Linear贡献，数值下降88.57%。随后Kimi-Linear和Llama 4各自在自己的原生范围
内随上下文增长。因此262K与524K之间必须断线。

### 5.3 2026YTD：先是winner切换，后是候选集合退出

| `C` | eligible数 | FLOPs上包络（TF/token） | winner | 解释 |
|---:|---:|---:|---|---|
| 4,096 | 3 | 0.103552 | GLM-5.2 | 三模型均eligible |
| 8,192 | 3 | 0.132964 | Kimi K2.6 | 集合不变，仅winner切换 |
| 131,072 | 3 | 1.176842 | Kimi K2.6 | 集合不变 |
| 262,144 | 3 | 2.290313 | Kimi K2.6 | 集合不变 |
| 524,288 | 1 | 0.193752 | GLM-5.2 | Kimi K2.6和Qwen3.6退出 |
| 1,048,576 | 1 | 0.284662 | GLM-5.2 | 只剩GLM-5.2 |

4K到8K只是同一三模型集合中的winner切换，所以可以连接；262K到524K下降91.54%
则来自候选集合由3个缩到1个，修订图明确断线。

这些例子说明：动态包络有意义，但意义是“条件样本上界”。如果问题是“相同工作负载
下不同年份的模型如何比较”，应使用下一节固定`C=2048`的模型明细图。

## 6. 公共工作负载锚点

为了避免把不同模型各自的最大上下文直接当成同一工作负载，下面固定
`C=2048`、使用`all_representative_models`年度inclusive max。该点五个年份的全部
样本均eligible。`fixed_context_comparison.csv`保存20个模型与三档Batch共60行
明细；`fixed_context_C2048_*_by_year`按年份分面展示每一个模型，黑边菱形才是
年度最大值，图中没有平均或加权。

### 6.1 `B=32`结果

| 年份 | FLOPs/token最大值（TF） | logical-HBM最大值（GiB/token） | Persistent最大值（GiB） |
|---|---:|---:|---:|
| 2022 | 1.092587（PaLM） | 17.918（BLOOM） | 573.286（BLOOM） |
| 2023 | 0.366835（Falcon） | 10.706（Falcon） | 342.584（Falcon） |
| 2024 | 0.824407（Llama 3.1 405B） | 15.041（Llama 3.1 405B，partial） | 631.209（DeepSeek-V3） |
| 2025 | 0.080770（Kimi K2 Thinking） | 7.986（Kimi K2 Thinking，partial） | 557.686（Kimi K2 Thinking） |
| 2026YTD | 0.103197（GLM-5.2，partial） | 14.098（GLM-5.2，partial） | 697.450（GLM-5.2） |

年度最大值可以由不同模型贡献，三列不能拼成一台同时具有所有最大值的“虚拟模型”。
精确贡献者、样本角色和支持状态见
`max_<metric>_model_release_ids`、`max_<metric>_sample_roles`与
`max_<metric>_support`。

### 6.2 Batch效应

同一`C=2048`下的年度logical-HBM和persistent最大值如下。每个单元按
`B=1 / 32 / 256`排列：

| 年份 | logical-HBM（GiB/token） | Persistent（GiB） |
|---|---:|---:|
| 2022 | 503.475 / 17.918 / 8.942 | 503.475 / 573.286 / 2288.286 |
| 2023 | 332.892 / 10.706 / 1.612 | 332.897 / 342.584 / 412.584 |
| 2024 | 450.774 / 15.041 / 2.742 | 627.054 / 631.209 / 705.768 |
| 2025 | 30.434 / 7.986 / 2.048 | 553.531 / 557.686 / 587.710 |
| 2026YTD | 38.626 / 14.098 / 2.800 | 694.476 / 697.450 / 718.942 |

每Token Weight read随`B`摊薄，所以logical-HBM通常下降；每请求Cache按`B`复制，
所以persistent容量上升。年度最大贡献模型可能随`B`变化，例如2022年的HBM最大值
从`B=1`的PaLM切换为`B=32/256`的BLOOM。

## 7. 各样本自身advertised ceiling

下面先在每个模型自身advertised max context取值，再按年取最大。它是条件压力
上包络，不是公共`C`横向对比。`advertised_ceiling_*_by_model`以每个模型自己的
`C`为x轴、工作量为y轴，不连接模型。除FLOPs列外，Traffic和容量列使用`B=32`。

| 年份 | FLOPs最大值（TF/token） | HBM最大值（GiB/token） | Cache/request最大值 | Persistent最大值 |
|---|---:|---:|---:|---:|
| 2022 | 1.093（PaLM@2,048） | 17.918（BLOOM） | 7.656 GiB（BLOOM） | 0.560 TiB（BLOOM） |
| 2023 | 0.412（Yi@200,000） | 47.752（Yi） | 45.776 GiB（Yi） | 1.493 TiB（Yi） |
| 2024 | 2.300（DeepSeek-V3@131,072） | 77.056（Llama 405B，partial） | 63.000 GiB（Llama 405B） | 2.412 TiB（Llama 405B） |
| 2025 | 2.609（Llama 4@10,485,760） | 485.500（Llama 4，partial） | 1.875 TiB（Llama 4，partial） | 60.196 TiB（Llama 4，partial） |
| 2026YTD | 2.290（Kimi K2.6@262,144） | 25.008（Kimi K2.6，partial） | 49.125 GiB（GLM-5.2） | 2.213 TiB（GLM-5.2） |

该表说明“最大模型计算量”“最大单Token流量”和“最大常驻容量”不一定由同一机制
决定。2025年的Llama 4 full-retention长上下文同时推高三类sequence相关指标，但
它的Traffic和Cache均为P3 partial，且不能当成典型服务场景。

## 8. 主导项交叉

交叉定义：

```text
Non-parameter FLOPs =
    Attention + Index + State + Extra

Non-weight logical-HBM =
    KV read/write + Index read/write
  + State read/write + Other read

Batch Cache =
    B × (KV + Index + State capacity per request)
```

下表中的数量按20个模型计；“达到”包括区间交叉和左删失。区间是相邻离散扫描点，
不是插值后的精确阈值。

| 条件 | `B=1` | `B=32` | `B=256` | 最早达到 |
|---|---:|---:|---:|---|
| Non-parameter FLOPs ≥ Parameter FLOPs | 12/20 | 12/20 | 12/20 | DeepSeek-V2：2,048→4,096 |
| Non-weight HBM ≥ Weight read | 4/20 | 12/20 | 18/20 | Qwen3：40,960→65,536；GLM-130B：512→1,024；`B=256`时GLM-130B阈值≤128 |
| Request Cache ≥ Decode weight | 2/20 | 2/20 | 2/20 | Qwen3：65,536→131,072 |
| Batch Cache ≥ Decode weight | 2/20 | 11/20 | 18/20 | Qwen3：65,536→131,072；GLM-130B：512→1,024；`B=256`时GLM-130B阈值≤128 |

FLOPs和Request Cache条件本身不随`B`变化，所以三列模型数相同。Traffic和Batch
Cache交叉会随`B`提前，分别来自Weight read摊薄和请求状态复制。

`crossover_points.csv`同时保存首次达到与稳定达到：

- 无达到：`not_reached_within_advertised`；
- 首个扫描点已达到：`at_or_below_min_scanned`，属于左删失；
- 相邻扫描点之间达到：`crossed_in_grid`；
- `dominance_reversal_count`记录首次达到后的回落次数；
- `stable_*`字段记录此后不再回落的交叉区间。

全量结果只有一个反转案例：Llama 4在`B=256`的Non-weight HBM/Weight read首次于
`C=4096→8191`达到，`C=8192`因chunk边界重置回落，随后于
`C=16384→32768`稳定达到。图中`8191/8192/8193`附近的锯齿是分块注意力公式的
预期行为，不是绘图错误。

## 9. KDA、Index与State如何进入结果

KDA由显式`recurrent_state`机制进入引擎，不会被伪装成普通KV Cache。Kimi-Linear
profile包含40层recurrent state与7层MLA：

- 每请求KDA/short-conv state为43,417,600 bytes；
- 在advertised `C=1,048,576、B=32`，persistent容量中Weight占26.537%，
  KV Cache占73.087%，State占0.375%；
- 同一点logical-HBM中State traffic占0.825%。

因此引擎支持KDA的固定State capacity、State read/write和显式State FLOPs。
但KDA非线性与归一化FLOPs没有足够公开事实，Kimi-Linear的FLOPs和Traffic P3状态
仍是`partially_supported`；Cache和Weight Capacity为`supported`。这是一种
“公式路径已支持、公开事实仍不完整”的有限支持，不应表述为端到端内核精确建模。

同理，GLM-5.2的DSA/IndexShare单独进入Index容量与流量。在其advertised ceiling、
`B=32`下，persistent容量中Index Cache占7.413%，不会与KV或Weight混为一项。

## 10. Llama 4多重上下文边界

Llama 4必须区分evaluated、trained、deployed与advertised四类事实。下表使用
`B=32`：

| 边界 | `C` | FLOPs（TF/token） | HBM（GiB/token） | Cache/request（GiB） | Persistent（TiB） |
|---|---:|---:|---:|---:|---:|
| evaluated max | 131,072 | 0.064488 | 11.500 | 24.000 | 0.946 |
| trained max | 262,144 | 0.096700 | 17.500 | 48.000 | 1.696 |
| deployed max | 3,600,000 | 0.919749 | 170.804 | 659.180 | 20.795 |
| advertised max | 10,485,760 | 2.609256 | 485.500 | 1920.000 | 60.196 |

`context_boundary_points.csv`以profile标量事实生成canonical标签，因此即使原始
131,072 anchor只标了`common_power_of_two`，仍会正确输出`evaluated_max`。
3,500,000 Token的`aws_deployed_max`保留为辅助anchor，不替代canonical
`deployed_max=3,600,000`。

该profile保守假设所有层保留full-history KV，36个local层按8192 Token chunk计算。
10M是发布宣称边界，不表示模型在10M上训练、评测有效或存在常见生产负载。

## 11. 产物与字段追溯

| 产物 | 用途 |
|---|---|
| `analysis_manifest.json` | 输入版本、输入哈希、分析脚本哈希、scope、行数和图表清单 |
| `quality_summary.json` | 冻结验证、分析校验、筛选数量、P3与warning统计 |
| `model_summary.csv` | 每模型静态事实、角色、机制、上下文边界和P3状态 |
| `annual_envelope.csv` | 两种年度scope、动态eligible集合、公共`C/B`包络、贡献模型与supported-only最大值 |
| `fixed_context_comparison.csv` | 全部20个模型在公共`C=2048`、`B=1/32/256`下的60行明细 |
| `native_max_points.csv` | 各模型advertised max下的完整分项 |
| `context_boundary_points.csv` | evaluated/trained/deployed/advertised边界压力 |
| `component_shares.csv` | FLOPs、Traffic及Weight/KV/Index/State容量占比 |
| `crossover_points.csv` | 首次与稳定交叉区间、ratio、删失和反转状态 |
| `figures/*.png`、`*.svg` | 29组图表的位图与矢量版本 |

关键结论与字段的对应关系：

| 结论 | 来源字段 |
|---|---|
| 年度weight与active ratio范围 | `model_summary.csv`中的capacity、active ratio、year |
| 动态条件上包络 | `annual_envelope.csv`的`max_*`、贡献模型、support、eligible IDs/count |
| 公共工作负载模型横比 | `fixed_context_comparison.csv`的模型、`C=2048`、`B`与指标列 |
| advertised ceiling压力 | `native_max_points.csv`的`context_C`与各资源分项 |
| Batch摊薄与容量复制 | 相同模型/`C`跨`concurrency_B`对比 |
| 主导项转换 | `crossover_points.csv`的首次字段、`stable_*`与reversal字段 |
| KDA/DSA容量归属 | `component_shares.csv`的State/Index Cache bytes与share |
| Llama 4证据边界 | `context_boundary_points.csv`的`boundary_tags` |

## 12. 限制

1. 每年只有3–5个研究样本，不代表全行业分布，因此不报告P90或统计显著性。
2. 年度inclusive max是“本研究样本中的条件可审计上界”，不是行业总体分位数；
   eligible集合变化前后的点不能解释为同一总体的连续趋势。
3. release-specific精度profile不等于所有部署；统一`C/B`也不是观测流量分布。
4. logical-HBM是算法逻辑边界，不含Activation、通信、kernel重复读取、
   workspace、调度和Host I/O。
5. `B=256`的超大persistent值表示逻辑超大芯片边界内的总常驻量，不表示单GPU
   可部署显存。
6. P3 partial仍进入inclusive包络，精确使用时必须同时读取支持状态、
   known gaps和supported-only列。
7. 本阶段没有能力/质量归一化，不能从年度资源下降直接推出算法效率提升。
8. 本阶段没有真实Token份额，不能从模型数量或发布数量推出部署采用率。

## 13. 复现

在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/analyze_decode_trend_envelope.py \
  --release-dir studies/decode_trend/releases/v1.0.0 \
  --output-dir /tmp/decode_trend_p8_envelope
```

只生成表格、不加载matplotlib：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/analyze_decode_trend_envelope.py \
  --release-dir studies/decode_trend/releases/v1.0.0 \
  --output-dir /tmp/decode_trend_p8_envelope \
  --no-plots
```

完整验证：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v
```

`/tmp`产物不提交Git；正式解释、口径和复现入口由本报告、实施计划和分析脚本承载。
下一阶段若研究“算法效率”，必须先补模型能力/质量控制字段并使用统一精度；
“部署采用”必须另行收集真实部署量或Token份额，不能与本包络合并成一条年度曲线。
