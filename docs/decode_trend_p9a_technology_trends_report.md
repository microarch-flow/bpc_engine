# P9A LLM 技术轨迹函数报告

版本：v0.2

观测窗口：2022–2026YTD（截止 2026-07-17）

冻结输入：`studies/decode_trend/releases/v1.0.0/`

正式输出：`/tmp/decode_trend_p9a_technology_trends/`

## 1. 先看结论

P9A以20个模型release、13个组织为样本，生成43条技术函数记录和3个组成轴函数。
最重要的结论不是“所有技术都在按一条曲线增长”，而是：

> 当前精选release显示，LLM演进更接近降低每Token激活比例、扩大上下文边界、
> 增加MoE出现倾向，并在Softmax层内部提高MLA组成份额；当前证据还不能证明
> Token Mixer整体从Softmax转向线性注意力、Attention访问稳定转向稀疏化，或
> 低比特精度的行业采用率稳定增长。

正式保留的方向信号为：

- `parameters.active_ratio`下降；
- advertised、trained、evaluated和所选profile deployed四种上下文边界上升；
- 精选release中的`moe.presence`上升；
- KV-layout组成轴出现从MHA/MQA向MLA移动的group-level信号。

其中trained、evaluated和deployed上下文只是各自语义下的边际诊断函数，不允许直接
进入P9B联合配置；KV-layout四个分量共享一个组成轴证据，不能算成四条独立发现。

明确未通过主回测的假设包括：

- Decode常驻参数按稳定倍率逐年增长；
- 绝对激活参数按稳定趋势变化；
- Softmax Mixer稳定转向Linear/Recurrent或SSM；
- MLA“模型出现率”已经形成可靠扩散曲线；
- State-like Mixer已经形成可靠扩散曲线；
- Full访问稳定转向Bounded Local或Sparse/Top-k；
- 权重、KV、Index或State位宽存在可靠年度下降趋势；
- MoE采用后的层占比、Expert数、Top-k或`k/E`存在可靠年度趋势。

## 2. 如何理解“43条函数”

43条记录包含四种角色，不能只看一个总数：

| 角色 | 条数 | 解释 |
|---|---:|---|
| `independent_marginal` | 23 | 独立边际函数；其中只有active ratio、advertised context和MoE presence为`emerging` |
| `marginal_diagnostic` | 6 | 语义有用，但不能直接拼进联合未来配置；3条上下文边界为`emerging` |
| `composition_component` | 10 | 属于3个归一化组成轴；KV轴选趋势，Mixer与Access轴选常数 |
| `derived_identity` | 4 | 由其他正式函数严格派生，不是4条新增统计证据 |

按记录行计，证据等级为11条`emerging`、17条`unstable`、15条`insufficient`；
没有`established`。11条`emerging`中包含3个KV分量和2个参数派生恒等式，所以不能
解释成11项独立技术趋势。

更合适的证据单位是：

- 6条非组成趋势函数记录；
- 1个通过回测的KV-layout组成轴；
- 4条保证代数闭合的派生函数；
- 其余指标保留常数或描述性结果。

## 3. 数据和声明边界

### 3.1 统计单位

一条统计观测是：

```text
model_release_id + deployment_profile_id
```

冻结`decode_results.csv`中的3357行是同一批模型在不同`C/B`上的确定性工作量计算，
不是3357个独立模型，因此不进入P9A拟合。该CSV只用于冻结哈希和行数一致性复核。

### 3.2 冻结输入

分析器先验证整个`SHA256SUMS`，再只从：

```text
studies/decode_trend/releases/v1.0.0/source_snapshot/configs/
```

读取配置。每个配置同时满足：

```text
SHA256(snapshot config)
  = model profile中的config_sha256
  = SHA256SUMS中的对应条目
```

20个配置全部通过。分析不再读取工作树中的实时`configs/`，因此后续代码或配置修改
不会静默改变这次冻结研究。

### 3.3 这不是行业采用率

20个模型混合了前沿包络、机制锚点、长上下文、效率对照和采用观察角色，是策展式
代表样本，不是固定纳入规则下的完整行业release census，也没有真实部署量或Token
份额。因此：

- `presence`只表示精选release中是否出现；
- 年度比例不是市场份额；
- 共现不是因果；
- 2026只有截至7月17日的YTD样本；
- 没有能力/质量控制，不能把本报告称为算法效率研究。

`deployed_max_context_tokens`还是所选历史profile的配置边界。20个模型中只有1个
deployed低于advertised，只有2个模型带effective-context观察，所以它不能解释成
真实生产有效上下文趋势。

## 4. 特征如何提取

### 4.1 参数基

原始观测：

```text
P_resident = decode_resident_parameter_elements
R_active   = active_matrix_parameter_elements_per_token / P_resident
```

正式投影强制派生：

```text
P_active(t)          = P_resident(t) × R_active(t)
sparsity_multiplier  = 1 / R_active(t)
sparsity_gap_bits    = -log2[R_active(t)]
```

这样不会再出现active ratio、active参数和sparsity multiplier彼此矛盾的未来代入值。
三项直接边际拟合仍保留在`trend_candidates.csv`中，但只作为诊断。

这里的`P_resident`是所选Decode profile常驻参数口径，不等同于论文标题中的完整模型
总参数或完整checkpoint参数。

### 4.2 Attention三轴

同一物理层只在Token Mixer轴计一次：

```text
Token Mixer = Softmax / Linear-Recurrent / SSM-Mamba
```

仅对Softmax层再分类：

```text
KV layout = MHA / MQA / GQA / MLA
Access    = Full / Bounded Local / Sparse Top-k
```

物理层锚点：

- Kimi Linear：27层，即20个KDA层加7个MLA层；
- Qwen3.6：40层，即30个Gated DeltaNet层加10个GQA层；
- Jamba：72层，即63个Mamba层加9个GQA层。

纯Linear或纯SSM模型也允许进入分析；此时Softmax层数为0，KV和Access条件组成保持
空值，不会被伪造为0%组成观测。

### 4.3 MoE两段式口径

MoE拆为：

```text
P_MoE(t)
E[layer_share | has_moe = 1, t]
population_routed_layer_mass(t)
  = P_MoE(t) × E[layer_share | has_moe = 1, t]
```

Dense release的非条件层质量为0，但given-MoE层占比、Expert数、Top-k和`k/E`保持
空值。这样不会把“是否采用MoE”和“采用后有多少层使用MoE”混成一条不透明强度曲线。

### 4.4 实际使用位宽

权重按显式参数组元素数加权。KV、Index和State只有在配置实际包含对应存储组件时
才形成观测：

```text
effective_bits
  = Σ(component_elements × component_bits) / Σcomponent_elements
```

没有Index或State的模型保持空值；deployment默认16 bit不会制造伪观测。当前非空数：

| 存储类型 | 非空模型数 | 当前事实 |
|---|---:|---|
| KV | 20 | 19个16-bit，GLM-5.2为8-bit |
| Index | 1 | 仅GLM-5.2，16-bit |
| State | 3 | Jamba 16-bit、Kimi Linear 30.9489-bit、Qwen3.6 16-bit |

Kimi Linear的状态有效位宽接近31 bit，是因为FP32 KDA大矩阵状态远大于BF16短卷积
状态；不能把`[16, 32]`简单平均成24 bit。

## 5. 拟合、回测和证据

令发布时间为十进制年份`t`，`u=t-2024`。

正值比较：

```text
M0: X(t) = 2^α
M1: X(t) = 2^(α + βu)
```

`M1`使用log2空间Theil–Sen中位斜率。比例和二元指标比较：

```text
M0: p(t) = Jeffreys-smoothed constant
M1: p(t) = σ(α + βu)
σ(z) = 1 / [1 + exp(-z)]
```

Logistic使用`L2=0.5`斜率正则。全0或全1训练集的时间斜率不可识别，必须返回
`insufficient`；优化未收敛也不能当作有效fit。

组成轴比较两套预注册候选：

```text
G0: 所有分量使用常数
G1: 所有分量使用趋势

p_k(t) = σ(α_k + β_k u) / Σ_j σ(α_j + β_j u)
```

每个fold先归一化，再以mean-component Brier计分。训练fold中尚未出现的分量只在该
fold使用Jeffreys常数cold-start fallback，不伪造斜率。

主回测测试年只有完整年度2024和2025，每个fold严格只用更早release训练；2026YTD
不进入完整年度回测。趋势至少需要两个有效fold、相对常数改善5%，并通过leave-year
与leave-organization方向稳定性。

`emerging`只表示方向稳定且相对常数有skill，不表示高预测精度。所有可评估函数的
accuracy maturity都是`provisional_two_folds`。

## 6. 正式趋势函数

### 6.1 非组成函数

| 函数ID | 正式函数 | 相对常数改善 | 绝对回测误差 | 角色 |
|---|---|---:|---:|---|
| `parameters.active_ratio` | `σ(1.023771 - 1.322891u)` | 46.3% | RMSE约42.0个百分点 | independent |
| `context.advertised_max` | `2^(14.720779 + 2.187812u)` | 55.6% | 典型×4.87 | independent |
| `context.trained_max` | `2^(14.095904 + 2.187812u)` | 45.4% | 典型×7.30 | marginal diagnostic |
| `context.evaluated_max` | `2^(14.543101 + 1.953743u)` | 60.9% | 典型×3.04 | marginal diagnostic |
| `context.deployed_max` | `2^(14.720779 + 2.187812u)` | 57.7% | 典型×4.26 | marginal diagnostic |
| `moe.presence` | `σ(-0.773194 + 1.420619u)` | 46.3% | RMSE约46.3个百分点 | independent |

上下文中心斜率对应：

- advertised、trained和profile-deployed每年约乘4.56；
- evaluated每年约乘3.87。

但3–7倍的典型回测误差说明它们只能用于方向与数量级情景，不能作为精确增长定律。

### 6.2 KV-layout组成函数

KV-layout轴正式选择group trend：

```text
p_k(t) = σ(α_k + β_k u) / Σ_j σ(α_j + β_j u)
```

| 分量 | α | β/year | 应用后方向 | 分量证据 |
|---|---:|---:|---|---|
| MHA | -2.440954 | -1.336121 | 下降 | emerging |
| MQA | -3.169349 | -1.149277 | 下降 | emerging |
| GQA | 0.227908 | -0.051339 | mixed | unstable |
| MLA | -1.850368 | 1.185935 | 上升 | emerging |

表中的分量证据是同一个KV轴被整组选中后的分量方向检查，不能把3个`emerging`
标签计成3项相互独立的趋势发现。

整个KV轴相对常数组成改善13.56%，两个fold的应用值RMSE约38.10个百分点。主增量是
MLA，主减量是MHA；二者通过留年和留组织方向检查。GQA的归一化方向会转折，不能
声称它稳定下降。

这支持的是“精选模型的Softmax物理层组成向MLA移动”，不是“MLA行业采用率增长”。
单独的`technology.mla_presence`在预测2024时没有历史正例，只剩1个有效趋势fold，
因此正式选择常数并标`insufficient`。

### 6.3 代数派生函数

正式目录还包含4条派生函数：

```text
parameters.active_elements
  = parameters.resident_elements × parameters.active_ratio

parameters.sparsity_multiplier
  = 1 / parameters.active_ratio

parameters.sparsity_gap_bits
  = -log2(parameters.active_ratio)

moe.unconditional_layer_share
  = moe.presence × moe.layer_share_given_moe
```

它们保证恒等式闭合，不增加独立证据。特别是常驻参数本身为`unstable`，MoE条件层
占比为`insufficient`，所以相应派生值也不能被包装成高置信预测。

## 7. 分技术方向结论

### 7.1 总参数越来越大、激活参数不变吗

年度中位数：

| 年份 | Decode常驻参数 | 每Token激活矩阵参数 | 激活占比 |
|---|---:|---:|---:|
| 2022 | 176.25B | 176.23B | 99.99% |
| 2023 | 46.70B | 33.93B | 98.66% |
| 2024 | 398.56B | 69.50B | 23.49% |
| 2025 | 78.45B | 23.91B | 10.65% |
| 2026YTD | 743.38B | 31.69B | 5.42% |

常驻参数趋势回测差于常数基线，绝对激活参数趋势也差于常数。当前只能正式支持
`active/resident`比例下降，不能写成“总参数稳定增长、激活参数稳定不变”。

### 7.2 上下文在变长吗

四种上下文的边际趋势都优于常数，但语义不同、误差很宽。P9B只能把advertised作为
候选primitive，再用明确情景约束trained、evaluated、deployed；不能把四条独立曲线
直接拼成一个上下文配置。

### 7.3 MHA正在被线性注意力替代吗

当前答案是：没有足够证据。

- Token Mixer组成趋势比常数组成差6.8%，正式保留常数；
- 显式Linear Attention只有Qwen3.6一个样本；
- State-like只有Jamba、Kimi Linear和Qwen3.6三个样本；
- State-like二元趋势只有1个有效回测fold，降为`insufficient`；
- KV轴中的MLA上升不等于Softmax被线性注意力替代，MLA仍是Softmax Attention。

可以说架构开始出现MLA、KDA、DeltaNet和Mamba等混合路径，不能说已经形成统一的
Softmax到线性注意力替代曲线。

### 7.4 MoE越来越常见吗

精选样本MoE出现数：

```text
2022: 0/3
2023: 1/5
2024: 3/5
2025: 3/4
2026YTD: 3/3
```

`moe.presence`通过回测，但它是精选样本出现倾向，不是行业采用率。given-MoE层占比
只有10个样本，预测2024时只有一个训练样本，无法形成两个有效fold，因此保留常数。
Expert数、Top-k和`k/E`同样没有可靠年度趋势。

### 7.5 稀疏访问越来越多吗

Bounded Local只有2个样本，Sparse/Top-k只有GLM-5.2一个样本。Access组成趋势相对
常数恶化51.8%，正式保留常数。当前只能记录机制里程碑，不能拟合扩散速度。

### 7.6 低比特越来越多吗

显式矩阵参数多数不高于8 bit的profile有11/20，多数不高于4 bit的有4/20，但年度
分布不单调。权重有效位宽、KV有效位宽和两条低比特presence都没有正向回测skill；
Index只有1个样本，State只有3个样本。

此外，profile混合原生精度和部署后量化，不能称为原生低比特行业采用率。

## 8. 技术共现

样本内：

- 6个MLA样本全部同时采用MoE；
- 3个State-like样本全部同时采用MoE；
- 唯一显式Linear Attention样本也采用MoE。

这些是`technology_cooccurrence.csv`中的描述性关联，不能解释为因果，也不能外推为
行业条件概率。

## 9. 如何代入未来年份

`trend_projection_grid.csv`区分：

- `historical_fit`：观测区间；
- `conditional_extrapolation`：最后观测后两年内；
- `speculative`：超过推荐外推期。

以`t=2028`为函数使用示例：

| 边际函数 | 2028代入 | 必须同时保留的限制 |
|---|---:|---|
| active/resident ratio | 1.38% | 常驻参数规模本身不稳定 |
| advertised context | 11.64M | 典型回测误差约×4.87 |
| evaluated context | 5.38M | marginal diagnostic，典型约×3.04 |
| KV-layout MLA share | 65.17% | 轴内组成，RMSE约38.1个百分点 |
| MoE selected-sample presence | 99.27% | 不是行业采用率，RMSE约46.3个百分点 |

参数派生函数在2028会严格满足：

```text
P_active = P_resident × active_ratio
resident / active = 1 / active_ratio
gap_bits = log2(resident / active)
```

这只说明代数闭合，不说明`P_resident`或最终模型配置准确。

不能把表中的边际值拼成“一台2028模型”。P9B仍需决定：

- 哪些边际函数作为primitive；
- 参数、上下文和三个组成轴如何联合约束；
- 低/中/高技术情景及相关性；
- 是否采用MoE、MLA、State-like、低比特和稀疏访问的离散组合。

## 10. 图表怎么读

分析器输出10组PNG/SVG：

| 图名 | 主要问题 |
|---|---|
| `parameter_scale_and_active_trends` | 常驻参数基与派生active参数 |
| `parameter_sparsity_decoupling` | resident/active解耦 |
| `context_boundary_trends` | 四种上下文边际边界 |
| `token_mixer_composition` | Softmax、Linear/Recurrent、SSM组成 |
| `kv_layout_composition` | MHA、MQA、GQA、MLA归一化组成 |
| `attention_access_composition` | Full、Bounded Local、Sparse/Top-k组成 |
| `moe_presence_and_intensity` | MoE出现、given-MoE强度与派生总体层质量 |
| `deployment_profile_precision` | 权重、KV、Index、State实际使用位宽 |
| `technology_presence_timeline` | 精选样本中的机制出现时间 |
| `trend_evidence_summary` | 43条记录的证据等级 |

所有时间图显式区分2026YTD、两年条件外推和更远期speculative区间。组成图的线是
group-level归一化函数，点是年度release等权均值；没有按参数量、下载量、部署量或
Token份额加权。

## 11. 机器产物与追溯

```text
/tmp/decode_trend_p9a_technology_trends/
├── analysis_manifest.json
├── quality_summary.json
├── technology_observations.csv
├── annual_sample_summary.csv
├── trend_candidates.csv
├── selected_trend_functions.csv
├── selected_trend_functions.json
├── composition_group_summary.csv
├── trend_backtests.csv
├── trend_sensitivity.csv
├── fitted_observations.csv
├── trend_projection_grid.csv
├── technology_milestones.csv
├── technology_cooccurrence.csv
└── figures/                    # 10组PNG/SVG
```

追溯入口：

- 正式函数、角色、公式、参数和证据：
  `selected_trend_functions.csv/json`；
- 组成轴选择：`composition_group_summary.csv`；
- 常数与趋势候选：`trend_candidates.csv`；
- 每个完整年度fold：`trend_backtests.csv`；
- 留release、年度和组织敏感性：`trend_sensitivity.csv`；
- release级canonical特征：`technology_observations.csv`；
- 输入、脚本、配置和每个artifact的SHA-256：
  `analysis_manifest.json`。

复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/decode_trend_p9a_mpl \
python3 -B scripts/analyze_decode_technology_trends.py \
  --release-dir studies/decode_trend/releases/v1.0.0 \
  --output-dir /tmp/decode_trend_p9a_technology_trends
```

## 12. 后续边界

P9A产出的是可审计的边际、组成和派生函数，不是硬件需求预测。后续阶段保持：

1. P9B：构造满足跨轴约束的低/中/高联合未来模型配置；
2. P9C：把P9B配置送入Decode引擎，计算FLOPs、logical-HBM和容量，再换算硬件；
3. P8算法效率：补能力/质量控制后单独研究；
4. P8部署采用：取得部署量或Token份额后单独研究。
