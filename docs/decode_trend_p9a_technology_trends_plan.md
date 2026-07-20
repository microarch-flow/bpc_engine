# P9A LLM 技术轨迹函数实施计划

版本：v0.2

观测窗口：2022–2026YTD（截止 2026-07-17）

冻结输入：`studies/decode_trend/releases/v1.0.0/`

实验输出：`/tmp/decode_trend_p9a_technology_trends/`

## 1. 目标

P9A先建立可代入年份`t`的“LLM技术轨迹函数集”，描述代表模型的技术特征如何随
发布时间变化。它回答的是未来模型可能向哪些技术方向演进，而不是直接预测
PFLOPS、HBM带宽、HBM容量、GPU数量或具体芯片型号。

本阶段分别建模以下存在约束和相关性的技术维度：

- Decode常驻参数、每Token激活矩阵参数及二者的稀疏解耦；
- advertised、trained、evaluated、deployed四种上下文边界；
- Token Mixer组成：Softmax、Linear/Recurrent、SSM/Mamba；
- Softmax层的KV表示：MHA、MQA、GQA、MLA；
- Softmax层的访问方式：Full、Bounded Local、Sparse/Top-k；
- MoE是否出现、MoE层占比、Expert数、Top-k和`k/E`；
- 选定历史部署profile中的权重、KV、Index和State位宽；
- MoE、长上下文、低比特、压缩KV和状态模型之间的样本内共现。

P9A输出函数及证据等级。未来把这些函数组合成模型配置、再接入计算引擎和硬件换算，
分别留给P9B和P9C。

## 2. 统计单位、总体和声明边界

### 2.1 统计单位

一个`model_release_id + deployment_profile_id`只算一个release级观测。3357条
`C/B`扫描结果是同一批模型的确定性重复计算，不能作为3357个独立统计样本；
`annual_envelope_*`也不进入技术趋势拟合。

P9A主要读取冻结`model_profiles.jsonl`。Canonical物理层组成只读取profile中
`config_path`对应的`source_snapshot/configs/`冻结配置；分析前完整校验
`SHA256SUMS`，再逐profile核对配置哈希、profile中的`config_sha256`与校验和三方
一致。禁止回退到工作树中的实时`configs/`。
`decode_results.csv`只记录输入哈希和release一致性，不参与拟合。

### 2.2 当前总体的含义

当前20个模型混合了前沿包络、机制锚点、长上下文锚点、效率对照和采用观察，是
策展式代表样本，不是固定纳入规则下的完整release census，也没有真实Token份额。
因此：

- 二元曲线必须称为`selected-sample presence`，不能称行业采用率；
- 年度比例必须称为精选release样本出现率；
- 技术共现只能解释为样本内关联，不能解释为因果；
- 2026是YTD右删失年度，年度比例不能与完整的2022–2025直接等同；
- `established`等级保留给未来具备预定义分母的release census，本轮曲线最高为
  `emerging`。

未来若要把“越来越流行”升级为行业结论，需要另建：

1. 能力前沿总体，用于规模和上下文前沿；
2. 主要基础模型release census，用于发布采用率；
3. deployment或Token-share总体，用于真实使用率。

## 3. 技术特征提取

新增：

```text
scripts/analyze_decode_technology_trends.py
```

脚本保持标准库核心依赖，matplotlib仅用于可选绘图。默认读取冻结`v1.0.0`，默认
写入`/tmp`，不得修改冻结release或`outputs/`。

### 3.1 参数和上下文

每个release直接使用已审计字段：

```text
P_total  = decode_resident_parameter_elements
P_active = active_matrix_parameter_elements_per_token
R_active = P_active / P_total
D_sparse = log2(P_total / P_active)
```

这里的`P_total`严格是Decode profile常驻参数口径，不偷换成完整checkpoint总参数。

四种上下文分别保存和拟合；`null`保持缺失，禁止互相填充：

```text
C_advertised
C_trained
C_evaluated
C_deployed
```

另保存`C_evaluated/C_advertised`和`C_deployed/C_advertised`，用于衡量宣称、
评测与部署边界的差距。当前任务无关的effective-context观察不足，不拟合统一
`C_effective(t)`。

### 3.2 Canonical Attention三轴

冻结profile中的`mechanism_layer_counts`是成本组件计数，不是物理层组成。例如
Kimi Linear的同一KDA层包含两个State成本组件，Qwen3.6的同一DeltaNet层同时包含
Linear与State组件。P9A必须按配置`layer_groups[]`逐组分类，每个物理层只计一次。

Token Mixer大类：

```text
softmax
linear_recurrent
ssm
```

并保留Softmax、KDA、Gated DeltaNet、Mamba等明细。Softmax层再分别按以下两轴
分类：

```text
KV layout: MHA / MQA / GQA / MLA
Access:    Full / Bounded Local / Sparse Top-k
```

每个模型必须满足：

```text
softmax_share + linear_recurrent_share + ssm_share = 1
sum(KV layout layers) = softmax_layers
sum(access layers)    = softmax_layers
```

访问范围与缓存保留分开表达；例如Chunked Attention访问局部块，不自动表示旧KV已
被释放。

### 3.3 MoE

MoE存在性由`routed_expert_groups`结构化判断。首版只接受每模型零或一个Routed
Expert组；未来若同一模型存在多个无法判重的组，必须补层身份而不能盲加。

```text
moe_presence
moe_layer_share_given_moe = routed_layers / physical_sequence_layers
population_routed_layer_mass =
    moe_presence × moe_layer_share_given_moe
routing_density = selected_per_token / expert_count
```

Dense模型的非条件层质量为0，但条件层占比、Expert数、Top-k和`k/E`保持空值，
不能写0。出现率、采用后的层强度以及二者派生的总体层质量分开表达。

### 3.4 位宽

权重位宽不能用单一默认值代表混合精度模型。显式参数组按元素数加权：

```text
matrix_effective_bits = Σ(N_g × bits_g) / ΣN_g
explicit_share_le8    = Σ(N_g | bits_g <= 8) / ΣN_g
explicit_share_le4    = Σ(N_g | bits_g <= 4) / ΣN_g
```

同时保存：

```text
resident_effective_storage_bits =
    8 × decode_profile_weight_capacity_bytes
      / decode_resident_parameter_elements
```

并记录显式参数组对常驻参数的覆盖率。KV、Index、State只在配置实际存在相应存储
组件时形成观测；deployment默认值不能让没有Index/State的模型产生伪16-bit观测。
实际使用位宽按存储元素加权：

```text
effective_bits = Σ(component_elements × component_bits) / Σcomponent_elements
```

KV、Index、State位宽分别保存，禁止合成一个“模型精度”。现有数据不能结构化区分
原生低比特和发布后量化，不从文件名猜测；所有曲线都明确限定为
`selected deployment profile precision`。

## 4. 函数族

丰富性来自多条低维、可解释的函数，而不是在20个样本上拟合高阶总曲线。

### 4.1 连续正值

对参数、稀疏倍数和四种上下文比较两个候选：

```text
M0: log2 X(t) = α
M1: log2 X(t) = α + β(t - t0)
```

M1使用Theil–Sen中位斜率；同日release对跳过零时间差。输出：

- `β`（bits/year）；
- 年度倍率`2^β`；
- 倍增或减半时间；
- 观测期内拟合误差；
- 滚动年度回测误差；
- leave-year和leave-organization方向稳定性。

### 4.2 二元出现与比例强度

MoE、MLA、GQA、State-like和低比特profile等使用：

```text
M0: p(t) = Jeffreys-smoothed constant
M1: p(t) = sigmoid[α + β(t - t0)]
```

M1使用固定弱L2斜率正则，避免小样本完全分离。正例或负例少于3、只在一个年份
出现、或优化失败时，只输出描述性常数并标`insufficient`。

Mixer、KV layout和Access层占比使用同一低维fractional-logit分量。每个组成轴只
比较两套预注册候选：所有分量常数与所有分量趋势；每个回测fold先归一化，再使用
group-level mean-component Brier选择，正式函数为：

```text
p_k(t) = sigmoid(α_k + β_k(t-t0))
         / Σ_j sigmoid(α_j + β_j(t-t0))
```

训练fold中某一分量全0或全1时，该分量只在该fold使用Jeffreys常数cold-start
fallback，不伪造时间斜率。技术“是否出现”和“出现后的层占比/强度”分别保存。

参数投影使用`resident_parameters + active_ratio`为基，严格派生active参数、
resident/active倍数和bit gap；直接边际拟合仍保留在候选表中作为诊断。MoE总体
层质量同样由presence与given-MoE条件层占比派生。跨技术轴的联合约束仍留给P9B。

当前只有一个显式Linear Attention样本和一个DSA样本，这两项只记录first-seen和
`insufficient`，不强行输出扩散S曲线。

### 4.3 选择、回测和证据等级

滚动回测只用过去预测下一完整年度：

```text
train: release_year <= Y
test:  release_year == Y + 1
```

连续值使用年度等权`MAE(log2 X)`及倍数误差；二元值使用年度等权Brier score和
log-loss。2026YTD不进入主完整年度回测，只进入release级拟合和敏感性。

单类全0或全1的Logistic训练集必须返回`insufficient`，不能把有限但未收敛的截距
冒充有效拟合。趋势候选至少有两个有效完整年度fold，且相对常数改善至少5%，才可
选为主函数；
否则保留常数基线。证据等级：

- `insufficient`：样本、年份或正负事件不足；
- `unstable`：可拟合但回测不优于常数，或留组后方向频繁翻转；
- `emerging`：方向与回测相对稳定，但历史短或总体仍是精选样本；
- `established`：预留给未来release census，本轮不授予。

函数可以代入任意年份，但首版推荐外推有效期只到最后观测日期后两年；更远年份标
`speculative`。

证据等级只描述方向稳定性和相对常数基线的skill，不认证数值预测精度。正值函数
另报`MAE(log2)`与典型倍数误差，比例函数另报Brier/RMSE percentage points；
当前最多只有两个完整年度fold；具有两个有效fold的可评估函数标为
`provisional_two_folds`，派生函数继承来源状态，无法回测的函数标为
`not_evaluable`。

## 5. 输出产物

```text
analysis_manifest.json
quality_summary.json
technology_observations.csv
annual_sample_summary.csv
trend_candidates.csv
selected_trend_functions.csv
selected_trend_functions.json
composition_group_summary.csv
trend_backtests.csv
trend_sensitivity.csv
fitted_observations.csv
trend_projection_grid.csv
technology_milestones.csv
technology_cooccurrence.csv
```

关键含义：

- `technology_observations.csv`：每个release一行的全部canonical技术特征；
- `annual_sample_summary.csv`：年度样本数、中位数、范围和精选样本出现数；
- `trend_candidates.csv`：常数与趋势候选、系数、误差和选择结果；
- `selected_trend_functions.*`：可机器读取的正式函数目录、claim scope和证据等级；
- `composition_group_summary.csv`：三个组成轴的联合候选、归一化回测和主增减分量；
- `trend_backtests.csv`：每个时间fold的训练截止、测试年和误差，证明没有未来泄漏；
- `trend_sensitivity.csv`：leave-year、leave-organization和关键离群点敏感性；
- `fitted_observations.csv`：观测值、拟合值和残差；
- `trend_projection_grid.csv`：历史拟合与未来代入展示网格，不是新的观测数据；
- `technology_milestones.csv`：first-seen、第二独立组织和可识别的`t10/t50/t90`；
- `technology_cooccurrence.csv`：四格表、条件比例、Jaccard和lift。

`analysis_manifest.json`必须保存冻结输入哈希、脚本哈希、拟合截止日期、YTD状态、
候选函数、选择规则、claim scope、行数和全部artifact。输出目录如果位于冻结
release内部必须拒绝。

## 6. 图表

matplotlib可用时同时输出PNG和SVG：

```text
parameter_scale_and_active_trends
parameter_sparsity_decoupling
context_boundary_trends
token_mixer_composition
kv_layout_composition
attention_access_composition
moe_presence_and_intensity
deployment_profile_precision
technology_presence_timeline
trend_evidence_summary
```

历史观测、拟合区间和外推区间必须使用不同线型或底色；2026YTD显式标记。组成图按
release级模型等权，不把几十个物理层伪装成几十个统计样本。

## 7. 正式解释文档

新增：

```text
docs/decode_trend_p9a_technology_trends_report.md
```

报告必须：

- 先说明精选样本、YTD、能力控制和部署采用数据缺口；
- 区分Observed、Fitted、Conditional extrapolation和Speculative；
- 给出每条主函数、系数、回测、敏感性、适用期和证据等级；
- 解释参数解耦、上下文边界、Attention三轴、MoE和量化分别说明什么；
- 对Linear、DSA等单点机制只报告描述性里程碑并标`insufficient`，不虚构增长率；
- 所有关键结论都指向CSV字段或函数ID。

同步更新`decode_trend_research_todo.md`，把后续工作拆成P9A技术轨迹、P9B未来配置
组合和P9C引擎/硬件换算；P8真实采用趋势仍保持未完成。

## 8. 测试与验收

新增：

```text
tests/test_decode_technology_trends.py
```

至少覆盖：

1. release/profile/config哈希、唯一键、null和正值校验；
2. Theil–Sen精确趋势、离群稳健性和同日release；
3. Logistic完全分离、稀有事件fallback和有限参数；
4. 回测训练集不包含测试年，2026YTD不进入完整年度主回测；
5. Kimi Linear、Qwen3.6和Jamba物理层不重复计数；
6. Mixer、KV layout和Access三轴占比闭合；
7. MoE层占比、Expert数、Top-k和`k/E`锚点；
8. Llama 4四种上下文边界不混用；
9. 权重组加权位宽、覆盖率以及KV/Index/State分离；
10. 函数JSON重算值与CSV投影一致，不出现NaN或Infinity；
11. 输出目录隔离、artifact清单和PNG/SVG有效性。

实施完成后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -v

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  scripts/analyze_decode_technology_trends.py \
  --release-dir studies/decode_trend/releases/v1.0.0 \
  --output-dir /tmp/decode_trend_p9a_technology_trends
```

验收条件：

1. 冻结release和已有`outputs/`未修改；
2. 20个release各生成一条canonical技术观测；
3. 三个Attention组成轴和位宽加权恒等式闭合；
4. 每个函数都有公式、参数、claim scope、回测或明确缺失原因；
5. Linear、DSA和行业采用率没有被过度声明；
6. 表格、图表、报告和manifest能够相互追溯；
7. 完整单元测试通过。
