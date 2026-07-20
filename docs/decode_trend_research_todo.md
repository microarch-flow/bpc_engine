# Decode 架构趋势研究推进待办

版本：v0.2

更新时间：2026-07-20

相关文档：

- [模型架构演进图谱](LLM模型架构演进图谱_2023-2026_v0.1.docx)
- [Decode 趋势指标定义与计算边界](decode_trend_metrics.md)
- [Decode 趋势研究样本清单](decode_trend_sample_manifest.md)
- [Decode 趋势结果字段能力矩阵](decode_trend_field_capability_matrix.md)
- [Decode 趋势字段数据字典](decode_trend_data_dictionary.md)
- [P3 模型与机制支持审计](decode_trend_p3_mechanism_audit.md)
- [P8 行业需求包络实施计划](decode_trend_p8_envelope_plan.md)
- [P8 包络可视化修订计划](decode_trend_p8_visualization_revision_plan.md)
- [P8 行业需求包络报告](decode_trend_p8_envelope_report.md)
- [P9A LLM 技术轨迹函数实施计划](decode_trend_p9a_technology_trends_plan.md)
- [P9A LLM 技术轨迹函数报告](decode_trend_p9a_technology_trends_report.md)
- [20模型数据生成说明](../studies/decode_trend/releases/README.md)

## 1. 已确认的研究边界

- [x] 只研究 LLM 推理的 Decode 阶段。
- [x] 将单芯片或节点抽象成承载完整模型的“逻辑超大芯片”。
- [x] 第一阶段不统计片间通信。
- [x] `C` 表示 Decode step 开始前的上下文长度。
- [x] `B` 表示本 step 同时推进的并发请求数。
- [x] 每个请求每 step 生成一个 Token。
- [x] 核心输出是 `FLOPs/token = F(C,B)`、`Bytes/token = G(C,B)` 和 `Capacity = H(C,B)`。
- [x] 指标定义和计算边界已固化到 `decode_trend_metrics.md`。
- [x] 研究窗口首先聚焦 2022–2026YTD，其中 2026YTD 截止到 2026-07-17。

## 2. 推进原则

1. 一次只推进一个阶段；当前阶段验收后再进入下一阶段。
2. 精确模型 release 是样本单位；不同精度 profile 不是独立模型样本。
3. 官方事实、公式推导、部署假设和预测结果必须分开保存。
4. 若未来增加统一理论 profile，必须与当年真实部署 profile 分开计算。
5. 未被引擎支持的机制不能强行套用相近公式。
6. 所有总量必须保留计算、流量和容量分项。
7. 超过模型官方上下文的计算点必须标记为理论外推。

## 3. 总体路线

| 阶段 | 状态 | 主要交付物 |
|---|---|---|
| P0 指标合同 | 已完成 | `decode_trend_metrics.md` |
| P1 代表模型样本清单 | 已完成 | 模型纳入规则与 `decode_trend_sample_manifest.md` |
| P2 原始数据字典 | 核心字典已完成 | 能力控制字段与字段级置信度仍待补齐 |
| P3 引擎机制覆盖检查 | 已完成 | 模型 × 机制 × 引擎支持矩阵 |
| P4 统一计算协议 | 已完成 | `C/B` 网格、外推和 deployment profiles |
| P5 小样本试算 | 已完成 | Llama 2 70B 与 GLM-5.2 端到端试算 |
| P6 全量数据计算 | 已完成 | 可复现的20模型数据与冻结发布链 |
| P7 质量验证 | 自动验证已完成 | 哈希、归一化、容量恒等式和机制锚点 |
| P8 三类趋势统计 | 进行中 | 行业需求包络v0.2已完成；效率和采用趋势待实施 |
| P9A 技术轨迹函数 | 已完成 | 43项函数记录、3个组成轴、回测、敏感性、里程碑和共现 |
| P9B 联合未来配置 | 待开始 | 低/中/高情景配置、联合约束与可复算验证 |
| P9C 引擎与硬件换算 | 待开始 | Workload分布、芯片指标包络与敏感性报告 |

## 4. P1：建立代表模型样本清单

本阶段已经完成，样本增删必须重新经过人工确认。

### 4.1 定义样本单位

- [x] 每个样本必须对应精确的模型家族、版本、尺寸和 release。
- [x] 区分 Base、Instruct、Reasoning 等变体，避免混成一个模型。
- [x] 同一 release 的量化版本只作为 deployment profile。
- [x] 记录精确发布日期；无法确定时记录日期区间和原因。

### 4.2 定义样本用途

每个模型可属于一个或多个用途：

- [x] `frontier_envelope`：代表当年旗舰资源需求。
- [x] `efficiency_comparison`：可与前代或同能力模型比较架构效率。
- [x] `adoption_observation`：能够获得实际部署或采用证据。
- [x] `mechanism_anchor`：代表重要架构机制，但不用于行业采用率估计。

### 4.3 定义纳入规则

- [x] 模型发布时间位于研究窗口内。
- [x] 至少存在官方论文、模型卡、配置或代码中的一种可审计来源。
- [x] 能确定主要层数、隐藏维度、Attention/KV、FFN/MoE 等关键结构。
- [x] 能确定或合理界定总参数量和激活参数量。
- [x] 能确定官方上下文上限。
- [x] 能确定实际部署精度或给出有证据的部署口径。
- [x] 研究原型、闭源估计和公开 checkpoint 分层标记，不能混用。
- [x] 后续机制样本按能力、机制效果、可部署性三项审查，并允许机制级有条件纳入。

### 4.4 初始样本规模

- [x] 每年以 3 个核心角色为基础；只为不可替代的上下文或机制增加样本。
- [x] 每年至少包含一个行业需求包络样本。
- [x] 包含可形成家族纵向比较的模型。
- [x] 同时保留主流但没有采用激进新机制的基线模型。
- [x] 不因同一模型存在多个尺寸或精度而虚增样本量。

### 4.5 P1 交付物

计划生成一份 sample manifest，每行至少包含：

```text
model_family
model_release_id
variant
organization
release_date
parameter_scale
architecture_summary
sample_roles
availability
primary_sources
inclusion_reason
known_gaps
```

### 4.6 P1 验收标准

- [x] 2022–2026YTD 每年都有代表样本。
- [x] 每个样本都能定位到精确 release。
- [x] 没有把精度 profile 当成独立样本。
- [x] 每个样本的用途和纳入理由明确。
- [x] 每个样本至少有一个一手来源。
- [x] 已列出缺失字段和不可审计风险。
- [x] 样本清单经人工确认后再进入 P2。

## 5. P2：定义原始数据字典

- [x] 建立目标字段相对现有引擎、配置和输出载体的能力矩阵。
- [x] 定义模型标识、版本和来源字段。
- [x] 定义总参数、激活矩阵参数和各参数组字段。
- [x] 定义 Attention、KV layout 和 access mechanism 字段。
- [x] 定义 MoE Expert 总数、Top-k、Shared Expert 和层数字段。
- [x] 定义 Linear Attention/SSM 的状态字段。
- [x] 定义 Weight、KV、Index、State 精度字段。
- [x] 定义 advertised/trained/evaluated context 字段。
- [ ] 定义模型能力或质量档位字段。
- [ ] 补齐字段级来源、置信度、估算范围和缺失原因；当前已有类别级状态与known gaps。
- [x] 明确 unknown 使用 `null`，不能使用 0。

验收标准：每个 P1 样本都能按同一份字典填写，且每个数值都能追溯来源或假设。

## 6. P3：检查引擎机制覆盖

- [x] 把每个样本拆成逐层或 layer-group 机制组合。
- [x] 将四类指标分别标记为 `supported`、`partially_supported`、`unsupported` 或 `not_applicable`。
- [x] 核对当前样本使用的 GQA、MLA、SWA、DSA、MoE、Linear State 和 Mamba 实现。
- [x] 列出 YOCO、MoD、MTP、Diffusion、BLT 等尚未建模机制。
- [x] 确认 `fixed_cost` 只表达小型显式加项，不能掩盖动态机制缺失。
- [x] 禁止用模型名称触发隐藏公式。
- [x] 形成部分支持项和缺口清单；本阶段未修改既有机制公式。

验收标准：每个入选样本都知道哪些指标能精确计算、哪些只能估算、哪些暂时不能计算。

## 7. P4：确定统一计算协议

- [x] 确定 `C` 的离散扫描点：公共点为 `128 × 2^n`，到 16M；另加模型与机制锚点。
- [x] 确定 `B` 的离散扫描点：主扫描为 `1,2,4,8,16,32,64,128,256`；`512,1024` 仅作高并发压力点。
- [x] 保留每个模型的 `advertised_max_context_tokens_at_release`。
- [x] 定义当年真实部署 profile：只使用各模型已审计并落盘的历史实际精度。
- [x] 本轮不建立统一理论精度 profile；若未来需要反事实比较，另行定义且不虚增模型样本。
- [x] 定义 MoE routing 的标准假设；真实 trace 和上下界仍留待 P3/P7。
- [x] 定义 quantization metadata 的处理方式。
- [x] 定义理论外推标记规则。
- [x] 固定引擎版本、配置版本和输出字段。

验收标准：任何模型在相同输入下都能得到可复算、可比较的结果。

## 8. P5：小样本端到端试算

- [x] 选择Llama 2 70B与GLM-5.2作为稠密基线和现代稀疏机制的两个互补试点，替代原“每年一个”的冗余试算计划。
- [x] 完成两个试点的原始数据录入、配置转换和引擎计算。
- [x] 两个试点均计算已审计的当年真实部署 profile；其余模型在P6统一计算。
- [x] 输出全部计算、流量和容量分项。
- [x] 手工复核一个 `C/B` 点。
- [x] 检查总参数、激活参数、权重容量和 Cache 容量是否一致。
- [x] 记录无法由当前引擎输出的目标指标。

验收标准：四个年份的试算结果能用同一 schema 保存，并通过人工数值审计。

## 9. P6：全量数据计算

- [x] 将全部 20 个模型回填到机器事实清单。
- [x] 为每个模型建立可审计配置。
- [x] 已在 `/tmp/bpc_engine_decode_trend_full` 试运行全部主网格。
- [x] 保存模型级事实与计算结果的关联关系。
- [x] 不把不同 profile 计为新的模型样本。
- [x] 对超出上下文范围的点设置 `is_extrapolated=true`。

验收标准：所有入选模型均有完整结果或明确的不可计算原因。

## 10. P7：质量验证

- [x] 检查参数量、层数和机制层数是否闭合。
- [x] 检查总权重容量与位宽换算。
- [x] 检查完整 step 与每 Token 归一化关系。
- [x] 检查 Weight、KV、Index、State 分项之和。
- [x] 检查 Cache capacity 未混入 traffic。
- [x] 检查 `C=0`、小 `C`、机制饱和点和最大 `C`。
- [x] 为主要机制建立数值锚点。
- [x] 输出异常、估算和缺失值清单；由验证报告、P3状态和known gaps共同承载。

验收标准：核心指标能复算，所有估算和外推均有显式标记。

## 11. P8：三类趋势统计

### 行业需求包络

- [x] 使用release-specific模型、证据化部署精度profile和advertised-native边界。
- [x] 输出年度inclusive最大值、supported-only最大值、范围和预先指定的前沿样本；
  年度样本仅3–5个，不报告不可靠的P90。
- [x] 分析总权重、Cache、FLOPs/token、logical-HBM Bytes/token、
  persistent容量和主导项交叉。

### 算法效率趋势

- [ ] 使用统一精度和统一 `C/B`。
- [ ] 控制模型能力或质量档位。
- [ ] 优先进行同家族升级和机制反事实比较。
- [ ] 分解规模、精度、上下文和架构机制的贡献。

### 部署采用趋势

- [ ] 单独收集模型部署量或 Token 份额。
- [ ] 无真实份额时使用明确的低/中/高采用情景。
- [ ] 不用发布模型数量代替实际采用率。

验收标准：三类趋势分别输出，不能合并成一条含义不清的年度曲线。

## 12. P9：技术轨迹、未来配置与芯片指标换算

### P9A：技术轨迹函数

- [x] 以release/profile为统计单位提取参数、上下文、Attention三轴、MoE和位宽特征。
- [x] 比较常数与低维趋势函数，执行完整年度滚动回测和留组敏感性分析。
- [x] 输出43项正式函数记录、3个组成轴摘要、证据等级、里程碑、共现表和10组PNG/SVG。
- [x] 明确精选样本出现率不是行业采用率，2026YTD不进入完整年度回测。
- [x] 强制参数基与MoE两部分模型的声明内恒等式；跨技术轴联合约束留给P9B。

实施与结论分别见
[P9A计划](decode_trend_p9a_technology_trends_plan.md)和
[P9A报告](decode_trend_p9a_technology_trends_report.md)。

### P9B：联合未来模型配置

- [ ] 先编写P9B实施计划，明确primitive、diagnostic、条件分支和验收标准。
- [ ] 把P9A边际函数组合成低/中/高情景，而不是单一模型名称。
- [ ] 保留P9A的参数/MoE声明内恒等式和三个轴内组成闭合，并新增跨技术轴联合约束。
- [ ] 对上下文、MoE、精度和机制共现施加可解释的联合约束。
- [ ] 区分历史拟合、两年条件外推和更远期speculative情景。
- [ ] 输出机器可读的情景配置、来源函数与假设清单、约束验证结果和解释报告。
- [ ] 验证每个情景均可被当前引擎解析，且不会把互斥技术或诊断函数直接拼接。

### P9C：引擎计算与芯片换算

- [ ] 只接收通过P9B约束验证的联合情景，不直接消费独立边际函数。
- [ ] 输出 FLOPs/token、Bytes/token 和 Capacity 的情景分布。
- [ ] 给定目标 Decode Token/s 和并发数。
- [ ] 换算 Peak Compute、Peak HBM Bandwidth 和 HBM Capacity。
- [ ] 计算所需 Bandwidth/Compute。
- [ ] 分析哪些结论对模型采用率、上下文和精度假设敏感。
- [ ] 在进入真实芯片设计前补充利用率、功耗、面积和通信模型。
- [ ] 输出公式、输入假设、结果表、图表和结论解释文档，保持可复算追溯。

## 13. 当前行动

当前数据准备阶段已经形成可复现的正式数据生成链：

> 已于2026-07-20从干净提交`4ca035f`生成本地正式版本`v1.0.0`：20个模型、3357行结果、1440行显式外推、完整来源快照和SHA-256，验证状态为`pass`。版本目录由Git忽略，不提交仓库。

该版本在两个`C=16M、B=256`极端外推点报告IEEE-754整数精度警告，涉及BLOOM 176B和GLM-130B的batch cache；进入统计时应保留警告标记。

P8行业需求包络v0.2已按[实施计划](decode_trend_p8_envelope_plan.md)和
[可视化修订计划](decode_trend_p8_visualization_revision_plan.md)完成，正式解释见
[结果报告](decode_trend_p8_envelope_report.md)。分析保留`partially_supported`
数据并逐指标标记，同时输出supported-only年度最大值；没有把partial状态静默视为
完整支持。可复现表格和图表默认生成到`/tmp/decode_trend_p8_envelope/`。

P9A已按[实施计划](decode_trend_p9a_technology_trends_plan.md)完成，正式解释见
[结果报告](decode_trend_p9a_technology_trends_report.md)。20个release生成43项
函数记录和3个组成轴；按记录行计为11项`emerging`、17项`unstable`、15项
`insufficient`，没有`established`结论。真正的趋势证据单位是6条非组成趋势记录
和1个通过回测的KV-layout组成轴，另有4条代数派生函数；不能把函数行数解释成独立
发现数。可复现产物默认写入
`/tmp/decode_trend_p9a_technology_trends/`。

若继续未来预测，下一步是P9B联合配置约束，不能直接把独立边际函数拼成一台未来
模型。P8算法效率仍需先补能力/质量控制字段，再使用统一精度和统一`C/B`实施；
部署采用趋势仍需另行收集部署量或Token份额。

后续优先级固定为：

1. 编写P9B实施计划并冻结联合情景合同；
2. 生成低/中/高情景配置、约束验证和解释报告；
3. P9B验收后实施P9C workload与芯片指标换算；
4. P2能力字段、P8算法效率和部署采用数据作为并行数据补强，不阻塞P9B情景方法设计。
