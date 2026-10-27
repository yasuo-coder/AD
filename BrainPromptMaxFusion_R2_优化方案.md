# BrainPromptMaxFusion R2 优化方案

状态：方案与实际诊断先提交；提交成功后启动固定 FIT 内的完整内部研究。此文件不是训练完成报告。

依据：[Issue #3 下一轮指导](https://github.com/yasuo-coder/AD/issues/3#issuecomment-6435101924)。用户进一步明确选择“固定 FIT 内的 9 折完整内部研究”。事实基线为 `main@db5bbf161efb7d7a723f94aabcfdd51512d1dd4c`；不改写 E0–E5 历史结果，不复用 R1 权重或规则。

## 1. 代码与协议

- 本次提交的 `experiments/BrainPromptR2/` 是完整执行代码；训练依赖模型已复制到 `legacy/`，不修改旧实验源代码。
- [机器配置](experiments/BrainPromptR2/configs/study.json)、[逐文件代码与环境哈希](experiments/BrainPromptR2/public/source_manifest.json)、[分区清单](experiments/BrainPromptR2/public/data_isolation.public.json)、[实际预检](experiments/BrainPromptR2/public/preflight.public.json)。本文件的 Git commit SHA 随 Issue 回复提供，避免文档自引用哈希。
- 旧模型来源：本地 `BrainPromptNComms2025@69411e8d614110d929afa263c1fc0184dcc819fd`；复制的模型代码保持原样。新的 `MinimalG`、训练和选择逻辑均在新项目内。
- 中断时保留模型、优化器、学习率调度器、随机数状态与完整历史；恢复会核对协议、数据、分区、代码哈希。失败或未收敛单元不能参加排序。

## 2. 固定人群和用途

只取既有 **seed42 / outer fold0 的 FIT**：715 名唯一受试者、1051 次扫描。CN/AD/SMC/EMCI/LMCI 的人数为 **401/52/16/223/23**。保持 original_diagnosis、MCI→EMCI、真实 LMCI 的标签口径。

在这 715 人中以 seed 42、2024、3407 各做一次分层受试者级三折，共九个内部单元。每个内部二分之三再分出 15% 的 STOP；其余为优化 FIT。优化 FIT 训练和估计标准化、先验及 reference；STOP 只用于 checkpoint、学习率与停止；内部 validation 只用于跨候选、τ 和提示条件比较。一个人的全部扫描始终同属一个分区。原外折 VAL 完全不用，原 TEST 完全不用。

主报告单位是受试者：先逐扫描 softmax，再在同一个人内平均概率。训练时神经网络每轮每人随机抽一条扫描，Logistic 用每扫描 `1/该人扫描数` 权重；标准化也采用该权重。这样多次扫描不增加受试者权重。各重复仍共享患者；九折不是九个独立试验，不做普通独立样本 t 检验。历史 895 人已暴露，R2 仍只能称内部探索。

## 3. TEST 的实际隔离

`r2.prepare` 是独立可信准备步骤，从既有分区导出仅含 FIT 的文件。训练数据不含原患者 ID、原始行号、外 VAL/TEST 行或历史预测。仅保留局部匿名 group 编号用于受试者聚合。

每个训练单元通过 `r2.isolation.command` 启动 bubblewrap：

1. 文件系统采用挂载白名单：系统库、Python 环境、封存代码、FIT-only 数据、当前单元 task 和输出；数据与代码只读，当前输出可写。
2. 没有挂载旧实验 out、其他研究目录、`/data2`、私有历史诊断或宿主 home；PID 命名空间也隔离，不能经宿主 `/proc/<pid>/root` 绕回去。
3. 网络命名空间隔离；外部网络连接实际失败。GPU 在隔离环境内实际可用。
4. 训练入口没有 TEST 参数或外折评估器；每个单元检查数据、分区、代码哈希以及 FIT/STOP/validation 的受试者互斥性。

已实际尝试读取完整 cohort、历史 TEST 预测和其他私有路径，均失败，见预检 JSON。这里是对误读/误用的运行隔离，不声称能抵抗拥有宿主权限的人主动重配挂载。可信控制器只读取新 FIT 与新内部输出。历史只读诊断在独立步骤执行，不作为新选模输入。

## 4. R2-A：先诊断，再解耦

[实际诊断报告](experiments/BrainPromptR2/public/诊断结果.md)及 CSV/JSON 已先完成。

- E2/E3：30 个历史外折选择均按原 VAL 规则重放成功；逐折配方、选中次数、VAL 与历史 TEST 指标、每 seed 合并 OOF、概率尺度、温度均已输出。没有新 TEST 推断或重新选择。旧 raw logits 未归档，明确记 NA，仅提供 centered log-probability 代理，不能称原 logits。
- E4：30/30 折第一次到学习率下限是 55，第一允许停止是 84，真实停止也是 84；已重放停止判据。不是写死 epoch cap。
- T：纯 ROI mean → FC bias → 患者 token → disease query 都能拟合 25 人，但 **E4 自身原架构在 1000 次更新仍仅 20% 准确率**。停止 T 路线，本轮不启动 E4/E5，也不把简化 T 的成功当作 E4 通过证据。诊断是 CE-only、固定小样本、无权重衰减的容量检查，不是泛化比较。

A 的正式内部矩阵为 **original/soft_topk × last/best**。两种图各训练九次，同一次轨迹同时保存 last 与 best，合计 **18 次拟合、36 组内部验证预测**。保留原 G 架构及原缓存 ROI/患者提示，统一 CE、初始化种子、受试者采样和停止规则，仅改人口图；不加入 E2 损失或 E3 结构。

soft_topk 训练时邻居来自当前 FIT batch；评估时是仅由优化 FIT 构建的五类受试者加权 reference，不含 STOP 或 validation。original 保留原实现供解耦；本轮 B 仍固定 population off，不用 A 的最好组合扩大 B 搜索。

## 5. R2-B：强制简单基线

全部采用相同九个受试者分区。

| 候选 | 预声明规格 | 拟合次数 |
|---|---|---:|
| Logistic | 6670 维非对角 FC，FIT-only 标准化，L2 multinomial，C=0.1/1/10 | 27 |
| 小 MLP | 同一 6670 维 FC，64 隐藏单元、ReLU、dropout=0.1 | 9 |
| 最简 G | 两层、hidden=64、dropout=0.1、无 batch norm，population off，数值 ROI，无患者/ROI 文本和疾病辅助损失 | 9 |
| T | 自身容量门槛失败，停止 | 0 |

不宣称不同优化器的 FLOP 相同；它们共享分区、输入样本预算和预先固定搜索清单。Logistic 用 LBFGS、tol=1e-6、最多 10000 次迭代；未收敛标失败，不把迭代上限当成功停止。神经网络统一 AdamW，lr=3e-4，weight_decay=0.005，batch=32，梯度裁剪 1，无额外搜索。

checkpoint 用 STOP Macro-NLL 最小；数值保留八位后同分，以 Macro-AP、Macro-AUROC 递减破同分。优化控制的最小改善量为 1e-4；连续 10 轮平台后按 ReduceLROnPlateau 的实现降 lr×0.5，最低 1e-6；在 lr 下限累计至少 30 个实际训练 epoch，且 30 轮无足量改善，才停止。无固定 epoch cap。单次进程超过 24 小时触发资源保护并标为未完成，保留恢复状态，不能进入比较；不能用保护触发的 last 作为完成结果。

跨候选按九折均值 Macro-NLL 最小、同分 AP/AUROC 最大选择最佳数值基线。G 必须同时满足：总体 Macro-NLL 更低，且至少 2/3 次重复的三折平均 Macro-NLL 更低、AP/AUROC 均不低，才可选作唯一主干。否则停止给 G/T 增加复杂度，选数值基线继续低自由度分析。

## 6. R2-C：单参数先验校正

只对 B 选定的唯一主干及其 best checkpoint 使用：

`logits'_c = logits_c - tau * log(pi_c)`，`tau ∈ {0, 0.25, 0.5, 0.75, 1}`。

`pi` 仅由当前**优化 FIT 的唯一受试者**类别人数计算，不含 STOP、内部 validation、外 VAL/TEST，不按扫描频次计算。先逐扫描校正 logits，再 softmax、按人平均；代码用 `log(prob)` 等价实现并保留原 logits。τ 按九折内部 validation 的相同主规则选择，同分取更小 τ。不使用温度搜索、五个 Youden 阈值或 TEST 信息。

预先设定进入校正的描述性排序门槛：至少 2/3 次重复的平均 AUROC>0.55 且 AP>0.20。它是保守的流程门槛，不是显著性检验；未过则保留 τ=0 并报告弱排序，不能靠校正拯救随机模型。T 已停止，不参与此阶段。

## 7. R2-D：唯一主干从头训练

没有复用 B 的权重。六组各九折，最多 **54 次拟合**：

1. no_prompt：纯数值主干，既无患者提示，也无 ROI 文本；
2. roi_only：加入 ROI 条件，仍无患者信息；
3. roi_age_sex：ROI + 年龄/性别，不含 site；
4. full：ROI + 年龄/性别 + site；
5. roi_age_sex_shuffled：同尺寸 ROI 文本打乱；
6. roi_age_sex_random：同尺寸、匹配行范数的随机 ROI 向量。

这是对“无 subject prompt”和“ROI only”重叠措辞的明确实现：第 1 组进一步移除 ROI，构成纯数值锚点；第 2 组是严格的仅 ROI 提示组。两者均从头训练。

年龄/性别采用结构化数值编码，full 增加仅在 FIT 出现过的 site one-hot；未见 site 编码为全零。**不重新调用 LLM，也不把数值编码称作患者文本语义。** site 的真实采集含义仍未核实。

若选出 G，沿用原 ROI 文本注入位置，固定 G 层数与宽度，患者编码经原投影进入；若数值主干胜出，则不新增神经编码器，而用固定 rank-8 ROI 投影构成 `Bᵀ FC B` 的 36 个上三角特征，附加到同一个 Logistic/MLP 输入。打乱与随机对照使用相同特征维度和下游可学习参数量。数值主干中，这只检验固定语义条件是否带来增量，不能外推成原 BrainPrompt-G/T 机制有效。

本轮不引入疾病语义分支、不主张疾病知识增益。ROI 真语义须按同一稳定收益规则超过两个同参数量对照，才有进一步解释的基础；否则只报告条件化或负结果。各提示条件保留 C 已选 τ，同时报告 τ=0；不为每组重新挑 τ。

若 C 后九折 validation 仍全部只预测一个类别，则停止进入 D，保留结果并请求机制审阅；不能把停机伪装成已完成 D。

## 8. 输出、执行和结束状态

上限 **117 次从头拟合**：A18 + B45 + D54；C 不训练。T 的容量检查与启动预检不计入性能矩阵，其权重不复用。调度严格 A→B→C→D；三个设备各运行一个 worker，共享同一封存协议。

每单元输出：训练/STOP 历史、best/last、选中 epoch、停止原因、优化器与 RNG、匿名私有预测、原 logits、概率尺度、temperature=1、FIT 先验和数据/代码哈希。公开报告只含汇总指标：Macro-NLL/NLL、AUROC/AP、Macro-F1、Balanced Accuracy、QWK、每类 Precision/Recall/F1/AP/AUROC、混淆矩阵、真实及预测人数。SMC/LMCI 单列。未预测类别的 precision/F1 按 zero_division=0。

每候选同时保留九折逐折指标、三个重复的折均值和各重复的合并 OOF。选模、τ 和条件比较共用内部验证，因此选择后的最好结果存在乐观偏差，不能包装为独立验证或显著改进。

成功终态为 `AWAIT_KIMI_REVIEW`；错误为 `FAILED_REQUIRES_INSPECTION`，全折单类为 `STOPPED_COLLAPSE_REQUIRES_REVIEW`。到终态后生成内部汇总，交 Kimi 审阅；本执行器不会自动开放任何外折 TEST。是否冻结一个方案做一次探索性外折评估另行决定。
