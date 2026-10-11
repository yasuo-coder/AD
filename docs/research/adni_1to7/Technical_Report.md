# ADNI_1–7 P2/P3/P5：实现与验证说明

本研究是独立的七批 FC 研究，历史五批训练和测试锁保持原状。本文描述实际代码和已核验机制，不能据此宣称三个有效创新；结果以完整三seed验证表、配对消融和冻结后的评估记录为准。

## 1. 数据与输入

ADNI_1–7纳入1323次rs-fMRI acquisition，对应895位生物学受试者。类别顺序CN/AD/SMC/EMCI/LMCI，数量850/65/39/61/308；AD占4.913%。下载CSV的Group为既定标签来源，MCI并入LMCI。不同访视和不同日期的有效采集保留；同一扫描、清洗时序和FC内容经过重复审计。原CSV同时包含T1等影像行，不能将影像行数当FC样本数。

各批有效FC为200/198/200/200/200/198/127；排除项及冲突裁决证据留在受控审计。统一从已有清洗时序计算经验Pearson相关，保持AAL116顺序、正负相关与float32；没有混入Nilearn收缩相关。FC、标签和文本与历史纳入的998项逐项对齐，新增批次使用相同流程。

实际输入包括FC[B,116,116]、ROI语义[116,2048]、受试者描述[B,2048]和固定五疾病描述[5,2048]。所有五个疾病query同时使用，不能按真实诊断选择query。Source_Dataset只用于追踪；Sample_ID只用于reference自排除，不能作为预测特征。历史site字段的实际采集站点含义未独立确认；P5不额外加入metadata边，但原基线受试者prompt仍保留既定信息。

## 2. 基线与公共骨干

`nets/connectome_p2p3p5.py`通过`OriginalBrainPromptAdapter`复用官方GCN、提示映射、疾病辅助分类及预测头。隐藏维116、4层GCN、监督Dropout .5；三开关关闭时保持适配基线的计算。CPU等价与CUDA容差检查已通过。官方population构图的sigmoid乘积与.8阈值造成当前配置空图，这一既有行为已审计，不通过修改基线来制造新模块收益。

比较方法包括A0原BrainPrompt适配、LegacyPro、历史M2、历史M3v3及其no_metadata对照。原工程不移动、不覆盖；历史代码来源及许可由仓库已有来源记录保留。七批采用统一的新训练流程，不能把这里的分数称为官方十折末轮复现分数。

## 3. P2：数值掩码学习

对FC上三角非对角连接以比例ρ采样隐藏位置，镜像到下三角；对角与已有无效位置按契约处理。隐藏连接同时从节点FC行和构图输入中移除，避免用原始拓扑暴露目标。额外visibility投影以零权重初始化，区分真实零与人为隐藏。常规ρ=.10，有限预训练探索覆盖.05/.10/.15。

连续解码器预测有符号连接，损失仅在隐藏上三角计算：

`L_rec = mean_hidden (FC_hat_ij - FC_ij)^2`。

原始dot使用投影后的ROI内积与tanh；pair解码器利用对称成对特征。诊断候选在解码器内LayerNorm并减ROI均值，normalized-dot采用有界相似度和可学习缩放。它们不改变FC目标，也不把所有负边截成正数。

TRAIN-only预训练固定40轮；best按TRAIN损失选择，last为固定第40轮。预训练一致性使用两掩码视图归一化池化表示的平方差。监督joint模式使用clean分类，附加`0.1 L_rec + 0.1 symmetric_KL(masked_logits_1, masked_logits_2)`；rec-only和cons-only分开登记。finetune载入GCN、visibility与解码器，分类阶段不添加预训练重建目标。

已验证事实：原始7项及2项中心化候选的验证重建均未优于TRAIN逐边均值。仅预训练关闭Dropout的pair候选在固定last40改善了重建，但两种checkpoint微调分类均较差，停止扩搜。三seed掩码探针显示完整P2只在部分扰动水平更好，未证明全面鲁棒性。详见P2_Reconstruction_Diagnostics、Mechanism_ThreeSeed。

## 4. P3：类别任务token

P3复用历史M2的DiseaseTokenReadout，不将已有疾病prompt重新包装成独立首创。设ROI特征H[B,116,116]、五类文本P[5,2048]：

`q_c = normalize(W_q (P_c - mean_class(P)))`，query宽度32；
`k_i = normalize(W_k (H_i - mean_ROI(H)))`；
`a_ci = softmax_i(4 q_c^T k_i)`；
`t_c = Σ_i a_ci W_v(H_i - mean_ROI(H))`。

输出token为[B,5,116]，并非[B,5,32]；32是query/key维度。历史残差读出继续使用五token拼接；新增类别c的logit只由`t_c`与对应class_weight产生，保留类别身份。真实标签只进入损失，不进入query选择。

random固定随机prompt、learned_query取消疾病语义、mean-MLP参数匹配均已实现。参数匹配不是新增疾病query：语义模块150801参数，MLP150775参数，差异小于0.1%，没有用未参与前向的参数凑数。三个对照仍保留原BrainPrompt的疾病辅助分类，以隔离新增读出的作用。

已验证事实：五类class_weight均有梯度，ROI注意力和token存在差异；不能将其解释为疾病相关性已验证。语义token的三seedMacro-F1高于random/learned-query，但没有稳定超过基线或历史/参数对照。完整背景边际以A2−A0、A4−A1、A6−A3、A7−A5判断。

## 5. P5：稳定性代理与TRAIN参考关系

P5的query和reference均由同训练seed的正式A0 checkpoint固定编码，不使用正在变化的分类器特征缓存。参考成员等于当前协议的TRAIN，版本绑定encoder checkpoint SHA；禁query-query传播，禁直接把当前FC作为value残差。

对三种固定对称扰动视图的归一化表示z，使用
`v = mean_k ||z_k - mean_view(z)||²`，`r = exp(-v / .05)`。
这是表示稳定性代理，不是经过临床标注验证的扫描质量或疾病置信度。扰动由固定seed31013生成，对所有query一致且与batch无关。

关系得分：有界可学习对角metric `1+.5*tanh(γ)`作用于query/reference，再取cosine；
`score_ij = cosine_ij/.5 + tanh(MLP(cosine_ij,r_i,r_j)) + log(max(r_j,.05))`。
按分数确定性选择top16并softmax。去除当前TRAIN样本自身后，以`(attention - uniform_allowed) @ reference`作为消息；线性value和有界增益形成残差，加入原融合特征。不能使用query-only的共同标量声称改变归一化后的邻居相对权重。

fixed关闭学习metric/scorer及可靠性；no_reliability保留学习得分但把可靠性输入置1；两者独立训练。无metadata默认方案与历史M3v3/no_metadata对照共同报告。

已验证事实：TRAIN参考重算、版本/成员隔离、batch independence通过。当前三个seed中可靠性偏高、top16内边权接近均匀，metric几乎未改变；population残差约占融合表示范数0.29%。拟合后移除分支对硬预测影响很小。其绝对有界残差与融合表示尺度的差异、微小edge梯度是弱作用的直接证据；不能据此断言任何population模型均无效。固定图对照优于完整P5的现象必须保留。

## 6. 训练、指标与时间成本

统一Adam lr5e-5、weight decay .005、batch20、梯度clip5；最多400轮，验证patience100且min_epochs120，loss驱动LR patience25/factor.5，LR低于1e-6也可提前停止。实际checkpoint按验证Macro-F1、固定综合平局规则、较低loss选择；不是最后一轮。新runner的共同分类目标为两个TRAIN计数平方根逆频率加权CE，疾病辅助权重1。旧config的graph sparse字段未在新runner使用，不把遗留字段误写成实际loss。

训练seed42/3407/2026共享split seed1729；mask/loader RNG分别为训练seed+1000003/+2000003。保存模型、优化器、scheduler和全部RNG，恢复要求source/config/split/依赖hash一致。计时含训练循环与逐轮验证，初始数据准备和P5 bank编码不在训练循环计时中；预训练额外40轮单列。共享GPU负载下不作严格速度benchmark。

P2在joint中增加掩码GCN前向；P5每个query含clean加三扰动的固定encoder计算，在联合P2时重复于各视图，因此完整组合更昂贵。主要pairwise连接或reference打分规模分别为O(BN²)、O(BTD)，N=116、T为TRAIN参考数；这仅是主要张量规模，不是整网严格FLOP结论。实际参数/时间由每run产物报告。

报告Accuracy、Macro-P/R/F1、Balanced Accuracy、Weighted-F1、Macro OvR AUC、逐类P/R/F1、预测分布及塌缩。五类均存在时Macro-R与Balanced Accuracy相同。三seedSD只描述训练随机性，不是患者置信区间；类别稀少使单例改变产生较大Recall跳变。混淆矩阵和逐例预测在私有结果，公开仅匿名白名单。

## 7. 协议与泛化边界

主协议为固定sample-level holdout，允许不同访视独立纳入，同时存在同人生物学关联跨split。历史五批及四批开发暴露已审计；本研究的holdout不能称为未见外部验证。另有冻结subject-disjoint协议，当前A0三seed验证Macro-F1约26.82%，主协议约35.63%；不能把二者差值全归因同人重叠，因为划分样本构成也不同。补充协议单独汇总，不改变主协议或暗中筛选主要模型。两独立划分之间存在样本角色交叉；TRAIN-only与test封存只按各自协议定义，不能宣称整个开发历史从未使用过这些FC。

参数/模块/checkpoint只依据TRAIN/VAL。最终评估入口默认只做VAL重载；TEST须有selection_lock、精确锁SHA、checkpoint/dataset/split/evaluator SHA绑定和全新指定输出。预先重载VAL不一致时拒绝TEST；部分输出不能自动覆盖重跑。锁建立前不构建TEST loader。

## 8. 来源、复用与排除边界

BrainPrompt保留来源与许可；P3明确复用历史M2，P5复用TRAIN-only/self-mask等契约并采用独立的稳定性与连续关系设计。ncomms2025启发与代码实现差异已在METHOD_AUDIT审查，未复制其AGPL代码，也未将缺模态任务说成FC掩码任务。GraphMAE、ST-JEMA、SpaRG等相关研究已纳入边界，不能宣称领域首创。

排除BrainPrompt+的频谱统计文本节点、FC空间连接文本边及五提示语义图；本次P2/P5为数值机制。所有实现的新颖性与有效性需分别判断，模块可运行或注意力有差异均不能直接成为论文贡献。

## 9. 验证和可复现证据

20模块单测、CPU/CUDA八组合、真实TRAIN反向、断点RNG一致性及16份短程入口smoke已有记录。3项配对统计测试检查交互、行顺序和缺失/重复seed；4项指标篡改拒绝检查防止伪正确汇总。新增冻结评估器4项锁契约测试，20个代表性正式checkpoint的CUDA VAL重载均通过。所有这些与真实性能实验分开登记。

代码、哈希、运行目录、恢复命令见STATE、EXPERIMENTS和RUNBOOK。原始影像、FC、样本身份、完整划分、权重和逐例概率不上传GitHub。任何仍在运行或未做的测试均明示状态，不补造数字。

## 10. 已完成的冻结评估
63项TEST和逐预测身份/标签/计数公式复核全部通过，当前无本研究作业。完整新版A7主TEST Macro-F1 27.84%，A0 28.33%；候选A1 28.25%。subject A1为23.75%、A0为21.77%，但部分指标/seed未提高。上述实现与机制检查不构成有效优化证明；最终表格、十项验收判断及局限见[Final_Report](Final_Report.md)。原始失败日志保留；不再依据本轮TEST更改模型。
