# 技术决策
每条记录：ID、日期、状态、依据、影响、复查条件。未验证的选择不得写成性能结论。

## D001 / 2026-10-09 / 已确认
在用户选定的 yasuo-coder/AD- 仓库管理 BrainPromptPro。
上下文入口位于根目录，新实现位于 BrainPromptPro/。保留基线来源及版本。

## D002 / 2026-10-09 / 用户约束
输入为 ADNI_1–5 合并数据的 FC 矩阵。
初始化时待核验；现已由D007和数据协议确认五类与MCI→LMCI。

## D003 / 2026-10-09 / 候选方向
研究语义引导自适应脑图、疾病感知读出、学习化 population graph；
允许其他有效候选，以验证集和消融决定最终三个，尚未证明收益。

## D004 / 2026-10-09 / 排除约束
不得采用 BrainPromptPlus 提出的方法。
初始化时需查明；现已完成全文/代码审查，见D009与docs/research。

## D005 / 2026-10-09 / 数据评估约束
不同访视可视为独立样本；保留受试者关联用于跨划分重叠审计。
扫描独立不代表统计独立；不得静默更改用户协议，也不得隐瞒相关性。
如比较 subject-disjoint 协议，另建 split 并分开报告。

## D006 / 2026-10-09 / 工程原则
上下文由版本化文档恢复；长时训练由独立作业管理和 checkpoint 恢复。
STATE 用最新快照，详细决策/实验按 ID 追加，避免无限追加聊天记录。

## 新决策模板
- ID / 日期 / 状态：
- 问题与候选：
- 选择及依据（论文、代码、日志路径）：
- 验证实验与否决条件：
- 影响的文件/协议：
- 复查触发条件：

## D007 / 2026-10-09 / 恢复既有已冻结协议
继承服务器2026-10-07研究协议：ADNI1–5、998个FC样本、MCI→LMCI、固定样本分层798/100/100，训练三seed，验证macroF1选checkpoint。初始化拟议五折和更早十折末轮均不是本轮协议。依据config/protocol/split哈希及实际checkpoint。

## D008 / 2026-10-09 / 非破坏性公开接入
保留原工程运行；独立克隆内复制必要源码/配置和匿名汇总。Pro原无Git历史，用逐文件SHA说明导入来源；BrainPrompt保留官方commit及dirty状态。公开副本外置私有扫描例外清单；原代码不改，历史训练不受接入影响。

## D009 / 2026-10-09 / 方法边界已核验
完整审查本地BrainPrompt+.pdf和官方代码。排除频谱统计文本节点提示、空间FC文本边提示与相关五提示语义图。M1–M5及M3V3的具体信息流已先审查ALLOW；不是领域首创或成功创新的证明。通用readout不自动成为论文贡献。

## D010 / 2026-10-09 / 有限研究结束与冻结规则
保留训练前声明的Final_Selection_Rule。四个三元体系均有完整八配置；Top-3须五指标等级合格、各模块四背景平均F1边际为正且机制/创新证据成立。三元失败不强凑模块，优先合格较简实验候选。测试一次后不改架构、选型、阈值或checkpoint。

## D011 / 2026-10-09 / 迁移兼容性
实际nets/layers/data/cli布局优先于初始化空models/training目录。保留--split与历史入口；验证代码位于根tests，历史相对导入已调整。server_research脚本保持服务器冻结路径，不能误当跨机器免配置入口。公开副本的历史路径配置保留为模板，使用前明确映射私有工件，不自动移动数据。

## D012 / 2026-10-10 / 七批独立研究
用户新任务使用独立分支、数据版本、划分与输出。五批已冻结选择不变；旧test未运行。

## D013 / 2026-10-10 / Pearson与采集配对
复用清洗时序统一经验Pearson；不能与既存收缩相关混用。同访视不同采集日期分别保留。QC失败不补造，原998FC保持相同。

## D014 / 2026-10-10 / P5固定编码器
初版query与TRAIN reference都由同seed新A0固定checkpoint编码；分类主干独立训练，禁过期参考缓存与直接current-FC value捷径；报告额外编码器成本。

## D015 / 2026-10-10 / P2数值诊断候选，尚未证明有效
原始dot、mask.05、40轮TRAIN预训练的验证重建MSE .056895，TRAIN逐连接均值对照 .035413；预测标准差 .02538、相关性 .06303，接近常数。初始tanh饱和比例 .98105。依据受控 probes/P2_pretrain_rec_05.json，追加两个有限候选：仅解码器内对ROI特征LayerNorm后减ROI均值；dot用可学习缩放cosine、初始scale .1，pair用同一中心化输入。两候选均mask.10、seed42、40轮，原始7项负面结果保留；不修改运行中的代码或A0。先比较验证重建和TRAIN均值，再验证分类收益。P3同时登记random、无语义learned_query和均值MLP近参数匹配对照；这些仅控制新P3 ROI查询，原基线疾病辅助损失保持不变。

## D016 / 2026-10-10 / 补齐P2分类机制对照
原始七项预训练的验证重建均劣于TRAIN逐连接均值，不能把下降的训练loss称作有效预训练。但重建失败本身不能替代分类验证；按用户要求登记joint rec-only、joint cons-only、固定40轮rec10/cons10后finetune四项seed42对照。权重/优化预算沿用共同协议，额外40轮成本单列。与A1 joint full及A0比较后再决定配方，不直接给八组合使用未经验证的修订；test保持封存。

## D017 / 2026-10-10 / 仅预训练Dropout诊断
归一化pair验证MSE .111547；TRAIN-only四模式探针：eval .114435、train .049894、固定BN统计但开启Dropout .050739、batch统计且关闭Dropout .219797。Dropout状态变化具有明显影响，BN也可能交互，不能只怪验证过拟合。新增一个受限候选normalized pair / mask.10 / TRAIN dropout0 / 40轮 / seed42，原始结果保留；监督训练dropout仍.5。证据N/probes/pretrain_modes_normalized_pair.json。候选尚未证明有效，不能把关闭Dropout推为全模型最优配置。

## D018 / 2026-10-10 / 预训练checkpoint敏感性对照
no-dropout normalized-pair的验证重建：best_training MSE .036301，固定第40轮last MSE .028170，TRAIN边均值 .035461；last相关性 .75434，高于均值 .61381。两checkpoint均由既有探针预先计算，未使用test。不能只报告较好last并隐去best失败；登记best/last两个下游微调对照，原best规则的在跑任务不改。分类益处尚未证明。新增排队依赖，仅在Phase3全部正常退出后使用GPU4/6运行此两项，防止与既有队列争用或重复启动。

## D019 / 2026-10-11 / 固定三seed完整消融配方
Phase1–4的21分类/10预训练全部结束，21分类独立复算通过。P2最佳重建候选的两种checkpoint均未改善下游分类，停止预训练扩搜。原A1 joint full、A2 semantic class token、A3 learned reliability单seedF1均未超过A0，不能宣布成功。为完成用户指定八组合及交互判断，保持原登记完整配方：P2 mask.10/dot、rec.1+cons.1、监督dropout.5；P3语义逐类token；P5同seed冻结A0/TRAIN-only/top16/tau.05。固定图或rec-only单列对照，不改名替代完整模块。
A0–A7和LegacyPro/HistoricalM2/HistoricalM3v3/HistoricalM3v3_no_metadata/P3_parameter_matched/P3_random/P3_learned_query/P5_fixed/P5_no_reliability/P2_joint_rec_only/P2_joint_cons_only各3seed，共57项，17项已有且40项补齐。使用现存已验证不可变29f34f9源码；不改数据/划分/训练预算/停训条件。recipe_freeze.private.json在Phase5启动前写入，不是最终selection_lock。全部仅TRAIN/VAL；待完整配对结果再决定保留模块，失败方向保留。

## D020 / 2026-10-11 / 机制复核与受试者隔离补充
25个checkpoint机制探针（7方法三seed及4项seed42）复核完成；P2全面鲁棒性/P3相对参数对照/P5连续边权收益均未证明。CPU重载P5_no_reliability偏差保留；原CUDA复核通过，未降低阈值。主配置仍按D019，不根据诊断中途修改在跑实验。
冻结subject-disjoint划分在dataset_v1既有文件，启动A0三seed补充基线，GPU1/2/5、独立subject_experiments、--split1，避免闲置资源；不重划分、不混合指标、不改变主模型选型口径。主矩阵完成后决定有证据的候选补充比较。主study尚无selection_lock，仍不评估test。

## D021 / 2026-10-11 / 验证集选型与冻结测试范围
在TEST尚未访问时明确最终比较规则：完整A0–A7各三seed均完成并独立复核后，按三seed平均Macro-F1选择主实验候选；严格平局依次用平均Balanced Accuracy、平均AUC、更少参数。此规则属于观察验证结果后的模型选型，不能倒称训练前预注册。候选仅是待holdout核验对象，不因少量或部分指标提升就宣布有效创新。
主候选补充运行既有subject-disjoint协议三seed，与已完成A0单独比较；该补充不用于改变主候选。所有TRAIN/VAL开发结束后封存精确checkpoint/source/dataset/split/evaluator hash。冻结测试范围为登记的主协议19方法各三seed（A0–A7+11关键对照），及subject协议A0与主候选；这些比较对象在test前固定，不按test高低换冠军。保留P2四项单seed微调为验证探索，不混入三seed正式test矩阵。
测试入口默认validation，真实test需要完整selection_lock和显式锁SHA，先复核VAL预测；拒绝覆盖输出。所有后续test结果只能用于固定比较和报告，不继续本研究调参或用test选模型。历史五批测试锁不变。

## D022 / 2026-10-11 / 主矩阵完成与隔离范围审计
61份分类主协议结果独立核验完成，其中57项为19方法各3seed。A1按D021入选研究候选，F1仅较A0高约0.60个百分点且三项主指标较低；A7 F1约33.00%，P2/P3/P5平均边际均为负，拒绝“三个稳定有效方向”的结论。Phase7仅训练冻结subject协议下A1三个seed，不更改主要候选。
交叉角色审计确认sample测试样本有部分处于subject协议的TRAIN/VAL，反向亦然。因此test封存及TRAIN-only等声明限定各自协议；不能宣称七批全部数据在项目开发中未被使用。完整私有交叉表留N/reports，最终锁绑定其hash；公开只写清边界，不上传成员明细。该审计不改变已冻结划分，也不使subject结果参与主选型。
