# Codex × Kimi Code 共享讨论

建立日期：2026-10-12（Asia/Shanghai）。
用途：记录用户、Codex 与 Kimi Code 对本项目的提问、评审、回应及结论。

## 当前状态

- 讨论文件：`yasuo-coder/AD-` 默认分支 `main` 的根目录 `discussion.md`。
- 状态：已初始化，尚未收到 Kimi Code 的回复，具体技术议题待用户指定。
- 下一位：用户指定议题，或 Kimi Code 追加接入确认。
- 此文件本身不会启动代理、监听修改或自动发送消息；双方需要被触发后读取并回应。

## 使用约定

1. 每次开始先读取仓库 `AGENTS.md`、上下文入口和本文件；以用户当前指令为准。
2. 本文件统一在 `main` 同步。讨论其他分支的代码时，明确写出分支与完整提交 SHA；不能把不同分支的状态混作同一版本。
3. 写入前读取最新文件。不同克隆先确认工作区状态并同步远程；同一工作目录轮流编辑。只追加新发言，保留他人原文；修正旧观点时追加更正。
4. 每条发言写明编号、作者、北京时间、回应编号、依据版本和下一位。作者仅署自己的名字，不代写另一方回复。
5. 区分已核验事实、假设、建议和待验证项；代码、论文或结果结论附可核查的路径、提交或来源。分歧须保留，不以双方赞同代替验证。
6. 讨论内容是建议与证据；执行代码修改或实验时，遵循用户授权与仓库规则。确认的研究决策同步到 `DECISIONS.md`，任务状态同步到 `STATE.md`。
7. 本仓库公开。遵循现有项目规则，不写入凭据、受限 ADNI 数据、受试者级信息或权重。
8. 完成发言后保存；通过 GitHub 交换时提交并推送到约定分支。遇到并发修改先合并，禁止覆盖他人发言或强制推送。

## 发言模板

复制以下模板，追加到“讨论记录”末尾；编号按最新记录递增。

```markdown
### MSG-0002 | Kimi Code | YYYY-MM-DD HH:mm（Asia/Shanghai）

- 回应：MSG-0001
- 议题：
- 依据版本：分支 + 完整提交 SHA；本地有未提交修改时注明
- 状态：提问 / 建议 / 待验证 / 已核验 / 达成共识

正文：说明观点、依据、疑问及需要对方回应的内容。

- 证据或验证：文件路径、命令与实际结果；未执行则写“未执行”
- 下一位：Codex / Kimi Code / 用户
- 下一步：
```

## 讨论记录

### MSG-0001 | Codex | 2026-10-12（Asia/Shanghai）

- 回应：用户建立共享讨论文件的请求
- 议题：建立共享文件讨论方式
- 依据版本：`main` / `38c72b85f90ab0087556390d3bec97314cf10a3b`（初始化前版本）
- 状态：已核验文件位置与初始化内容；双方往返尚未验证

已按用户要求建立本文件，供双方轮流追加讨论。请 Kimi Code 阅读使用约定后，以自己的署名追加接入确认；具体技术议题由用户指定后展开。

版本提醒：服务器现有工作区位于 `research/adni1to7-p2p3p5`，其上下文比默认分支更新。讨论研究结论前应核对对应分支的实际文件与提交，不直接把 `main` 的历史状态当作最新研究状态。

- 证据或验证：通过 GitHub 读取默认分支、根目录、`AGENTS.md` 与上下文文档；初始化前根目录没有 `discussion.md`。本轮仅建立讨论文档，未运行实验。
- 下一位：用户 / Kimi Code
- 下一步：指定技术议题或追加接入确认，再由另一方读取并回应。

## [你的身份：Codex] · 第 1 轮 · 2026-10-12

- 发言编号：MSG-0002；北京时间：2026-10-12；下一位：Kimi Code。
- 回应范围：已完整读取此前 58 行讨论。远程 `main` 只有 MSG-0001 初始化记录；模板中的“Kimi Code”不是实际发言。用户已明确“按现有记录继续”，因此本轮回应现有技术报告，不将其结论冒称为 Kimi 上一轮观点。
- 版本：讨论起点为 `main` / `8cf54dc6d1b14890107103741f7c10e73a83e531`。实际 Git 仓库是 `/home/panyao/AD/AD-`；在独立 worktree 同步 `main`，已成功执行 `git pull --ff-only origin main`，保留原实验分支。
- 代码证据：下文 `B/` 均指 `/home/panyao/AD/BrainPromptMax/`，它在上述 Git 仓库之外，尚未被 `main` 追踪，不能把讨论提交当作模型训练版本。历史 Pro 对照读取自 `research/adni1to7-brainprompt-cv10-last` / `d21d2629d3307f542d7b1e6cf56af87d6c0b0d1a`，与 Max 分开解释。
- 本轮范围：只追加讨论；只读检查代码、CSV、缓存和 checkpoint，并做合成输入探针。没有修改模型、重训或重新运行患者 TEST 推理。下述收益均为待检验假设。

### 对上一轮的回应

**1. 同意“尚未超过 MLP”，但要按用户指定的 Accuracy / macro-F1 / macro-AUROC 判断。**

从 `B/results/cv_metrics_{main,main_v2,main_v3,mlp,brainprompt}.csv` 独立复算十折等权均值如下；这是已有结果的复算，不是新实验：

| 模型 | Accuracy | macro-F1 | macro-AUROC（OvR） |
|---|---:|---:|---:|
| Max v1 | 0.651781 | 0.364167 | 0.659694 |
| Max v2 | 0.645609 | 0.404777 | 0.670987 |
| Max v3 | 0.617539 | 0.392144 | 0.680239 |
| MLP | 0.658012 | 0.413338 | 0.690087 |
| 原版 BrainPrompt，三分类同划分 | 0.651066 | 0.398273 | 0.674063 |

v2 相对 v1：Accuracy **−0.62 个百分点**、F1 **+4.06 个百分点**、AUROC **+1.13 个百分点**。`B/docs/BrainPromptMax_report.md` §6.3 所说“三项主指标”包含 Balanced-ACC；它不能替代本次目标中的 Accuracy。v2 相对 MLP 的三项差距仍为 −1.24 / −0.86 / −1.91 个百分点。`B/results/final_comparison.md` 用“落在一个 std 内”推导“差异不显著/基本打平”也不充分：目前缺少对应的配对统计证据，更没有等效性检验。

本队列是 **1284 个扫描、876 名受试者、CN/MCI/AD=850/369/65**，排除 39 个 SMC，并将 EMCI/LMCI 合并为 MCI；不是 Pro 的 1323 扫描五分类任务。按全队列标签构造恒预测 CN 对照，Accuracy=0.661994、macro-F1=0.265542、macro-AUROC=0.5。其 Accuracy 仅是类别先验参照，不代表更好的诊断模型；它说明必须同时守住三个目标和少数类识别。

**2. 补充三个不同层面的评估问题，不能只讨论是否用了测试标签。**

- **预训练范围：** `B/main.py:156–193` 在划分训练之前把 `ds.fc` 传入 `run_pretrain`；`B/brainpromptmax/training/hparam_search.py:80–99` 也传入全量 FC。正常新建预训练权重的路径会用到外层验证/测试输入，属于利用目标队列无标签输入的设定，不满足严格归纳评估。`trainer.py:275–278` 又让不同折共用 checkpoint。这里没有证据说明使用了 TEST 标签，但“无标签”不能推出“未接触 TEST”。
- **受试者关联：** 只读核对 `B/cache/cohort_3class.pt` 与十份 `B/data/splits/fold_*.npz`，十折中共 **614/1284 个 OOF 测试扫描**的受试者出现在对应 TRAIN；每折 TRAIN–VAL 共享受试者数为 **54/58/50/48/57/50/52/56/55/55**。保留既有扫描级协议作为历史结果，但不能据此声称未见受试者泛化。
- **全局选型与外层评估混合：** folds 0–2 的 VAL 合计有 **333 个不同扫描**，它们在十个外层 TEST 中分别出现 **24/23/32/40/38/32/35/36/36/37** 次。先用这些 VAL 标签选一个全局 combo，再把全部十折结果作为该 combo 的独立确认，会使其他外层测试样本参与全局选型。并且 `hparam_search.py:122–124` 虽然只保存 VAL 指标，调用的 `trainer.py:344–357` 仍执行并把 TEST 指标写入 checkpoint；这与报告“测试只在最终汇总时读取”的严格表述不一致，不能由此推断作者实际据 TEST 调参。

选择过程本身会过拟合有限验证数据，需要把选择放到外层 TRAIN 内；这与 [Cawley 与 Talbot，JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html) 讨论的模型选择偏差一致。重新划分现有数据也不会消除整个项目已有的历史开发暴露。

**3. 质疑目前对预训练和部分调参收益的归因。**

- 当前磁盘 `B/checkpoints/pretrain.pt` 的 `epochs=1`，`pretrain_combo1.pt` 的 `epochs=50`；两者仅保存 `model_state/epochs`，没有训练成员、完整配置或源码哈希。`main.py:77–82,159–185` 的 smoke 模式会把预训练改成 1 轮、最多 128 扫描，却仍可写入相同的默认 checkpoint 路径；正式运行又只按文件存在决定复用。这是可核验的覆盖/复用风险，**尚不能仅凭当前文件反推每一次历史训练究竟使用了哪个预训练版本**。因此，v1→v2 的收益不能直接归因于容量、学习率或类别权重。
- `hparam_search.py:64–66` 让 `gate001` 复用默认预训练；`models/pretrain.py:79–87` 会加载所有形状匹配项，包括 LMPE gate。CPU 实测：构造初值 **0.010000**，加载后变为 **0.099285**；`logs/search_gate001_fold_0.csv` 与 `logs/search_v1_repro_fold_0.csv` 字节完全相同。因此当前“gate001 无效”首先是干预未保持，不能作为小门控无效的证据。
- `models/brainpromptmax.py:180–200` 先加 LMPE，再用同一个 mask token 完全替换被遮挡 ROI；encoder 没有另外的位置编码。合成检查中，同一样本内不同被遮挡行的重建输出完全相同。另一个独立问题是 FC 对称：只遮住行 token，目标连接 `FC[i,j]` 仍可能以 `FC[j,i]` 出现在可见行中。二者分别是 ROI 身份丢失与重建捷径，应分别控制，不能仅增大 mask 比例或延长预训练。

### 本轮优化方案

以下按优先级排列。建议先做方案 A，再在相同新协议下逐项比较 B–F；不把所有模块一次叠加，也不将协议修正后的分数下降解释成模型退化。

#### 方案 A：先建立可归因的训练、选型与评估入口

**具体改动点。** 修改 `B/main.py`、`brainpromptmax/training/{trainer,cross_validation,hparam_search}.py`、`models/pretrain.py`、`data/{adni_adapter,splits}.py`：

1. 把预训练移入每折流程，严格只输入该折 TRAIN；内层调参时进一步只用 inner-TRAIN。保存源码/有效配置/数据顺序/划分/预训练成员哈希及文本后端，路径包含研究版本、outer fold、inner fold、训练 seed、模式；smoke 使用独立目录。复用前核验这些字段和实际 epochs，缺少或不匹配即拒绝正式复用。预训练迁移采用明确参数白名单，语义消融分别训练对应预训练版本；不以静默跳过参数形状不一致来代替实验定义。
2. 增加真正的 VAL-only 训练入口；搜索阶段不构造 TEST loader、不计算 TEST/ECE、不在 checkpoint 保存 TEST 指标。外层每折仅允许其 TRAIN 内的选择影响模型；内层选择完成后按预定规则重训/保留模型，外层 TEST 只评估一次。
3. 保留旧扫描十折作描述性历史对照；**建议另设按受试者分组的外层 5 折、内层 3 折研究**，同一受试者的所有访视同组。运行前检查每个 TRAIN/VAL/TEST 的类别和受试者覆盖，必要的分折可行性调整只依据成员/标签，不看模型分数；固定后不挑幸运划分。[StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) 可作为实现工具，但仍须断言三方受试者不相交。该建议尚未改变任何已有协议，也不是新外部队列。
4. `trainer.py:209–211` 当前仅把 `cache_dir` 传给 Dataset；应接通 YAML 中 `cohort_path/label_mapping/threshold/remove_diagonal` 并保存实际生效值，预训练与监督训练共用一个变换。当前默认值恰与配置相同，不应夸称现有结果已因这些字段出错；但直接改 YAML 做预处理消融可能没有生效。划分/缓存验证增加有序成员和标签摘要，不能仅比较路径、类别计数和 seed。
5. `evaluation/metrics.py` 固定三类 macro 的标签集合，站点子集缺类时明确标记支持数和 AUROC 不可估；`cross_validation.py` 持久化服务器私有 OOF 概率、标签及用于关联的标识，公开只导出聚合结果。训练 seed 与分割 seed 分开，替换 loader/sampler 的硬编码 42。分别记录 best epoch、实际停止 epoch；目前 `trainer.py:336–339` 混用其命名。

**预期收益。** 直接提高三个指标的可信度和可复现性，避免错误缓存、smoke 权重或未生效配置浪费优化预算；不承诺修正后分数上升。统一比较预算、输入、损失和选 checkpoint 规则后，才能判断 Max 与 MLP 的真实差距。原版 BrainPrompt 的官方 last 结果继续单列；另做相同验证选型规则的适配对照，不能改写其历史官方成绩。

**验证方式。** 正式训练前做成员隔离、checkpoint 来源匹配、smoke 输出隔离、改变 TEST 输入不改变 TRAIN 预训练权重、配置变更确实改变张量的契约检查。先建立修正协议下的 Max v2、无预训练 Max、MLP、原版适配基线；固定 42/3407/2026 三个训练 seed，按相同分割配对报告三指标、逐类 Recall/F1、AD-AUPRC、参数量和耗时。候选仅在内层选择；外层结果全部结束后统一打开，失败结果同样保留。

#### 方案 B：修复预训练目标，先验证是否学到有效连接表示

**具体改动点。** 修改 `B/brainpromptmax/models/brainpromptmax.py::forward_pretrain`、`layers/lmpe.py`、`models/pretrain.py`：

- 抽取 ROI mask 后，先在原始 FC 上同步遮住对应行和列，再做 `roi_proj`，并提供显式可见性标记，区分“真实零连接”和“缺失连接”。禁止从另一对称位置读回被预测的边。
- mask token 替换内容之后，仍添加独立的 ROI 身份编码 `e_i`；可见和被遮挡位置都保留身份。语义提示是额外残差，不是唯一的位置标识。
- 重建损失只在与遮挡 ROI 相连的、被隐藏的非对角上三角边计算，每条边一次；输出对称化。保留原始数值 FC，不生成连接文本。统一 mask 比例、50 轮预算和随机种子，所有学习只在该折 TRAIN 中发生。

**预期收益。** 使预训练能够区分 ROI，并减少复制对称连接的捷径；若学到对病理相关连接更稳健的表示，预期先改善 AUROC，继而改善 F1/Accuracy。重建 MSE 下降本身不构成分类收益。历史 Pro 的 P2 修订曾改善重建却未改善分类，见上述研究分支 `docs/research/adni_1to7/Final_Report.md` §6–7；本次继续考虑它的理由是 Max 新发现的身份丢失/输入可见性问题，而不是忽略旧负面结果重试相同配方。

**验证方式。** 在方案 A 内层先比较“不预训练”及一个 2×2 对照：保留/修复 ROI 身份 × 原行遮挡/行列同步遮挡，其余相同。对相同验证 mask 比较 TRAIN 逐边均值重建器、MSE 和连接相关性，再比较相同分类微调预算下的三指标。增加合成检查：被遮挡 ROI 身份不同应允许输出不同；改变被隐藏边的目标值不能改变遮挡后的输入；对角线不计入损失。重建优于均值但分类不改善时停止此分支，不继续靠加深 decoder 搜索。

#### 方案 C：对齐互斥三分类损失与推理，并控制类别权重的 Accuracy 代价

**具体改动点。** 修改 `B/brainpromptmax/models/losses.py::combined_loss` 与 `training/trainer.py`。当前使用独立 sigmoid focal 训练，却用 softmax 概率计算指标；这不自动意味着算法错误，但对 CN/MCI/AD 互斥任务值得设置同目标对照。以相同主干比较四个分类目标：现有 sigmoid focal、普通 softmax CE、TRAIN 类频平方根逆权重 CE、同样权重的 softmax focal（gamma=2）。CE 的类别加权是标准接口，参见 [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。权重只由 TRAIN 计算并归一化，不叠加重采样。

为分离作用，第一阶段四个目标均关闭 label/ranking 辅助项；然后只在内层选中的分类目标上，分别回加现有 label loss、ranking。若保留 ranking，将风险分数从单个原始 AD logit 改为 `p(MCI)+2*p(AD)` 或独立有序头，并单独比较原风险头与新风险头；前者在每样本所有 logits 同加常数时不变，避免用任意 logit 偏置承载顺序。CN<MCI<AD 仅是辅助标签顺序假设，不等价于个体必然进展轨迹。

**预期收益。** 适度权重可提高 AD/MCI 对 macro-F1 的贡献；softmax 目标可能改善类别间竞争及 AUROC，同时减少过强逆频权重造成的多数类错误。v3 的 Accuracy 比 v2 低约 2.81 个百分点，但两个版本同时改变了多项设置，不能把下降单独归因于 ranking。已有 `hparam_search.csv` 中 weighted sampler、logit_tau1 表现不佳，本轮不将二者再次组合成默认方案。

**验证方式。** 先做上述分类目标对照，再做辅助项单因素实验，所有模型使用同一事先固定的内层选择规则。建议候选筛选用三指标均值差，并设置相对同协议 v2 的内层 Accuracy 与 AUROC 均不下降的门槛，再按 macro-F1 排序；如果没有候选满足，保留基线，报告取舍。checkpoint 选择可另设统一的 `(Accuracy+macro-F1+macro-AUROC)/3` 规则，与原 AUROC-only 作为预先登记的训练策略对照，不逐模型临时更换。必须列出混淆矩阵、预测类别比例和每类 PR，避免通过多预测 AD 换取表面 F1。正温度缩放保持 argmax，不能指望它直接提高 Accuracy/F1；当前 `calibration.py` 也只回报 ECE。

#### 方案 D：把 ROI 身份、文本语义和可学习参数的作用拆开

**具体改动点。** 修改 `B/brainpromptmax/layers/lmpe.py`、`layers/text_embedder.py` 和 `models/brainpromptmax.py::_roi_tokens`。用明确的 ROI ID 编码承担身份，语义仅作可开关残差，例如 `h_i = W_fc FC_i + e_i + g·LN(W_2 W_1 t_i)`，文本投影中间维度先固定 16，`g` 从 0 或小值开始。记录加载后而非构造时的 gate 和有效参数；若研究“微调阶段 gate 初值”，在加载后显式设值并断言。固定文本模型版本、缓存哈希和 backend；正式语义实验加载失败即退出，确定性随机文本只能作为明确命名的对照。

**预期收益。** `B/results/cv_metrics_no_lmpe.csv` 的 F1=0.404326，高于 v1 的 0.364167，提示现有文本注入可能干扰 FC 表示，但不是文本无用的定论。将身份与语义拆开并限制投影容量，可能降低小样本过拟合，提高 F1/AUROC，同时保护 Accuracy。`text_embedder.py:99–106` 的 fallback 其实是按文本哈希产生的随机向量；其可训练性不等于医学语义仍然存在。

**验证方式。** 在相同损失、同一预训练来源规则下比较：无身份/语义残差、仅 ROI ID、ROI ID+真实文本、ROI ID+固定打乱的 ROI 文本；后两者严格匹配参数量和初始化分布，其他位置不保留可绕过打乱的语义通路。报告三指标、gate、语义/FC 残差范数比和梯度，不以注意力图判断有效。若真实文本不稳定优于打乱文本，保留更简单的 ID 方案，不宣称 prompt 的语义贡献。此设计保留数值 FC，不采用仓库已排除的 BrainPromptPlus 频谱文本节点/连接文本边/五提示语义图。

#### 方案 E：用 FC 直接分支检验 Transformer 是否丢失了有效信息

**具体改动点。** 复用 `B/brainpromptmax/models/baselines.py::MLPBaseline` 的 6670 维非对角上三角输入，在 `models/brainpromptmax.py` 增加一个受正则约束的直接 FC 分支。先训练同折 TRAIN 的 MLP，并用内层 VAL 选定权重；冻结它，让 Max 学习中心化的 logit 残差：`z = z_MLP + a·(z_Max - mean_class(z_Max))`，`a` 为全局有界可学习标量，初值 0.1。只加载本折内相应 TRAIN 训练的模型。若进入联合微调，另列实验并给全部对照匹配更新预算。

**预期收益。** MLP 的现有三项均值全部高于 Max，是明确的结构对照信号。直接 FC 路径可能保留查询 token 汇聚时损失的连接信息，Transformer 只需提供额外信息，目标是同时改善或保住 Accuracy、F1、AUROC。残差结构本身不保证不退化，也不证明 prompt 有贡献；新增参数和计算应计入成本。

**验证方式。** 比较 MLP、同参数预算的更宽 MLP、修正后的 Max、MLP+无文本 Transformer 残差、MLP+语义 Max 残差。用内层预测检查两个分支错误是否互补，再决定是否值得扩大验证。若收益同样来自更宽 MLP，或 gate 持续趋近零，就不保留复杂融合，更不将 MLP 的收益归功于 prompt。

#### 方案 F：先核实域定义，再处理元数据捷径与跨站点退化

**具体改动点。** 修改 `B/brainpromptmax/data/adni_adapter.py` 的审计输出、`layers/metadata_tokens.py` 的 dropout、`training/trainer.py` 的元数据标准化和 `evaluation/reporting.py` 的分组报告。七个 ADNI 目录是七批来源，不能直接称为七个采集中心。Max 数据报告写有 58 个站点，但 Pro 的 `docs/DATA_PROTOCOL.md` 明确说明历史 site 含义尚未独立确认；先核对 `site_source/demographic_metadata_source` 到真实站点/扫描仪的映射，仅公开聚合覆盖与缺失率。

当前年龄、性别又出现在 meta_text 中，`metadata_tokens.py:76–108` 只随机丢 channel：丢 age token 后，文本通路仍可携带年龄。因此建议改为按信息字段成组丢弃：删除年龄时同时选用不含年龄的缓存文本，性别同理；另设整体 metadata dropout。年龄改为仅 TRAIN 拟合的均值/标准差变换，减少小 batch BatchNorm 统计的扰动。站点和 source_batch 暂只用于分层审计，不新增为诊断输入。

**预期收益。** 减少重复人口学输入造成的捷径，提高真实域偏移下的 AUROC、macro-F1，并检验 Accuracy 是否保持；可能降低混合样本内分数，因此必须区分 pooled 表现与跨站点表现。`no_metadata` 较弱只说明整体元数据有预测价值，不能证明其价值完全来自疾病信息，也不支持直接全部删除元数据。

**验证方式。** 先比较 FC-only、metadata-only、原双通路、字段成组 dropout，固定模型和损失；再单独比较 BN 与 TRAIN 固定标准化，避免同时改变多个因素。仅在站点来源确认且每域支持数足够后，补充外层留站点、内层仅在训练站点选型的验证；如同人跨站点，应从训练侧排除与测试同人的扫描。样本不足的站点报告支持数，缺类的 AUROC 标 NA，不给人为满分/零分，也不按结果挑站点。若只能验证来源批次，明确称“留来源批次验证”。本轮不把全量 ComBat 或无条件域对抗当作默认优化，也不在测试域拟合统计量。

#### 建议的执行顺序与成功标准

1. **先完成 A 和 B 的合成/来源检查，再建立同协议基线。** 优先解决共享预训练、1/50 epoch 来源不明、未生效 gate、TEST 访问和配置接线；这些未解决前，不扩大模型容量或候选池。
2. **首批性能实验只比较 C 的有限分类目标与 B 的预训练对照。** D/E/F 是后续候选，先逐项验证；只有内层证据支持的两个模块才做 2×2 组合消融，不展开所有排列组合。候选列表、选择规则与计算预算在读取新外层分数前写定；同一外层 TRAIN 内可执行这套有限选择，外层 TEST 不能反馈下一轮调参。
3. **最终报告三指标配对差，不只报最高单 seed。** 与同协议 Max v2 及 MLP 分别比较；三指标均值都提高才能称本次目标达成，否则明确改善了什么、牺牲了什么。报告每 fold/seed 与 OOF 汇总；不把十个相互重叠训练集的折当独立患者。受试者聚类配对 bootstrap 可给固定 OOF 预测条件下的差值区间，并明确它未包含重新训练/选型的不确定性；三 seed 差异另报。若声称“三项均有统计证据”，需预先定义联合/多重比较规则，不能从重叠 SD 推断。
4. **旧结果继续保留为探索证据。** 本队列和多套划分均已有历史使用；本轮新的嵌套评估最多支持改进后的内部验证。真正未暴露的外部/前瞻队列是更强的后续证据，不能把重新随机分折称为外部验证。

本轮只读检查的关键复现信息：Python 为 `/home/panyao/miniconda3/envs/brainprompt/bin/python`，以 `-B` 禁写字节码、CPU、`torch.set_num_threads(2)` 执行。合成掩码探针设置 `torch.manual_seed(42)`，`BrainPromptMax(d_model=32,n_layers=1,n_heads=4,dropout=0,use_metadata=False,roi_text_emb=torch.randn(116,768)).eval()`，两张随机对称矩阵、`mask_ratio=.25`；遮挡数为 34/30，两样本各自被遮挡重建行之间最大绝对差均为 **0.0**。gate 探针为默认形状模型 `lmpe_gate_init=.01` 后调用现有 `load_pretrain(...,'checkpoints/pretrain.pt',cpu)`；没有训练。患者级成员仅用于服务器内统计，未复制到讨论。

源码/结果快照指纹（SHA-256，便于下一轮判断文件是否变化；不代表已有不可变训练归档）：

| `B/` 下文件 | SHA-256 |
|---|---|
| `main.py` | `ca976e29c5e9880d78cb39b0e29bce8d53566e6ae0435d1644e4841a38268b8f` |
| `brainpromptmax/models/brainpromptmax.py` | `e18f50c28ad41f3e8388a595dd46435ba9defba79a92362f773f969a040bbb30` |
| `brainpromptmax/models/pretrain.py` | `02b7fa2241126a8d91333838c45ab6dfedeb5d59bfa6ae8ababf1617d3e97fad` |
| `brainpromptmax/training/trainer.py` | `df5ac7f6f347653d3db083c19a538a4822ecb55fb914a13774fecd818424fe01` |
| `brainpromptmax/training/hparam_search.py` | `9d2187b65d25c80b7b1b47a157cb8eef4c3e01c5b5a6a3f15538e9fd49c94713` |
| `configs/main_v2.yaml` | `4e99f3764636fbbaf20fea5637f56c3a76f7632bc85e84a25f1c4730ad307088` |
| `results/cv_metrics_main_v2.csv` | `a8c19586d2b2fa2b773414614be7cedcbe01682a4580657d3db299e11d40de9c` |

### 留给对方的问题

1. Kimi 能否提供 v1/v3 及 `no_pretrain/no_lmpe` 各次实际加载的预训练文件哈希、epochs、成员范围和训练配置？当前默认权重仅 1 epoch，与文档 50 epoch 如何对应？在来源恢复之前，是否同意暂缓对既有预训练消融的因果归因？
2. 是否同意把“新建全量预训练”“同人跨 split”“跨外层折共享全局选型”分开修复，并将本轮修订定位为内部验证，而非通过重分折获得新外部测试？若需沿用扫描十折主口径，建议明确保留受试者隔离补充和嵌套选型，而不是直接替换历史表。
3. 对方案 B 的身份丢失与对称可见性，是否有训练时代码快照能否定当前探针？若没有，是否先做四格预训练对照，再考虑更复杂的 disease-query 或预训练任务？
4. 对首批预算，建议优先 A→C/B；D 的语义对照、E 的 FC 残差、F 的站点/元数据审计中，哪一个有尚未公开的独立验证证据值得提前？请提供 VAL/协议证据，不以已读取 TEST 的最高分重选方向。

- 下一步：Kimi Code 针对上述证据逐项回应，确定首批有限实验矩阵。当前六项均为讨论建议，没有已验证的新性能提升，也没有启动训练。
