# BrainPrompt 优化 Agent 任务书（本地环境与受试者十折修订版）

修订日期：2026-10-21。以本次用户要求及已核验的服务器环境替换原稿中的占位路径、seed 41、5 seeds、扫描级缓存和未证实的性能保证。原稿保存在 `docs/TASKBOOK_ORIGINAL.md`。本文件规定研究方案；完成状态只以实际日志、预测及核验报告为准。

## 1. 目标与固定协议

在 ADNI_1–7 的 AAL116 rs-fMRI Pearson FC 上优化 BrainPromptG，借鉴 [BrainPrompt](https://github.com/AngusMonroe/BrainPrompt) 和 [ncomms2025](https://github.com/vkola-lab/ncomms2025) 的组件，完成 E0–E5 与消融。仅使用 fMRI FC 和已有 ROI、疾病、年龄/性别/站点文本提示，不引入 ABIDE、sMRI、PET 或诊断提示作为患者输入。

- **受试者级分层十折；随机种子 42、2024、3407。** 同一生物学受试者全部扫描必须属于同一 fit/validation/test 分区。每个 seed 每名受试者恰好有一次外折测试预测。
- **MCI→EMCI；真实 LMCI 保留。** 按原始 `original_diagnosis` 重映射，不能沿用源 bundle 的旧 MCI→LMCI 标签。
- 内部类别索引固定 `[CN, AD, SMC, EMCI, LMCI]`；序数损失、QWK 与相邻采样使用 `[CN, SMC, EMCI, LMCI, AD]`，显式映射再逆回内部索引。此顺序是研究假设，不意味着每位患者必定按该路径进展。
- 已核验 895 名受试者/1323 次扫描。按内部索引，受试者数 `[503,65,19,279,29]`，扫描数 `[850,65,39,310,59]`。SMC、LMCI 是最稀少类别，不能把合并后的 EMCI 误称为最小类。
- 使用与 `adni7_subject_cv10_mci_emci_v2_20261020` 相同的受试者外折及约 80%/10%/10% fit/validation/test；原 BrainPrompt 的扫描级 `.index` 文件保留但不用于本研究。
- 主报告先对同一受试者的扫描概率求均值，再在每个 seed 拼接十折 OOF 计算指标，报告三个 seed 均值±样本标准差；另提供逐折和扫描级结果。禁止把扫描数写成受试者数。
- 所有模型/配方/阈值/校准/checkpoint 选择仅使用当前外折 fit/validation，不使用测试数据或跨外折测试汇总来选择。该队列已有历史开发暴露，结果属于探索性内部比较，不是独立外部验证。
- 训练 **不设最大 epoch**；按预声明验证平台规则终止。15 轮冻结、5 轮 warmup、余弦衰减周期是优化阶段长度，不是训练上限。

## 2. 实际路径与隔离边界

|用途|实际路径|
|---|---|
|新工程 `$REPO`|`/home/panyao/AD/BrainPromptNComms2025`|
|原版只读参考|`/home/panyao/AD/BrainPrompt`|
|已有数据/划分审计实现（只读）|`/home/panyao/AD/BrainPromptMaxFusion/subjectcv`|
|ncomms2025 只读参考|`/home/panyao/AD/BrainPromptMax/references/ncomms2025`|
|Python|`/home/panyao/miniconda3/envs/brainprompt/bin/python`|
|审计后的七批 cohort|`/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010/dataset_v1/cohort.private.pt`|
|ROI prompt|`/home/panyao/AD/BrainPrompt/data/prompts/ROI_prompts/AAL116_short_datatoken.pt`|
|ROI 文本|`/home/panyao/AD/BrainPrompt/data/prompts/ROI_prompts/AAL116_short.txt`|
|疾病 prompt|`/home/panyao/AD/BrainPrompt/data/prompts/label_prompts/adni_label.pt`|
|疾病文本|`/home/panyao/AD/BrainPrompt/data/prompts/label_prompts/adni_label.txt`|
|患者 prompt|上述 cohort 的 `subject_prompt`，形状 1323×2048，与 records 一一对应|
|实现|`$REPO/bpstudy/`；原版必要依赖的可追溯副本在 `$REPO/vendor/`|
|配置|`$REPO/configs/study.json`|
|实验产物|`$REPO/out/adni7_ncomms_subject10_20261021/`|

源 cohort SHA256：`217b662f09aa3c359b71b1d49cb51e1b77a37334c67f59663bed5aae5c6bd273`。不修改原始数据、其他研究源码或现有任务。原版 `data/prompts/meta_prompts/adni_meta_datatoken.pt` 为1327条，不能用于本研究。全局旧 `.bin` 或其占位路径也不是本研究输入；直接使用已核验的 Pearson FC，不能对 FC 行再次 corrcoef 而改变输入定义。

GitHub 当前只同步本任务书；本地新工程按阶段提交代码，禁止上传 ADNI 数据、身份信息、逐例预测、患者提示、权重或大缓存。

## 3. 前置核验

执行 `python -B -m bpstudy.run prepare --config configs/study.json`，写入 `out/.../env_manifest.md`、数据/切分审计及配置/源码哈希。

1. 核对原版 `llm-main.py`、`train_TUs_graph_classification_llm.py`、`metrics.py`、`baseline.sh`、G/C 模型、GCN/attention/readout 层与配置存在，记录源文件哈希。
2. 验证 bundle SHA、1323个唯一 acquisition、895名生物学受试者、原始标签来源、七批来源、FC形状116×116、有限/对称/[-1,1]/对角1，ROI顺序、prompt来源与形状、无患者诊断文本泄漏。
3. 验证 ROI 116×2048、疾病5×2048、患者1323×2048。疾病文本顺序必须与内部索引相同；不重新随机生成或按错误顺序加载。
4. 验证30个外折：fit/validation/test受试者交集全为0，各分区五类齐全；三个seed测试覆盖完整。与历史受试者划分逐索引比较，不改旧缓存。
5. 环境包括 PyTorch 2.4.0+cu121、DGL 2.4.0+cu121、scikit-learn 1.7.2、SciPy、captum、tensorboardX。实际GPU检查在宿主环境进行；不将沙箱CUDA不可见误报为设备故障。可选 tangent 未纳入本次主矩阵，nilearn 缺失不阻碍必选任务。
6. 原版真实 fit/validation 数据小批冒烟并保存 loss、人口图统计。原版 G 的关系图实测应恒零；修复版必须有非零邻接/有效梯度。每个阶段实现后执行独立的一折短冒烟，不把有限轮次的冒烟成绩作为正式结果。

出现数据完整性、来源或模型接口错误时拒绝启动正式训练，修复可修复的工程问题并重验；不得用随机数据/提示替代正式数据。

## 4. Phase 0：基线与修复

原版 `sigmoid(cosine)` 两矩阵乘积上界约0.5344，G阈值0.8、C阈值0.7都会生成全零人口图。`global_gcs` 仍有 residual，不应误称整个分支输出恒零。C还存在 `hg` 未定义与 `F.dropout` 在eval时仍随机的问题。

- E0保留原版G架构/病灶，参考 `baseline.sh`：L4、hidden/out116、mean readout、batchnorm/residual、dropout0.5、batch20、Adam lr5e-5、weight_decay0.005、edge_ratio0.2、disease loss权重1。受试者切分、标签和收敛协议按本文件适配，因此称“原版架构协议适配对照”，不冒称论文数值的逐位复现。
- `pop_fusion=original/soft_topk/off` 显式配置。soft_topk使用直接cosine平均、top-k选择及选中值softmax，不在邻接上dropout；允许对特征dropout。离散top-k索引不可微，选中相似度与权重可微。
- 修复版训练使用当前fit批图；验证/测试只连接由fit样本生成的5个类别原型（脑表征及meta表征），不使用验证/测试标签建立原型、不连接测试患者彼此。以整批/逐例预测一致性检验推理是否摆脱测试batch组成影响。原型在选定checkpoint后重新构建。
- 在vendor副本修复C的 `hg→out` 与eval dropout；C只做运行验收，不新增主矩阵。
- 保存 `last.pt`（恢复用）和 `best.pt`（验证受试者macro-AUROC最优，同分按NLL）。E0主结果按末轮、同时留best诊断输出；E0p及后续按best测试。E0p同时改变人口图与选模，解释时不得把总差值全归因于其中一项。
- 同时记录train/val loss、macro-AUROC/AP/F1、每类recall、lr、最佳/停止epoch及完整停止状态。

## 5. Phase 1：推理侧对照

E1复用E0p权重，不额外训练模型。单独报告原始、温度校准、Youden、TTA及组合结果；禁止因测试表现下降而删掉分支。

- 仅validation拟合一个温度T，优先受试者聚合后 log(probability)/T 的再归一化方式，使校准单位明确。保存T及拟合索引哈希；测试采用完全相同公式。不用仅2名SMC的停止集拟合5个自由温度。
- validation上每类one-vs-rest ROC最大化Youden J。单标签决策预声明为 `argmax(p_c−t_c)`，同分优先原概率更高的类；另报各独立二分类阈值recall，不能混为五分类recall。验证缺类时明确失败/不可估计。
- TTA：K8、对称edge dropout p0.1、保留self规则一致，平均概率；G在邻接上作用，T在FC输入及对应bias上作用。随机种子/重复顺序固定。组合校准/阈值在对应TTA验证概率上拟合。
- 校准不保证多分类macro-AUROC改善，Youden不保证少数类测试recall提升；这些均为结果而非软件验收条件。

## 6. Phase 2：损失与采样

配置均显式声明，候选只用当前外折validation macro-AUROC选择（同分macro-AP，再NLL，再固定顺序）。选出的配方用于后续阶段，测试不参与。该validation同时用于checkpoint与候选选择，需承认小验证集选择方差；不把validation成绩当无偏泛化结果。

- `ce`、`focal`（gamma2，按fit唯一受试者频率倒数归一化alpha）、`cb_focal`（有效样本数beta0.9999）分别实现，不把两种重加权默认为同一公式。
- 相邻对采样覆盖CN–SMC、SMC–EMCI、EMCI–LMCI、LMCI–AD，四对等比例；先采fit受试者再采该受试者扫描。batch20不能整除8，跨batch轮转余额并记录实现，不声称每批严格四等份；禁止误称过采样增加了真实受试者数。
- CORAL式4个累积边界，共享权重、单调阈值；所有已知五分类标签的4个二值边界都是有效监督。加入 `lambda_ordinal*BCE`（默认1），以及跨不同真实阶段样本的期望阶段分数排名项 `lambda_rank*mean(relu(margin*真实阶段差−预测期望阶段差))`（0.025、0.25）。期望阶段为4个累积概率之和；不使用原稿会惩罚正确CN/AD端点的同样本累积概率间距约束。不能只加ranking项而遗漏有标签BCE，也不能以缺失标签mask抹掉合法负例。
- 主预测保持softmax五分类；另报序数头 `sum(p_k>0.5)` 的自然顺序预测及两头一致率。一致率>70%是观察指标，不是筛除失败模型条件。
- 疾病文本对比head与疾病query共享投影；soft target按自然序距离 `softmax(-distance/decay)`（decay1）构造，以KL替代hard CE。总loss显式含分类、疾病、ordinal BCE、ranking四项。
- E2候选：focal、cb_focal、仅adjacent、仅ordinal、仅soft_label、全组件组合；其余继承E0p。也允许E0p原CE入选，防止强迫选择更差的复杂配方。

## 7. Phase 3：FC表征

在选定E2配方上逐项测试 Fisher-z、多尺度、符号双通道及三者组合；E2本身也作为候选。只用当前validation选择。

- Fisher-z：clip±0.999再atanh，对角置0，所有参数不从test学习。
- 多尺度：10%/20%/30%邻接、共享编码器、视图读出均值聚合，以控制参数增长；这替代原稿会扩大分类头的拼接，明确记录实现差异。
- 正负边独立聚合、共享基础变换后融合，避免抹去符号；参数增长须实际统计。
- 可选tangent暂不纳入必选矩阵；若未来加入，参考均值只能fit分区拟合，不能全队列拟合。

## 8. Phase 4：BrainPrompt-T

独立 `bpstudy/models.py`，原模型代码及CLI不修改。hidden64、2层、4头、FFN128起步，所有值落配置。

- ROI token = FC行投影 + ROI文本投影 + 可学习ID；ROI prompt训练dropout0.2，患者prompt dropout0.5，均用可学习null替换。患者prompt作为额外token进入attention。
- Attention logits加 `beta_head*Fisher-z(C)`；每头beta初始化0.5，可学习。保留连续正负FC；这取代GCN的硬稀疏邻接，多尺度/双关系GCN开关不机械搬到T中。E4保留选定E2损失与Fisher特征选择，并记录其与E3的架构差异。
- 5个疾病prompt经共享投影构造query，对ROI tokens cross-attention，得到类条件表示及5类logits；同投影用于疾病对比head。默认人口图off，患者输入通过token融合。
- 验收梯度、mask生效、eval确定性、真实一折loss可优化；参数总数小于1.5倍G。不把prompt dropout“必须显著改善”当程序正确性的依据。

## 9. Phase 5：两阶段训练与消融

- Stage1仅当前外折fit中CN/AD预训练；validation也只用当前validation的CN/AD。禁止使用外折test或其他外折的预训练权重。复用同一编码器，但二分类训练采用CN/AD两logit监督。
- Stage2迁移编码器/提示投影，重新初始化5类输出与ordinal头；冻结迁移参数15轮，仅训练新head，再全解冻。Stage2 lr为Stage1的1/10。完整保存迁移参数名、源checkpoint hash及冻结/解冻证据。
- AdamW beta(0.9,0.98)、5轮warmup、cosine衰减、gradclip1；余弦衰减到minlr后保持，不用周期结束作为停止条件。保留legacy Adam配置供G对照。
- 无最大epoch：legacy调度在验证NLL无实质改善15轮后减半lr；cosine默认50轮衰减至1e-6。达到最低lr且验证macro-AUROC与NLL均连续30轮无1e-4改善、在最低lr已运行30轮才停止。冻结期内不触发终止，解冻后重置平台观察状态。报告实际达到的“预声明收敛判据”，不宣称数学全局收敛。
- E5无prompt_dropout、无FC_bias、无ordinal各独立两阶段训练；若选定配方本身没有ordinal，明确标为无有效组件差异的对照，不伪造消融收益。
- ncomms原实现的15轮冻结用于其Stage1，且其任务为多个二元端点；此处CN–AD预训→五分类是本研究适配，不冒称逐行复刻或直接迁移其医学影像预训练权重。其二阶采样针对多任务二元标签对，permutation masker随机保留非空模态子集，calibration主要为isotonic；本研究相邻疾病采样、Bernoulli prompt dropout与scalar temperature均是明确的新适配。age/sex辅助任务为可选，默认不纳入本次矩阵。

## 10. 必选矩阵与报告

依次执行：E0；E0p；E1后处理；E2六候选及验证选择；E3四候选及验证选择；E4；E5；E5_no_prompt_dropout、E5_no_fc_bias、E5_no_ordinal。每个独立训练变体覆盖3seed×10外折。选择流程自身也保存完整OOF输出，禁止在报告时挑选测试最好的变体作为“最终模型”。

对E0、E5额外执行患者prompt缺失评估（相同checkpoint，零/null按模型契约），报告性能变化。保留E1不同后处理变体及序数预测。

每实验输出 Accuracy、balanced accuracy、macro-F1、macro-AUROC、macro-AUPRC（明确采用average precision/AP）、每类AUROC/AP/Recall/Precision/F1、QWK、NLL、混淆矩阵；同时保留argmax与阈值口径。缺失可估计条件标为NA，不能伪造0或1。

95%区间与模型差值采用**受试者聚类、类别分层的配对bootstrap**，每次重采样同一批受试者并保留其三seed预测，默认2000次；不能把30个相关fold当独立样本做普通t检验。DeLong不直接套到三seed×重复受试者的macro结果。显著性作为探索性输出，若输出多项p值进行Holm校正，并说明未包含完整重新训练的不确定性。

结论必须如实包含负结果与少数类失败。macro-AUROC、macro-AP、SMC召回的提升及p<0.05是研究目标，**不是保证，也不是完成任务的必要条件**。

## 11. 工程、恢复与完成定义

- 新工程独立Git提交：协议/数据、Phase0、Phase1/2、Phase3/4、Phase5/runner/report；保存vendor来源、配置、源码及输入哈希。
- 每个组件原子保存model/optimizer/scheduler/RNG/best/epoch/history；恢复前检查数据、分区、配置、代码来源。独立锁防重复；已完成任务校验后跳过。
- 训练前通过梯度/公式/分组/恢复测试、CPU及GPU真实数据冒烟；正式与smoke目录分离。准备可恢复后台控制器与日志，不以Markdown代替进程保活。不停止其他研究进程。
- 一键入口 `bash scripts/run_study.sh`；检查状态 `python -B -m bpstudy.run status --config configs/study.json`；最终核验及汇总由控制器自动执行，报告放在 `out/.../results/REPORT.md`。
- 完成定义：数据与来源核验通过；全部30外折所有必选变体达到停止判据；预测覆盖/受试者隔离/选模与候选选择/指标复算核验通过；报告完整、可恢复可复现。若资源/程序失败，记录具体任务与错误并恢复，不把部分完成当整体完成。
