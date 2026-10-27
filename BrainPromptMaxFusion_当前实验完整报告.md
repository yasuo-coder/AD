# BrainPromptMaxFusion 当前实验完整报告

> **当前事实快照，不是下一轮优化方案。** 核验时间：2026-10-27 11:19:05（北京时间，UTC+8）。本次仅读取现有工程、配置、队列、日志、checkpoint 和已有预测，复算指标并生成匿名报告；没有启动训练、重新运行模型 TEST 推理或按 TEST 改选模型。本报告不预写本次提交自身 SHA；实际提交 SHA 将在 [Issue #3](https://github.com/yasuo-coder/AD/issues/3) 回复。

## 1. 执行摘要、版本与证据层次

**最重要的状态纠正：新任务书 E0–E5 已经运行完成，不能写成 0/30。** 对应工程是 `BrainPromptNComms2025`，不是历史 `BrainPromptPro` 的 004/005，也不是大小写不同的 `BrainPromptMaxFUsion_v2` 中同名 E0–E5。已完成 30/30 外层 seed×fold、630/630 训练组件、780/780 预测组。本次重新核验 best/last/不可变 best checkpoint、签名、首次停止、最佳轮、分区、验证选型与校准重放、逐折及 OOF 指标，全部通过，最大数值差 `8.881784197001252e-16`。证据：[本次核验](docs/reports/issue3-20261027/snapshot_verification.public.json)、[完整执行矩阵](docs/reports/issue3-20261027/E0_E5_execution_matrix.csv)、[训练组件](docs/reports/issue3-20261027/E0_E5_training_components.csv)。

### 1.1 版本与工程位置

| 对象 | 分支 / 完整 commit 或可复核版本 | 实际位置与说明 |
|---|---|---|
| GitHub AD 当前任务书 | `main`，`27ac3e2f27de74d744e816c6880960530a5d536d` | 核验时 main 只有任务书；不能认为训练代码已经在 main 发布 |
| 最新公开历史结果 | `research/adni1to7-brainprompt-cv10-last`，`19d2395c864d9b33fc64547ff84af0b683c4c716` | 本报告历史链接全部固定此 commit，避免分支移动改变证据 |
| 较旧历史分支 | `research/adni1to7-p2p3p5`，`c90a97574f4a9ecc4f3ba5dbb6582e102de32f49` | 仅作历史定位，不覆盖上行更新事实 |
| 本地历史仓库 | `research/adni1to7-pro-cv10-last`，`1ccbe081b1b30aa82c76239c803290f1210083aa` | `/home/panyao/AD/AD-`，origin 仍指 `yasuo-coder/AD-`；存在既有未提交改动。本次只读指定 Git 对象，不把工作树当历史版本，不修改或同步这些改动 |
| 004 主要训练 / 控制器恢复 | `29f34f9322d4b797329c8bd405fa64ea1bbbd582` / `39ee2f2e8fc69ec7c2a7023d5d995787ca42aa7a` | 各早期运行另有独立 source hash，见历史逐 seed JSON |
| 005 实际训练 | `8e5d8c35810bc6f8e5d4bfdffe2b54cf61a8c1c6` | 公开等树训练归档 `f628bcda06d9abe7bcb62351aa988e934ae283ef`；不能以报告 commit 替代训练 commit |
| E0–E5 当前本地 Git | `master`，`69411e8d614110d929afa263c1fc0184dcc819fd` | `/home/panyao/AD/BrainPromptNComms2025`；无 origin；未跟踪 analysis 和后续分析文档，不纳入训练源码声明 |
| E0–E5 封存训练代码对应版本 | `82bdfafee09d5a97f4d2a32bd282d57819b123f4` | 32 个项目内封存文件逐一与该 commit 一致；加外部只读依赖共 41 个封存文件。训练前验收记录 commit `c61030b96bd4e2aa29fb9cf3fa6167acb7dd6ae2` 的这些文件也一致 |
| MaxFusion / PatientRepair | 各工程不是独立 Git checkout，完整 commit 为 NA | 分别以配置、源码封印、输入哈希标识；不能借用父目录或其他仓库 SHA 充当训练版本 |

受控目录缩写：

- `N=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010`：004 数据、训练、锁、冻结 TEST、机制探针。
- `C=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012`：005；`C/reports/verified_cv10_20261012` 为已核验报告。
- `M=/home/panyao/AD/BrainPromptMaxFusion`：MaxFusion 实际工程，训练在 `M/experiments`，报告在 `M/results`；10 月 26 日实验在独立实验目录内保存报告。
- `E=/home/panyao/AD/BrainPromptNComms2025/out/adni7_ncomms_subject10_20261021`：本任务书 E0–E5；`units/seed_{seed}/fold_{00..09}/components/{component}` 保存训练，`results/summary.json`、`verification.json`、`REPORT.md` 保存汇总。
- `P=/home/panyao/AD/BrainPromptPatientRepair2026/out/adni7_patient_readout_nested3_20261024`：后续独立患者读出修复研究。

### 1.2 当前实际状态

| 研究 / 工作 | 已完成证据 | 在跑 / 未完成 |
|---|---|---|
| 004 | 主 19×3 + 4 单 seed 分类探索、10 预训练、subject A0/A1 各 3 seed；63/63 冻结 TEST 及独立复算 | 无在跑；四项探索未扩成三 seed，subject 未扩至 19 方法，均属范围边界 |
| 005 | 10/10 扫描级外折末轮 TEST，已核验 | 无在跑；不可拿作新独立外部验证 |
| MaxFusion S20 / S21 | S20 受试者三模型十折完成；S21 30 外折、1364 拟合、240 预测组，既有核验 PASSED | S21 30 折均未通过语义融合门槛，最终均等于所选 FC 分支 |
| 新任务书 E0–E5 | 30 外折、630 组件、26×30 预测组；2026-10-23 15:17:41 完成 | 缺失 0，当前失败 0，排队 0；不是“方案待执行” |
| PatientRepair S24 | 30 外折、270 拟合，2026-10-24 23:36:51 完成；既有核验 PASSED | 不是 E0–E5 的补齐折；多数类塌缩仍在 |
| 10 月 26 日固定 FC 选模 | 30 共同训练轨迹、60 选中 checkpoint、30 外 TEST 配对预测；既有核验通过 | SMC、真实 LMCI 召回仍为 0；不能依据 B/A 测试排名追加选择 |
| 10 月 26 日 R1 最小修订 | 实际 90 拟合、30 外折结果、3 worker 退出码均 0；本次核对 90 个 last hash 与停止锁一致 | 已训练；旧 `BASELINE_COMPARISON.md` 中“未训练”滞后。没有找到同等级独立全流程核验，不能仅据 COMPLETE 写成全面验收 |
| 本 Issue | 事实核查、匿名汇总与报告 | 等待 Kimi 审阅；下一轮新增优化矩阵未启动 |

宿主 `/proc` 检查：E0–E5 控制器 2748876、worker 2748929/2748932/2748938，PatientRepair 控制器 1645466 与初始 worker，R1 worker 1770275/1770276/1770277 均已不存在；未发现 AD 工程在跑的 Python 训练任务。宿主有 8 张 RTX 4090，GPU 上仍有其他服务，不能把显存占用等同于本研究运行。沙箱内 CUDA 不可见不是宿主故障证据。E0–E5 历史启动为 2026-10-22 11:16:49，使用 GPU 0/3/7，日志 `E/logs/`；当前 PID、启动时间、GPU、待执行队列均为 **NA（没有活跃任务）**。其他研究日志及核验范围见[附加研究证据](docs/reports/issue3-20261027/additional_studies.public.json)。

### 1.3 目前可以说的科学结论

1. 004 中 A7 未全面超过 BrainPrompt，预先选定 A1 主 TEST 五项均值均未超过 A0。P2/P3/P5 的验证四背景 F1 边际均为负，不支持“三个稳定有效创新”。
2. 原版 population graph 的空邻接是已核实实现问题；修复实现通过梯度与推理批次不变性验收，但 E0p 并未由此获得全面性能收益。修复正确与研究有效是不同结论。
3. E5 受试者 Macro-AUROC `50.35±0.36%`、Macro-F1 `15.14±1.64%`，均低于 E0p；完整方案改善的目标未达成。
4. MaxFusion S21 最终 Macro-F1 `26.38±1.08%`，但 30/30 折语义门槛未通过；不能把最终 FC 分支成绩归功于语义融合。
5. PatientRepair 与 10 月 26 日 R1 仍存在严重多数类预测问题。固定 FC 平滑 Macro-NLL 选模有部分描述性改善，仍未解决 SMC/LMCI 零召回，也不是独立确认性证据。
6. 全部结论属于历史暴露 ADNI 队列的内部探索性比较；没有独立外部泛化成立的证据。

## 2. 数据、标签、提示与来源

### 2.1 历史和当前标签分别列示

类别顺序均为内部 `[CN, AD, SMC, EMCI, LMCI]`。下表的扫描数与受试者数绝不混用。

| 协议 | 有效 FC / 扫描 | 生物学受试者 | 扫描类别计数 | 受试者类别计数 | 标签定义 |
|---|---:|---:|---|---|---|
| 历史五批 | 998 | 685 | 本报告不重列旧分布 | 本报告不重列 | 背景；不能用作七批结果 |
| 旧七批 004/005 | 1323 | 895 | `[850,65,39,61,308]` | `[503,65,19,61,247]`，当前源审计所核对的旧标签 | 下载 `Group`，MCI→LMCI |
| 新任务书 / E0–E5 | 1323 | 895 | `[850,65,39,310,59]` | `[503,65,19,279,29]` | 读取 `original_diagnosis`，MCI→EMCI，真实 LMCI 保留 |

原始标签包含 MCI 249 次扫描、EMCI 61、LMCI 59；因此旧 LMCI=`249+59=308`，新 EMCI=`249+61=310`。同样的 895 人和 1323 次扫描发生重映射，不是新增人或移除类别。受试者类别差值为 218 人从旧合并 LMCI 转入 EMCI。现在最少类为 SMC 19 人、真实 LMCI 29 人；不能把 EMCI 当最少类，也不能拿 005 的 LMCI Recall 31.17% 解释真实 LMCI 的召回。当前聚合核验未发现同一受试者混合标签，但来源仍是下载元数据，**不是独立临床裁决或逐扫描时点诊断核实**。

七批有效 FC 数为 `200/198/200/200/200/198/127`。ADNI_2 两次扫描缺可信映射，ADNI_6 两次预处理失败，不填造 FC；同访视名但不同采集日期在映射核实后保留。历史审计未发现重复 acquisition、清洗时序或 FC 内容，998 个历史 FC 与标签逐项一致。当前七批含 290 名多扫描受试者，保留纵向关联，不把多次扫描视为更多独立患者。原始影像 CSV 含结构像等配对行，不能拿影像行数当 FC 数。来源：[旧数据审计](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/DATA_AUDIT.md)、[旧匿名数据版本](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/public_data_summary.json)、`E/data_audit.json`。

### 2.2 输入与 QC

输入是 AAL116 rs-fMRI **经验 Pearson FC**，由已有清洗时序统一计算，float32，完整形状 `[1323,116,116]`；保留正负相关、对称性和对角 1。当前审计最大不对称为 0，范围 `[-0.9519595503807068,1]`，有限值与重复检查通过。既存 Nilearn 收缩相关矩阵与这一输入不同；不对 FC 行再计算 corrcoef。ROI 顺序与 AAL116 文本、嵌入对齐，未用 TEST 重排 ROI。本报告没有重新完成影像层面运动混杂或临床 QC；不能将矩阵数值检查等同全面影像质量验证。

| 提示 | 形状 | 来源 / 使用方式 |
|---|---|---|
| ROI | `[116,2048]` | `BrainPrompt/data/prompts/ROI_prompts/AAL116_short_datatoken.pt`，文本 `AAL116_short.txt` |
| 疾病 | `[5,2048]` | `BrainPrompt/data/prompts/label_prompts/adni_label.pt`，文本 `adni_label.txt`；顺序固定内部五类，所有类共同输入；禁止按患者真实标签挑 query |
| 患者 | `[1323,2048]` | 原始审核 cohort 的 `subject_prompt`；严格与 records 对齐；不使用原版 1327 条的旧全局 meta 缓存 |

年龄、性别和 `site_id` 是已有患者文本提示的来源，另用于元数据完整性审计；E0–E5 未增加独立 age/sex 辅助任务，也不将 Source_Dataset 当预测特征。`site_id` 共 58 个不同值，字段完整不代表已确认其真实 acquisition site 含义；仍标记**站点语义未独立核实**。Source_Dataset 只用于批次追溯和范围核对，身份只用于分组、对齐和历史 reference 自排除，不进入预测特征。患者提示不包含真实诊断；疾病类别提示作为共享字典与按标签选择患者输入是不同事情。

### 2.3 可追溯哈希

| 对象 | SHA256 |
|---|---|
| 原始七批 source cohort | `217b662f09aa3c359b71b1d49cb51e1b77a37334c67f59663bed5aae5c6bd273` |
| 004 冻结 protocol/split 文件 | `4c73c0e59329886799cf98edd472dbe5b0ca42b37fcf0e667929dd43fbb5a130` |
| 005 扫描十折 split | `f44a16439d0d5f651532ca571673b4588ecdf1f5874fbe64a4b09467ff5aa26a` |
| E0–E5 重映射后的受控 cohort | `a5c7df3f484ae462a22d9f8841f3705ecb1b53d3d36ff2f1c3cac931b31d19da` |
| E0–E5 split 文件 | `b497ef68b15478a66ea7568e63acf168945c3fca866ecad2b5e8e426fcb0e0d9` |
| 所对齐的 S20 split 文件 | `0938c45dbf0955a07567629c600420eab9a00665681744cfbe1b50998d9bb3ca` |
| ROI 顺序 | `cb807eaaa2e56660e9fd88621d22d37a737a4b643a60207d51229a8b7c4e3a7d` |
| ROI embedding | `e5855857af8e4364c016cd93a551bcbcac11fadeee7792c0c0d74a2ef69b1e1d` |
| 疾病 embedding | `fbe964659223b4e3a67da9fff64afa64a2fca0293458c3ec120d1fdfe2f79411` |
| E0–E5 配置 | `2b049183e3dd436f3697073a0c4f76e478d0f195b6c6c5a599717f7ccdcaa986` |
| E0–E5 源码与输入 seal ID | `e066e317bc38cde1f7aa5d7fecc04e83ce2ab485b930030aeb313b2e68023f54` |

S20 与 E0–E5 split 文件序列化格式不同，文件哈希不同；30 个分区已逐索引对齐，而不是仅凭名称推定一致。公共附件只列聚合信息和哈希；原始数据、患者身份、逐例预测、提示、checkpoint、私有缓存均保留受控路径。

## 3. 划分、选择和不可跨越的 TEST 边界

| 协议 | 划分与种子 | 选择 / 报告规则 | 已知限制 |
|---|---|---|---|
| 004 主协议 | 扫描分层 80/10/10 固定 holdout，split seed 1729，训练 seed 42/3407/2026 | VAL Macro-F1 及固定平局规则选 checkpoint；三 seed 样本 SD；A1 在 TEST 前从 A0–A7 的验证均值选定 | 同人可跨分区；规则在观察验证结果后声明，不能倒称预注册 |
| 004 subject 补充 | 独立冻结的 subject-disjoint holdout，A0/A1 各 3 seed | 分开汇总；不以补充结果重新选主候选 | 与主协议样本角色交叉；不能混成 6 个 seed 或外部验证 |
| 005 | seed 41，扫描分层十折 | 每扫描一次 OOF；重新加载**实际停止 last.pt**；十折均值±总体 SD，另报合并 OOF | 624/1323 个 OOF TEST 扫描在相应 TRAIN 中有同人关联；官方循环每轮两次 TEST 监测，测试虽不驱动调度/停止/选模，已被暴露 |
| E0–E5 | 受试者分层外十折 × 42/2024/3407；约 80/10/10 fit/validation/test | 所有访视同分区；每 seed 每人一次外折 TEST。先平均同人扫描概率，再拼十折 OOF，最后对三 seed 指标求均值±样本 SD | 各折 fit/validation/test 人员交集为 0；队列历史开发暴露仍在；验证集兼做 checkpoint 和配方选择，选择方差未消除 |

E0 按 last 为主，E0_best 只作预先保留的诊断。E0p 及后续按 validation 受试者 Macro-AUROC 最优、同分 NLL 最低选择 best。E2/E3 配方只按本外折 validation AUROC、再 AP、再 NLL、再固定顺序选择；E0p/E2 原方案也可保留。E1 温度和 Youden 阈值只在本折 validation 拟合；温度在受试者扫描概率平均后作用于 `log(p)/T` 再归一化。单标签 Youden 决策为 `argmax(p_c−t_c)`，并保留独立 OvR 阈值召回口径。TTA K=8、对称 edge dropout=0.1；组合后处理用相应 TTA 验证预测拟合。主预测始终为五类 softmax/预声明阈值规则，序数头另报。

实际 `bpstudy/pipeline.py` 会在流程推进时保存各已训练候选的 TEST 预测，再进行后续候选训练；它**没有全矩阵训练完成后才统一访问 TEST 的屏障**。本次代码与选型重放确认自动选择仅依赖 validation，没有把 TEST 指标送入选择函数；但这不等于 TEST 从未可见。因此之后不能据这些表重新选冠军、改阈值、校准、checkpoint 或追加优化并声称独立确认。004 的一次冻结 TEST 锁为 `fb71040c47d235b382dac5f487b01ff9f925d262f305f134254a67250bc63031`，不得删除锁或再次运行 `finalize_study`。不同标签、切分、统计单位、训练配方和选模规则之间不能直接作模块收益比较。

## 4. 模型实现、超参数与工程边界

### 4.1 原版 BrainPrompt 与历史 P2/P3/P5

原版每个患者建立 FC 脑图，节点为 ROI 的 FC 行；ROI prompt 经投影增强节点表示，4 层 GCN 后 mean readout 得到患者脑表征。患者文本经线性投影参与融合与 population 相似度，固定五类疾病 prompt 经投影构成辅助分类。损失为五类分类 CE + 疾病对齐 CE（权重 1）；具体是否加类权重取决于协议，不能把 004 加权 CE 写成 005 的无权重 CE。原版 FC 稀疏规则是先对有符号 FC 排序取分位阈值，再比较绝对值，不等价于严格保留绝对 FC 前 20%。E3 多尺度则明确按非对角绝对 FC 边数保留 10/20/30%，差异已落代码。

历史入口：`BrainPrompt/llm-main.py`、`train_TUs_graph_classification_llm.py`、`nets/brainpromptg_net.py`。004 入口为 `BrainPromptPro/scripts/adni_1to7/train.py`，配置 `BrainPromptPro/configs/adni_1to7/study_v1.json`，模型 `nets/connectome_p2p3p5.py`、`nets/original_adapter.py`；005 为 `scripts/adni_1to7/cv10_baseline.py` 和 `brainprompt_cv10_last_v1.json`。完整公式见[历史技术报告](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/Technical_Report.md)。

| 模块 | 实际定义和主要参数 | 机制证据边界 |
|---|---|---|
| P2 | 对称掩码 FC，同时遮节点 FC 行和构图输入；默认 mask 0.10，探索 .05/.10/.15；监督 joint 为 clean CE + `0.1×隐藏上三角重建MSE + 0.1×双掩码预测对称KL`；预训练 40 轮 | 10 项预训练仅 TRAIN；重建数值改善未转成分类收益，不称已验证鲁棒性 |
| P3 | 全五类疾病 query，中心化、归一化 Q/K，query/key 维 32、softmax 温度尺度 4；输出 `[B,5,116]` 类条件 token；保留类身份的读出 | 真实标签不选 query；random、learned-query、参数匹配对照已跑。语义模块 150801 参数、匹配 MLP 150775；有梯度不等于语义特异收益 |
| P5 | 同 seed A0 固定编码器、仅 TRAIN reference、自排除；3 个固定对称扰动估计稳定性 `exp(−v/.05)`；cosine 温度 .5、top16、可学习关系评分、中心化消息及有界残差 | 无 query-query 边；reference 版本、成员、batch independence 验收通过；可靠性约 .96、边权近均匀、残差约主融合范数 .29%，实际作用弱 |

| 004 ID | 组合 |
|---|---|
| A0 | 原版适配基线 |
| A1 | P2 |
| A2 | P3 |
| A3 | P5 |
| A4 | P2+P3 |
| A5 | P2+P5 |
| A6 | P3+P5 |
| A7 | P2+P3+P5 完整实现 |

注意：S21 也使用 A0/A1/A2/A3 等名称，但定义为 scan/subject CE、loss 选择、FC 表征选择等新流程，**不是 004 的同名消融**。

004 共同训练为 Adam lr5e−5、wd.005、batch20、dropout.5、clip5；两个 CE 使用 TRAIN 计数平方根逆频率权重，疾病辅助系数1。最多400轮，min120、验证无改善patience100；NLL调度patience25/factor.5，lr<1e−6也可停止；按验证Macro-F1及固定平局规则/loss选best。P2预训练额外40轮单列成本。A0/A1/A7参数818095/835264/1817814，训练循环平均约6.67/18.13/70.16分钟；共享GPU且P5 bank构建未计入，不能当严格效率benchmark。来源：[历史最终分析](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/final_test/Final_Analysis.md)。

### 4.2 空 population graph 与修复验证

原版 `sigmoid(cos_brain)×sigmoid(cos_meta)` 上界 `sigmoid(1)^2≈0.5344467`，G 阈值 .8、C 阈值 .7 均生成全零邻接。`global_gcs` 还有 residual / bias，不能写成整个分支输出恒零。C 的 `hg` 未定义及 eval 仍随机 dropout 已在 vendor 修复；C 只作接口验收，未加入新主矩阵。

E0p 的 `soft_topk` 取直接 cosine 平均，选中 top-k 相似度经 softmax；训练用当前 fit 批图，推理只连由 fit 构建的五类 brain/meta 原型，选定 checkpoint 后重建。默认 k=5，因此对五个原型全数赋权；top-k 索引不可微，选中值可微。fit 标签只用于 fit 内原型分组，不使用 validation/test 标签或测试患者之间的连边。该修复与历史 P5 的逐样本 TRAIN reference 不是同一算法。

原始 37 项验收包括空邻接、非零修复梯度、C 确定性；真实 CPU/GPU 各 21 组件、26 变体冒烟独立于正式训练。GPU checkpoint 重放最大差 `2.851749020260641e-08`，测试 batch invariance 最大差 `3.4215931449255166e-08`。这是已记录的真实小样本验收结果；本次未对全部正式 checkpoint 再跑批次不变性推理，不能扩写为 630 个正式模型逐一接受了该测试。E0p 同时改变 population 图与 best/last 选模，E0p−E0 差值不能全归因于修图。

### 4.3 当前 E0–E5 实现

源码：`BrainPromptNComms2025/bpstudy/{run,pipeline,matrix,training,models,objectives,calibration,report}.py`；配置 `configs/study.json`；原版依赖在 `vendor/brainprompt/`，来源在 `vendor/SOURCE_MANIFEST.json`。对应[已发布任务书](https://github.com/yasuo-coder/AD/blob/27ac3e2f27de74d744e816c6880960530a5d536d/BrainPrompt优化_Agent任务书.md)，以下项目均已有代码和正式训练/评估产物。

| 实验 | 实际配方 / 机制 |
|---|---|
| E0 / E0_best | 原版 G 架构适配新标签与受试者协议；保留空人口图；同一训练的 last / best 输出 |
| E0p | soft_topk + fit-only 原型推理 + best |
| E1 | 复用 E0p；原概率、温度、Youden、TTA、组合全部保留；不新增训练 |
| E2 六候选 | focal(γ2)、CB-focal(β.9999)、仅相邻对采样、仅 ordinal、仅 soft-label、全组件；允许原 CE E0p 入选 |
| E3 四候选 | Fisher-z(clip±.999、对角0)、共享编码器 10/20/30% 多尺度均值、正负独立聚合双通道、三者组合；允许 E2 入选 |
| E4 | BrainPrompt-T：hidden64、2 层、4 头、FFN128；FC 行+ROI文本+可学习ID；患者额外 token；attention 加逐头 β×Fisher-z(FC)，β初值.5；五疾病 query cross-attention；population off |
| E5 | 本外折 fit CN/AD 二分类 stage1；迁移 encoder/prompt projection，重置五类输出与 ordinal head，冻结15轮后全解冻；stage2 lr=stage1/10 |
| E5 三消融 | no_prompt_dropout、no_fc_bias、no_ordinal 各自独立两阶段训练；不是事后只关开关。25/30 折上游未选 ordinal，no_ordinal 没有有效组件差异 |
| 缺失患者提示 | E0_missing_subject、E5_missing_subject 复用各自选定权重，按模型零/null契约评估 |

ordinal 是 4 个共享权重、单调阈值的累积边界，内部标签先映射到假设顺序 `[CN,SMC,EMCI,LMCI,AD]`。所有已知标签的四边界 BCE 都有效；λordinal=1、λrank=.025、margin=.25；ranking 比较不同真实阶段样本的期望阶段差。主输出仍为 softmax。疾病 soft target 按自然序距离 `softmax(−distance/decay)`（decay1）构造，用 KL 替代 hard disease CE。相邻采样先采 fit 受试者再采扫描，四对轮转余额；batch20不意味着每批严格四等份。逆频率权重由 fit 唯一受试者计数，不由测试分布确定。

G 训练 hidden/out116、L4、mean、BN/residual、dropout.5、batch20、Adam5e−5、wd.005、clip1。T 为 AdamW3e−4、wd.01、β=(.9,.98)，prompt dropout ROI .2 / patient .5；Stage2 lr3e−5。legacy NLL 平台15轮学习率减半；T warmup5、cosine50至1e−6后保持；到最低 lr 且 AUROC/NLL 均连续30轮无1e−4实质改善、最低 lr 已运行30轮才停止。冻结期不终止，解冻重置平台状态。**没有最大 epoch**；实际停止84–555轮，不能将84、15、50写成固定训练上限。best 按精确 AUROC/NLL 选取，与停止时的最小改善阈值分开。

恢复保存 model/optimizer/scheduler/RNG/history/control，原子 last 日志与不可变 best 版本绑定，恢复前校验源码/配置/数据/分区/签名并检查进程锁。完整组件结果已存在时验证后跳过。本次没有使用恢复命令；当前所有组件都完成，矩阵恢复栏为 NA。若将来仅定位异常，维护入口为在该工程用 `/home/panyao/miniconda3/envs/brainprompt/bin/python -B -m bpstudy.run controller --config configs/study.json`；**不是本报告要求执行的下一步**，不能删除完成标志、覆盖结果或启动新配置。

### 4.4 MaxFusion 名称和新颖性边界

`M/bpmfusion/{data,models,layers,training,evaluation}` 保留早期工程；`M/subjectcv/{models,trainer,pipeline,run,report}.py` 对应 S20；`M/subjectopt/{planning,training,pipeline,run,report,selection,calibration}.py` 对应 S21。FC 分支使用 116 ROI 非对角上三角 6670 维 FC-MLP，语义 Transformer 分支及固定概率融合/受约束残差依不同研究分别实现。不能把这些架构、004 的 GNN P2P3P5、NComms E0–E5、`BrainPromptMaxFUsion_v2` 的 5 外折 RID 研究当成一个模型。

BrainPrompt+ 排除边界沿用[公开审计](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/BrainPromptPlus_Novelty_Exclusion_List.md)：不将频谱统计语言化节点、FC 空间连接语言化边、五提示语义图重新包装成贡献；保留原版 ROI/subject/disease prompt 与疾病 CE 也不是新贡献。数值 FC 掩码、数值正负通道、类别 query 或 reference 构图与 Plus 机制不同不等于全球首创。ncomms 的多任务二元端点、二阶采样、permutation masker、isotonic calibration 与本研究 CN/AD→五分类、相邻疾病采样、Bernoulli prompt dropout、scalar temperature 是明确适配，不能说逐行复现或直接使用其医学影像预训练权重。

## 5. 已完成实验与完整结果

### 5.1 004 主 TEST：完整 19 方法×3 seed

以下百分数为三个训练 seed 的均值±样本 SD，P/R/F1 均为五类 macro；AUC 为概率 OvR macro。该协议每次评估单位是扫描。QWK、NLL 未在这些历史公开指标中报告，统一 **NA**，不以新标签补算历史序数结果。本次确认服务器 `N/reports/frozen_test_complete_20261011/test_summary.public.json` 与固定公开版本完全一致，独立历史 audit 为63/63通过。

| 方法 | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall / BalAcc | Macro OvR AUC |
|---|---:|---:|---:|---:|---:|
| A0 BrainPrompt | 56.14 ± 5.84 | 28.33 ± 0.65 | 31.19 ± 3.70 | 29.15 ± 1.98 | 68.88 ± 1.54 |
| LegacyPro 旧版 | 57.64 ± 7.22 | 22.45 ± 2.05 | 22.38 ± 2.99 | 23.89 ± 2.42 | 63.71 ± 8.03 |
| A1 P2（预先选定候选） | 55.64 ± 7.17 | 28.25 ± 2.11 | 30.10 ± 2.99 | 29.10 ± 2.68 | 67.65 ± 1.58 |
| A2 P3 | 58.15 ± 3.86 | 25.91 ± 7.43 | 28.19 ± 7.43 | 28.15 ± 9.47 | 67.96 ± 1.80 |
| A3 P5 | 56.64 ± 4.14 | 27.75 ± 2.30 | 31.44 ± 0.23 | 28.57 ± 3.87 | 68.53 ± 1.43 |
| A4 P2+P3 | 58.15 ± 3.86 | 30.49 ± 2.40 | 32.77 ± 3.76 | 31.06 ± 4.50 | 68.36 ± 0.35 |
| A5 P2+P5 | 52.63 ± 5.21 | 27.50 ± 2.46 | 29.55 ± 2.38 | 29.86 ± 2.99 | 66.87 ± 0.63 |
| A6 P3+P5 | 55.39 ± 5.54 | 27.22 ± 0.49 | 28.21 ± 0.32 | 27.50 ± 0.45 | 66.89 ± 1.43 |
| A7 新版完整 P2+P3+P5 | 57.64 ± 1.89 | 27.84 ± 4.43 | 28.61 ± 3.10 | 29.57 ± 5.83 | 66.27 ± 0.97 |
| HistoricalM2 | 63.16 ± 3.45 | 30.57 ± 2.06 | 36.09 ± 2.78 | 30.06 ± 1.96 | 68.80 ± 1.62 |
| HistoricalM3v3 | 58.40 ± 2.85 | 28.04 ± 2.26 | 31.95 ± 2.43 | 27.26 ± 2.77 | 68.38 ± 0.74 |
| HistoricalM3v3_no_metadata | 59.40 ± 4.93 | 28.84 ± 2.25 | 33.28 ± 2.38 | 30.04 ± 3.36 | 69.07 ± 1.30 |
| P2_joint_cons_only | 56.39 ± 6.02 | 28.32 ± 0.59 | 30.74 ± 1.16 | 28.81 ± 1.03 | 68.05 ± 1.22 |
| P2_joint_rec_only | 55.64 ± 4.19 | 26.69 ± 4.62 | 28.25 ± 4.08 | 27.82 ± 3.44 | 66.48 ± 1.53 |
| P3_learned_query | 51.38 ± 11.31 | 26.28 ± 1.49 | 32.83 ± 9.79 | 28.18 ± 0.81 | 67.24 ± 2.80 |
| P3_parameter_matched | 55.64 ± 3.01 | 29.69 ± 4.44 | 35.91 ± 3.42 | 29.45 ± 4.56 | 66.90 ± 0.91 |
| P3_random | 57.39 ± 5.12 | 30.18 ± 2.37 | 33.92 ± 1.38 | 29.73 ± 4.61 | 69.74 ± 1.91 |
| P5_fixed | 56.14 ± 2.64 | 28.67 ± 1.03 | 30.45 ± 1.71 | 29.59 ± 2.00 | 68.07 ± 1.94 |
| P5_no_reliability | 57.89 ± 0.75 | 28.41 ± 1.38 | 32.10 ± 1.58 | 28.25 ± 1.55 | 68.50 ± 2.38 |

完整逐 seed、逐类和成本：[匿名 TEST JSON](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/final_test/test_summary.public.json)；[混淆矩阵](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/final_test/test_confusion.svg)；[19 方法验证表](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/validation_complete/Validation_Results.md)；[配对消融](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/validation_complete/Ablation_Effects.md)。各方法三 seed 平均 TEST Recall 如下，均为百分数，类顺序未改。

| 004方法 | CN | AD | SMC | EMCI | LMCI |
| --- | --- | --- | --- | --- | --- |
| A0 | 75.69 | 38.10 | 8.33 | 0.00 | 23.66 |
| A1 | 72.55 | 42.86 | 0.00 | 0.00 | 30.11 |
| A2 | 82.75 | 19.05 | 8.33 | 16.67 | 13.98 |
| A3 | 76.47 | 33.33 | 8.33 | 0.00 | 24.73 |
| A4 | 76.08 | 28.57 | 8.33 | 11.11 | 31.18 |
| A5 | 67.06 | 33.33 | 16.67 | 0.00 | 32.26 |
| A6 | 70.20 | 28.57 | 0.00 | 0.00 | 38.71 |
| A7 | 76.08 | 33.33 | 8.33 | 0.00 | 30.11 |
| HistoricalM2 | 86.67 | 33.33 | 0.00 | 5.56 | 24.73 |
| HistoricalM3v3 | 80.39 | 33.33 | 0.00 | 0.00 | 22.58 |
| HistoricalM3v3_no_metadata | 81.18 | 38.10 | 8.33 | 0.00 | 22.58 |
| LegacyPro | 77.65 | 9.52 | 0.00 | 0.00 | 32.26 |
| P2_joint_cons_only | 75.69 | 38.10 | 0.00 | 5.56 | 24.73 |
| P2_joint_rec_only | 74.12 | 38.10 | 0.00 | 0.00 | 26.88 |
| P3_learned_query | 60.39 | 23.81 | 8.33 | 0.00 | 48.39 |
| P3_parameter_matched | 74.51 | 28.57 | 8.33 | 11.11 | 24.73 |
| P3_random | 73.33 | 23.81 | 8.33 | 5.56 | 37.63 |
| P5_fixed | 74.90 | 33.33 | 8.33 | 5.56 | 25.81 |
| P5_no_reliability | 78.43 | 38.10 | 0.00 | 0.00 | 24.73 |

A0/A1/A7 主 TEST EMCI Recall 均为0；A1 SMC Recall也为0。A7 Accuracy/Macro-F1/AUC=`57.64/27.84/66.27%`，A0=`56.14/28.33/68.88%`；A1=`55.64/28.25/67.65%`，主五项均值均未超 A0。不能因为 A4 或 HistoricalM2 在 TEST 表更高而改变已锁定候选。63 项测试无全单类塌缩，不等于少数类识别成功。

### 5.2 004 subject-disjoint 补充

| 方法 | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall / BalAcc | Macro OvR AUC |
|---|---:|---:|---:|---:|---:|
| A0 BrainPrompt | 42.28 ± 17.72 | 21.77 ± 4.80 | 23.55 ± 1.75 | 24.20 ± 5.38 | 56.50 ± 1.14 |
| A1 P2（预先选定候选） | 51.22 ± 1.63 | 23.75 ± 1.59 | 23.52 ± 1.42 | 25.58 ± 2.60 | 58.29 ± 0.68 |

该表按 subject-disjoint 划分，但沿用历史测试样本指标，不等于本任务书的受试者平均概率 OOF。A1 F1 相对 A0 为 `+1.99±3.23 pp`，不是三 seed 全提高；Precision `−0.03±0.33 pp`；两者 AD Recall均27.78%、EMCI均0。QWK/NLL=NA。来源：[补充完整结果](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/final_test/Test_Results_subject.md)、[逐类与成本](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/final_test/Final_Analysis.md)。

### 5.3 四项单 seed 探索、十项预训练与机制探针

主61项分类验证=19×3正式比较+下列4项seed42探索，另有subject6项。四项未扩三seed，未进入63项冻结 TEST；不是待补齐的正式任务。

| 探索 | seed | VAL Acc | F1 | P | R/BalAcc | AUC | best | stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2_finetune_rec10 | 42 | 42.86 | 33.73 | 33.26 | 43.25 | 64.99 | 87 | 187 |
| P2_finetune_cons10 | 42 | 56.39 | 33.30 | 35.63 | 33.57 | 64.73 | 125 | 225 |
| P2_finetune_no_dropout_pair_best | 42 | 50.38 | 31.57 | 30.34 | 34.19 | 66.42 | 153 | 233 |
| P2_finetune_no_dropout_pair_last40 | 42 | 58.65 | 32.33 | 46.64 | 35.72 | 62.98 | 89 | 189 |

十项预训练全部 TRAIN-only、固定40轮，best按TRAIN loss、last40同时报告；以下是验证重建指标，不是TEST分类成绩。

| 实验 | best轮 | best MSE | last40 MSE | best相关 | last相关 | TRAIN边均值MSE |
|---|---:|---:|---:|---:|---:|---:|
| P2_pretrain_cons_05 | 40 | 0.056887 | 0.056887 | 0.0593 | 0.0593 | 0.035413 |
| P2_pretrain_cons_10 | 40 | 0.056781 | 0.056781 | 0.0465 | 0.0465 | 0.035461 |
| P2_pretrain_cons_15 | 40 | 0.056738 | 0.056738 | 0.0370 | 0.0370 | 0.035472 |
| P2_pretrain_no_dropout_pair_10 | 39 | 0.036301 | 0.028170 | 0.7093 | 0.7543 | 0.035461 |
| P2_pretrain_normalized_dot_10 | 37 | 0.053836 | 0.053696 | 0.2750 | 0.2780 | 0.035461 |
| P2_pretrain_normalized_pair_10 | 40 | 0.111547 | 0.111547 | 0.2852 | 0.2852 | 0.035461 |
| P2_pretrain_pair_10 | 37 | 0.056171 | 0.056078 | 0.1438 | 0.1513 | 0.035461 |
| P2_pretrain_rec_05 | 40 | 0.056895 | 0.056895 | 0.0630 | 0.0630 | 0.035413 |
| P2_pretrain_rec_10 | 40 | 0.056730 | 0.056730 | 0.0535 | 0.0535 | 0.035461 |
| P2_pretrain_rec_15 | 40 | 0.056710 | 0.056710 | 0.0428 | 0.0428 | 0.035472 |

仅 no_dropout_pair 的 last40 重建 MSE .028170 优于 TRAIN边均值 .035461，但 best .036301 未胜；两种 checkpoint 的下游验证F1为31.57/32.33%，均低于同seed A0 35.86%。原始7项及两个中心化解码候选未胜均值，不能声称重建已普遍成功。来源：[预训练原报告](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/P2_Reconstruction_Diagnostics.md)、[四项探索来源](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/exploration_20261011/Validation_Results.md)。探索来源中的 PARTIAL/NOT_RUN 是当时快照；19方法最终状态以最终报告为准。

25份机制探针与三seed机制汇总覆盖 P2 扰动鲁棒性、重建/Dropout/BN、P3语义与参数对照、P5可靠性/残差/梯度。P2只在部分扰动强度有优势；P3的真实语义读出未稳定击败参数/随机语义控制；P5可靠性图弱作用，fixed对照测试F1 28.67%高于A3 27.75%。P2/P3/P5四背景平均验证F1边际为 `−0.28±0.29 / −1.31±1.58 / −0.48±1.19 pp`；先每seed平均四背景，再跨seed算SD，不能当12个独立数据。P3加入P2背景 `−1.99 pp`、加入P2+P5 `−2.47 pp`；三阶交互 `−2.27±5.19 pp` 是描述性结果。来源：[三seed机制](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/Mechanism_ThreeSeed.md)、[机制诊断](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/Mechanism_Diagnostics.md)、[最终报告](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/research/adni_1to7/Final_Report.md)。

### 5.4 005 扫描级十折实际末轮

| 测试折 | 停止轮数 | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall / BalAcc | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 245 | 61.65 | 24.63 | 24.86 | 25.19 | 71.47 |
| 2 | 240 | 68.42 | 32.87 | 36.38 | 31.91 | 69.15 |
| 3 | 297 | 66.17 | 30.24 | 35.26 | 29.23 | 75.21 |
| 4 | 253 | 62.88 | 21.96 | 21.33 | 23.22 | 72.28 |
| 5 | 268 | 66.67 | 23.98 | 23.33 | 25.21 | 71.78 |
| 6 | 183 | 63.64 | 19.98 | 19.87 | 21.81 | 70.05 |
| 7 | 250 | 67.42 | 31.21 | 31.47 | 31.44 | 73.73 |
| 8 | 253 | 56.06 | 19.00 | 17.93 | 20.28 | 62.60 |
| 9 | 307 | 61.36 | 22.95 | 21.82 | 24.24 | 72.19 |
| 10 | 228 | 59.85 | 20.61 | 19.67 | 22.04 | 70.19 |
| 十折均值±SD | — | 63.41 ± 3.66 | 24.74 ± 4.72 | 25.19 ± 6.38 | 25.46 ± 3.87 | 70.87 ± 3.23 |
| 合并OOF | — | 63.42 | 25.62 | 31.42 | 25.55 | 70.13 |

十折均值±**总体SD（ddof=0）**与合并OOF分别报告；QWK/NLL=NA。OOF Recall CN/AD/SMC/EMCI/LMCI=`86.82/4.62/5.13/0/31.17%`。无全CN折、无全单类折，但 AD/SMC/EMCI 识别明显不足。共同参数 L4/hidden116/dropout.5、batch20、Adam5e−5、wd.005、两个无权重CE；无梯度裁剪/平滑；最多1000轮或每折6小时、val-loss Plateau(.5,25)、lr严格<1e−6停止。实际停止183–307轮，last重载；不是 E0–E5 的无 epoch 上限规则。来源：[原十折报告](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/experiments/adni1to7_brainprompt_cv10_last_20261012/CV10_Test_Results.md)、[含混淆计数的匿名JSON](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/experiments/adni1to7_brainprompt_cv10_last_20261012/cv10_summary.public.json)、[运行证据](https://github.com/yasuo-coder/AD/blob/19d2395c864d9b33fc64547ff84af0b683c4c716/docs/experiments/adni1to7_brainprompt_cv10_last_20261012/RUNBOOK.md)。本次服务器匿名 JSON 与仓库固定版本一致。

### 5.5 E0–E5 全部26变体：当前受试者 OOF 结果

每seed先完整895人OOF，再计算3seed均值±样本SD；主表百分数，NLL为nat，QWK原尺度。Macro-Precision 原管线未汇总，且未预测类别的 per-class precision 按契约为 null，故主表明确 **NA**，不临时选择zero-division口径造出一个数；逐类precision完整保存在附件。Macro-Recall=Balanced Accuracy。

| 实验 | Acc | Macro-F1 | Macro-P | Macro-R/BalAcc | Macro-AUROC | Macro-AP | QWK | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 56.39±0.56 | 17.73±0.98 | NA | 21.03±0.37 | 60.71±0.56 | 25.24±0.56 | 0.074±0.033 | 1.091±0.004 |
| E0_best | 54.23±0.91 | 18.15±0.41 | NA | 20.67±0.19 | 56.01±0.19 | 23.14±0.41 | 0.039±0.020 | 1.102±0.017 |
| E0p | 55.94±0.39 | 16.95±1.88 | NA | 20.71±0.59 | 57.23±1.17 | 23.63±0.60 | 0.036±0.037 | 1.091±0.006 |
| E1_calibrated | 55.94±0.39 | 16.95±1.88 | NA | 20.71±0.59 | 58.43±0.53 | 24.29±0.32 | 0.036±0.037 | 1.082±0.005 |
| E1_youden | 45.25±0.70 | 23.78±1.01 | NA | 24.59±0.75 | 57.23±1.17 | 23.63±0.60 | 0.163±0.030 | 1.091±0.006 |
| E1_tta | 56.01±0.28 | 17.01±1.76 | NA | 20.73±0.58 | 57.26±1.23 | 23.65±0.65 | 0.034±0.038 | 1.091±0.006 |
| E1_combined | 45.14±1.65 | 23.29±1.33 | NA | 24.09±0.95 | 58.44±0.49 | 24.29±0.27 | 0.169±0.044 | 1.081±0.005 |
| E2_focal | 36.05±5.00 | 16.38±1.02 | NA | 22.90±0.44 | 54.79±1.37 | 22.81±0.05 | 0.100±0.051 | 1.448±0.021 |
| E2_cb_focal | 43.24±1.95 | 18.19±0.71 | NA | 23.89±1.27 | 54.88±1.70 | 23.43±0.45 | 0.099±0.023 | 1.439±0.009 |
| E2_adjacent | 20.48±1.93 | 12.18±0.56 | NA | 23.40±3.56 | 53.83±0.75 | 21.71±0.32 | 0.021±0.013 | 1.684±0.067 |
| E2_ordinal | 54.90±1.19 | 16.84±0.84 | NA | 20.33±0.19 | 55.28±0.88 | 22.92±1.41 | 0.021±0.013 | 1.101±0.021 |
| E2_soft | 55.68±1.06 | 16.08±1.13 | NA | 20.30±0.32 | 55.94±0.89 | 23.25±0.19 | 0.024±0.009 | 1.100±0.010 |
| E2_all | 3.39±0.17 | 3.08±0.89 | NA | 24.63±2.62 | 54.66±1.71 | 21.76±0.29 | 0.058±0.023 | 2.777±0.145 |
| E2_selected | 42.98±3.72 | 18.63±0.66 | NA | 20.28±1.07 | 53.35±0.57 | 22.74±0.41 | 0.043±0.016 | 1.323±0.054 |
| E3_fisher | 44.21±2.11 | 19.94±1.16 | NA | 20.77±0.69 | 53.78±0.43 | 22.78±0.95 | 0.052±0.026 | 1.315±0.016 |
| E3_multiscale | 44.13±2.83 | 20.72±1.73 | NA | 22.13±1.93 | 55.69±0.97 | 23.54±0.65 | 0.080±0.049 | 1.281±0.019 |
| E3_signed | 42.31±4.20 | 20.41±1.52 | NA | 21.73±1.74 | 55.72±0.60 | 23.05±0.44 | 0.101±0.007 | 1.319±0.031 |
| E3_all | 43.76±3.28 | 20.28±1.54 | NA | 21.94±1.29 | 55.22±1.04 | 22.97±1.13 | 0.091±0.028 | 1.312±0.030 |
| E3_selected | 44.36±2.74 | 20.03±1.07 | NA | 20.79±1.49 | 53.93±0.49 | 22.85±0.93 | 0.078±0.057 | 1.307±0.024 |
| E4 | 36.01±3.94 | 14.96±1.88 | NA | 20.37±0.18 | 50.43±0.29 | 20.82±0.55 | -0.000±0.003 | 1.511±0.063 |
| E5 | 39.03±3.97 | 15.14±1.64 | NA | 19.63±0.36 | 50.35±0.36 | 20.98±0.62 | -0.001±0.006 | 1.366±0.061 |
| E5_no_prompt_dropout | 36.05±5.99 | 15.46±3.30 | NA | 20.29±1.04 | 51.80±0.80 | 21.84±0.47 | 0.016±0.017 | 1.441±0.045 |
| E5_no_fc_bias | 36.69±6.42 | 13.97±1.41 | NA | 19.97±0.23 | 50.15±0.36 | 20.94±0.34 | -0.007±0.009 | 1.444±0.078 |
| E5_no_ordinal | 34.00±4.83 | 14.23±3.10 | NA | 19.08±0.84 | 50.37±0.41 | 20.80±0.05 | -0.004±0.003 | 1.411±0.084 |
| E0_missing_subject | 50.39±1.55 | 23.78±2.03 | NA | 24.26±1.62 | 58.18±0.49 | 24.12±0.11 | 0.203±0.054 | 1.385±0.033 |
| E5_missing_subject | 38.99±3.93 | 15.02±1.52 | NA | 19.28±0.75 | 50.39±0.38 | 20.69±0.16 | -0.005±0.001 | 1.366±0.061 |

| 实验 Recall(%) | CN | AD | SMC | EMCI | LMCI |
| --- | --- | --- | --- | --- | --- |
| E0 | 94.30±2.88 | 0.00±0.00 | 0.00±0.00 | 10.87±4.34 | 0.00±0.00 |
| E0_best | 87.94±3.09 | 0.00±0.00 | 0.00±0.00 | 15.41±2.93 | 0.00±0.00 |
| E0p | 94.57±5.19 | 0.00±0.00 | 0.00±0.00 | 8.96±8.14 | 0.00±0.00 |
| E1_calibrated | 94.57±5.19 | 0.00±0.00 | 0.00±0.00 | 8.96±8.14 | 0.00±0.00 |
| E1_youden | 53.68±2.76 | 8.21±6.22 | 3.51±3.04 | 44.92±2.16 | 12.64±5.27 |
| E1_tta | 94.70±4.56 | 0.00±0.00 | 0.00±0.00 | 8.96±7.46 | 0.00±0.00 |
| E1_combined | 53.55±3.69 | 8.72±7.27 | 1.75±3.04 | 44.92±1.15 | 11.49±5.27 |
| E2_focal | 60.50±9.54 | 7.18±3.20 | 14.04±6.08 | 0.60±0.41 | 32.18±13.94 |
| E2_cb_focal | 73.49±5.12 | 8.72±4.70 | 12.28±8.04 | 0.84±1.15 | 24.14±13.79 |
| E2_adjacent | 0.00±0.00 | 5.13±1.78 | 29.82±16.08 | 60.22±8.07 | 21.84±8.68 |
| E2_ordinal | 92.71±4.27 | 0.00±0.00 | 0.00±0.00 | 8.96±4.07 | 0.00±0.00 |
| E2_soft | 96.02±3.97 | 0.00±0.00 | 0.00±0.00 | 5.50±4.42 | 0.00±0.00 |
| E2_all | 0.00±0.00 | 3.08±5.33 | 64.91±12.15 | 0.00±0.00 | 55.17±9.12 |
| E2_selected | 64.61±8.96 | 4.62±6.71 | 10.53±13.93 | 19.35±11.99 | 2.30±1.99 |
| E3_fisher | 63.68±6.65 | 4.62±5.55 | 7.02±3.04 | 25.09±5.92 | 3.45±5.97 |
| E3_multiscale | 62.29±6.83 | 5.13±6.41 | 14.04±3.04 | 26.88±5.06 | 2.30±3.98 |
| E3_signed | 58.98±8.27 | 2.56±0.89 | 5.26±5.26 | 26.88±10.65 | 14.94±7.18 |
| E3_all | 63.29±5.84 | 1.03±0.89 | 5.26±0.00 | 24.01±8.27 | 16.09±1.99 |
| E3_selected | 62.89±8.34 | 4.62±6.71 | 7.02±8.04 | 27.12±7.69 | 2.30±1.99 |
| E4 | 59.84±10.04 | 3.59±6.22 | 14.04±12.15 | 3.70±5.22 | 20.69±10.34 |
| E5 | 66.53±5.80 | 3.08±5.33 | 15.79±9.12 | 2.39±4.14 | 10.34±0.00 |
| E5_no_prompt_dropout | 59.77±10.04 | 3.59±3.20 | 12.28±13.25 | 3.94±4.35 | 21.84±23.47 |
| E5_no_fc_bias | 63.22±11.54 | 3.08±5.33 | 14.04±16.08 | 0.00±0.00 | 19.54±1.99 |
| E5_no_ordinal | 56.39±5.80 | 6.67±11.55 | 14.04±19.93 | 3.35±5.79 | 14.94±11.08 |
| E0_missing_subject | 72.76±2.76 | 12.82±1.78 | 0.00±0.00 | 26.52±8.60 | 9.20±1.99 |
| E5_missing_subject | 66.53±5.80 | 3.08±5.33 | 14.04±10.96 | 2.39±4.14 | 10.34±0.00 |

全部逐seed受试者与扫描级指标、每类precision/recall/F1/AUROC/AP、混淆矩阵见 [E0–E5匿名指标](docs/reports/issue3-20261027/E0_E5_metrics.public.json) 的 `per_seed`；`variants` 为三seed汇总。概率 AP 使用 average precision，不混称梯形 PR-AUC。扫描结果未混入主表。E5−E0p 的预声明配对受试者分层 bootstrap：AUROC `−6.882 pp`，95%CI `[-10.431,-3.172]`；F1 `−1.817 pp`，CI `[-3.135,-0.634]`。2000次重采样保留同一受试者的三seed预测；区间以已训练OOF为条件，不含重训/重新选型不确定性。E5不支持完整方案改善，不能用E1 Youden较高硬分类F1改写主方案。

### 5.6 MaxFusion 与后续独立研究的最新事实

这些研究共享历史暴露队列，但输入、FIT规模、损失、选模预算等不同，不能据整行差值归因单模块。S21 A0–A7等新名不与004合并；`BrainPromptMaxFUsion_v2` 的旧MCI→LMCI RID五外折研究也不与新受试者十折混排。

| 独立研究（受试者） | Acc | Macro-F1 | Macro-P | Macro-R/BalAcc | Macro-AUROC | Macro-AP | QWK | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S21 A3_capacity_selected | 51.62±1.07 | 26.38±1.08 | NA | 26.54±0.79 | 59.02±0.85 | 24.69±0.24 | NA | NA |
| S21 MaxFusion_optimized | 51.62±1.07 | 26.38±1.08 | NA | 26.54±0.79 | 59.02±0.85 | 24.69±0.24 | NA | NA |
| S24 G_original | 55.79±0.26 | 15.87±0.42 | NA | 20.25±0.20 | 62.23±0.66 | 25.53±0.35 | 0.032±0.004 | 1.044±0.002 |
| S24 T_query | 56.20±0.00 | 14.39±0.00 | NA | 20.00±0.00 | 50.85±0.67 | 21.04±0.62 | 0.000±0.000 | 1.076±0.004 |
| S24 T_patient | 56.09±0.19 | 14.42±0.06 | NA | 19.97±0.05 | 55.45±0.16 | 22.97±0.51 | 0.002±0.003 | 1.066±0.002 |
| S24 fit_prior | 56.20±0.00 | 14.39±0.00 | NA | 20.00±0.00 | 48.58±0.00 | 19.80±0.00 | 0.000±0.000 | 1.071±0.000 |
| 10/26 FC f1_best | 55.01±0.55 | 24.25±0.68 | 25.46±0.60 | 24.60±0.66 | 61.70±0.44 | NA | NA | 2.004±0.106 |
| 10/26 FC macro_nll_best | 55.01±1.27 | 24.97±0.20 | 27.37±0.69 | 24.99±0.29 | 62.91±0.47 | NA | NA | 1.446±0.045 |
| 10/26 R1 | 56.20±0.11 | 14.45±0.08 | 17.92±11.54 | 20.01±0.06 | 52.95±2.54 | 21.68±0.63 | NA | 1.132±0.043 |

以上数值、逐类召回和可用的混淆计数见[附加研究匿名汇总](docs/reports/issue3-20261027/additional_studies.public.json)。NA表示该引用汇总未提供，不能解释为0。

S21 30/30折融合未启用，最终等于A3；SMC/真实LMCI平均召回仅1.75/5.75%。PatientRepair T_patient相对T_query的AUC提升未超G，T_patient 28/30折仅预测一类，AD/SMC/LMCI召回为0。固定FC方案B相对A的F1变化 `+0.7176 pp`，仅2/3seed提高；AUC `+1.2088 pp`，自然NLL下降，但两者SMC/LMCI召回都为0。其B在全部30折选到第5轮，不能据此再把epoch5确立为新TEST驱动超参数。

R1 在10月26日实际已完成，采用sum readout和原始cosine乘积人口图，阈值 .4/.6/.8 在本折STOP Macro-NLL选择，使用各候选末轮。当前受试者CN Recall约99.93%，AD/SMC/LMCI均0，EMCI约0.12%，仍接近多数类预测。它和 E0p 的 soft_topk/reference-only 修复不同。本次对R1仅重新核验90个完成结果、last hash、30组TEST产物及worker退出码，未扩写为完整独立全流程复算通过。

旧MaxFusion目录还保留20轮预算的早期subject_nested、样本级5×3×3seed GNN/Transformer收敛研究及F0–F11（660次训练、停止151–165轮），它们均为另外的旧标签/划分协议，不进入新主表。F0 F1 30.71±2.43%、F1 MLP-only30.52±2.10%、F7参数匹配MLP31.76±2.23%，同样不能支撑语义必然有效；原报告在`M/results/adni7_f0_f11_v1/REPORT.md`。历史旧五批锁定TEST仍属于未运行的历史路线，不以七批结果替它完成。

## 6. E0–E5 当前执行矩阵与恢复状态

**逐项明细不是计划矩阵。** [780行执行CSV](docs/reports/issue3-20261027/E0_E5_execution_matrix.csv) 每行一个实际 experiment×seed×outer_fold，列出模型/配方、选择来源、代码完整SHA、状态、best/last、实际选中与停止epoch、验证依据、TEST状态、指标、结果路径、失败与恢复。外折在CSV为0–9，与目录一致。E1和selected输出共享训练来源，不能把780组预测说成780次训练。

[630行组件CSV](docs/reports/issue3-20261027/E0_E5_training_components.csv) 单列各 stage1/stage2、实际best/stop、参数数、验证选模值、best/last哈希及封印。每外折13个单阶段组件+4×2个E5两阶段组件=21，30×21=630。下表为真实计数与范围；确切每seed每fold轮次以CSV为准，不用范围替代逐项记录。

| 训练组件 | seed×外折 | 完成 | 状态 | 使用权重 | 实际停止epoch范围 | TEST用途 | 失败/恢复 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | last主结果 / best诊断 | 123–295 | 已评估 | 无 / NA |
| E0p | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 123–256 | 已评估 | 无 / NA |
| E2_focal | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 128–234 | 已评估 | 无 / NA |
| E2_cb_focal | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 128–243 | 已评估 | 无 / NA |
| E2_adjacent | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 135–238 | 已评估 | 无 / NA |
| E2_ordinal | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 121–212 | 已评估 | 无 / NA |
| E2_soft | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 121–234 | 已评估 | 无 / NA |
| E2_all | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 121–274 | 已评估 | 无 / NA |
| E3_fisher | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 133–309 | 已评估 | 无 / NA |
| E3_multiscale | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 139–555 | 已评估 | 无 / NA |
| E3_signed | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 131–235 | 已评估 | 无 / NA |
| E3_all | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 145–201 | 已评估 | 无 / NA |
| E4 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 84–84 | 已评估 | 无 / NA |
| E5_stage1 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 84–84 | 不单独TEST；仅迁移 | 无 / NA |
| E5_stage2 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 99–361 | 已评估 | 无 / NA |
| E5_no_prompt_dropout_stage1 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 84–511 | 不单独TEST；仅迁移 | 无 / NA |
| E5_no_prompt_dropout_stage2 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 99–323 | 已评估 | 无 / NA |
| E5_no_fc_bias_stage1 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 84–84 | 不单独TEST；仅迁移 | 无 / NA |
| E5_no_fc_bias_stage2 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 99–362 | 已评估 | 无 / NA |
| E5_no_ordinal_stage1 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 84–84 | 不单独TEST；仅迁移 | 无 / NA |
| E5_no_ordinal_stage2 | 42/2024/3407 × 0–9 | 30/30 | COMPLETE | best | 99–351 | 已评估 | 无 / NA |

E0_best、E0_missing_subject为E0复用；E1四种各30/30复用E0p；E2_selected/E3_selected各30/30为验证选中结果；E5_missing_subject 30/30复用E5 stage2。独立训练、后处理、selection输出均齐全，缺失0。全部 TEST 已评估，不可再用于选型。各组件失败原因=`none`，恢复=`NA: complete; do not rerun`。

smoke、37项单元/公式/梯度/恢复测试不计正式拟合；CPU/GPU各21组件×26变体短程验收在 `E/smoke/`，没有复用其权重。正式目录在 `E/units/`；实际epochs与首次收敛由历史逐轮记录重放，本次没有新训练或新推理。PatientRepair的50项测试、20名FIT受试者容量诊断100%拟合也是软件/容量证据，不进性能表。

## 7. 失败、局限与文档冲突

| 已核实问题 | 证据与正确解释 |
|---|---|
| 原 population 空图 | 数学上界+代码/真实验收；不等于整个残差分支恒零，修复不自动等于有效 |
| 004 控制器故障 | 首次 Phase8 因混读队列schema `KeyError running` 在建锁前失败，零TEST访问；39ee2f2独立恢复目录完成63项，未重训、未重复TEST |
| P2重建饱和 | 原dot接近常数/均值；关预训练Dropout仅改善部分重建，不改善下游分类 |
| P3语义对照失败 | 随机/参数对照未被稳定击败；注意力不同或梯度非零不能证明疾病语义机制 |
| P5弱作用与组合负交互 | 可靠性偏高、边权近均匀、残差小、edge梯度弱；P2/P3/P5四背景平均收益为负 |
| S21 汇总脚本失败 | 原训练完成后局部变量命名冲突；独立`analysis/subjectopt_report_fixed_20261021.py`两行命名修复，保留原封印，`report_recovery.json`记录；这是报告恢复，不是训练失败或新增模型 |
| 最新文档滞后 | `M/STATE.md`止于10月21日，未覆盖10月26日；`BASELINE_COMPARISON.md`写R1未训练，但90结果/锁/日志证明已完成；本报告依据实际产物明确更正，不改写原记录 |
| 同名ID冲突 | 004 A0–A7、S21 A系、旧v2 E0–E5、新任务书E0–E5分别定义，不能合表计数 |
| E5消融可解释性 | 25/30折没有ordinal可去除；不能把no_ordinal随机重训差异全称ordinal收益 |

四个证据等级必须分开：**代码验收通过**（形状/梯度/确定性/恢复）≠**验证集有效**（受选择偏差影响）≠**冻结测试有效**（004主候选未获确认、E5负结果）≠**外部泛化成立**（本项目没有此证据）。性能目标未达不应伪装成工程未完成，工程完成也不应伪装成研究成功。

已开发/测试暴露的队列包括历史五批开发集、004主/补充holdout、005扫描十折、MaxFusion旧样本研究与S20/S21、E0–E5、PatientRepair、10月26日FC和R1。新旧协议患者/扫描角色交叉，改变seed或重切十折不会使队列恢复“未见”。不得继续依据这些TEST的排名调模块、阈值、校准、checkpoint或统计口径。旧五批锁TEST即使未正式评价，也不能因历史多协议交叉就宣称它为独立外部队列。

895人中SMC19、真实LMCI29，单个验证折通常仅约2–3人，少数类一次错判就会大幅改变召回、AUROC和阈值；年龄/性别/来源/未明确站点可能形成混杂。三seed重复同一批患者，30折训练集高度重叠，不能普通t检验当30独立样本；三seedSD不是患者CI。bootstrap也没有覆盖重新训练、重新选型的不确定性。内部自然序只是研究假设，不证明SMC必然按该路线进展。缺失指标以NA记录，不造0/1，不混用扫描与受试者指标。

本次交付的可复核边界：004/005与受控服务器聚合报告一致；E0–E5本次完整重放630组件和780组预测及OOF；其他研究以已存在报告/核验为基础，R1另核对90个checkpoint但没有追加全流程独立核验。没有公开数据、受试者ID或逐例预测，也没有将本地私有缓存上传作证据。

## 8. 给 Kimi 3.1 Pro 的待审问题

1. 在已完成E0–E5、S24和R1均存在负结果的情况下，下一阶段应优先审查population信息流、序数监督、FC表征/选模，还是BrainPrompt-T患者读出？需要什么证据才能决定优先级，而不依据已见TEST重新挑选？
2. 哪些历史模块应只保留为负结果对照，哪些因稳定性不足、成本或BrainPrompt+重合停止；如何避免把P2/P3/P5已失败的机制换名重试？
3. SMC19人、真实LMCI29人的条件下，macro-AUROC、macro-AP、QWK、少数类召回、自然NLL/Macro-NLL和校准应如何预先约束选择；验证集如此小是否需要改变未来研究的选择设计？
4. 下一阶段最小消融怎样区分修图与best/last、机制与参数增长、prompt语义与一般参数、读出与损失、单次随机性与选择方差？E5的25折无ordinal有效差异应如何处理？
5. 如何建立尚未参与历史开发的确认性评估边界；是否应先补充R1独立停止/来源/指标核验及下载标签/站点时点审计，再决定任何新优化训练？
6. 是否接受本报告以实际产物纠正文档滞后，并将新任务书E0–E5、MaxFusion S21、PatientRepair及10月26日研究分别作为事实基线？
