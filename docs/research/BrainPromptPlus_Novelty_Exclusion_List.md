# BrainPromptPlus full-text novelty audit

审计日期：2026-10-08。审计对象为 BrainPrompt+（BrainPromptPlus），不是原 BrainPrompt，也不是其他同名论文。此稿只审计方法和重叠，不修改模型、不训练、不声称任何候选已获得性能提升。

## 1. 核验来源与范围

- 用户提供完整论文：`/data2/panyao_data/ADNI/BrainPrompt+.pdf`，11 页，期刊页码 4150–4160，DOI `10.1109/TMI.2026.3692958`；SHA256 `16f5b593f95b45431f2938243c36c16ecbfa427ed296349de3026ee1361979df`。
- 官方出版信息：[IEEE](https://ieeexplore.ieee.org/document/11518539/)，[PubMed](https://pubmed.ncbi.nlm.nih.gov/42127068/)。全文按用户提供 PDF 阅读；网络摘要未代替全文。
- 官方代码：[AngusMonroe/BrainPromptPlus](https://github.com/AngusMonroe/BrainPromptPlus)，固定 commit `87932c1e6d03d0f4fb5c9aeaca00b043e51b6b14`（2026-05-14 11:05:00 +08:00）。只读 checkout：`/tmp/brainprompt_final_audit_plus/BrainPromptPlus`。
- 已阅读 PDF 11 页提取文本，包括引言、相对会议版扩展、全部方法与公式 1–8、实验设置/结果解释/消融说明、局限与未来工作、参考文献；官方网络框架图已视觉核对。PDF 的表格数字未在文本提取中完整保留，本报告不转录或据此报告表格数值。
- 已阅读该 commit 全部 `.py`、`.sh`、`.json` 和 `spectral_spatial_prompt.ipynb` 代码单元。仓库框架图已视觉核对。未声称其默认入口能直接复现论文，代码存在接线差异。
- 原 BrainPrompt 对照：官方 `c85304a023636a2701d465e4cb0acb13b637127e`，对应论文 [arXiv](https://arxiv.org/html/2504.16096v2)。当前 Pro 对照为 `/home/panyao/AD/BrainPromptPro/layers/{adaptive_brain_graph,disease_aware_readout,multirelation_population_graph}.py` 的实际实现。

## 2. 核心判断

论文 §I-D（4151–4152）明确声明：相对原 BrainPrompt 的方法扩展是增加 **spectral prompt 与 spatial prompt**，把原来的三层提示扩展为五层，并以语言嵌入构建脑图节点/边表征；其他扩展是 PPMI、泛化、解释性等实验验证。

论文没有在方法段提出疾病查询 ROI 注意力读出、按 age/sex/site 分别学习连续人群图、新的 supervised-contrastive 或 prototype 损失。其 readout 是普通 sum，population graph 仍为两个相似度乘积的固定阈值二值图，总 loss 仍是分类 CE 与疾病文本对齐 CE 之和。这些不是根据函数名字猜测，而是 §III-B/C/D、公式 3–8 与 forward/loss 调用链共同支持的结论。

**限制：未在此论文/此 commit 中发现，不代表在全部相关文献中首次提出。** 通过 Plus firewall 只是排除重复 Plus 的必要条件，不能代替全领域新颖性检索、实验有效性或因果证明。

## 3. BrainPromptPlus_Novelty_Exclusion_List

以下为新的 Pro 不得直接采用或改名复用的 Plus 核心结构。基础 GNN、MLP、attention 不因同名自动禁止。

| ID | 需排除的具体机制 | 论文证据 | 官方代码证据 | 禁止范围 / 边界 |
|---|---|---|---|---|
| PX-01 | 将每个 ROI 的频谱统计及 Slow-5/Slow-4/Classical 频带信息组织成自然语言，再用冻结 LLM 编码作为个体脑图节点特征 | §I-D，§III-A，Eq. (1) 中节点 `x_v=Enc(p_node_v)`；4153 | `data/prompts/spectral_spatial_prompt.ipynb` cell 10 的 `generate_spectral_prompts_from_bold`；cell 12/15 保存；`nets/brainprompt_net.py:41,49`；训练入口 `train_TUs_graph_classification_llm.py:35–37` | 不得改成不同 FFT 带宽、提示措辞、MLP 层数后宣称新模块。普通数字频谱特征本身不是 Plus 独占，但任何基于频谱的新方案仍需独立审核，且本轮 FC-only 数据契约不能凭空恢复频谱。 |
| PX-02 | 把 ROI 对的功能连接强弱、正负及脑区名称语言化，再将冻结 LLM 的编码作为边特征 | §III-A，Eq. (1) 中边 `e_vu=Enc(p_edge_vu)`；4153–4154 | 同 notebook cell 11 `generate_spatial_prompts_from_bold`、cell 12；`nets/brainprompt_net.py:32,51` | 不得仅更换相关性阈值、同义句、边投影维度后重命名。保留 signed FC 数值并学习数值门控，与把连接描述文本送入冻结 LLM 不是同一机制，但要逐候选比较。 |
| PX-03 | 以 spectral 节点提示 + spatial 边提示构建语义脑图，再由边提示表征调制 GNN 消息的联合结构 | §III-A Eq. (1)–(2) 与 §III-B Eq. (3) | notebook cell 10–12；`nets/brainprompt_net.py:49–57`；`layers/gcn_layer.py:28–81` | 禁止原样复用这条信息流、重排或轻微修改。一般的 edge-weighted GNN、数值边门控并不自动属于禁用范围。 |
| PX-04 | 将上述 spectral/spatial 输入图与继承的 ROI 身份增强、subject population、disease semantic supervision 组合成五层提示系统并声称新的系统创新 | §I-D、§III、Fig. 1、Eq. (1)–(8) | `nets/brainprompt_net.py` 完整 forward/loss；prompt notebook；`layers/attention_layer.py:263–282` | 不得仅改变模块顺序、投影、激活、loss 权重、维度后将五层提示组合包装为 Pro 新创新。 |

### 继承机制：可以作为原 BrainPrompt 基线保留，但不能作为新 Pro 核心创新

| ID | 机制 | 论文位置 | 代码位置 | 审计归类 |
|---|---|---|---|---|
| BP-I1 | 解剖/功能 ROI 文本经冻结编码器与线性映射，加到 ROI 节点表示 | §III-B Eq. (4)，4154 | `layers/attention_layer.py:263–282`; `nets/brainprompt_net.py:38,54` | 原 BP 已有，非 Plus 相对 BP 新增 |
| BP-I2 | 将 age/sex/site 写成 subject prompt，图表示与提示表示的相似度相乘、阈值二值化，单个 population GCN，再融合 metadata | §III-C Eq. (5)–(7)，4154 | `nets/brainprompt_net.py:62–78,127–132` | 原 BP 已有，Plus 延用；不可仅调阈值或重命名当新创新 |
| BP-I3 | 将全部疾病阶段文本编码，与最终受试者表示进行归一化相似度分类，辅助 CE | §III-D Eq. (8)，4154–4155 | `nets/brainprompt_net.py:81,86–106` | 原 BP 已有；不等于新的 ROI 查询读出或 supervised contrastive |
| BP-I4 | 冻结 Llama-encoder-1.0B，提示离线计算一次、训练/推理复用 | §III 总述，4152–4153 | `language_models/text2emb.py:8–32`；notebook cell 6–7 | 原 BP 共用组件；不能把 encoder 冻结/缓存当新论文创新 |

### 未被 Plus 提出为新的方法的类别

- **Readout：** Eq. (5) 明确普通 sum pooling；代码允许 sum/max/mean，当前配置 mean。没有 disease prompt query 的 `softmax(QK^T)V` ROI 聚合。
- **Population：** Eq. (6)–(7) 是 image/subject-text 两个相似度的乘积阈值图。没有三个 age/sex/site 独立可微 adjacency learners，也没有 relation-specific message passing + learned fusion 的方法。
- **Loss：** Eq. (8) 只有 `L_cls + λ L_disease`。代码 `compute_dis_loss` 只是未调用的函数；`lambda2` 也未参与返回 loss，不能记成已使用的新损失。普通 CE、Adam、scheduler、dropout 均非核心创新。
- **Representation：** Plus 新增的是频谱及空间连接的语言化语义表征。没有在方法部分加入类别原型学习、supervised contrastive 学习或多模态对齐新目标。§II 中讨论这些相关工作不等于自己提出或采用。
- **训练框架：** 固定 LLM 的离线预计算，加端到端训练 GNN/投影器；§IV-B 用 8:1:1、十折、Adam、验证损失调度停止。没有另一个新颖的多阶段预训练/蒸馏/元学习训练框架。

## 4. 逐式对应与实现差异

| 论文公式 | 论文计算含义 | 可定位代码 | 注意事项 |
|---|---|---|---|
| (1) | spectral prompt→节点、spatial prompt→边，LLM + 对齐线性层 | notebook cells 10–12; `brainprompt_net.py:32,41,49–51` | 生成器实现清楚；发布的通用 dataloader 会覆盖/丢失这些信息，见下文 |
| (2) | 节点集、边集、节点矩阵、边矩阵构成图 | notebook cell 12 DGL 图写入 | 论文 top 20% absolute FC；notebook cell 11 实际选最强 10% 和最弱 10% absolute FC，共约20%，两者不等价 |
| (3) | 边特征经映射调制邻居节点消息后聚合 | `layers/gcn_layer.py:53–65` | 论文写求和、边映射为 scalar、先变换节点；代码是向量边逐通道乘源节点、mean 聚合后 Linear。不能称公式严格逐字实现 |
| (4) | ROI 语义增强每层 GCN 输入 | `brainprompt_net.py:54–57`; `LMPE.forward` | 公式按层描述；代码在进入 GCN 循环前加一次 ROI prompt |
| (5) | sum readout 得到 subject representation | `brainprompt_net.py:62–69` | 代码支持 mean/sum/max，仓库默认配置为 mean |
| (6) | brain/subject prompt 两个 cosine similarity | `brainprompt_net.py:74–75,127–132` | 代码返回 sigmoid(cosine)，不是原始 cosine |
| (7) | 两种相似度乘积大于阈值时连边 | `brainprompt_net.py:76` | 代码固定 0.5 并 dropout；论文 th 是验证集搜索参数 |
| (8) | 分类 CE + λ × 疾病对齐 CE | `brainprompt_net.py:86–106` | 已接入；没有 `compute_dis_loss` 调用、没有监督对比项 |

关于 Eq. (7)：§IV-G（4159）文字称增大阈值导致更密、减小阈值丢更多边，与 `A'=1[score>th]` 的数学关系相反。这是论文解释与公式的内部不一致，应按公式/代码判断，不能复制相反因果解释。

### 发布代码的连通性问题，不改变 novelty 排除结论

1. `nets/gcn_net.py:13` 导入 `GCNLayer`，但该 commit 的 `layers/gcn_layer.py` 只定义 `GCNELayer`、`simpleGCNLayer`；`nets/load_net.py` 无条件导入前者。
2. `baseline.sh:1,9` 使用 `_prompt` 目录/文件名，而配置只在非 `_prompt` 目录；配置 model 又是 `BrainPromptG/C`，registry 仅保留 `BrainPrompt+`。
3. `llm-main.py:335` 调用 `LoadData_llm(..., spatial=...)`，但 `data/data.py` 不接收该关键字。
4. `llm-main.py:98–101` 仅在旧 `BrainPromptG/C` 名称下装载 `label_embs`，而 Plus 构造器必需此值。
5. `data/BrainNet.py:146–160` 对稀疏 prompt 图重新构建全图，仅拷贝原 BOLD `N_features`，导致 `ALFF`（此处实际是 spectral text embedding）丢失；随后用 Pearson 数值覆盖 `E_features`，spatial text embedding 不再存在。
6. 训练函数 `train_TUs_graph_classification_llm.py:35–37` 给 Plus 输入 `ndata['ALFF']`；两个评估函数没有对应分支，用 `ndata['feat']`。pearson 设置为 116 维，网络 `emb_transform` 需要 2048 维，路径不一致。
7. notebook cell 10 的 `fALFF` 与 `PowerRatio` 实际采用相同表达式，故数值必然相同；cell 9 的早期“0–80Hz”模板被 cell 10 重定义覆盖，不能把早期草稿模板当最终方法。
8. 多分类评估函数将 P/R/F1/AUC 直接赋值 Accuracy (`train_TUs_graph_classification_llm.py:132–136`)，不能用于本项目的 macro 指标；其二分类 AUC 也用 argmax 而非概率 (`metrics.py:roc_auc`)。本项目应坚持自己的固定、真实指标定义。

这些问题说明不能以此 commit 默认脚本能否跑通来否认论文已公开的创新，也不能凭“修复 Plus 代码”把该方法变成 Pro 原创。

## 5. 对现有 M1/M2/M3 的具体 overlap check

以下仅针对已经存在、明确可读的 V1 代码；未来 V2/V3 必须带确切公式、输入和信息流重新审核。此处 ALLOW 只表示该具体机制未复现 Plus 核心创新，不表示允许免实验进入最终 Top-3，也不证明全球首创。

### M1 V1 — 当前 SemanticGuidedAdaptiveBrainGraph

- Problem addressed：固定脑图不能按个体数据及任务学习有效边。
- Core mechanism：完整数值 FC 行与 ROI text embedding 分别投影融合；每对节点输入 `[FC_ij, |FC_ij|, |h_i-h_j|, h_i⊙h_j]` 到可训练 MLP，sigmoid 后对称化，得到个体连续 adjacency；在分类训练中反向传播。
- Closest BrainPromptPlus component：spectral/spatial 语义图 + edge-feature GCN。
- Similarity：**Medium（目标及脑图层级相近，核心输入与计算不同）**。
- Does it reproduce a BrainPromptPlus novelty：**NO，仅针对这段 V1 实现**。
- Decision：**ALLOW for further validation of the specified V1; V2/V3 not yet reviewed**。
- Evidence：当前 `/home/panyao/AD/BrainPromptPro/layers/adaptive_brain_graph.py:21–45` 没有 spectral 文本编码、没有 ROI-pair coupling 文本编码；Plus §III-A 先离线编码频谱/连接句子，固定相关性选边，§V明确 learned graph structures 未系统研究。
- 必须保留边界：不能将 M1 V2 改成 `LLM(连接描述)` 边特征 + `LLM(频谱描述)` 节点，再改名语义自适应图。

### M2 V1 — 当前 DiseaseAwareSemanticReadout

- Problem addressed：等权聚合可能抹去疾病相关 ROI 差异。
- Core mechanism：全部 K 类 disease embeddings 作为 query，ROI 表示为 key/value；类条件 ROI attention 产生 K 个向量，再按自身兼容性融合，不使用真实类别选择 query。
- Closest BrainPromptPlus component：Eq. (5) 普通 readout + Eq. (8) 最终 subject/disease semantic alignment。
- Similarity：**Low（共用 disease prompt 来源，但作用位置/计算明显不同）**。
- Does it reproduce a BrainPromptPlus novelty：**NO，仅针对这段 V1 实现**。
- Decision：**ALLOW for further validation of the specified V1; V2/V3 not yet reviewed**。
- Evidence：当前 `/home/panyao/AD/BrainPromptPro/layers/disease_aware_readout.py:19–30`；Plus `brainprompt_net.py:62–69` 无 class-conditioned pooling，`81,95–106` 只对末端表示做语义分类。
- 不能把保留 BP 的疾病对齐 CE 本身列为 M2 新贡献；需要验证疾病间 attention 差异和实际梯度。

### M3 V1 — 当前 MultiRelationPopulationGraph

- Problem addressed：单一乘积相似度图把不同关系混为一体且硬阈值不传递拓扑梯度。
- Core mechanism：age、sex、真实 site 分别形成 metadata pair，与 patient embedding pair 联合输入不同 MLP，分别学习连续 adjacency，分别 GNN，再通过可学习 relation gate 融合。
- Closest BrainPromptPlus component：subject-prompted population graph Eq. (6)–(7)。
- Similarity：**Medium（同是 metadata population 建模，但特定结构不同）**。
- Does it reproduce a BrainPromptPlus novelty：**NO，仅针对这段 V1 实现**。
- Decision：**ALLOW for further validation of the specified V1; V2/V3 not yet reviewed**。
- Evidence：当前 `/home/panyao/AD/BrainPromptPro/layers/multirelation_population_graph.py:24–62`；Plus 只有一个 joint-text similarity 和一个阈值 adjacency，既没有三关系 edge learners，也没有 relation fusion。
- 额外新颖性限制：Plus §II-B明确已有 SEGC、SAC-GCN、MMP-GCN 等多 metadata 图融合相关工作，因此“多关系图”这个宽泛概念绝不能仅因不同于 Plus 就宣称首次提出。

## 6. 不提前放行的候选与论文未来工作边界

论文 §V（4159）明确讨论未来的 group/network/hierarchical ROI prompts、continuous disease embeddings、uncertainty-aware/longitudinal prompts，以及其他图构建方式。这些没有作为完整提出并验证的方法，但已经公开作为研究方向。若新候选采用这些方向，应标为 **future-work overlap / Medium**，说明相对这段公开建议到底新增了哪个明确计算机制；不能声称该想法未被 Plus 提及。

| 候选描述 | 审计决定 |
|---|---|
| “加 spectral prompt”；“把强/弱正/负 FC 变成 LLM 边向量”；五层提示重新组合 | REJECT，PX-01/02/03/04 直接覆盖 |
| “将原 subject/disease prompt + CE 改名为 contrastive/prototype module” | REJECT as claimed core novelty，继承 BP-I2/I3；名字变化不构成新机制 |
| “generic supervised contrastive”“prototype learning”“dynamic graph”“hierarchical graph”只有方向名 | NOT REVIEWED，必须先提供具体正负对定义/原型定义/图变量/信息流；不预先填 NO/ALLOW |
| “network-level prompts”“continuous disease embeddings”“uncertainty-aware disease prompt” | NOT REVIEWED，并明确标注 §V 已提及的 future-work overlap；需要结构性区别，不得称方向首创 |
| 优化器、损失权重、类平衡、归一化、warmup、clip、scheduler | Supporting optimization；可按授权研究，但不能充当三项结构创新 |

## 7. 最终审核要求

每个 V2/V3 或新增模块编码前，必须给出确切输入、可计算公式、与 PX-01–04 / BP-I1–I4 的逐条区别；High overlap 或复现 Plus 核心创新则 REJECT。保留原型/梯度/消融/validation结果后，才允许筛选最终三个模块。未定义模块、仅方向名、尚未训练模块不能在报告中出现“已证明独立创新”或“性能已提升”。

本审计不要求把旧 M1/M2/M3 固定为最终答案。它们与 Plus 不直接同构，不等于当前实现有效；最终取舍仍由相同数据、划分、指标和验证流程下的单模块、组合、完整消融与多 seed 证据决定。

## 8. 五个具体候选的编码前审查（2026-10-08 补充）

审查对象是根 agent 在 2026-10-08 发来的以下完整机制说明，不是对任意同名候选的许可。审查只判断是否复现 BrainPromptPlus；五个候选均尚未通过当前数据上的有效性或 Top-3 资格验证。后续若输入、公式或信息流跨越本表的明确边界，须重新审核。

### Candidate M1 V2：数值 FC 正负分路、ROI 语义与个体 FC 条件门控

- **Problem addressed：** 当前 sigmoid 稠密图可能抹去正负连接含义、压缩个体差异；固定脑图不能学习分类需要的数值连接权重。
- **Core mechanism：** 从数值 FC 直接构成 `C_pos=max(C,0)`、`C_neg=max(-C,0)`；将 ROI 语义投影与该患者完整 FC 特征输入可训练门控器，分别调制正/负消息通道。边学习直接接收 FC 数值和个体节点/图状态。冻结文本编码仅用于原 BP 已有的 ROI 知识；不生成频谱描述，不生成 ROI-pair coupling 句子，不对数值 FC 进行 Plus 式边语言编码。
- **Closest BrainPromptPlus component：** §III-A Eq. (1)–(2) 的空间连接语义表征，以及 §III-B Eq. (3) 的边调制传播。
- **Similarity：Medium。** 都针对连接噪声/负相关和提示引导图表征；但 Plus 把统计与连接描述离线语言化、固定阈值选边，再映射文本边向量，候选保留数值 FC 的正负通道并按患者表示在线学习 gate。
- **Does it reproduce a BrainPromptPlus novelty：NO，在上述具体机制内。**
- **Decision：ALLOW，限定此数值 signed-FC 条件门控设计进入实现与验证。**
- **审计边界：** 不得将 signed coupling 变成文本边 embedding，也不得加 spectral-text nodes 后称新语义脑图。signed-GNN 是广泛研究方向，Plus §V也列举 signed/learned graph 为未系统评估方向；本决定不能支撑“首次使用负连接/首次提出learned graph”的主张。需验证正负分路的梯度、各分支效应以及图不坍缩。

### Candidate M2 V2：中心化疾病查询、保留疾病 token 身份的 ROI 读出

- **Problem addressed：** 当前疾病 query 可能语义近似且产生相似 attention，过早把全部疾病读出再平均会丢失类别身份。
- **Core mechanism：** 对全部 K 个 disease prompt 的查询表示做预先定义的中心化；所有患者均使用全部 K 个查询，以 `softmax(QK_roi^T/sqrt(d))V_roi` 得到 K 个有固定类别顺序的读出 token；将 K 个 token 按固定身份拼接、回投影，与 mean pooling 残差融合。真实标签不参与 query 选择或输入。
- **Closest BrainPromptPlus component：** §III-C Eq. (5) 的普通 sum readout；§III-D Eq. (8) 的末端 disease embedding 对齐。
- **Similarity：Low。** 同样使用 disease prompt，但 Plus 不将疾病文本作为 ROI 聚合查询，也不保留 K 个疾病条件 readout token；其疾病语义仅对末端受试者向量施加辅助 CE。
- **Does it reproduce a BrainPromptPlus novelty：NO，在上述具体机制内。**
- **Decision：ALLOW，限定此全类别查询读出设计进入实现与验证。**
- **审计边界：** 原 BP 已有的末端疾病 CE 不能被当作这一候选的新贡献。类别 token 的顺序必须与固定 label/prompt/output mapping 一致。不能按真实类别或测试预测后验先挑选一个 query；需记录 attention 相似度、熵、疾病token相关性、中心化前后范数与梯度。

### Candidate M3 V2：固定 TRAIN 参考库的可学习多关系人群读出

- **Problem addressed：** 批内 population graph 的表示依赖碰巧同批的患者，硬阈值/稠密平均容易使信息丢失；现有人群边无法解释各 metadata 关系的独立作用。
- **Core mechanism：** reference 库只由训练 split 的 FC/metadata 构成，reference 样本 ID 集固定，不向模型提供其标签。每个 query 仅连到 TRAIN references；分别根据 age、sex、真实 acquisition site 与数值脑表示学习连续 relation score，并通过 relation-specific attention 聚合 value、再融合。query 之间无边；reference 编码器与 score/value 路径可学习，推理保持 reference 全集及编码方式确定，从而预测不随 query batch 同伴改变。
- **Closest BrainPromptPlus component：** §III-C Eq. (6)–(7) 的批内 subject-prompted population graph。
- **Similarity：Medium。** 均使用患者关系及 demographics，但 Plus 使用当前 minibatch 的无向、阈值化单图；候选为 train-reference 条件的 query→reference 映射、多关系可微权重、无 query→query 消息，信息流及邻接定义不同。
- **Does it reproduce a BrainPromptPlus novelty：NO，在上述具体机制内。**
- **Decision：ALLOW，限定该 TRAIN-only reference 结构进入实现与验证。**
- **审计边界：** 不将 Source_Dataset、diagnosis、target label 放入 scorer/value；validation/test 的 FC 不能进入 reference 库，也不能用于 reference 选择、统计量拟合或原型构建。这里“fixed”是 reference membership 固定，网络表示可随训练更新。若训练 query 自身也在 reference 中，是否去掉 self-reference 必须预先规定并统一执行。batch independence 是要由 eval-mode 的不同 batch size/order/composition 测试验证的性质，不能只凭无 query 边声称，尤其注意 BatchNorm、dropout、reference子采样/缓存更新。
- **新颖性保留：** Plus §II-B已列多 metadata 图融合先前工作，整个“多关系人口图/参考库”的一般思想并非凭此获得领域首创保证；最终论文须说明这里的具体局限与可验证改进。

### Candidate M4：按固定 AAL ROI 身份保留空间模式的低维 slot 读出

- **Problem addressed：** mean readout 对各 ROI 一视同仁，可能让不同局部空间模式产生近似患者向量。
- **Core mechanism：** GNN 后每个 ROI 表示经共享低维投影形成 slot，按同一 AAL116 索引顺序拼接成固定位置向量，再压缩为 patient embedding。slot 是数值 ROI 特征，不是新增文字描述、network-level prompt 或 Plus spectral embedding。
- **Closest BrainPromptPlus component：** §III-B Eq. (4) ROI 身份增强与 §III-C Eq. (5) sum readout。
- **Similarity：Low。** ROI 身份已是原 BP 输入先验；候选把位置身份显式保持到读出，Plus sum/mean 聚合没有这个固定顺序拼接路径。没有 PX-01/02 的新增 prompt，也没有五提示融合。
- **Does it reproduce a BrainPromptPlus novelty：NO，在上述具体机制内。**
- **Decision：ALLOW，作为有序数值读出候选进入实现与验证。**
- **审计边界：** AAL ROI 顺序必须与 FC 和 ROI prompt 严格一致；不能按测试统计/标签重排。参数增加需要与参数量匹配的通用 MLP/readout 控制比较。**固定顺序拼接+投影本身是通用读出方式，不能仅因不同于 Plus 就直接称为新的核心科学创新**；若最终仅是标准 flatten/dense 替换，其贡献定位应谨慎，并依据相对原 BP 的结构区别及消融证据决定是否满足用户 Top-3 要求。

### Candidate M5：投影后 ROI 二阶协方差读出残差

- **Problem addressed：** 只保留 ROI 均值的一阶摘要可能丢失同一患者 ROI 表征分布的差异。
- **Core mechanism：** 对 GNN 的 ROI hidden states 做低维共享投影；在每个患者内部按 ROI 维中心化，计算低维 hidden-channel covariance；取固定上三角向量经投影产生 patient readout 残差，与一阶读出融合。它是 GNN 表征的数值二阶统计，不使用 BOLD FFT、ALFF/fALFF、频带文本、ROI 对连接描述文本。
- **Closest BrainPromptPlus component：** Eq. (5) 的一阶聚合；从“统计量构成输入”的远距离类比看，还有 §III-A 的频谱描述，但统计对象/输入层级/编码形式不同。
- **Similarity：Low。** Plus 的频谱统计来源是 ROI 时间序列、组织成语言后冻结 LLM 编码，候选统计的是已学习 ROI 表征、用数值矩阵运算构成二阶 readout，不复现语言图输入机制。
- **Does it reproduce a BrainPromptPlus novelty：NO，在上述具体机制内。**
- **Decision：ALLOW，限定数值二阶读出残差进入实现与验证。**
- **审计边界：** covariance 必须每个 patient 独立计算；不能跨 validation/test 患者估计公共统计后回灌训练。中心化/归一化/epsilon 的约定预先固定；注意 scale、秩和 hidden-channel维度带来的参数规模。**协方差/二阶 pooling 本身属于既有通用方法族**，通过 Plus firewall 不代表可无条件宣称原创，需要明确区别和效果证据。

### 五候选审查汇总

| Candidate | Similarity to closest Plus component | Reproduces Plus novelty? | Decision before coding | 限定 |
|---|---|---|---|---|
| M1 V2 signed numeric FC + patient/ROI gate | Medium | NO | ALLOW | 禁 spectral/spatial 文本化图输入；只批准具体数值门控信息流 |
| M2 V2 centered disease-query tokens + identity-preserving readout | Low | NO | ALLOW | 所有疾病共同查询；不能使用真实标签选 query |
| M3 V2 fixed TRAIN-reference multi-relation query mapping | Medium | NO | ALLOW | 只用TRAIN库、无reference labels、无query-query边；需检验batch independence |
| M4 ordered low-dimensional ROI slots | Low | NO | ALLOW for candidate experiment | 通用有序读出家族，最终“核心创新”资格仍待论证 |
| M5 within-patient projected ROI covariance residual | Low | NO | ALLOW for candidate experiment | 通用二阶pooling家族，最终“核心创新”资格仍待论证 |

没有一项本次候选使用 Plus 的 spectral-text 节点、spatial-text 边或五提示语义图结构。代码实现之后还应对实际调用链再做一次审查，确认实现确实保持上述边界；后续任何变体不是自动继承 ALLOW。

主审补充：已核对 PDF 页2/4/6/7/8/9/10 的框架与表格渲染，正文11页全部读取。具体实验表数值不作为本队列训练结果。五候选边界经主审确认，许可仅用于结构实验，不证明全领域首创或满足最终三核心创新。
