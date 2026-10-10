# Implemented Candidate Specification

核对日期：2026-10-09（Asia/Shanghai）。这是对已实现源码的说明，不修改训练算法、超参数、注册许可或冻结 novelty 审查，也不作最终选型。路径基于 `/home/panyao/AD/BrainPromptPro`；实现文件为 `nets/research_candidates_v1.py`，实际训练/损失调用链为 `candidate_runner.py`。报告中的 M1/M2/M3 V2 与旧 Pro 同名模块是不同实现，不可混用诊断数据。

## 1. 证据、共同主干与许可范围

- [冻结 novelty 审查](BrainPromptPlus_Novelty_Exclusion_List.md)第 8 节：五个具体候选获 ALLOW；M1/M3 为 Medium，M2/M4/M5 为 Low。许可绑定哈希，不证明领域首创或性能提升。
- [registry](../../BrainPromptPro/configs/research_snapshot_20261007/stage2_registry.json)：BP_control 无模块、无创新主张；M1/M2/M3 version=2，M4/M5 version=1。
- [protocol](../../BrainPromptPro/configs/research_snapshot_20261007/stage2_protocol.json)：沿用固定数据/划分与训练设置，seeds=42/3407/2026，profile=`sqrt_inverse_frequency`；test 在模型和消融决策冻结前 CLOSED。
- 候选契约记录（服务器工件：`/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007/stage2_20261008/verification/candidate_contract_tests.json`，不在公开仓库）：已记录的测试通过；此文引用结果，不声称再次执行训练测试。

`ResearchBrainPrompt` 先构造 `OriginalBrainPromptAdapter`；active 为空时直接委托原 adapter。所有候选保留原 BP 完整 FC 行输入、ROI prompt、原图主干、subject prompt 融合、疾病辅助 CE 和分类头。当前共同 hidden_dim=116、L=4、AAL116 ROI。候选参数初始化置于 CPU `torch.random.fork_rng(devices=[])` 内，意图保留共享主干/dropout RNG 流；输出残差的末端 Linear 为零初始化。契约记录证明其覆盖的初始输出与公共梯度条件，不等于任意设备/运行的最终训练轨迹数值相同。

候选分支不接收 diagnosis、target label 或 Source_Dataset。M3 acquisition ID 只用于 self-reference 屏蔽；不能因此声称 reference 是 subject-disjoint，因为当前协议允许不同 acquisition 来自同一 biological subject。AAL/label/prompt 顺序沿用固定数据契约。

## 2. M1 V2：SignedConditionalMessages

输入为患者数值 FC `C∈R^(116×116)`、冻结 ROI embedding `P` 与每层原主干的节点表示。FC 行与 ROI 语义各投影至 24 维，相加后 GELU 并 L2 归一化为 `z_i`。边输入为 `[|z_i-z_j|, z_i⊙z_j, C_ij, |C_ij|]`，共 50 维，经 `50→32→2` MLP；score 对称化后产生两路 gate `g=0.25+0.75·sigmoid(score)`。

去除对角线后，`A_pos=max(C,0)⊙g_pos`，`A_neg=max(-C,0)⊙g_neg`。两路各自以非负 degree 归一化 `D^(-1/2) A D^(-1/2)`，避免负 degree。每层使用卷积前节点表示，构造 `m=N(A_pos)W_pos h − N(A_neg)W_neg h`；经过 LayerNorm、GELU、零初始化输出 Linear 后，加到该层原 BP 卷积输出。数值 FC 的负连接通过消息减号体现，不把两路合计矩阵误称为带符号邻接。

实际 gate 的下界为 0.25，因此它不会把非零 FC 边硬剪为零；图在当前输入下可能仍然稠密。runner 的 brain 正则作用于返回的 `A_brain=A_pos+A_neg` 非对角绝对均值，不直接惩罚 signed 消息或原 BP sparse topology。原 BP 路径保留，M1 是添加消息残差，不能写成彻底替换原图。

Plus 边界：无 BOLD 频谱、无频谱文字、无 FC 强弱/正负句子与 LLM 边 embedding；ROI 文本仅使用原 BP 已有知识。signed GNN/learned graph 的一般思想不获得首创保证。

## 3. M2 V2：DiseaseTokenReadout

对全部 K=5 个冻结 disease embeddings 沿类别维减均值，再投影至 32 维并 L2 归一化为 Q。ROI hidden states 沿患者自身 ROI 维减均值，再由 key Linear 投影至 32 维并归一化为 K_roi；value 为中心化 hidden states 的 116 维线性映射。

实际注意力为 `softmax(4·Q K_roi^T)`，不是未核对的默认 `1/sqrt(d)` 缩放。所有患者均使用 CN/AD/SMC/EMCI/LMCI 全部 query，不用真实类别选 query。得到 5 个疾病条件 token，每个 token 做 LayerNorm(116)，按固定类别身份拼成 580 维，经过零初始化的 `580→116` Linear，作为原 mean readout 的残差。

中心化由 disease 输入 prompt 与患者 ROI 表示各自完成；末端原 BP 疾病 CE 继续存在，不作为 M2 新贡献。输出保留 `roi_attention` 与 `disease_tokens`，可供后续熵、相似度、梯度与身份保留检查。该机制与 Plus 普通 sum/mean readout 不同，但需要实验与相关工作支持创新判断。

## 4. M3 V2：ReferencePopulation

固定 reference membership 为全部 798 个 TRAIN acquisition；缓存数值 FC 严格上三角 6670 维及 train-fitted age/sex/site metadata，不缓存 diagnosis 或 Source_Dataset。TRAIN membership 固定，FC/metadata buffer 固定；FC projection、scorers、value transforms、融合 gate 与输出 projection 随训练学习。reference 表示由专用 `6670→32` FC 投影后 L2 归一化得到，不是重新通过完整 ROI GNN。

Query 为当前 subject representation 经 `116→32` query projection 后归一化；pair 是 query/ref 的 32 维绝对差与 32 维逐元素积。age scorer 另接 3 维年龄绝对差、积与均值；sex/site 各用 4 维 embedding 构成差、积与非缺失同类指示，共 9 维。metadata age 使用共同训练 encoder 的变换值；site 来自项目历史 `site_id`，原 acquisition site 来源尚未独立核实，不能把该字段命名当成已核验真实采集站点的证据。

每路 MLP 输出 `score_r(q,j)`；按 Sample_ID 对 query 自身 acquisition 的 reference 屏蔽，然后 `attention_r=softmax(masked_score_r)`。各路消息为 `Σ_j attention_r(q,j)·V_r(k_j) − V_r(k_currentFC)`，其中 currentFC 也用共享 FC projection。三路由 `softmax(Linear(subject))` 融合，经 LayerNorm 与零初始化输出 Linear 后，加到原 BP fused representation。没有 query-query 边，也不使用 reference 标签；同一 biological subject 的其他 acquisition 不按 subject ID 全部剔除。

**稀疏正则的准确含义：** 模型返回的 `population_adjacencies` 是 `sigmoid(score_r)·permitted`，runner 惩罚其绝对均值；实际消息邻接却是上述 softmax attention。降低 sigmoid 分数均值不保证 softmax attention 稀疏。例如，同一 query 的全部 score 同减常数会降低 sigmoid 均值，而 softmax 完全不变。这里没有 top-k、阈值或稀疏 attention 算子；有限 score 的 permitted references 通常都得到正权重。不能把该 loss 称为已证明对真实 softmax 邻接施加稀疏约束。

当前契约报告 `train_reference_only=true`、`self_reference_masked=true`，最大 batch-size logits 差 `1.4901161193847656e-07`。它支持被测 eval 设置下的数值 batch 一致性，不证明所有 batch order/composition、训练 dropout 行为或 metadata 因果性。保留原 BP population 路径，不把 M3 写成删除了全部原 population 运算。

## 5. M4：OrderedSlots

GNN 后每个 ROI 经共享 `116→8` Linear 与 LayerNorm(8)，按原 AAL116 顺序展平为 928 维；经零初始化 `928→116` Linear，加入原 mean readout。不存在新文字描述、network-level prompt、测试统计重排或额外 ROI 分区。

这是保留位置的通用有序数值读出对照。即使验证指标更高，也需要参数量匹配、与通用 MLP/readout 比较及相关工作证据，不能因不同于 Plus 就自动满足核心科学创新资格。

## 6. M5：CovarianceReadout

GNN 后每个 ROI 经无偏置 `116→16` Linear；在每名患者内部沿 116 ROI 维中心化。计算 `S=X^T X/max(N_ROI−1,1)`，然后逐患者以 `mean(diag(S))+1e-4` 归一化。取含对角线的固定上三角 136 个数，经零初始化 `136→116` Linear，作为原 mean readout 残差。

这里协方差描述 learned hidden channels 的二阶统计，不是直接重新计算 ROI×ROI FC，也不是 BOLD 时间频谱、ALFF/fALFF 或文字化统计。没有跨 validation/test 患者拟合公共协方差。它属于既有 covariance/second-order pooling 通用方法族，当前定位为读出对照，不能自动称为原创模块。

## 7. 实际损失、训练与验证边界

runner 使用 `CE(logits,y)+1.0·CE(disease_logits,y)`，两项均采用仅按 TRAIN 类别计数计算、均值归一化的平方根逆频率权重。M1 激活时加 `1e-4·mean_offdiag(|A_pos+A_neg|)`；M3 激活时每路返回 sigmoid 矩阵的绝对均值按关系数平均、再乘 `1e-4`。没有新增 contrastive/prototype loss；relation diversity 系数为 0。

正式训练沿用 Adam lr=5e-5、weight_decay=0.005、batch=20、dropout=0.5、clip=5、最多400轮；验证 macro-F1优先，同分按固定综合分，再按较低验证 loss 选 checkpoint。ReduceLROnPlateau factor=0.5、patience=25；LR<1e-6，或至少120轮且100轮未改善时停止。metadata 只在训练记录上拟合。开发 train 只实例化 train/val loader；test 函数要求先有最终 selection lock，并校验所选 checkpoint、spec/source/data/split 绑定。

截至核对时间，6 组两轮 smoke 均完成；接手时 18 组正式实验 16 组完成，M4/M5 seed2026 待补；最新核对时两项已补齐，现为 18/18。所有已完成正式 result 的 test 字段为 `NOT_EVALUATED`；RUN/STAGE 均无 `selection_lock.json`，STAGE 无 `test_result.json`。这些状态不授权提前创建选择锁或访问测试集。

契约文件记录全关闭 train/eval 等价、零残差初始输出/公共梯度等价、TRAIN-only reference/self mask、五候选三步后分类梯度非零。三步后的梯度范数分别为 M1 `4.474142386357016e-05`，M2 `6.770737648010254`，M3 `0.00019047122947313255`，M4 `0.399260887876153`，M5 `0.02466726489365101`。有限步骤局部非零梯度不等于长期有效学习或分类改善。

原 BrainPrompt 三 seed 是主基线；新 BP_control 是数值复跑对照。两者保持独立，报告所有 seed 和两套差值；不选择较低者，不拼接为六 seed，不选最好 seed。本文不决定候选排名、组合、Top-3 或最终创新结论。

## 8. 本次只读核对的文件哈希

状态核对时间：2026-10-09 10:54:18（Asia/Shanghai）。

| 文件 | SHA256 |
|---|---|
| `nets/research_candidates_v1.py` | `07498aadb5522ed12bba346091a53a8095f926363110fd7c43980dae74522887` |
| `candidate_runner.py` | `751199429d6425d3f23486dc64caca2fef625d182c751ef3977687304307ee79` |
| `outputs/research/final_optimization_20261007/stage2_20261008/registry.json` | `2792d3773be1b6743e736b21df379fbc5d77c6df4aa7fc9abb864e14acbec441` |
| `outputs/research/final_optimization_20261007/stage2_20261008/protocol.json` | `61a1b5a065ad418fd2f14d96bff1a271970b059e56e0a09912d5b40d8492dadf` |
| `outputs/research/final_optimization_20261007/stage2_20261008/audits/BrainPromptPlus_Novelty_Exclusion_List.md` | `4dee091cb65b76b1eade3050bc7985517d4f5f725ecec1ddc32a44b2bf49ffba` |
| `outputs/research/final_optimization_20261007/stage2_20261008/verification/candidate_contract_tests.json` | `7147de565dd21171087f207bcc63a8afea25d77279ddd312abd3e5c79a88549f` |
