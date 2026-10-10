# P2/P3/P5 编码前方法审查

2026-10-10；历史 M 编号与本次 P 编号不同。以下是待验证设计，未宣称性能或全球首创。

## 论文和代码证据

- [BrainPrompt](https://arxiv.org/html/2504.16096v2)，已核对 ROI 数值 FC 行、ROI 身份提示、subject 图和疾病辅助分类。服务器官方基点 c85304a，原实现不改，通过 OriginalBrainPromptAdapter 复用。
- [ncomms2025](https://www.nature.com/articles/s41467-025-62590-4)，已读取全文 HTML 的方法、训练及限制；官方[代码](https://github.com/vkola-lab/ncomms2025/tree/497e0f23380a122a8069eb15902eaae590a45851)固定497e0f2。其随机特征丢弃是应对缺失多模态信息，预测任务为多标签 PET 生物标志物，并非 FC 掩码自监督重建。P2是本项目的数值图适配假设。
- 官方 adrd/nn/transformer.py 创建 emb_aux 任务向量、拼接源特征再self-attention，但当前 forward_trf 在Transformer结果后取 `[0]`，forward_cls将同一表示送给各头。因此不能称该commit已经逐类输出不同任务Token。P3逐类读取是本项目明确接口，主要复用已有M2，而非照搬代码。仓库AGPL-3.0；仅审查，未复制其源码或权重。
- [Plus排重](../BrainPromptPlus_Novelty_Exclusion_List.md)已含全文及官方87932c1依据。禁频谱文字节点、FC连接文字边及五提示语义图；本次三模块不生成这些输入。
- 相关方法已有[GraphMAE](https://arxiv.org/abs/2205.10803)、[ST-JEMA](https://arxiv.org/abs/2403.06432)、[SpaRG](https://pmc.ncbi.nlm.nih.gov/articles/PMC11694515/)。掩码图学习/自监督connectome不是本项目首次提出；这些文献仅用于定位方法族，不移用其性能。

## 复用关系与具体边界

| 方案 | 已有机制 | 复用 | 必要变化 |
|---|---|---|---|
| P2 | 原GCN以FC行作节点特征，历史无掩码重建 | 官方embedding_h/pos_emb/layers | 对称可见性mask；图边与节点FC同时屏蔽；只依赖可见边重算阈值；独立mask投影区分真零；上三角重建与双视图损失，TRAIN预训练 |
| P3 | M2 V1用五疾病query后概率加权求一个向量；M2 V2中心化query/key并保留tokens再flatten | DiseaseTokenReadout的q/key/value/中心化注意力 | 明确第c个token只产生第c个新增logit；残差全局头作对照；单头M2注意力为初版，不伪称新Transformer；随机/零语义与参数匹配对照 |
| P5 | M3 V3 TRAIN PCA32、top32、self-mask、reference-only聚合，禁V2直接当前FC value | 参考成员校验、确定性top-k、自排除、TRAIN拟合PCA契约 | 基于固定版本表示多视图方差的可靠性代理，连续scorer，fixed/no-reliability对照；冻结编码器和reference版本共同绑定，避免过期缓存；纯数值，无diagnosis/source入图 |

## 已审查公式和验收边界

P2：M只在有效上三角随机产生后镜像，对角保持可见；输入X=A*M，邻接仅由X及M决定，不能先从未掩码A建立拓扑。额外mask特征只含可见性。对称decoder为点积或MLP(h_i+h_j,|h_i-h_j|,h_i*h_j)，tanh限制[-1,1]；loss仅在被遮上三角，目标detach。对称KL采用log_softmax。mask比例候选.05/.10/.15，预训练和联合训练分别记录，不与分类塌缩混淆。

P3：直接调用历史DiseaseTokenReadout，所有5疾病同时查询；新增逐类线性w_c^T R_c，无真实标签输入。原类别順序固定，att[B,5,116]、tokens[B,5,d]。如果此改动无益则如实认定M2复用无提升，不称新独立贡献。

P5：固定TRAIN成员，表征版本hash绑定。r=exp(-mean_k ||normalize(z_k)-mean(normalize(z))||²/tau)，只视作扰动稳定性代理。s=cos(Wq,Wref)，score=s/t + boundedMLP(s,rq,rr)，top-k按稳定顺序选，softmax归一，message仅来自reference values，无query直接value捷径；模块输出受控残差。无metadata是默认，site不可当作已独立核验站点。需要分别检查query和reference可靠性是否在归一中抵消、极少数类是否被削弱。

具体三条数值路径不复现Plus PX01–04，允许实现验证；P5与Plus未来uncertainty-aware方向有概念邻近，不能声称想法首创。现有历史M2/M3失败诊断将进入机制消融；未运行实验不能宣布有效。
