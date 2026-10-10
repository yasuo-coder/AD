# 实际架构与候选
`BrainPrompt/`为必要官方基线及服务器改动快照；`BrainPromptPro/`保留真实包布局，未套入第二层同名目录。

| 实现 | 文件 | 机制与状态 |
|---|---|---|
| BrainPrompt适配 | BrainPromptPro/nets/original_adapter.py | 复用原GCN/提示/分类器；固定拓扑；population sigmoid乘积上界低于.8阈值导致实际空图 |
| legacy Pro | nets/brainprompt_pro_net.py | 正值稠密自适应图、disease readout、batch内三关系；存在图/注意力同质化 |
| stable Pro | nets/stable_brainprompt_pro_net.py | FC先验、残差、LayerNorm；图统计改善未解决整体分类塌缩 |
| M1 V2 | nets/research_candidates_v1.py | FC/ROI语义条件正负双路消息；最佳checkpoint gate近常数，需机制反事实 |
| M2 V2 | 同上 | 中心化疾病query、归一化ROI key、保留五类token的残差readout；单项较弱 |
| M3 V2 | 同上 | TRAIN参考三关系；直接current-FC value捷径与过拟合已发现 |
| M3 V3 | nets/research_population_v3.py | TRAIN PCA32固定key/value、top32关系、rank8有界残差，删除直接query-value；fixed/no_metadata/direct三对照 |
| M4/M5 | nets/research_candidates_v1.py | ordered slots / hidden covariance通用读出；不自动构成核心创新 |

Plus排重见 docs/research/BrainPromptPlus_Novelty_Exclusion_List.md。所有候选先审查再编码；不使用Plus频谱/空间语言提示图。
V3正式入口为candidate_v3_initialized_runner，早期small-init入口保留历史，不能混用其初始化。冻结输入LLM嵌入不是后训练encoder。
主输入fc[B,116,116]、ROI[116,2048]、subject prompt[B,2048]、五类disease prompt[5,2048]，age/sex/site和仅用于selfmask的Sample_ID。输出logits[B,5]与辅助/解释结构。Source_Dataset只在导出追踪使用。

## 七批独立P2/P3/P5实现
`nets/connectome_p2p3p5.py`复用官方GCN，三开关均关闭时等价基线。P2在节点FC和拓扑同时掩码，以显式visibility区分真零；对称数值重建和双视图一致性，原始/中心化解码器分开登记。P3复用历史M2读出并新增逐类token logit；random、learned_query和近参数匹配mean-MLP仅替换新读出，原基线疾病辅助分类保持相同。P5采用同seed固定A0编码器生成TRAIN参考和query，多视图稳定性只是可靠性代理；reference-only残差、self-mask、版本与batch independence均受检查。新增文件、公式和排重见research/adni_1to7/METHOD_AUDIT.md。代码通过不等于有效增益；真实收益由正式验证/消融判断。
