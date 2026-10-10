# 当前状态
核对日期：2026-10-10（UTC）。

## 活跃任务
ADNI_1–7 的 P2 Masked Connectome / P3 Task Token / P5 Reliability Population 独立研究。
分支 `research/adni1to7-p2p3p5`；活跃计划 [004](plans/004_adni1to7_p2p3p5.md)。
M0–M3 已完成；模块已实现，基础验证和真实数据 smoke 通过；Phase1的3个A0和7个P2预训练全部完成，未使用test。

## 已核验事实
- 新数据版本 `ADNI_Combined_1to7_v1`：1323 FC、895 biological subjects；CN850/AD65/SMC39/EMCI61/LMCI308。MCI→LMCI，五类顺序不变。
- 复用已有清洗时序统一计算经验 Pearson；既存 Nilearn 收缩相关不能与经验 Pearson 混用。旧五批纳入样本的 FC、标签、prompt 已逐项对齐。
- 新主协议为独立固定 sample-level 80/10/10 holdout，split seed1729；训练seed42/3407/2026。subject-disjoint 补充划分另存。详细成员及关联审计仅在受控目录；不能声称主协议无同人重叠或外部验证。
- P2/P3/P5 三开关可组合；20模块单测（扩展后重跑通过）；初版16项、CPU/CUDA各8组合、真实TRAIN小样本反向通过。CUDA比较容差1e-6；CPU基线关闭开关等价。
- A0连续/恢复、A7、P2重建、LegacyPro、历史M2/M3v3及no_metadata入口共8份smoke结果完成；均标记 NOT_EVALUATED，不作为正式性能证据。
- CPU A0两轮连续与断点恢复模型张量及RNG完全一致，证据 `resume_validation.json`。此验证绑定当时源码快照，不等于所有未来修订均已验证。

## 路径、作业与下一步
受控根目录 N=`/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010`。
数据审计 `N/data_audit_v4`；冻结数据 `N/dataset_v1`；契约 `N/contracts_cpu.json`、`N/contracts_cuda.json`；恢复证据 `N/resume_validation.json`；smoke `N/smoke`。
正式队列 `N/queues/phase1_baseline_p2/state.private.json`，controller PID1153238；启动时GPU0/3/5运行A0 seed42/3407/2026，GPU7依序运行P2预训练；最新队列10完成、0运行、0待启动、failed=false。以state、ps和result.json核实当前进度，不能仅凭此PID重启。
运行源码 `N/code/074d2e0`，本地commit074d2e00302463219edc370f73db42bbde29f81d；公开commit4ad8ec19df90396f6c53560c1cf50f900e9ade81，tree相同，API作者时间导致commit不同；本地提交保留。独立分支已推送，[草稿PR1](https://github.com/yasuo-coder/AD-/pull/1)。
下一步运行新增两个数值解码器候选、必要历史基线、A1/A2/A3单seed及P3/P5机制对照，再依据验证集固定正式消融配方。A0三seed指标已独立复算，公开表见docs/research/adni_1to7/phase1。P2验证重建探针正在补齐，首个失败不能替代全部候选结论。新增实现使用下一个不可变代码副本，不能修改正在跑的074d2e0。正式训练禁止引用smoke checkpoint；全部test暂不评估。

## 历史边界与限制
接入 main 已发布 `38c72b85f90ab0087556390d3bec97314cf10a3b`，原工程保留。五批研究12+72正式训练已核验；两个selection_lock及frozen_decision已创建，最终test未运行，计划003保留。
没有七批正式指标，不宣称P2/P3/P5有效。首个P2 mask.05/40轮验证重建劣于TRAIN逐边均值，见D015与受控probes；初始化饱和和近常数输出有实证，不宣布P2有效。已新增P3三对照及P2两数值候选，5个CPU单轮smoke全部通过；这些不计正式性能。P3和P5机制收益仍待正式验证。
公开只包含代码、配置路径模板、总体匿名分布、hash和测试状态；完整协议、样本身份、split、权重保留服务器。
