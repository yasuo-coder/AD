# 当前状态
核对日期：2026-10-10（UTC）。

## 活跃任务
ADNI_1–7 的 P2 Masked Connectome / P3 Task Token / P5 Reliability Population 独立研究。
分支 `research/adni1to7-p2p3p5`；活跃计划 [004](plans/004_adni1to7_p2p3p5.md)。
M0–M3 已完成；模块已实现，基础验证和真实数据 smoke 通过；正式训练尚未启动。

## 已核验事实
- 新数据版本 `ADNI_Combined_1to7_v1`：1323 FC、895 biological subjects；CN850/AD65/SMC39/EMCI61/LMCI308。MCI→LMCI，五类顺序不变。
- 复用已有清洗时序统一计算经验 Pearson；既存 Nilearn 收缩相关不能与经验 Pearson 混用。旧五批纳入样本的 FC、标签、prompt 已逐项对齐。
- 新主协议为独立固定 sample-level 80/10/10 holdout，split seed1729；训练seed42/3407/2026。subject-disjoint 补充划分另存。详细成员及关联审计仅在受控目录；不能声称主协议无同人重叠或外部验证。
- P2/P3/P5 三开关可组合；16模块单测、CPU/CUDA各8组合、真实TRAIN小样本反向通过。CUDA比较容差1e-6；CPU基线关闭开关等价。
- A0连续/恢复、A7、P2重建、LegacyPro、历史M2/M3v3及no_metadata入口共8份smoke结果完成；均标记 NOT_EVALUATED，不作为正式性能证据。
- CPU A0两轮连续与断点恢复模型张量及RNG完全一致，证据 `resume_validation.json`。此验证绑定当时源码快照，不等于所有未来修订均已验证。

## 路径、作业与下一步
受控根目录 N=`/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010`。
数据审计 `N/data_audit_v4`；冻结数据 `N/dataset_v1`；契约 `N/contracts_cpu.json`、`N/contracts_cuda.json`；恢复证据 `N/resume_validation.json`；smoke `N/smoke`。
最新进程核对：无本项目训练作业；GPU 0/3/5/7有余量。下一步公开审查提交本轮实现，使用不可变代码副本运行A0三seed及P2有限预训练探索。正式训练禁止引用smoke checkpoint；全部test暂不评估。

## 历史边界与限制
接入 main 已发布 `38c72b85f90ab0087556390d3bec97314cf10a3b`，原工程保留。五批研究12+72正式训练已核验；两个selection_lock及frozen_decision已创建，最终test未运行，计划003保留。
没有七批正式指标，不宣称P2/P3/P5有效。P2两轮重建误差偏高，需要与TRAIN均值重建对照诊断；P3语义/参数对照和P5正式可靠性机制仍待验证。
公开只包含代码、配置路径模板、总体匿名分布、hash和测试状态；完整协议、样本身份、split、权重保留服务器。
