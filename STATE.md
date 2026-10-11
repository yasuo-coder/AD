# 当前状态
核对时间：2026-10-11T04:54:18.678748+00:00（UTC）。续接须重新核实队列、PID和产物。

## 当前阶段
ADNI_1–7独立研究，活跃计划004。主协议57项三seed比较已全部完成，另4项单seed微调探索与10项预训练保留；61份主协议分类指标已独立复算。subject-disjoint A0三seed也已完成并重载核验。M8/M9完成；正在补齐验证选定A1的subject补充，完成后按D021冻结并一次评估固定清单。当前尚无新七批selection_lock，尚未进行TEST角色评估。

## 真实结论
A0/A1/A2/A3/A4/A5/A6/A7的三seed验证Macro-F1均值依次35.63/36.24/34.34/34.80/34.25/35.47/35.29/33.00%。[完整指标及配对消融](docs/research/adni_1to7/validation_complete/Validation_Results.md)。
P2/P3/P5四背景平均F1边际分别−0.28/−1.31/−0.48个百分点，均未证明稳定正贡献。A7不优于A0；不能宣称三个有效创新。A1仅F1/Recall比A0较高，Accuracy/Precision/AUC较低，按D021作为实验候选继续holdout核验，不视为已证实改善。候选已由主验证集锁定，不因补充或测试结果换模型。
25项机制探针及另外3项subject A0重载通过。P5边权近均匀、残差比例约0.29%；P3 token有梯度/差异但无稳定比较优势；P2预训练重建改善未转成下游提升。技术细节见[Technical_Report](docs/research/adni_1to7/Technical_Report.md)。
subject A0 Macro-F1 26.82%，与主协议35.63%分开报告；划分/人群差异不能全归因同人重叠。[补充基线](docs/research/adni_1to7/subject_baseline/Validation_Results.md)。

## 数据与隔离
1323 FC/895人，CN850/AD65/SMC39/EMCI61/LMCI308，MCI→LMCI；统一经验Pearson。split seed1729，训练seed42/3407/2026。主协议存在同人跨split及历史开发暴露。两套协议之间另有样本角色交叉，隔离只在各自协议内成立；不能称为全研究从未使用或外部独立测试。私有交叉审计N/reports/cross_protocol_exposure_20261011.private.json，完整成员/标签/FC不公开。

## 作业与路径
N=`/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010`。Phase1–6训练均完成；Phase5的40项全部退出0，controller1408894已退出，主观察器1428866已完成；生成N/reports/phase5_complete_host_20261011及ablation。Phase6 controller1425910已退出；subject基线报告N/reports/subject_baseline_complete_20261011。
Phase7 controller1432091，2026-10-11T04:49:10Z启动，GPU1/2/5，A1三个seed的--split1；输出N/subject_experiments/A1/seed*。训练源码仍N/code/29f34f9，不可修改；配置和恢复命令在queues/phase7_subject_candidate/launcher.json、jobs.private.json及各日志。主候选证据N/candidate_selection.private.json。
新finalize_study仅等待Phase7成功、复核完整开发结果，再建立锁并运行D021固定测试范围。实际是否启动Phase8及其PID必须查看N/queues/phase8_frozen_test；不能因脚本存在就认为任务已运行。测试入口默认VAL，20代表性checkpoint的CUDA重载通过；4项锁契约、3项配对分析测试、4项指标篡改拒绝和后续队列失败拒绝检查通过。

## Git与下一步
独立分支research/adni1to7-p2p3p5，[PR1](https://github.com/yasuo-coder/AD-/pull/1)。上一已发布本地49f8eff/公开def6eff的tree均158b446f3911af56dca275ee39a9a5aa8fef2d93；本次新改动以git status和后续提交为准。旧五批12+72训练及锁保留，旧test未运行。
下一步：完成Phase7并独立汇总subject A0/A1；冻结source/checkpoint/data/split/evaluator后执行固定test清单；独立复算、报告真实正负结果，之后不再依据test调参。最终报告需要明确当前协议局限，不能因测试某对照较高就重新挑选主模型。运行恢复与不可变副本规则见RUNBOOK。
