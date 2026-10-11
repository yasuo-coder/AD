# AD- / BrainPromptPro
基于ADNI rs-fMRI功能连接和语义提示的多类别分类研究。已接入服务器实际BrainPrompt、BrainPromptPro、固定协议、候选实现及匿名验证汇总；受控数据和权重不公开。

先读[STATE](STATE.md)、[现有工作审计](docs/EXISTING_WORK_AUDIT.md)和[活跃七批研究计划](plans/004_adni1to7_p2p3p5.md)，不能从初始化骨架重新开始实验。

- [背景](CONTEXT.md)、[路线](ROADMAP.md)、[决策](DECISIONS.md)、[实验索引](EXPERIMENTS.md)
- [数据协议](docs/DATA_PROTOCOL.md)、[实际架构](docs/ARCHITECTURE.md)、[复现与恢复](docs/REPRODUCIBILITY.md)
- [BrainPrompt来源](BrainPrompt/UPSTREAM.md)、[BrainPromptPro源码](BrainPromptPro/README.md)、[Plus排重](docs/research/BrainPromptPlus_Novelty_Exclusion_List.md)
- [三seed验证汇总](docs/experiments/final_optimization_20261007/Validation_Results.md)

当前七批研究（1323 FC、895人）已完成完整三seed消融、subject补充及63项冻结测试，逐预测标签和指标核验通过。新版三模块没有稳定超过BrainPrompt；不宣称三个有效创新或达成60%目标。历史五批与原工程保留，公开树不含ADNI影像/FC/身份/私有划分/权重。

- [最终研究报告与完整比较](docs/research/adni_1to7/Final_Report.md)
- [主测试19方法](docs/research/adni_1to7/final_test/Test_Results_sample.md)、[受试者隔离补充](docs/research/adni_1to7/final_test/Test_Results_subject.md)
- [完整验证](docs/research/adni_1to7/validation_complete/Validation_Results.md)、[配对消融](docs/research/adni_1to7/validation_complete/Ablation_Effects.md)
- [技术说明](docs/research/adni_1to7/Technical_Report.md)、[三seed机制](docs/research/adni_1to7/Mechanism_ThreeSeed.md)、[P2负面证据](docs/research/adni_1to7/P2_Reconstruction_Diagnostics.md)
- [作业证据与恢复约束](docs/research/adni_1to7/RUNBOOK.md)

本轮是固定holdout/验证选checkpoint，不是历史十折末轮。TEST已使用，禁止继续据其选型或调参；sample同人跨split、协议间角色交叉和历史开发暴露限制外部泛化解释。
