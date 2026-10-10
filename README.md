# AD- / BrainPromptPro
基于ADNI rs-fMRI功能连接和语义提示的多类别分类研究。已接入服务器实际BrainPrompt、BrainPromptPro、固定协议、候选实现及匿名验证汇总；受控数据和权重不公开。

先读[STATE](STATE.md)、[现有工作审计](docs/EXISTING_WORK_AUDIT.md)和[活跃七批研究计划](plans/004_adni1to7_p2p3p5.md)，不能从初始化骨架重新开始实验。

- [背景](CONTEXT.md)、[路线](ROADMAP.md)、[决策](DECISIONS.md)、[实验索引](EXPERIMENTS.md)
- [数据协议](docs/DATA_PROTOCOL.md)、[实际架构](docs/ARCHITECTURE.md)、[复现与恢复](docs/REPRODUCIBILITY.md)
- [BrainPrompt来源](BrainPrompt/UPSTREAM.md)、[BrainPromptPro源码](BrainPromptPro/README.md)、[Plus排重](docs/research/BrainPromptPlus_Novelty_Exclusion_List.md)
- [三seed验证汇总](docs/experiments/final_optimization_20261007/Validation_Results.md)

当前活跃研究使用独立ADNI_1–7版本（1323 FC、895人），固定holdout及三训练seed；P2/P3/P5实现和基础验证已完成，正式性能待运行。历史998 FC五批研究和冻结选择完整保留。研究收益和Top-3资格按实际验证/消融判定，未宣称已经达到60%目标。公开树不含ADNI影像、FC、受试者信息、私有划分、密钥和checkpoint。
