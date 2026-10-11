# 当前状态
核对：2026-10-11T11:27:30.526404+00:00（UTC）。先核实Git和真实产物；以下为七批研究最终状态。

## 当前任务与完成范围
ADNI_1–7/P2P3P5独立研究，计划004的训练、验证、消融、冻结测试和最终报告已完成。主协议61份分类验证（19方法×3seed+4项单seed探索）、10项预训练、subject A0/A1各3seed均已核验。63项固定TEST（主57+补充6）于2026-10-11T11:21:28Z全部成功；独立复算逐份对齐冻结标签/成员与七项指标、混淆矩阵和聚合，63/63通过。最终公开同步以本轮Git提交及PR1核验为准。

## 结果与结论
主TEST三seed均值：A0 Accuracy56.14/Macro-F1 28.33/AUC68.88%；旧Pro57.64/22.45/63.71%；A1(P2、VAL预先选定候选)55.64/28.25/67.65%；A7(完整新版)57.64/27.84/66.27%。A7未全面超过BrainPrompt；A1主TEST五项均值也未超过A0。subject A1 F1=23.75%，A0=21.77%，但不稳定且Precision未提高，不能声称全面泛化收益。
P2/P3/P5验证四背景F1边际−0.28/−1.31/−0.48pp；不成立三个稳定有效创新。63项无全单类塌缩，SMC/EMCI识别仍差；A0/A1/A7主TEST的EMCI Recall均0。保留失败证据，不根据TEST改选A4/历史M2等。
[最终报告](docs/research/adni_1to7/Final_Report.md) · [主测试全19方法](docs/research/adni_1to7/final_test/Test_Results_sample.md) · [subject补充](docs/research/adni_1to7/final_test/Test_Results_subject.md) · [逐类/配对/成本](docs/research/adni_1to7/final_test/Final_Analysis.md) · [验证消融](docs/research/adni_1to7/validation_complete/Ablation_Effects.md)。

## 数据与隔离
1323 FC/895人，CN850/AD65/SMC39/EMCI61/LMCI308，AD65是FC数；MCI→LMCI、统一经验Pearson。split seed1729，训练seed42/3407/2026。本轮是固定holdout+验证选checkpoint，不是历史十折末轮。sample存在同人跨split和历史开发暴露，两协议之间另有样本角色交叉；只在各协议内部定义TRAIN/VAL/TEST隔离，非全新外部验证。site真实含义未独立确认。完整成员/FC/权重/私有锁不公开。

## 作业、故障和恢复
N=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010。Phase1–7训练均结束，当前无本研究在跑作业。Phase7 PID1432091三项退出0。首次Phase8 PID1433530在写锁前遇到队列schema KeyError running，零TEST访问；日志保留N/queues/phase8_frozen_test，未改fe8bf25快照。
D023修复区分训练队列/控制器，3新回归+3统计+4锁测试通过。新快照N/code/39ee2f2在GPU0/3/6恢复，controller1629946于11:18:42Z开始，11:21:28Z完成并退出；launcher/state/log在queues/phase8_frozen_test_recovery_20261011。没有重训，也没有重复TEST。
selection_lock SHA256=fb71040c47d235b382dac5f487b01ff9f925d262f305f134254a67250bc63031。固定63输出N/final_test；完整汇总N/reports/frozen_test_complete_20261011；逐身份/标签/公式复核N/reports/final_analysis_20261011。不可删除锁或重跑finalize_study；报告重建只读已有预测，用report_final写新目录。候选A1与范围已冻结，不再用TEST调参。

## Git与下一行动
本地分支research/adni1to7-p2p3p5，[PR1](https://github.com/yasuo-coder/AD-/pull/1)。原工程保留；五批12+72训练及锁完整，旧TEST未运行。已发布历史本地fe8bf25/公开dacbef3具有相同tree bcb68a619bf3159ade77f0e99197849665daa026。此次修复本地39ee2f2已作为实际评估快照；最终报告提交不能预写自身SHA，请读git log与PR，API同步按tree校验，不做force/reset。
当前无需继续训练。下一明确行动：审阅最终报告/PR；如开展新研究，先定义与本轮已使用TEST隔离的新评估范围。缺少外部验证和稳定增益属于科学限制，不能以继续重复失败配方或测试选型来补齐。旧五批任务边界保持不变。

最终报告校验：3项独立公式回归（完美分类、全CN、无效输入）与3队列schema、3配对统计、4锁契约共13项测试通过；最终匿名JSON字段白名单与新增Markdown链接检查通过。
