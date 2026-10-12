# 当前状态
核验UTC：2026-10-12T11:22:05.566133+00:00。用户任务005（BrainPrompt在ADNI_1–7合并队列上的官方十折、停止末轮测试）已完成训练与独立结果核验；十折全部退出0，无正在运行的005训练。报告与交接待本次提交/远程同步，不能用旧holdout替代本表。

## 005结果与验证
1323 FC/895人，seed41，扫描分层十折，CN/AD/SMC/EMCI/LMCI、MCI→LMCI。实际停训last.pt，不恢复best。十折等权均值±population SD：Accuracy=63.41±3.66%；Macro-F1=24.74±4.72%；Macro-Precision=25.19±6.38%；Macro-Recall/Balanced Accuracy=25.46±3.87%；ROC-AUC=70.87±3.23%。全CN 0/10、单类塌缩 0/10；OOF逐类Recall：CN 86.82%，AD 4.62%，SMC 5.13%，EMCI 0.00%，LMCI 31.17%。
全部逐折与OOF结果见[十折结果](docs/experiments/adni1to7_brainprompt_cv10_last_20261012/CV10_Test_Results.md)、[匿名JSON](docs/experiments/adni1to7_brainprompt_cv10_last_20261012/cv10_summary.public.json)、[证据/命令](docs/experiments/adni1to7_brainprompt_cv10_last_20261012/RUNBOOK.md)。624个OOF测试扫描存在对应TRAIN同人关联，队列有历史开发暴露；不能声称未见受试者/外部泛化，也不能与004三seed holdout混用。

## 005版本、产物与下一步
C=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012；C/results链接至C/code/8e5d8c3/BrainPromptPro/outputs/comparisons/adni1to7_brainprompt_cv10_last_seed41。训练本地commit8e5d8c35810bc6f8e5d4bfdffe2b54cf61a8c1c6、公开等树归档f628bcda06d9abe7bcb62351aa988e934ae283ef（tree d0fdfb4fd333938bd158c4d556b8b2a93456d07f）。controller2830424于10:54:39Z启动，GPU0/3/4/5/6/7；launcher/master/queue_status及逐折结果均保留。报告在C/reports/verified_cv10_20261012；输入SHA=217b662f09aa3c359b71b1d49cb51e1b77a37334c67f59663bed5aae5c6bd273，新split SHA=f44a16439d0d5f651532ca571673b4588ecdf1f5874fbe64a4b09467ff5aa26a。3协议契约、TRAIN反向、报告合成输入检查通过；正式10/10输入/来源/锁/成员/标签/编码器、scheduler首次停止、last.pt、指标/混淆/OOF及CUDA重载通过。
[PR2](https://github.com/yasuo-coder/AD-/pull/2)叠加PR1；main的discussion.md协作设置已合入保留。原003/004研究及锁不变。无缺失折/失败折；无需继续或恢复训练。旧worker没有任意epoch完整RNG恢复入口，禁止覆盖已完成目录。下一步提交并核验公开树，再审阅结果；后续新优化应定义独立研究范围，不据此TEST调参。公开副本只含源码/配置规则与匿名聚合报告。

## 已完成的004历史研究（保留证据）
核对：2026-10-11T11:27:30.526404+00:00（UTC）。先核实Git和真实产物；以下为七批研究最终状态。

## 当前任务与完成范围
ADNI_1–7/P2P3P5独立研究，计划004的训练、验证、消融、冻结测试和最终报告已完成。主协议61份分类验证（19方法×3seed+4项单seed探索）、10项预训练、subject A0/A1各3seed均已核验。63项固定TEST（主57+补充6）于2026-10-11T11:21:28Z全部成功；独立复算逐份对齐冻结标签/成员与七项指标、混淆矩阵和聚合，63/63通过。最终报告已发布至PR1分支；本地dfb6c81与公开24121414文件树均为050ba6304e41d5a8fee8efac1859020fb30bb0a0，已核验一致。

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

发布验收：代码修复、最终报告、匿名63项结果及图表已同步公开24121414d9926d84b7485ea5188dbb7759f331c7；与本地dfb6c8123262675c08977c282e0142a0a14a98a7树一致。计划004 M0–M12完成；此条交接记录作为后续提交，不预写自身SHA。

## 2026-10-12 共享讨论文件（从 main 保留的历史协作记录）

用户本次任务：建立 Codex 与 Kimi Code 的共享讨论文件。新增根目录 [discussion.md](discussion.md)，约定在 main 同步、轮流追加、标注代码分支与提交，附发言模板和 Codex 初始化记录。已核对默认分支、文件不存在及仓库指令，文档内容已审阅；未运行实验，Kimi 往返尚未验证。上文为 main 的历史研究快照，服务器 research/adni1to7-p2p3p5 分支的更新状态应在对应版本单独核对。既有研究计划本轮未推进。

下一步：用户指定讨论议题，或 Kimi Code 读取 discussion.md 并追加接入确认；对方回复后由 Codex 继续回应。讨论文件没有后台监听能力。
