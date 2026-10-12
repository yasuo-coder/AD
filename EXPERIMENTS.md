# 实际实验索引
核对时间：2026-10-10T08:37:13+08:00；历史实验来源于服务器产物，接入Git不意味着这些实验由新提交训练。

| ID | 状态 | 配置/seed | 证据 |
|---|---|---|---|
| FOPT-20261007-stage1 | 12训练已完成，独立验证通过 | 3架构×加权3seed；另3未加权seed42 | R/experiments与verification/development_verification.json |
| FOPT-20261008-stage2 | 72训练已完成，正式72/72已独立核验通过 | 24配置×42/3407/2026 | S/experiments/result.json、source_sha256、best_validation.pt、last.pt、history.jsonl |
| FOPT-smoke | 6两轮入口检查，不进入性能均值 | seed42 | S/smoke与contract_tests |
| FOPT-final-test | 未运行，选择已冻结 | 同一test100，每模型3seed | 两个selection_lock已创建；不填测试数值 |

R/S定义见复现文档。bundle和split标识见数据协议。最后队列last_triples_20261009：6退出0，无活跃任务。本轮无新增训练；已接管现有作业并等待完成。

[验证三seed完整结果](docs/experiments/final_optimization_20261007/Validation_Results.md) · [逐seed/配置/最佳轮/停止轮/耗时/结果路径](docs/experiments/final_optimization_20261007/validation_per_seed.csv) · [双基线差值](docs/experiments/final_optimization_20261007/validation_three_seed.csv)。表中均为验证集，不是最终test。
原BP主基线始终保留；BP_control单独呈现，不能选择较低基线夸大收益。五批holdout与历史四批十折末轮不可混合。早期历史输出仅定位为存在；未重新逐项审核的不列作本轮实证。

## ADNI1–7 / 2026-10-10
数据审计与独立版本构建完成；[匿名汇总](docs/research/adni_1to7/DATA_AUDIT.md)。16模块单测、CPU/CUDA各8组合及TRAIN真实小样本梯度检查PASSED；8份真实数据smoke入口结果通过；CPU A0两轮断点恢复与连续训练模型/RNG完全一致。A0–A7正式指标PENDING，尚无七批测试结果。受控输出位于/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010。

### 七批 phase1 实际启动
队列 N/queues/phase1_baseline_p2，PID1153238；A0三seed和7个P2 seed42/40轮TRAIN-only探索，GPU0/3/5/7。不可变运行commit074d2e0，公开等价tree commit4ad8ec1，PR1。具体完成进度以queue/result为准；尚未汇总完整三seed，不提前填性能均值。P2首个mask.05验证重建MSE .056895、TRAIN边均值 .035413、相关性 .063031，未证明重建有效；probes/P2_pretrain_rec_05.json。新增P3三对照/P2两数值候选的5个CPU单轮smoke均通过，20单测和修订CPU8组合通过；这些单独归类为验证入口。

Phase1后续核对：10项训练均退出0，无残留running/pending。A0三seed原始payload指标独立复算通过，公开白名单汇总位于docs/research/adni_1to7/phase1；只含匿名标量/哈希，不含私有预测和split明细。其他模型正式结果仍未运行，不填写比较收益。

## 七批探索完成及Phase5 / 2026-10-11
Phase1/2/3/4分别10/14/5/2任务全部退出0，共21正式分类和10预训练。21份分类checkpoint/source/共同配置及概率指标独立核验通过，匿名[完整表](docs/research/adni_1to7/exploration_20261011/Validation_Results.md)；各次运行commit、checkpoint哈希、seed、最佳/停止epoch和训练耗时在同目录JSON。私人证据N/reports/exploration_complete_20261011。全部为validation，test NOT_EVALUATED。
Phase5：本地29f34f9/公开等价251dd6c8，config SHA cd5a2194a5f03f5ba51a20108c03e2ca8589eb76c044e0cec28b9009fc93f0da。controller1408894，启动2026-10-11T01:45:32Z，GPU0/3/4/5/6/7；40项新任务补齐A0–A7与11关键对照的三seed。使用已完成同seed A0为P5参考；数据与split哈希沿用DATA_AUDIT。命令与逐任务输出见N/queues/phase5_three_seed/launcher.json、jobs.private.json及RUNBOOK。不是已完成的三seed性能结论。

### 2026-10-11 后续实证
主协议已独立复算56分类结果，最新phase5_progress_20261011；25机制探针完成，三seed机制表与单seed对照分别公开。分析3单测与4项指标核验拒绝契约通过。Phase6受试者隔离A0三seed独立运行，controller1425910，GPU1/2/5，输出N/subject_experiments，命令见queues/phase6_subject_baseline/launcher.json；主协议结果不能与之合并。待Phase5完成生成完整配对消融，测试未运行。

### 主消融完成 / 2026-10-11
Phase5 40/40成功结束，主观察器1428866独立复核61分类结果，生成validation_complete与完整三seed配对消融。19正式方法各3seed完整；四项失败预训练微调只作为单seed探索保留，并非尚在排队的缺失正式组。Phase6的subject A0三seed完成及重载通过；Phase7 A1三个seed运行中，controller1432091，GPU1/2/5，训练代码29f34f9。
新最终评估入口20种代表性正式checkpoint仅VAL重载通过；4项锁合约测试通过。TEST还未评估；候选在任何TEST访问前按D021由主VAL选定。两协议开发角色交叉的限制及私有证据见D022。

### Phase7完成与Phase8前置失败 / 2026-10-11
subject A1三seed均退出0（best/stopped：42=154/195，3407=83/183，2026=188/211）。Phase8 controller1433530于05:07:13Z在subject汇总阶段失败：KeyError running，未生成lock且无TEST访问。错误源为控制器与训练队列状态schema混读。3项新回归+3分析+4锁测试通过，准备按D023使用独立queue目录恢复，原产物保留。

### 七批最终评估 / 2026-10-11
修复版本39ee2f2，仅改变队列schema汇总和独立恢复目录；原evaluator SHA不变。controller1629946、GPU0/3/6，11:18:42–11:21:28Z，63/63成功。Phase8旧失败与日志保留，未覆盖。锁SHA fb71040c47d235b382dac5f487b01ff9f925d262f305f134254a67250bc63031；N/reports/frozen_test_complete_20261011为自动核验结果，N/reports/final_analysis_20261011另按冻结成员/标签和直接计数公式复算，63/63通过，不再次执行模型。
主协议19×3、subject A0/A1×3；[最终全表](docs/research/adni_1to7/final_test/Test_Results_sample.md)、[补充](docs/research/adni_1to7/final_test/Test_Results_subject.md)、[逐类/成本/配对](docs/research/adni_1to7/final_test/Final_Analysis.md)、[最终结论](docs/research/adni_1to7/Final_Report.md)。本轮67份分类验证（主61+subject6）及10预训练完成，4项微调明确为单seed探索。没有全单类塌缩，但EMCI/SMC识别失败保留；最终负面结果不改写为成功。旧五批TEST仍未运行。
恢复命令与逐任务checkpoint路径见各queues/launcher.json、jobs.private.json、run/config.json与result.json；训练来源主要29f34f9，不将报告提交当训练来源。report_final只读现有预测生成新私有报告，模型测试不可重跑。当前本研究运行作业0个。

最终报告校验：3项独立公式回归（完美分类、全CN、无效输入）与3队列schema、3配对统计、4锁契约共13项测试通过；最终匿名JSON字段白名单与新增Markdown链接检查通过。

公开发布核验：最终结果与修复已接入research/adni1to7-p2p3p5；本地dfb6c81/公开24121414的tree一致（050ba6304e41d5a8fee8efac1859020fb30bb0a0）。源码快照、原私有实验与失败日志未改。

## ADNI1–7 BrainPrompt CV10-last / 2026-10-12
用户确认官方末轮规则。状态PREPARING，正式十折未运行；新配置BrainPromptPro/configs/adni_1to7/brainprompt_cv10_last_v1.json，计划005。新受控根/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012；输入沿用217b662f...数据版本，不覆盖004。

005实际启动：本地训练源码8e5d8c3，controller2830424，六卡0/3/4/5/6/7，10:54:39Z开始。新split SHA f44a16439d0d5f651532ca571673b4588ecdf1f5874fbe64a4b09467ff5aa26a；原数据及旧锁hash封存。3项官方协议检查通过，TRAIN-only 40扫描检查通过；正式10折运行中，未填写最终指标。
