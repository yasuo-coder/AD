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
