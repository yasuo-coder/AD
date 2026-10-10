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
