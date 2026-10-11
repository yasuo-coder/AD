# 七批研究验证结果

固定新协议；验证集选 checkpoint。以下不是最终测试结果。

| 方法 | 状态/完成seed数 | Accuracy | Macro-F1 | Macro-P | Macro-R / Balanced Acc | ROC-AUC |
|---|---|---|---|---|---|---|
| A0 | COMPLETE/3 | 57.89 ± 6.51 | 35.63 ± 1.59 | 38.28 ± 2.03 | 39.00 ± 2.95 | 67.71 ± 1.45 |
| LegacyPro | PARTIAL/1 | 66.17（单seed） | 34.60（单seed） | 43.58（单seed） | 32.44（单seed） | 62.25（单seed） |
| HistoricalM2 | PARTIAL/1 | 66.17（单seed） | 37.32（单seed） | 40.24（单seed） | 37.42（单seed） | 65.47（单seed） |
| HistoricalM3v3 | PARTIAL/1 | 58.65（单seed） | 32.92（单seed） | 32.27（单seed） | 36.35（单seed） | 66.04（单seed） |
| HistoricalM3v3_no_metadata | PARTIAL/1 | 60.15（单seed） | 37.70（单seed） | 38.47（单seed） | 42.61（单seed） | 64.24（单seed） |
| A1 | PARTIAL/1 | 53.38（单seed） | 35.31（单seed） | 32.91（单seed） | 39.47（单seed） | 68.17（单seed） |
| A2 | PARTIAL/1 | 60.90（单seed） | 33.01（单seed） | 34.57（单seed） | 35.81（单seed） | 65.21（单seed） |
| A3 | PARTIAL/1 | 63.16（单seed） | 35.59（单seed） | 38.49（单seed） | 36.49（单seed） | 67.58（单seed） |
| A4 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A5 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A6 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A7 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| P2_finetune_no_dropout_pair_best | PARTIAL/1 | 50.38（单seed） | 31.57（单seed） | 30.34（单seed） | 34.19（单seed） | 66.42（单seed） |
| P2_finetune_no_dropout_pair_last40 | PARTIAL/1 | 58.65（单seed） | 32.33（单seed） | 46.64（单seed） | 35.72（单seed） | 62.98（单seed） |
| P2_joint_rec_only | PARTIAL/1 | 52.63（单seed） | 36.17（单seed） | 34.86（单seed） | 40.68（单seed） | 66.32（单seed） |
| P2_joint_cons_only | PARTIAL/1 | 54.89（单seed） | 36.07（单seed） | 34.45（单seed） | 39.53（单seed） | 68.34（单seed） |
| P2_finetune_rec10 | PARTIAL/1 | 42.86（单seed） | 33.73（单seed） | 33.26（单seed） | 43.25（单seed） | 64.99（单seed） |
| P2_finetune_cons10 | PARTIAL/1 | 56.39（单seed） | 33.30（单seed） | 35.63（单seed） | 33.57（单seed） | 64.73（单seed） |
| P3_random | PARTIAL/1 | 45.86（单seed） | 30.82（单seed） | 31.60（单seed） | 33.62（单seed） | 66.98（单seed） |
| P3_learned_query | PARTIAL/1 | 52.63（单seed） | 30.78（单seed） | 29.19（单seed） | 35.71（单seed） | 68.57（单seed） |
| P3_parameter_matched | PARTIAL/1 | 64.66（单seed） | 37.63（单seed） | 39.61（单seed） | 39.23（单seed） | 65.23（单seed） |
| P5_fixed | PARTIAL/1 | 57.14（单seed） | 38.86（单seed） | 37.65（单seed） | 45.60（单seed） | 70.78（单seed） |
| P5_no_reliability | PARTIAL/1 | 57.14（单seed） | 35.14（单seed） | 34.32（单seed） | 38.99（单seed） | 69.64（单seed） |

单位为%；Balanced Accuracy与固定五类Macro-Recall在本验证集上相同。SD为训练种子间样本标准差，不是患者层面的置信区间。
未完成方法不参与收益结论；部分种子结果不代表完整三seed结果。受控逐样本预测与混淆矩阵保留在服务器。
