# 七批研究验证结果

固定新协议；验证集选 checkpoint。以下不是最终测试结果。

| 方法 | 状态/完成seed数 | Accuracy | Macro-F1 | Macro-P | Macro-R / Balanced Acc | ROC-AUC |
|---|---|---|---|---|---|---|
| A0 | COMPLETE/3 | 57.89 ± 6.51 | 35.63 ± 1.59 | 38.28 ± 2.03 | 39.00 ± 2.95 | 67.71 ± 1.45 |
| LegacyPro | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| HistoricalM2 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| HistoricalM3v3 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| HistoricalM3v3_no_metadata | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A1 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A2 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A3 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A4 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A5 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A6 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| A7 | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| P3_random | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| P3_learned_query | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| P3_parameter_matched | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| P5_fixed | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |
| P5_no_reliability | NOT_RUN/0 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

单位为%；Balanced Accuracy与固定五类Macro-Recall在本验证集上相同。SD为训练种子间样本标准差，不是患者层面的置信区间。
未完成方法不参与收益结论；部分种子结果不代表完整三seed结果。受控逐样本预测与混淆矩阵保留在服务器。
