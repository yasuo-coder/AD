# ADNI_1–7 数据审计

2026-10-10；以下为允许公开的整体匿名汇总。完整数据协议、逐扫描审计和划分信息仅保存在服务器。

| 数据集 | 有效FC | CN | AD | SMC | EMCI | LMCI |
|---|---:|---:|---:|---:|---:|---:|
| ADNI_1 | 200 | 130 | 4 | 5 | 5 | 56 |
| ADNI_2 | 198 | 126 | 4 | 15 | 15 | 38 |
| ADNI_3 | 200 | 111 | 11 | 18 | 6 | 54 |
| ADNI_4 | 200 | 127 | 7 | 1 | 16 | 49 |
| ADNI_5 | 200 | 132 | 15 | 0 | 7 | 46 |
| ADNI_6 | 198 | 134 | 10 | 0 | 11 | 43 |
| ADNI_7 | 127 | 90 | 14 | 0 | 1 | 22 |
| 总计 | 1323 | 850 | 65 | 39 | 61 | 308 |

895名生物学受试者，AD65（4.913%），非AD1258（95.087%）。不同访视保留，未发现纳入队列中相同扫描、时序或FC内容重复。
ADNI2有两个下载扫描缺少既有可信映射；ADNI6有两个采集未完成有效预处理。ADNI7同访视名但不同采集日期的记录依据实际预处理映射分别保留。未自动纳入缺少下载映射的存储文件。
全部纳入样本核验116×116、有限值、对称性、对角1、相关范围、非恒定矩阵及AAL116顺序。既存correlation_matrix.mat为Nilearn收缩协方差相关，非经验Pearson；本轮复用清洗时序统一np.corrcoef(ts.T)，输出float32并保留负相关，未重跑影像预处理。原五批998个FC及标签逐例相同。
MCI继续合并LMCI，CN/AD/SMC/EMCI/LMCI顺序不变。下载CSV Group作为统一标签来源；来源冲突保留在受控报告，并非临床独立裁决。site尚未独立核验，P5初版不使用metadata。
新建独立数据版本与评估协议。新队列含历史开发样本，并不构成全新外部验证；同人关联及历史使用审计均保存于服务器完整协议。各协议结果不得混合。

版本及哈希见[public_data_summary.json](public_data_summary.json)。基础检查见[public_validation_summary.json](public_validation_summary.json)。正式性能指标尚未运行。

受控产物根：`/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010`；data_audit_v4含审计明细，dataset_v1含cohort与完整protocol。公开仓库不包含这些私有文件。
