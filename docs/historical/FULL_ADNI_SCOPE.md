> 历史快照，非本轮当前协议。原文的相对路径以服务器原工程为基准；当前状态以根目录 STATE.md 为准。已去除扫描级标识。

# 完整 ADNI 范围更正与接入状态

> 2026-10-04 更新：下述原始实现／数据盘点保留为历史记录。当前策略将 MCI 并入 LMCI；
> 两套项目的相关源码和 ADNI_1/2 派生数据已更新，当前联合队列为 399 次扫描。
> 当前标签规则、备份和验证见 [MCI_LABEL_POLICY.md](MCI_LABEL_POLICY.md)。

用户明确指定数据位于 `/data2/panyao_data/ADNI/ADNI_1`～`ADNI_7`。
这七个目录均纳入正式目标。此前默认使用 ADNI_2 的做法已更正；历史 mini-training
仅验证模型和训练链路，不能视为论文完整 ADNI 的验证。

2026-10-02 的只读文件审计如下。预处理列表示发现输出文件，不表示通过了影像 QC。

| 批次 | 元数据扫描 ID | BIDS BOLD | 预处理 BOLD | AAL116 时序 | 现有筛选后 graph manifest |
|---|---:|---:|---:|---:|---:|
| ADNI_1 | 200 | 200 | 200 | 200 | 159 |
| ADNI_2 | 200 | 200 | 200 | 200 | 173 |
| ADNI_3 | 200 | 200 | 31 | 0 | 0 |
| ADNI_4 | 0 | 0 | 0 | 0 | 0 |
| ADNI_5 | 0 | 0 | 0 | 0 | 0 |
| ADNI_6 | 0 | 0 | 0 | 0 | 0 |
| ADNI_7 | 0 | 0 | 0 | 0 | 0 |

扫描清单共 600 个 subject/session、398 位 subject，没有跨批重复的同一 subject/session。
原始 CSV 共 1,200 行，包含结构与功能影像记录，不能将行数当成患者数。
ADNI_4～7 的 `data`、`ADNI_BIDS`、`ADNI_connectivity`、`BrainPrompt_data` 当前为空。
这描述当前挂载路径可见的状态，不判断其他设备或尚未迁移的数据。

可重复执行 `python3 scripts/audit_full_adni.py` 更新机器可读报告：

- `outputs/results/full_adni_audit/inventory.json`：七批覆盖情况、元数据哈希、官方 index 哈希和各折大小。
- `outputs/results/full_adni_audit/scan_inventory.csv`：跨七批扫描清单，包括原始诊断和预处理文件定位。

清单不删除 MCI，不合并或推断诊断，不自动去重，不冒充模型训练 manifest。
数据准备完毕后还须统一纳入/排除规则、标签映射、age/sex/site、ROI 顺序和 graph/prompt 身份来源。
特别是现有 ADNI_1/2 的筛选 manifest 不能直接拼接就宣称复现论文 cohort。

## 配置与防误跑

正式默认配置：`configs/adni_AAL116/BrainPromptPro_adni_AAL116.json`。
它登记七批 source root、论文目标 1,326 样本和原 `data/adni_AAL116` 十折 index 位置。
统一产物位置约定为 `@pro/outputs/data/full_adni`，当前尚未生成。
缺文件、缺七批 source_batch、样本数不符、缺生成链哈希或官方顺序证据时，入口在创建训练 run 前停止。
这些检查不能替代人工/来源审计；填入一个哈希并不能凭空建立 subject 身份。

`official_order_evidence` 就绪后为 `{ "path": "ordering_audit.json", "sha256": "..." }`；
所引用审计文件必须记录 `source_reference`、`alignment_method` 以及 `sha256` 中的
`manifest`、`graph_order`、`train_index`、`val_index`、`test_index`，绑定经过审阅的来源与具体文件。
`prompt_site_count` 也须据实际完整元数据/生成链确认，当前为 null，未照搬子集数值。
`alignment_provenance` 当前明确为 pending；不能仅修改该字符串绕过实际对齐工作。

小规模测试配置：`configs/smoke/BrainPromptPro_adni2_smoke.json`。
仅明确传入 `--config` 才选择 ADNI_2；旧 checkpoint 保留自己的历史配置。
新结果写入 cohort_name、scope、sample/subject 数，`full_adni_cv` 与一般 `complete_cv` 分开记录。
已新增测试确认 173 样本不能冒充完整数据，并验证七批盘点保留重复 ID 和未映射诊断。

## 与论文划分的对应关系

[BrainPrompt 论文](https://arxiv.org/html/2504.16096v2) 报告完整 ADNI 为 1,326 个样本。
本地官方十折每折确实覆盖 index 0～1325，但 index 文件自身不携带 subject ID。
发布的 subject embeddings 有 1,327 行。不能通过目录排序、拼接后顺序或截断一行来建立对应关系。
只有核验 cohort 和 ID 顺序后才使用这些 indices；如果核验发现问题，应明确报告，不自动重划分来掩盖。

当前完整数据接入未完成，未执行完整 ADNI mini-training、完整 CV 或完整数据消融。
等待 ADNI_4～7 的实际数据位置/迁移状态，同时需要补齐 ADNI_3 及其余批次的预处理和完整身份对齐来源。
