> 历史快照，非本轮当前协议。原文的相对路径以服务器原工程为基准；当前状态以根目录 STATE.md 为准。已去除扫描级标识。

# MCI 并入 LMCI：代码与数据迁移

2026-10-04，依据用户转述的作者邮件，BrainPrompt 和 BrainPromptPro 统一采用 `mci_to_lmci_v1`。
这是作者实验采用的标签合并规则，不是根据年龄或影像预测 MCI 的临床亚型。

| 原始诊断 | 训练诊断 | 数值标签 |
|---|---|---:|
| CN | CN | 0 |
| AD | AD | 1 |
| SMC | SMC | 2 |
| EMCI | EMCI | 3 |
| LMCI、MCI | LMCI | 4 |

保留官方 loader 中的历史数值标签 `5→4`。模型输出和疾病 prompt 顺序仍是五类，EMCI 独立保留。
原始 ADNI CSV、影像、预处理时序不修改；派生 manifest 的 `original_diagnosis` 保存来源诊断，
`diagnosis`、`label` 用于训练，同时记录 `label_policy`。站点、年龄等字段仍按原始受试者和访视匹配。

## 实际重建结果

| 数据 | 扫描数 | 受试者数 | 新纳入原始 MCI | CN | AD | SMC | EMCI | LMCI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ADNI_1 | 200 | 130 | 41 | 130 | 4 | 5 | 5 | 56 |
| ADNI_2 | 199 | 133 | 26 | 123 | 4 | 18 | 19 | 35 |
| 联合队列 | 399 | 263 | 67 | 253 | 8 | 23 | 24 | 91 |

ADNI_2 的 一张历史核验的重复扫描 为已核实的重复 CN 扫描，仍排除；MCI 排除数为 0。
原 ADNI_2 六个特殊访视按既有审核规则处理，其中五个保留、一个重复排除。
新增 MCI 经过完整的 stage1→stage2→finalize 元数据解析，并非直接给旧排除记录补数值标签。

图、subject prompt 文本、嵌入和两个批次的十折索引已一起重建。两批十折均按受试者分组；
ADNI_2 原先有受试者交叉的索引保留在备份中。样本数从 manifest 读取，不再写死 159/173。

Pro 新联合数据位于 `outputs/data/adni12_mci_to_lmci_v1/`，包括 FC、带 ID 的元数据嵌入、
manifest、排除清单、SHA256 来源记录与固定划分预览。配置为
`configs/adni12/BrainPromptPro_adni12_holdout.json`。

| 固定划分，seed=41 | 扫描数 | 受试者数 |
|---|---:|---:|
| train | 243 | 157 |
| val | 79 | 53 |
| test | 77 | 53 |

三组受试者完全不交叉，每组均包含五类。这里只是 ADNI_1+2，完整七批配置仍要求完整 ADNI 的数据与索引证据。

## 代码范围

- BrainPrompt：`adni_labels.py`；ADNI_1 preparation；ADNI_2 manifest 三阶段；两种 graph builder；元数据 prompt/embedding 生成器；ADNI 配置及日志、结果汇总入口。
- BrainPromptPro：`data/labels.py`；legacy/prepared 两个 loader；联合准备脚本；配置；checkpoint/result 的策略记录；报告生成器和标签回归测试。
- Pro 拒绝没有当前策略标记的历史配置和 bundle。仅给旧配置改策略名也不能通过检查。
- 报告按指定 run 保存的配置和 bundle hash 读取原队列，核对测试 ID 和标签，不再引用可变的当前 manifest 或写死样本数。
- `scripts/check_baseline.py` 检查本次授权修改后的快照。初次开发时“原仓库不修改”的快照保留为历史证据，不再把本次用户明确要求的源码更新判为非法变化。

## 备份与历史实验

本次修改前的源码、配置、图、嵌入和索引位于：

`/home/panyao/AD/BrainPromptPro/outputs/migrations/mci_to_lmci_v1/before/`

旧联合队列 `outputs/data/adni12/` 和旧训练 checkpoint/预测文件保持原样。
旧 332 扫描实验排除了 67 次 MCI 和 1 次重复 CN（并非 68 次 MCI）；旧指标不代表新策略。
正式比较需用新队列重新训练。旧十折日志不会被当前结果收集器当成新运行复用。

迁移目录保存重建日志、修改文件清单、前后 SHA256 和相对修改前源码的补丁。
如需回退，必须成套恢复源码、数据、嵌入、索引与配置，不能只恢复单个 `.bin`。

## 验证

- 27 项 Pro 回归测试全部通过，覆盖两套规则一致性、67 次 MCI 纳入、EMCI 保留、图标签/manifest 对齐、重复记录排除、旧策略拒绝、受试者无交叉、原模型前向/反向及兼容性。
- 原 BrainPrompt 的真实 `BrainDataset_llm` 成功加载 ADNI_1 的 200 图/嵌入、ADNI_2 的 199 图/嵌入和各自十折索引。
- Pro 使用新 399 扫描队列完成 2 epoch 链路测试，run 为 `mci_policy_sanity_20261004`。这仅验证训练/保存/报告流程，不是正式性能评估。
- `mci_policy_reload_20261004` 重新载入 checkpoint 后，预测 CSV 与前一运行逐字节一致。
- 原始时序与图逐元素核对、逐扫描 AAL116 ROI 名称顺序核对、完整 FC 和元数据嵌入有限值检查均通过。

当前数据已经准备完毕。复验代码：

```bash
cd /home/panyao/AD/BrainPromptPro
bash scripts/run_tests.sh
python3 scripts/check_baseline.py
```

后续正式训练使用当前联合配置：

```bash
/home/panyao/miniconda3/envs/brainprompt/bin/python llm-main.py \
  --config configs/adni12/BrainPromptPro_adni12_holdout.json \
  --split 0 --device cuda:1 --save_interpretability
```

`prepare_adni12.py` 对已有版本的输出目录拒绝覆盖；备份确认后才应重新准备数据。
