> 历史快照，非本轮当前协议。原文的相对路径以服务器原工程为基准；当前状态以根目录 STATE.md 为准。已去除扫描级标识。

# BrainPromptPro

当前 ADNI_1～4 合并输入的三种固定方法统一十折入口见 [MERGED_ADNI1234.md](BrainPromptPro_MERGED_ADNI1234.md)；该实验与此前 ADNI_1～3 的验证选型实验分开保存。

当前 ADNI_1–3 的官方十折结果、Pro 优化结构与运行方式见 [OPTIMIZATION.md](BrainPromptPro_OPTIMIZATION.md)。下文保留早期 grouped/best-checkpoint 与 ADNI_2 mini-training 的说明，不能直接当作本轮十折末轮实验结果。

基于 [AngusMonroe/BrainPrompt](https://github.com/AngusMonroe/BrainPrompt)，官方提交
`c85304a023636a2701d465e4cb0acb13b637127e`。2026-10-04 按用户转述的作者邮件，
已同步修改 BrainPrompt 和 BrainPromptPro：**MCI→LMCI（标签 4），EMCI 仍为标签 3**。
已有本地修改保留；没有使用或修改 BrainPrompt++。详情见 [MCI_LABEL_POLICY](MCI_LABEL_POLICY.md)。

项目位置：`/home/panyao/AD/BrainPromptPro`，与原 `BrainPrompt` 目录并列。
实现、测试、配置、实验输出均在本目录。原始影像和元数据保持原样；派生图、索引和 subject embeddings 已按新标签策略重建。
完整原架构、38 个问题的核对、论文差异见 [IMPLEMENTATION_ANALYSIS](IMPLEMENTATION_ANALYSIS.md)；
A–N 交付报告见 FINAL_REPORT（服务器历史产物：`/home/panyao/AD/BrainPromptPro/docs/FINAL_REPORT.md`，不在公开仓库）。

```text
Full Pearson FC + ROI prompt + individual FC rows
  -> SemanticGuidedAdaptiveBrainGraph (continuous, symmetric, subject-specific)
  -> Weighted GNN_r (D^-1/2 A D^-1/2 H W)
  -> DiseaseAwareSemanticReadout (all K disease queries for every subject)
  -> Subject representation
  -> MultiRelationPopulationGraph (age / sex / site)
  -> Separate weighted GNNs -> learned relation gate
  -> residual sum with subject representation and projected subject prompt
  -> Predictor
```

| 原实现 | Pro 实现 |
|---|---|
| `data/BrainNet.py`, `data/data.py` | `data/brainnet_pro.py`, `metadata.py`, `splits.py`, `data.py` |
| `nets/brainpromptg_net.py` | `nets/brainprompt_pro_net.py` |
| `layers/gcn_layer.py` | `layers/gcn_layer.py`：显式 dense weighted propagation |
| `layers/attention_layer.py:LMPE` | 关闭 adaptive 时保留一次 ROI projection + addition |
| `layers/mlp_readout_layer.py` | 原样复制到同名 Pro 文件 |
| `llm-main.py`, training helper | Pro 独立入口、`cli.py`、training helper |

环境与测试：

```bash
cd /home/panyao/AD/BrainPromptPro
export BRAINPROMPT_PYTHON=/home/panyao/miniconda3/envs/brainprompt/bin/python
bash scripts/run_tests.sh
python3 scripts/check_baseline.py
```

使用现有 `brainprompt` 环境，已验证 Python 3.10、PyTorch 2.4.0+cu121、DGL 2.4.0+cu121、
NumPy 1.26.4、SciPy 1.15.3、scikit-learn 1.7.2。测试使用 Python 标准库 unittest，无需 pytest。
DGL 仅用于读取原 `.bin`、原模型及对照测试，新 weighted GNN 使用 PyTorch batched matmul。
早期结构测试在 CPU 完成；沙箱内不可见 GPU，现已确认沙箱外可访问 RTX 4090 并用于 ADNI_1＋ADNI_2 训练。
默认 device 为 CPU，GPU 就绪时明确传 `--device cuda:0`。

ADNI_1＋ADNI_2 联合评估：

历史策略下的固定划分结果报告（服务器历史产物：`/home/panyao/AD/BrainPromptPro/outputs/results/adni12_holdout_seed41_20261002/fold_0/REPORT.md`，不在公开仓库）包含旧队列的指标和图表。该实验排除 MCI，结果不能作为当前合并策略的评估；新策略需重新训练。

已增加两个预处理批次的独立实验入口（不改变完整 ADNI 默认配置）：

```bash
cd /home/panyao/AD/BrainPromptPro
# 数据准备只需执行一次：输出带 ID 的 FC/prompt bundle，已有输出时拒绝覆盖
"$BRAINPROMPT_PYTHON" scripts/prepare_adni12.py --device cuda:1
# 两批按受试者联合做约 60/20/20 的固定划分，seed=41
"$BRAINPROMPT_PYTHON" llm-main.py \
  --config configs/adni12/BrainPromptPro_adni12_holdout.json --split 0 --save_interpretability
# 对指定 run 的测试预测重算指标并生成报告
"$BRAINPROMPT_PYTHON" scripts/report_adni12.py --run_id RUN_ID
```

ADNI_1 的 site 从现有元数据按完整 subject ID 恢复；两批均使用同一无标签模板重新编码 subject prompt。
报告包含 Accuracy、macro/weighted F1、Precision、Recall、OvR AUC、逐类指标、混淆矩阵、
扫描级及受试者级结果。当前联合输入为 399 次扫描／263 人，纳入原先排除的 67 次 MCI，
仅排除 1 次重复 CN 扫描。旧实验使用 332 次扫描／199 人，历史记录保留。
默认最多 200 epoch，验证损失连续 30 epoch 不改善时停止；测试集只在模型选择完成后评估。
当前 GPU 在沙箱外可访问；此前 CPU 环境验证的记录仍作为历史结果保留。

数据范围与当前状态：

- 正式目标是 `/data2/panyao_data/ADNI/ADNI_1`～`ADNI_7` 全部七批，并核验其与论文完整 ADNI 的对应关系。
  默认 `configs/adni_AAL116/BrainPromptPro_adni_AAL116.json` 已改为完整数据配置。
- **完整数据接入尚未完成。** 当前可见：ADNI_1/2 各有 200 份时序；ADNI_3 有 200 次 BIDS 扫描但无 FC；
  ADNI_4～7 的 data/BIDS/FC/brain graph 目录为空。七批审计和缺口见
  [FULL_ADNI_SCOPE.md](FULL_ADNI_SCOPE.md)。不能把目录存在当成数据已齐备。
- 默认完整配置缺少经审计的统一 manifest、脑图、带 ID 顺序证据的 metadata/prompt、
  官方 index 与实际 subject 顺序对应证据，**会在创建训练 run 前报错**，不会回退到 ADNI_2。
  `@pro/outputs/data/full_adni` 是待准备产物的位置约定，不表示已有这些文件。
- 官方十折 index 每折覆盖 1,326 个样本；只有证实 cohort/顺序一致后才可使用。
  不将七批目录排序直接套入官方 indices，也不截断 1,327 行匿名 subject embeddings 来凑数。
- ADNI_2 链路测试配置已更新为 199 session／133 subject，位于
  `configs/smoke/BrainPromptPro_adni2_smoke.json`，只用于明确指定的链路测试。
  它的 graph order、prompt order、manifest 通过 subject/session 匹配，原始时序逐元素核验，
  subject embeddings 通过已审阅生成脚本、order CSV、文本和 SHA256 绑定身份。
- 完整 FC 沿用原 loader 的 `np.corrcoef(N_features)`，不从阈值图恢复。
  age mean/std、site vocabulary 仅在当前训练集合 fit；缺 site 明确报错。
- ADNI_2 子集旧划分有 longitudinal subject overlap；重建的 baseline indices 和 smoke 均使用 subject-grouped CV。
  该发现不能推广到官方完整 ADNI 划分；完整配置保留官方 indices，接入后仍须检查 ID 泄漏。

运行：

```bash
# 七批只读盘点；未完成的数据也纳入报告
python3 scripts/audit_full_adni.py

# 明确选择已核验的 ADNI_2 小规模链路测试
SMOKE_CONFIG=configs/smoke/BrainPromptPro_adni2_smoke.json
"$BRAINPROMPT_PYTHON" llm-main.py --config "$SMOKE_CONFIG" --audit_only

# 3 epoch 的单折真实数据 sanity run；只缩小训练集，保留 val/test
bash scripts/run_single_split.sh --config "$SMOKE_CONFIG" --epochs 3 --batch_size 8 --max_train_samples 32 --save_interpretability

# 完整 ADNI 接入并核验成功后：单折训练；目前缺少输入会明确报错
bash scripts/run_single_split.sh --save_interpretability

# 完整 ADNI 的 10-fold（当前数据未齐，不会启动子集代替）
bash scripts/run_full_cv.sh --save_interpretability

# 完整数据就绪后：8 种消融，按相同配置和划分运行 fold 0
bash scripts/run_ablation.sh --split 0

# 不改原模型源码，在相同安全划分和训练流程下训练实际官方 BrainPromptG
"$BRAINPROMPT_PYTHON" llm-main.py --official_baseline --split 0

# 重载已交付的 mini checkpoint，只评估和导出，不训练
"$BRAINPROMPT_PYTHON" llm-main.py \
  --checkpoint outputs/checkpoints/mini_all/fold_0/best.pt --save_interpretability
```

入口可以从任意 cwd 调用；相对 config 路径按调用者 cwd 解析，默认 config 按本文件位置解析。
输出路径由 Pro 包位置决定，不能写出 `BrainPromptPro/outputs`。
每次自动创建 Asia/Shanghai 时间命名的 run；显式 `--run_id` 若已存在则拒绝覆盖。
训练不查看 test，按最低 validation 总 loss 保存 best checkpoint，训练结束后重载它，只评估一次 test。
checkpoint 保存模型、配置、数据哈希、fold 索引、实际训练索引、age/site 编码、optimizer。
`--checkpoint` 是只评估入口，不是断点续训入口。

消融：

| `--ablation` | Adaptive | Disease readout | Multi-relation |
|---|---:|---:|---:|
| `baseline` | 0 | 0 | 0 |
| `adaptive` | 1 | 0 | 0 |
| `disease` | 0 | 1 | 0 |
| `multi` | 0 | 0 | 1 |
| `adaptive_disease` | 1 | 1 | 0 |
| `adaptive_multi` | 1 | 0 | 1 |
| `disease_multi` | 0 | 1 | 1 |
| `all` | 1 | 1 | 1 |

全部关闭为 baseline-compatible mode，仍是 Pro 实现。真正原模型用上面的
`--official_baseline` 只读 adapter 或原仓库 `llm-main.py`。
Adaptive OFF 恢复原 FC 阈值图和 LMPE；Disease OFF 恢复原 readout 与末端 disease auxiliary loss；
Multi OFF 恢复 sigmoid-cosine 乘积加固定 0.8 阈值及无归一化 simpleGCN。
**官方 sigmoid-cosine 的乘积最高约 0.534，故原 0.8 阈值使 population graph 恒零。**
兼容模式保留此行为，不暗中修正官方 baseline。

Loss：`CE(class_logits,y) + lambda_disease*CE(disease_logits,y)`，加可配置的
brain/pop off-diagonal mean 稀疏项和可选 relation cosine-squared 相似性惩罚。
默认系数为 `1, 1e-4, 1e-4, 0`。四项可用同名 `--lambda_*` 参数覆盖。
关闭模块时相关图正则为零；batch size=1 时 population off-diagonal 正则为零。

解释性输出只在最终 test 且传入 `--save_interpretability` 时保存。
`.pt` 包含 `[S,N,N]` 脑图、`[S,K,N]` ROI attention、`[S,R]` gate、真实标签、预测、概率和 ID。
Population adjacency 按 batch 保存，各矩阵带自身 row/column ID；不能当作全数据集邻接。
CSV 保存每个 sample 的预测与 relation weights。weights 是归一化前的对称 `[0,1]` 矩阵，
传播时内部使用对称度归一化。关闭的解释性模块明确记为 None。

限制：模型预测依赖 batch composition，尤其最后一个 singleton batch 无跨患者消息；
固定评估排序和 batch size 才能复现实验。默认 BN 的训练统计跨 ROI batch 共享，
但脑图 message passing 仍按 subject 隔离。pair scorer 分块且训练时 checkpoint 重算，
没有用 hard threshold 缩小新图；真实 GPU 峰值内存尚未测量。
ADNI_2 类别计数 CN=123、AD=4、SMC=18、EMCI=19、LMCI=35，10-fold 很多 split 缺类别，
AUC 返回 null 并保存 present_classes。mini-training 仅证明训练链路可用，不能证明方法性能提升。
ROI 顺序依赖现有 AAL116 preprocessing 的约定；原 `.mat` 不含可独立验证的完整 ROI 名称映射。
