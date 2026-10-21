# BrainPrompt 优化 Agent 任务书

> 目标：在 **ADNI 数据集（rs-fMRI 预处理后的 AAL116 FC 矩阵，五分类 CN/SMC/EMCI/LMCI/AD）** 上，借鉴 vkola-lab/ncomms2025（Nature Communications 2025）的架构与训练范式优化 BrainPrompt（MICCAI 2025），提升 **Accuracy、AUROC、AUPRC 及少数类（SMC/EMCI）Recall**。
> > >
> 代码基线：https://github.com/AngusMonroe/BrainPrompt
> 参考实现：https://github.com/vkola-lab/ncomms2025
> **禁止使用 ABIDE 数据集；输入仅为 fMRI 功能连接图，不引入 sMRI/PET 等额外影像模态。**

---

## -1. 前置核验：本地代码与 ADNI 数据集检查（任何改动前必须完成）

> 本阶段只读不改。目的是确认本地代码、数据、环境三者齐备且与任务书假设一致。**任一检查失败：停止执行，输出失败项与缺失文件清单，等待补齐后再继续。**

### T-1.1 定位代码仓库

```bash
# 在常见位置搜索 BrainPrompt 仓库根目录（特征文件: llm-main.py）
find / -maxdepth 5 -name "llm-main.py" -not -path "/proc/*" -not -path "/sys/*" 2>/dev/null
```

- 找到后将仓库根目录记为 `$REPO`，后续所有相对路径以 `$REPO` 为基准。
- 核验目录结构完整性，必须存在：

```javascript
  $REPO/llm-main.py
  $REPO/train_TUs_graph_classification_llm.py
  $REPO/metrics.py
  $REPO/baseline.sh
  $REPO/nets/{brainpromptg_net.py, brainpromptc_net.py, load_net.py, ...}
  $REPO/layers/{attention_layer.py, gcn_layer.py, mlp_readout_layer.py}
  $REPO/data/{BrainNet.py, data.py}
  $REPO/configs/adni_AAL116/TUs_graph_classification_BrainPromptG_adni_AAL116_100k.json
```

- 核验代码版本与本任务书 §0 描述一致：`grep -n "sigmoid" nets/brainpromptg_net.py` 应能看到 `sim()` 内的 sigmoid 包装（即病灶 1 存在，未被预先改动）。

### T-1.2 核验 ADNI 图数据文件

`data/BrainNet.py` 中 `name2path['adni_AAL116']` 指向的 `.bin` 文件必须真实存在（注意：原仓库该路径为占位符 `/path/to/...`，本地必须已替换为实际路径，否则先修正 `name2path`）。用以下脚本核验：

```python
import dgl, torch
from dgl.data.utils import load_graphs

graphs, labels = load_graphs("<adni_AAL116.bin 实际路径>")
print("图数量:", len(graphs))
print("标签分布:", torch.tensor(labels['glabel'].tolist()).bincount())

g = graphs[0]
assert g.ndata['N_features'].size(0) == 116, "节点数应为 116 (AAL116)"
feat = g.ndata['N_features']
assert not torch.isnan(feat).any(), "节点特征含 NaN"

# 标签重映射核对: 五类 0..4 (原始 label 5 应被 remap 为 4, 见 BrainDataset_llm.__init__)
uniq = sorted(set(labels['glabel'].tolist()))
print("类别集合(重映射前):", uniq)

# FC 矩阵合理性: 由节点特征重建 corrcoef, 检查对称性、对角线、值域
import numpy as np
C = np.corrcoef(feat.numpy())
assert np.allclose(C, C.T, atol=1e-5), "FC 矩阵不对称"
assert C.min() >= -1.001 and C.max() <= 1.001, "FC 值域越界"
print("FC 矩阵检查通过, 形状:", C.shape)
```

- 记录：总样本数、各类样本数（判断 SMC/EMCI 少数类规模，供 Phase 2 采样器参数参考）。

### T-1.3 核验 Prompt 嵌入文件

```python
import torch
roi   = torch.stack(torch.load("$REPO/data/prompts/ROI_prompts/AAL116_short_datatoken.pt")).squeeze()
label = torch.stack(torch.load("$REPO/data/prompts/label_prompts/adni_label.pt")).squeeze()
meta  = torch.load("$REPO/data/prompts/meta_prompts/adni_meta_datatoken.pt")
assert roi.shape   == (116, 2048), f"ROI prompt 形状异常: {roi.shape}"
assert label.shape == (5, 2048),   f"label prompt 形状异常: {label.shape}"   # 五分类应为 5 条
assert len(meta)   == <T-1.2 的图数量>, "meta prompt 数量与图数量不一致"
```

- 若 `adni_label.pt` 条数 ≠ 5，说明 label prompt 是旧版（可能按二分类生成），需按 `language_models/text2emb.py` 用五分类文本模板重新生成（模板见论文 §3.3：CN/SMC/EMCI/LMCI/AD 五个进展期描述）。

### T-1.4 核验数据切分缓存

- 检查 `$REPO/data/adni_AAL116/{train,val,test}.index` 是否存在且各 10 行（10 折）：

```bash
  wc -l $REPO/data/adni_AAL116/*.index   # 每个文件应为 10 行
```

- 存在则**原样保留**（硬性约束，保证与历史实验可比）；不存在则首次运行会自动生成，生成后立即备份到 `$REPO/data/adni_AAL116/index_backup/`。

### T-1.5 核验运行环境

```bash
python3 -c "import torch, dgl; print(torch.__version__, torch.cuda.is_available(), dgl.__version__)"
python3 -c "import sklearn, captum, tensorboardX"   # metrics.py / 训练脚本依赖
```

- 需要：PyTorch + CUDA（或明确接受 CPU 慢速冒烟）、DGL、scikit-learn、captum、tensorboardX、nilearn（Phase 3 T3.4 可选）。
- 缺依赖先补齐；DGL 与 PyTorch 版本必须匹配（`dgl.__version__` 与 `torch.__version__` 对应关系查 DGL 官方表）。

### T-1.6 基线冒烟测试（修复前的原版复现）

```bash
cd $REPO && python3 llm-main.py --gpu_id 0 \
  --config configs/adni_AAL116/TUs_graph_classification_BrainPromptG_adni_AAL116_100k.json \
  --node_feat_transform pearson --edge_ratio 0.2 --lambda1 1.0 \
  --epochs 3 --max_time 1 --out_dir out/smoke_test/
```

- 验收：能跑完 ≥1 个 epoch 且 loss 正常下降；**同时验证病灶 1 存在**——在 `brainpromptg_net.py` 的 `forward` 中临时打印 `binary_matrix.sum()`，应为 0（打印完即移除临时代码）。
- 将以上全部核验结果（路径、样本数、类别分布、文件形状、环境版本、冒烟 loss）写入 `$REPO/out/env_manifest.md` 存档，作为后续所有实验的环境基线。

**前置核验完成判据**：env_manifest.md 生成，且 T-1.1 ~ T-1.6 全部通过。

---

## 0. 背景与已知代码病灶（动手前必读）

### 0.1 数据管线现状（`data/BrainNet.py`）

- 每个被试：116×116 Pearson FC 矩阵 `C`；节点特征 `X = C`（每行 = 该 ROI 与全部 116 个 ROI 的连接剖面），邻接 `A` = `C` 按 `--edge_ratio 0.2` 稀疏化（保留 top-20%）。
- 标签重映射 `5→4` 已实现（五类为 0..4），10 折 StratifiedKFold + 每折 80/10/10 划分，`data/adni_AAL116/*.index` 缓存切分。
- Subject 级元数据：`data/prompts/meta_prompts/adni_meta_datatoken.pt`（age/sex/site 文本经 Llama-encoder-1.0B 编码的 2048 维向量，LLM 全程冻结）。
- ROI 级 prompt：`data/prompts/ROI_prompts/AAL116_short_datatoken.pt`；疾病级 prompt：`data/prompts/label_prompts/adni_label.pt`。

### 0.2 必须首先修复的三个病灶（Phase 0）

1. **人口图死代码**（`nets/brainpromptg_net.py`）：`sim()` 输出 `sigmoid(cosine) ∈ [0.269, 0.731]`，乘积上界 ≈0.534，而二值化阈值为 0.8 → `binary_matrix` 恒为零矩阵，`global_gcs`（GNNp）从未真正传播信息。BrainPrompt-C（`brainpromptc_net.py`）阈值 0.7 同病。
2. **`brainpromptc_net.py` 中 `fused_repr = fused_repr + hg + meta_repr` 的 `hg` 未定义**（应为 `out`），C 变体运行即 NameError。本次优化以 **BrainPromptG 为主干**，C 变体仅修复 bug，不做主实验。
3. **last-epoch 评估**（`llm-main.py`）：每 epoch 删除旧 checkpoint 只存最新，最终报最后 epoch 指标；模型选择依赖 LR<1e-6 停止。须改为按验证集指标保存最优 checkpoint。

### 0.3 硬性约束

- 不改动 10 折切分文件与随机种子协议（seed=41，配置文件内）；所有实验沿用相同切分以保证与基线可比。
- 不改动 `main()` 的 CLI 参数解析结构；新增功能一律通过新 CLI 参数 / config 字段开启，默认行为保持与原版一致。
- 每个 Phase 完成后跑通 1 个 fold 的冒烟测试（`--max_time 1`）再跑完整实验。

---

## 1. 总体架构目标

```javascript
输入: FC 矩阵 C (116×116)
  │
  ├─ Phase 1 表征层: Fisher-z(C) → [可选] tangent 嵌入; 邻接多尺度 {10%,20%,30%}
  │
  ├─ Phase 2 编码器: ROI tokens = Linear(FC行) + ROI-prompt(可dropout) + 可学习identity
  │      Transformer Encoder, attention logits += β · Fisher-z(C)   (连续FC偏置, 可学习β)
  │      5 个类别 query token cross-attention 读出 (替代 sum/mean pooling)
  │
  ├─ Phase 3 提示融合: subject prompt (可dropout, null token占位)
  │      人口图 → top-k 软稀疏化(可微) + 类别原型 memory bank (推理去批化)
  │
  ├─ Phase 4 输出头:
  │      (a) 5分类softmax头 (focal loss, 类平衡α)
  │      (b) 4个序数累积头 P(y≥k), CORAL式 + margin ranking loss
  │      (c) 疾病prompt对比头 (soft target: 按进展轴距离衰减的KL)
  │
  └─ Phase 5 训练/推理协议:
         两阶段迁移(CN-vs-AD预训→冻结15ep→全解冻微调)
         相邻类对采样, AdamW+cosine+warmup+gradclip
         按 val macro-AUROC 选模 + early stop(15)
         温度校准 + per-class Youden阈值 + TTA(边dropout)
```

---

## 2. 分阶段执行任务

### Phase 0 — 修复与评估协议重建（最高优先级，预估 1 天）

**T0.1 修复人口图死代码** — `nets/brainpromptg_net.py`

- 删除 `sim()` 中的 sigmoid 包装，改为直接返回 cosine 相似度矩阵（值域 [−1,1]）。
- 二值化改为 top-k 软稀疏化：

```python
  def soft_topk_adj(self, hg, meta, k=5):
      s = 0.5 * self.cos_sim(hg) + 0.5 * self.cos_sim(meta)   # [bz, bz], 可微
      v, idx = torch.topk(s, k=k, dim=-1)
      adj = torch.zeros_like(s).scatter_(-1, idx, F.softmax(v, dim=-1))
      return adj  # 行归一化的软邻接, 替代 binary_matrix
```

- `self.dropout(...)` 从邻接矩阵上移除（dropout 施加在特征而非邻接）。
- 新增 CLI 参数 `--pop_fusion {original,soft_topk,off}`，默认 `soft_topk`；`original` 保留原逻辑仅用于复现对照。

**T0.2 修复 BrainPrompt-C 的 `hg` 未定义 bug**：`fused_repr + hg` → `fused_repr + out`。

**T0.3 Best-checkpoint 选模 + early stopping** — `llm-main.py`

- 每个 fold 内按 **验证集 macro-AUROC（one-vs-rest）** 保存最优权重 `best.pkl`（不再每 epoch 删除）；final 评估用 `best.pkl` 而非 last epoch。
- early stopping：验证 macro-AUROC 连续 15 个 epoch 无提升则停止（对齐 ncomms2025 `early_stop_threshold=15`），与原有 LR 停止条件取先触发者。

**T0.4 指标系统重建** — `metrics.py` + `train_TUs_graph_classification_llm.py`

- `evaluate_network_all_metric` 改为必报：per-class one-vs-rest **AUROC/AUPRC**、macro-AUROC、macro-AUPRC、balanced accuracy、per-class recall、macro-F1、QWK（quadratic weighted kappa）。
- 结果文件追加完整指标表（10 折均值±std），不再只报 accuracy。
- 验收：任意 fold 冒烟运行后结果文件包含全部指标；`soft_topk` 模式下人口图邻接非全零（打印非零率断言）。

### Phase 1 — 推理侧概率质量工程（预估 1 天，零训练成本收益）

**T1.1 温度校准**：每折在验证集上拟合 per-class 温度（或单温度 scaling），保存 `temperature.pt`；测试集概率 = softmax(logits/T)。参考 `ncomms2025/adrd/model/calibration.py`。
**T1.2 Per-class Youden 阈值**：验证集 ROC 上按 Youden's J 选每类阈值，替代 argmax 输出预测；同时报告 argmax 与阈值两种口径的 Recall。
**T1.3 TTA**：测试时对 FC 邻接做 K=8 次 edge dropout（p=0.1）前向，概率取均值。

- 新增 CLI：`--calibrate --youden --tta K`。
- 验收：同一 checkpoint 上，阈值口径下 SMC/EMCI 的 Recall 不低于 argmax 口径；macro-AUROC 不因校准下降超过 0.002。

### Phase 2 — 损失与采样（对标少数类 Recall / AUPRC，预估 3~5 天）

**T2.1 类平衡 focal loss**：多分类 focal（γ=2，α=各类频率反比，归一化），替换 `loss()` 中的 CE；保留 `--loss {ce,focal,cb_focal}` 开关。参考 `ncomms2025/adrd/nn/focal_loss.py`。
**T2.2 相邻类对采样器**：改造 ncomms2025 `Transformer2ndOrderBalancedTrainingDataset` → `AdjacentPairSampler`：每个 batch 按相邻期对（CN-SMC、SMC-EMCI、EMCI-LMCI、LMCI-AD）各占 25% 采样。实现为 `torch.utils.data.Sampler`，`--sampler {none,adjacent_pair}`。
**T2.3 序数累积头（CORAL）**：新增 4 个共享权重的累积二分类头 P(y≥1),P(y≥2),P(y≥3),P(y≥4)；

- 标签 mask：样本只对其真实边界两侧的头计损失（对齐 ncomms2025 `y_mask` 机制）；
- **margin ranking loss**：`L_rank = Σ_i relu(margin − (p_i − p_{i+1}))`，margin=0.25，权重 λ_r=0.025 起步（ncomms2025 初始量级），CLI `--lambda_rank` 可调；
- 推理：预测期 = Σ_k [p_k > 0.5]。
- 总损失：`L = L_focal + λ_r·L_rank + λ1·L_disease(见T2.4)`，CLI `--ordinal` 开关。
**T2.4 疾病 prompt 软目标对比损失**：`compute_label_loss` 的 hard CE 改为按进展轴距离衰减的 soft target 分布（例：LMCI 目标 [0, 0.05, 0.15, 0.6, 0.2]，用 KL 散度）；soft 分布的衰减系数作为 config 字段。
- 验收：focal+相邻采样下，SMC recall 较基线提升；序数头预测与 softmax 头预测一致率 >70%（否则检查实现）。

### Phase 3 — FC 表征层（数据侧，预估 3~5 天）

**T3.1 Fisher-z 变换**：`BrainNet.py` 中 FC 计算后 `C = atanh(clip(C, -0.999, 0.999))`，对角线置 0。CLI `--fisher_z`。
**T3.2 多尺度邻接**：`--edge_ratio` 支持列表 `{0.1,0.2,0.3}`，构建 3 个稀疏度视图；编码器共享权重分别前向后拼接读出（视图级集成）。
**T3.3 符号双通道**：正/负 FC 边拆分为两个关系通道，GCN 消息传递分通道聚合后拼接（避免正负边混淆）。
**T3.4（可选）tangent 嵌入**：用 nilearn `ConnectivityMeasure(kind='tangent')` 生成切空间向量作为节点特征替代 `X=C`，CLI `--node_feat_transform tangent`。

- 验收：Fisher-z 后特征值域检查；多视图模式下参数量增长 <15%。

### Phase 4 — 编码器重构：FC-biased Masked Transformer（主创新，预估 1~2 周）

**T4.1 新模型 `nets/brainpromptt_net.py`（BrainPrompt-T）**：

- Token 构造：`token_v = Linear(FC行_v) + W_r·ROIprompt_v·mask_v + identity_v`，其中 `mask_v ∈ {0,1}` 由 **prompt dropout** 产生（训练时 ROI prompt 以 p=0.2 屏蔽，subject prompt 以 p=0.5 屏蔽，被屏蔽项用可学习 null embedding 占位——移植 ncomms2025 `DropoutMasker` permutation 策略）。
- Encoder：L 层标准 Transformer encoder；**attention logits 加连续偏置 `β·Fisher-z(C)`**（β 每头可学习标量，初始化 0.5），完全替代 top-k 硬稀疏邻接。
- 读出：**5 个类别 query token**（用 `adni_label.pt` 的 5 个疾病 prompt 嵌入初始化）对 116 个 ROI token 做 cross-attention，得到 5 个类条件表示，拼接/加权后进分类头；同一组疾病 prompt 嵌入同时服务于 Phase 2 的对比损失（参数共享，两处监督）。
- Subject prompt：作为额外 1 个 token 插入 token 序列（参与 self-attention），替代原有的人口图融合；人口图仅保留为 `--pop_fusion` 消融项。
- CLI：`--model BrainPromptT --prompt_dropout 0.2,0.5 --fc_bias`。
- 验收：参数量 < 1.5× BrainPromptG；1 fold 冒烟可训练且 loss 下降；prompt dropout 开启/关闭有显著指标差（否则检查 mask 是否真正生效）。

### Phase 5 — 两阶段迁移训练（预估 1 周）

**T5.1 Stage 1（粗分期预训）**：CN vs AD 二分类子集上预训编码器 + prompt 投影；可选挂辅助头（age 回归 / sex 分类，label mask 处理缺失，`--aux_tasks`）。
**T5.2 Stage 2（细分期微调）**：加载 Stage 1 权重；**冻结迁移参数 15 epoch**（`--transfer_epoch 15`，复刻 ncomms2025 `pseudocode.md` 的 freeze→unfreeze），只训五分类头/序数头/对比投影；随后全解冻，初始 lr 降为 Stage 1 的 1/10 微调。
**T5.3 优化器协议**：AdamW(β=(0.9,0.98)) + cosine annealing + 5 epoch warmup + grad clip 1.0；替换原 Adam + ReduceLROnPlateau（保留 `--optim legacy` 复现开关）。

- 实现位置：`llm-main.py` 训练循环新增 `--stage {1,2}` 与 `--init_from <ckpt>`；冻结逻辑参考 `ncomms2025/adrd/model/adrd_model.py` 的 `params_copied`/`requires_grad` 模式。
- 验收：Stage 2 第 1 epoch 的验证 AUROC 显著高于从零训练的同 epoch（证明迁移生效）。

---

## 3. 实验与消融矩阵（必须按序执行）

| 实验 | 配置 | 目的 |
| --- | --- | --- |
| E0 | 原版 BrainPromptG（复现） | 基线 |
| E0' | Phase 0 修复后（soft_topk + best-AUROC 选模） | 修复收益 |
| E1 | E0' + Phase 1（校准+Youden+TTA） | 推理侧收益 |
| E2 | E0' + Phase 2 各项逐项（focal / adjacent_pair / ordinal / soft-label） | 损失侧消融 |
| E3 | E2 最优组合 + Phase 3 各项逐项 | 表征侧消融 |
| E4 | E3 + Phase 4（BrainPrompt-T） | 编码器收益 |
| E5 | E4 + Phase 5 两阶段 | 完整方法 |
| E5a/b | E5 去掉 prompt dropout / 去掉 FC bias / 去掉序数头 | 核心组件消融 |

**报告口径**（每实验）：10 折 × 5 seeds 均值±std；per-class AUROC/AUPRC/Recall、macro-AUROC、macro-AUPRC、balanced acc、macro-F1、QWK；AUROC 给 DeLong 95% CI；与 E0 的配对检验（paired t-test 或 Wilcoxon）。
**鲁棒性附实验**：测试时屏蔽 subject prompt（模拟元数据缺失），比较 E0 与 E5 的性能落差——用于证明 prompt dropout 带来的缺失鲁棒性（ncomms2025 的核心卖点之一）。

---

## 4. 工程纪律

1. 每个 Phase 一个独立分支/commit，配置写入 `configs/adni_AAL116/` 新 json，命名带 phase 前缀（例：`phase2_ordinal_focal.json`）。
2. 所有新超参只经 CLI/config 暴露，禁止硬编码；默认值写在任务书涉及的代码注释中。
3. 训练日志用 tensorboardX 记录：各折 train/val loss、macro-AUROC、per-class recall、lr。
4. 禁止改动测试折的任何样本用于超参选择；所有调参只用验证折。
5. 结果汇总到 `out/results/summary_<timestamp>.md`：含消融矩阵表与指标 CI。

## 5. 完成定义（Definition of Done）

- [ ] 前置核验（T-1.1 ~ T-1.6）全部通过，`out/env_manifest.md` 已生成（含数据路径、类别分布、prompt 文件形状、环境版本、基线冒烟 loss，及病灶 1 的 `binary_matrix.sum()==0` 实证记录）。
- [ ] Phase 0 三个病灶修复并通过断言检查（人口图非零率 >0、C 变体可运行、选模用 best.pkl）。
- [ ] 全部实验 E0–E5 完成，结果落盘 `summary_*.md`。
- [ ] 相对 E0：macro-AUROC、macro-AUPRC、SMC 类 Recall 三项均有统计显著的提升（p<0.05）。
- [ ] 代码可通过 `baseline.sh` 等价脚本一键复现 E5。