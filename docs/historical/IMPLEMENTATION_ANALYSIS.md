> 历史快照，非本轮当前协议。原文的相对路径以服务器原工程为基准；当前状态以根目录 STATE.md 为准。已去除扫描级标识。

# IMPLEMENTATION ANALYSIS

先读后改：本分析内容已在实现前输出到会话。论文正文、公式、Figure 1、实验与消融均已阅读；
随后阅读官方 Git tree 内 Python 源码、配置、prompt 文本与 embedding 格式，以及本地数据生成链。
检查了原 `BrainPromptC` 作为参照，没有以它或 BrainPrompt++ 为新方法基线。

来源：[原论文 arXiv v2](https://arxiv.org/html/2504.16096v2)、
[MICCAI accepted PDF](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、
[官方 GitHub](https://github.com/AngusMonroe/BrainPrompt)。
2026-10-02 核对远端 HEAD 与本地 HEAD 同为 `c85304a023636a2701d465e4cb0acb13b637127e`。
现有本地修改单独保存在 `git_status_before.txt`、`git_diff_before.patch`，不当成官方源码。

```text
FC / Individual Brain Graph
  -> ROI Prompt -> GNN_r -> Readout -> Subject Representation
  + Subject Prompt -> Population Graph -> GNN_p
  -> residual fusion -> Predictor + Disease Prompt Loss
```

符号 B=batch size、N=ROI 数、T=BOLD 时间点、D=hidden dim、P=prompt dim、K=类别数。
原 ADNI-G 默认 N=D=116、P=2048、K=5、L=4。

| Stage | File, class, function | Input → output | Trainable | Hard threshold |
|---|---|---|---|---|
| FC/graph | `data/BrainNet.py:BrainDataset_llm.__init__` | `[N,T] → [N,N]`，node FC rows `[N,N]` | no | yes, edges |
| Input projection | `nets/brainpromptg_net.py:BrainPromptGNet.forward`, `embedding_h` | `[B*N,N] → [B*N,D]` | yes | no |
| ROI prompt | `layers/attention_layer.py:LMPE.__init__/forward` | `[N,P] → [N,D]`；add to `[B,N,D]` | projection only | no |
| GNN_r | `BrainPromptGNet.layers`, `layers/gcn_layer.py:GCNLayer.forward` | 4 × `[B*N,116] → [B*N,116]` | yes | topology precomputed |
| Readout | `BrainPromptGNet.forward`, `dgl.mean_nodes` | `[B*N,D] → [B,D]` | no | no |
| Subject prompt | `BrainDataset_llm`, `linear_layer1` | `[B,P] → [B,D]` | projection only | no |
| Population graph | `BrainPromptGNet.sim/forward` | two `[B,D] → two [B,B] → A [B,B]` | threshold blocks topology gradient | `>0.8` |
| GNN_p | `layers/gcn_layer.py:simpleGCNLayer.forward` | `M [B,D], A [B,B] → [B,D]` | yes | no |
| Fusion | `BrainPromptGNet.forward` | `global_gcs(hg,A)+hg+meta_repr → [B,D]` | inherited | no |
| Disease loss | `llm-main.py:train_val_pipeline`; `label_transform`, `compute_label_loss` | `[K,P] → [K,D]`；cosine with fused `[B,D] → [B,K]` | projection only | no |
| Predictor | `layers/mlp_readout_layer.py:MLPReadout.forward` | `[B,116]→[B,58]→[B,29]→[B,5]` | yes | no |

逐项核对用户提出的 38 个问题：

1. FC 在 `BrainDataset_llm.__init__` 对每幅图的 `N_features` 调用 `np.corrcoef` 生成。
2. 使用 Pearson correlation，ROI 是行、BOLD 时间点是列。
3. 同一函数读取／补成完全图，然后 `remove_edges` 构造 adjacency；并非模型 forward 动态构图。
4. `edge_ratio` 非零时，在 flatten 的**有符号**相关系数排序中取 `int(N*N*(1-edge_ratio))` 分位；
   然后删 `abs(E_features)<threshold`。因此并不严格等于绝对相关 top20%。
5. 官方 AAL116 prompt 为 116 行，本地两个 cohort 图均为 116 ROI。
6. 默认 `original` 为时序；实际 `baseline.sh --node_feat_transform pearson` 为 FC row。
7. `pearson` 分支再次 `np.corrcoef(N_features)`，每个 node feature `[116]`，完整矩阵未被 edge threshold 截断。
8. ROI prompt 由 LMPE 加载 `data/prompts/ROI_prompts/AAL116_short_datatoken.pt`。
9. 文件是 tensor list，stack 后 `[116,2048]`。冻结输入；原代码不是 register_buffer，forward 手动迁移 device。
10. ROI prompt 在 input projection 后、GNN 层循环前仅加一次。
11. LMPE 用 Linear(2048,D) 投影 ROI identity semantics，广播至 batch、相加、dropout。
12. GNN_r 为 `BrainPromptGNet.layers` 中的 `GCNLayer`，底层 DGL GraphConv。
13. `embedding_h: [B*116,116]→[B*116,116]`，四层各自输入输出同形。
    hidden/out 不相等时原 LMPE 与 global_gcs 假设会产生不匹配，默认 config 恰好相等。
14. 默认 `dgl_builtin=True` 路径不使用 edge feature。
15. `self.conv(g,feature)` 不传 `e` 或 `edge_weight`；`g.edata['feat']` 也不被该调用自动使用。
    非默认自定义 message 分支才设置 `g.edata['e']` 并在 e_feat=True 时 u_mul_e。
16. sum/mean/max 都有分支；未知 readout 默认 mean，ADNI config 明确 mean。
17. 官方 subject embedding 路径为 `data/prompts/meta_prompts/adni_meta_datatoken.pt`；
    本地已有适配分别读取 ADNI_1 / ADNI_2 的自定义 `.pt`。
18. `batch_llms` 是逐 subject 预计算文本 embedding，通常 `[B,2048]`，不是 token ID 或模型实例。
19. `meta_repr=linear_layer1(batch_llms.squeeze(1))`，形状 `[B,D]`。
20. `hg_sim=sigmoid(cosine(hg,hg))`，`llm_sim=sigmoid(cosine(meta_repr,meta_repr))`；
    两者 Hadamard product 阈值化后为 binary_matrix，G 版本随后应用 dropout。
21. 固定 `>0.8` 在 `nets/brainpromptg_net.py:forward`，不是从 config 读取。
22. 阈值输出是 binary；dropout 在 train 时理论上把存活值缩放，因此一般不能说 dropout 后仍严格 0/1。
23. 阈值人工固定，而且 `sigmoid(1)^2=0.5344467<0.8`，故当前默认的 A 实际恒零。
24. GNN_p 是 `simpleGCNLayer`：`X+relu(A@(XW)+b)`，没有 degree normalization。
    外层再加一次 X，所以零邻接时融合成为 `2*X+relu(b)+meta_repr`，仍能分类但没有人口消息。
25. Disease embedding 在 `llm-main.py:train_val_pipeline` 加载 `data/prompts/label_prompts/adni_label.pt`。
26. `label_embs` 是五个类别的冻结 embedding `[5,2048]`，**顺序 CN, AD, SMC, EMCI, LMCI**。
27. `label_transform` 是 Linear(2048,D)。
28. `compute_label_loss` 对 fused graph representation 和投影的所有 class embeddings L2 normalize，
    点积得到 cosine logits，再以 label 为 CE target；没有 temperature，原 norm 没有 eps。
29. Disease prompt 不进入原 ROI readout。
30. 它是末端 auxiliary semantic loss；共享表示会接收该 loss 的训练梯度，但 forward readout 不查询 disease prompt。
31. 官方 `ADNI_process.csv` 文件未提供，无法确证 named columns。现有可审计数据是
    `/data2/panyao_data/ADNI/ADNI_2/BrainPrompt_data/brainprompt_manifest.csv` 的 `age/sex/site_id`；
    溯源 `pipeline_logs/ADNI_metadata.csv` 的 `Age/Sex/site_id`、`Subject`、`Visit`、`Image Data ID`。
32. `ADNI_process.csv` 结构无法读取。未被主模型调用的 `load_metadata` 只表明代码检查 row[3] 的 F/M，
    然后取 row[3:6]；这不足以认定余下列名或 ID schema。
33. 官方 dataset 仅按 graph index 取 extra_embedding[i]；本地旧适配增加长度断言，仍未严格 ID 验证。
34. 官方 prompts `.txt/.pt` 没有 subject ID。ADNI_2 具有 graph_order、meta_order、manifest 的 subject/session，
    并可验证 .bin 中时序与 ID 对应 .mat 完全相等。Pro 采用这一可审计路径。
35. `BrainDataset_llm.get_all_split_idx` 读写 `data/<dataset>/{train,val,test}.index`；
    原自动生成是 StratifiedKFold(10,shuffle=True)+train_test_split(test_size=.111)，未按 patient 分组。
36. 同一患者不同 session 有泄漏风险：本地 ADNI_2 fold0 train/val、train/test、val/test
    subject overlap 分别 12、10、4。原训练每 epoch 查看两次 test，且保存／报告最后 epoch。
    Pro 按 subject 分组，只用 validation 选 checkpoint，test 只最终评估。
37. DGL batch 节点是 subject-contiguous：0..115、116..231 等，边为独立 block；Pro 用 `[B,N,N]`。
38. 原 population graph 每个 mini-batch 单独构建，验证／测试 loader 也如此，不是全 dataset 图。

Paper vs repository（不能据论文替换工程行为）：

| Item | Paper implementation | Repository implementation |
|---|---|---|
| ROI Eq.(2) | Prompt in each layer's input | Once before all GCN layers |
| Readout | Sum | ADNI mean |
| Eq.(3) similarity | Cosine | Sigmoid of cosine |
| Population threshold | Validation-selected th | Hardcoded 0.8 in G, 0.7 in C |
| Predictor | Two-layer MLP | Two hidden layers plus output Linear |
| Individual graph | Sparsified FC | Signed quantile then absolute threshold; unweighted GraphConv |
| Full ADNI size | 1326 in paper table | 1327 supplied subject embeddings; full graphs unavailable locally |
| Ablation prose | Claims each isolated prompt helps | Disease-only ADNI table is 61.95 vs 62.87 baseline |

额外运行问题：未经本地适配的 upstream main 传 `spatial` 给不接受该参数的 loader；
upstream train pipeline 二次加载时未传 edge_ratio；官方 BrainPrompt-C 引用了未定义 hg，
且 functional dropout 未绑定 training flag。这些均不在原目录修复。
工程 baseline 使用不变的官方 BrainPromptG model，现有本地 loader 作为数据接入参考；
Pro 的原模型 adapter 用同一安全划分运行实际 original class，训练协议的改进单独说明。


数据范围更正：正式实验目标是 ADNI_1～ADNI_7 全部七批。本文关于 ADNI_2 的对齐和泄漏实测仅适用于该子集；不能据此认定完整论文 cohort 已接入。默认配置现为完整 ADNI，详细现状见 [FULL_ADNI_SCOPE.md](FULL_ADNI_SCOPE.md)。
