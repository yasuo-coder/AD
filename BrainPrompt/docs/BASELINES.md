# Table 2 基线接入说明

> 历史技术文档快照：以下目录、命令及46项测试结论来自原服务器基线接入记录，不是本次BrainPromptPro优化的运行结果。本轮未重新执行该46项测试；原运行根为`/home/panyao/AD/BrainPrompt`，未复制的脚本、测试和日志仍在该目录。人口学CSV片段为接口示例，不是ADNI记录。

本次只修改 BrainPrompt，不修改 BrainPrompt++。保留现有自定义 ADNI 数据路径、数据构建脚本和 `--split` 单折入口。

这里提供七个模型的统一训练接口及配置。新增模型是对公开实现/论文的适配，不代表已经复现论文 Table 2 的数值。尤其 fTSPL 是论文重建，LG-GNN 的评估协议有明确变化；不能把这些实现未经验证的结果标为原作者复现结果。配置中的学习率、数据划分、图稀疏度需要在训练/验证集上确认。

## 模型与入口

| 配置中的 model | 实现 | 内容 |
|---|---|---|
| GCN | `nets/gcn_net.py` | 保留原 GCN，兼容第四个参数；修正读出层输入维度 |
| BrainNetCNN | `nets/brain_net_cnn.py` | 保留 E2E/E2N/N2G；兼容第四个参数，eval 时关闭 dropout |
| BrainGNN | `nets/braingnn_net.py` | ROI 条件卷积、两级 TopK pooling、多层读出、pooling 正则 |
| BNT | `nets/bnt_net.py` | Transformer、正交聚类池化、分类器 |
| ContrastPool | `nets/contrastpool_net.py` | 训练集类别对比参考图、注意力汇总、可学习池化、GraphSAGE |
| LG-GNN | `nets/lg_gnn_net.py` | 局部 SABP/互信息、人口学边编码、全局 Chebyshev 卷积 |
| fTSPL | `nets/ftspl_net.py` | ABTG 文本、冻结 BiomedCLIP、深层双模态 prompt、PMFCC、GCN |

统一调用 `model(g, h, e, batch_llms=None)`，返回 `[batch, n_classes]` logits；训练调用 `model.loss(logits, labels)`。这些基线不使用原 BrainPrompt 的疾病/受试者 LLM 向量进行分类。现有训练入口仍加载其配套 meta 文件以保持数据接口兼容，因此原数据和 meta 文件仍需存在。fTSPL 使用自己从时间序列生成的 ABTG 文本。

新增配置覆盖 `adni_AAL116`、`abide_full_AAL116`、`adni_custom_AAL116`。前两个数据集原有 GCN/BrainNetCNN 配置继续使用；自定义数据集新增这两个配置。新增模型注册在 `nets/load_net.py`。

## 运行

在仓库根目录、已安装原项目依赖的环境中执行，例如：

```bash
cd /home/panyao/AD/BrainPrompt
source /home/panyao/AD/.venv-brainprompt/bin/activate
bash run_table2_baseline.sh GCN adni_custom_AAL116 --split 0 --epochs 1
bash run_table2_baseline.sh BrainNetCNN adni_custom_AAL116 --split 0
bash run_table2_baseline.sh BrainGNN adni_custom_AAL116 --split 0
bash run_table2_baseline.sh BNT adni_custom_AAL116 --split 0
bash run_table2_baseline.sh ContrastPool adni_custom_AAL116 --split 0
```

省略 `--split` 使用现有十折流程；省略数据集默认自定义 ADNI。脚本默认 Pearson FC 节点特征、`edge_ratio=0.2`，额外 CLI 参数可覆盖。输出在 `out/baselines/<模型>/<数据集>/`。CPU 环境会自动回退到 CPU；真实 fTSPL 需要较多内存，建议使用已有 CUDA 环境。

### LG-GNN 的人口学数据

不能用 LLM embedding 代替年龄、性别、站点。准备 CSV：

```csv
graph_index,age,sex,site
0,72,M,ADNI
1,68,F,ADNI
```

`graph_index` 必须对应原始 `.bin` 中从零开始的图顺序，且覆盖全部图、无重复。性别支持 M/F、male/female、1/0（1 表示男），年龄需为有限数值。站点作为类别相等关系使用，不把站点编号当连续数值。人口学编码器使用年龄、性别；均值与方差仅用当前训练折拟合。不要直接假定任意 manifest 行顺序与图顺序一致。

```bash
bash run_table2_baseline.sh LG-GNN adni_custom_AAL116 --split 0 \
  --baseline_metadata_csv /absolute/path/demographics.csv
```

默认 `lg_population_scope=split`：训练、验证、测试分别构建完整群体图。验证/测试样本之间可交换特征信息，不能视为逐样本独立推断。与原全体受试者传导式群体图协议不同。人口规模影响结果与显存；不要随意拆成小 batch，默认阈值在极小群体上可能删掉全部边。

### fTSPL 的额外依赖与预训练权重

```bash
python -m pip install -r requirements-ftspl.txt
```

需要相互匹配的 PyTorch/torchvision，以及官方 [BiomedCLIP 权重](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) 中的 `open_clip_pytorch_model.bin`。默认 tokenizer/config 来自 [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)。离线运行时，提前将后者的 config、tokenizer、词表下载到同一目录；传入 `--ftspl_tokenizer_path`。不需要另行加载该 BERT 的权重，因为 BiomedCLIP checkpoint 已包含文本编码器。

```bash
bash run_table2_baseline.sh fTSPL adni_custom_AAL116 --split 0 \
  --ftspl_pretrained_path /absolute/path/open_clip_pytorch_model.bin \
  --ftspl_tokenizer_path /absolute/path/pubmedbert_config_and_tokenizer
```

生产入口要求真实 checkpoint，严格核对 state_dict，不会缺失权重时偷偷使用随机模型。DGL 数据必须保留原始 `N_features` ROI 时间序列；加载时额外保存 `timeseries`，不从 FC 反推时间序列。ROI 顺序必须与 AAL116 名称文件一致。只训练 prompts、投影和 GCN，冻结编码器仍保留对输入/prompt 的反向传播。

## 来源及适配差异

- BrainPrompt：<https://github.com/AngusMonroe/BrainPrompt>，核对版本 `c85304a023636a2701d465e4cb0acb13b637127e`。
- BrainGNN：<https://github.com/xxlya/BrainGNN_Pytorch>，`1e337e7a13af5e4374343fda81ce11b62c5ce566`。根据结构独立编写密集 PyTorch 实现，避免引入旧 PyG 依赖；保留 ROI 身份。正则使用池化前完整分数，类别一致性在原 ROI 顺序上比较，不沿用上游仅选中节点再次 sigmoid 的分数。上游未发现许可证文件，未直接复制其源码。
- BNT：<https://github.com/Wayfear/BrainNetworkTransformer>，`8a588aadad0166209269fa114e5df4e42209e207`，MIT。使用 PyTorch Transformer 和聚类池化；中心全部正交归一化，冻结中心可配置。DEC KL 权重默认 0，与上游训练入口没有调用该附加损失一致。使用本项目数据与分类训练流程，未引入上游 mixup。
- ContrastPool：<https://github.com/AngusMonroe/ContrastPool>，`f3df45d5fc1573b3b6a05a5c214a9552c34a37d7`，GPL-3.0。参考图只取当前训练折，克隆为 checkpoint buffers；支持多分类。参考特征使用完整 Pearson FC，消息传递邻接使用当前稀疏图，按边索引还原，避免依赖 DGL 边排列。默认全部训练参考样本；可通过 `contrast_max_references` 限制数量（会改变方法）。保留对比图熵正则，未加入上游入口没有实际使用的附加池化损失。
- LG-GNN：<https://github.com/cnuzh/LG-GNN>，`664b70c76d00054456851bd9eaf863bb5717ff0f`，MIT。按上文使用分割隔离的群体图；学习到的边权实际参与全局卷积，互信息损失保留梯度，未沿用上游将标量损失归一化为符号的计算。使用固定配置的 SABP 比例和四级 Chebyshev 网络。属于方法适配，不是逐行复刻上游。
- fTSPL：<https://github.com/CUHK-AIM-Group/fTSPL>，`6ad09559970dea52e0c6c4b6d75f20833a16aee7`，检查时只有 README，没有运行源码。依据 [MICCAI 2024 论文](https://papers.miccai.org/miccai-2024/paper/2709_paper.pdf) 重建。未明确的选择：时间插值到 128；图像位置序列插值到 ROI 数；ABTG 的 Hc 默认 5 且排除自相关；矩形 `[N+2,N]` PMFCC 补全为对称 `[N+2,N+2]` 图，边强度取绝对值。采用深度 4、长度 2 的 prompts，CE/对称 InfoNCE 权重各 0.5。这些选择公开为配置/代码，不能保证与未发布实现一致。

已保留 BNT、ContrastPool、LG-GNN 的上游许可证，见 `third_party/licenses/`；尤其分发 ContrastPool 相关适配时需遵守其 GPL 条款。

## 评估与验证边界

七个基线的二分类 ROC-AUC 使用 softmax 概率；多分类 Precision/Recall/F1 使用 macro，ROC-AUC 使用 macro OVR。某评估 split 缺少类别时 AUC 返回 NaN，而不是虚构数值。此修正仅作用于这七个基线，BrainPrompt 模型原有指标逻辑未改动。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_table2_baselines.py
```

测试涵盖二/五分类 forward/backward、单样本 batch、eval 确定性、checkpoint、真实训练/评估函数、训练参考集隔离、群体边权梯度、ABTG、缺失输入错误，以及每个模型通过真实 `llm-main.py` 数据加载和单折训练保存的集成测试。测试使用合成图；fTSPL 使用冻结的小型编码器，另用真实 timm ViT/HF BERT 类验证 prompt 梯度与冻结行为。这不验证下载权重后的完整 BiomedCLIP 数值，也不代表实际 ADNI/ABIDE 十折性能、GPU 显存或论文准确率已通过验证。

本次验证结果：**46 passed，无跳过项**（CPU，2026-09-20）。完整输出见 服务器测试记录 `/home/panyao/AD/BrainPrompt/docs/table2_baseline_test_results.txt`（未上传）。依赖一致性 `pip check`、Python 编译检查、shell 语法及 `git diff --check` 通过。环境版本：torch 2.2.2+cpu, torchvision 0.17.2+cpu, dgl 1.1.3, numpy 1.26.4, open_clip_torch 2.23.0, timm 0.9.12, transformers 4.35.2。两条 warning 来自固定版本 Transformers 对 PyTorch 弃用接口的调用。
