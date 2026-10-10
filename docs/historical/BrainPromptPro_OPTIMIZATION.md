> 历史快照，非本轮当前协议。原文的相对路径以服务器原工程为基准；当前状态以根目录 STATE.md 为准。已去除扫描级标识。

# ADNI_1–3：BrainPromptPro 优化入口

本轮十折验证已完成：验证选型后的 macro F1/Precision/Recall 略升，但 Accuracy 和 ROC-AUC 下降，未实现整体提升。旧 Pro 保持默认；`stable_v1` 是保留的实验配置。详见下方结果报告。

本轮使用已核验的 598 张扫描、398 位受试者，标签顺序 CN/AD/SMC/EMCI/LMCI，MCI 统一并入 LMCI。它与原 README 中的早期 ADNI_2 mini-training 是不同实验。

原版 BrainPrompt 源码不变。原 Pro 结构仍是默认 `legacy` 配置；新增 `net_params.stability_profile="stable_v1"` 启用优化结构。全部历史模型和结果保留，新结果写入 `outputs/optimizations/`。

新增实现：

- `layers/stable_graphs.py`：LayerNorm GCN、连续 FC 先验与 FC 残差、带残差的疾病引导读出、带软元数据先验的多关系图。
- `nets/stable_brainprompt_pro_net.py`：三个模块的整合、归一化和可学习分支门控。
- `optimization_cv.py`：训练频率类别加权、label smoothing、末轮训练、逐折验证选择和选择锁定后的测试。
- `tests/test_stable_profile.py`：分类梯度、图约束、无 BN 批统计依赖、权重计算及小样本学习。

## 运行

在 `/home/panyao/AD` 中，使用 `/home/panyao/miniconda3/envs/brainprompt/bin/python` 或激活 `brainprompt` 环境。为新实验指定未使用过的 RUN_ID：

```bash
python -m BrainPromptPro.optimization_cv prepare --run-id NEW_RUN_ID
python -m BrainPromptPro.optimization_cv train --run-id NEW_RUN_ID --candidate stable_weighted --fold 0 --smoke
python BrainPromptPro/scripts/run_optimization_queue.py --run-id NEW_RUN_ID
CUDA_VISIBLE_DEVICES=5 python -m BrainPromptPro.optimization_cv legacy-validation --run-id NEW_RUN_ID
python -m BrainPromptPro.optimization_cv select --run-id NEW_RUN_ID
CUDA_VISIBLE_DEVICES=5 python -m BrainPromptPro.optimization_cv evaluate --run-id NEW_RUN_ID
python BrainPromptPro/scripts/report_pro_optimization.py --run-id NEW_RUN_ID
```

队列使用 GPU 5、6、7：GPU 5 最多两个进程，GPU 6/7 各一个，运行前应检查设备占用。`--smoke` 只运行三轮并写入独立 smoke 目录，正式训练不会继承它的权重。

当前入口针对已经核验的 ADNI_1–3 数据及已冻结的官方扫描十折索引。`prepare` 从既有参考实验配置生成优化配置；它不是任意数据集的自动接入工具。

`stable` 和 `stable_weighted` 共享结构、初始化种子和训练参数，区别是后者用该折训练标签计算平方根倒数类别权重。每折从这两个候选及原 Pro 的最后一轮保留对照中按验证 macro F1、再按验证 AUC 选型。所有折选择保存到 `selection.json` 后才进行新的测试评估。

两候选均最多训练 1000 轮；Adam lr=5e-5、weight_decay=0.005、batch=20；验证损失调度 factor=0.5、patience=25，lr<1e-6 或每折 6 小时停止。最终使用实际最后一轮，绝不载入某个更早的 best checkpoint。

`configs/adni123/BrainPromptPro_adni123_stable_v1.json` 记录优化结构及基础参数。不要用原 `cli.py` 重现本轮：原 CLI 仍采用自己的 best-checkpoint 流程，也不处理本轮的验证选型和加权/平滑目标。

## 结果与解释范围

- 此前两模型官方训练规则十折结果（服务器历史产物：`/home/panyao/AD/BrainPromptPro/outputs/comparisons/adni123_official_scan10_last_seed41_20261005_v2/REPORT.md`，不在公开仓库）
- 本轮优化结果与详细原因（服务器历史产物：`/home/panyao/AD/BrainPromptPro/outputs/optimizations/adni123_pro_stable_v1_20261006/REPORT.md`，不在公开仓库）
- 预先固定的候选与选择方案（服务器历史产物：`/home/panyao/AD/BrainPromptPro/outputs/optimizations/adni123_pro_stable_v1_20261006/plan.json`，不在公开仓库）
- 旧 Pro 源码/配置归档：`outputs/optimizations/adni123_pro_stable_v1_20261006/source_before/`。

旧配置和 checkpoint 保持兼容。BrainPrompt 原模型源码未修改；旧官方流程报告生成器会拒绝使用已经变化的当前源码冒充旧版本，旧版本以归档源码及原实验哈希为准。

本轮保留官方扫描划分以便对照，仍有同一受试者随访跨集合的问题。此前已查看过该数据的测试结果，本轮属于探索性优化，不能当作新的独立受试者泛化或临床验证。固定评估 batch_size=20 和测试顺序，因为人群图仍在 batch 内工作。
