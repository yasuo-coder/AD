> 历史快照，非本轮当前协议。原文的相对路径以服务器原工程为基准；当前状态以根目录 STATE.md 为准。已去除扫描级标识。

# ADNI_1～4：三种固定方法统一十折

**已完成 30/30 折。新版在统一训练配置下十折均预测 CN；Accuracy 最高，但另外四项指标均低于 BrainPrompt 和旧 Pro，不支持整体性能提升。** 完整结果与核验（服务器历史产物：`/home/panyao/AD/BrainPromptPro/outputs/comparisons/adni1234_three_models_official_last_seed41_20261006/REPORT.md`，不在公开仓库）。

本入口按用户要求将四批数据合并成一个输入，运行 BrainPrompt、旧 Pro、固定新版 Pro；并不将四批分别训练，也不使用此前优化流程的逐折回退。

结果目录：`outputs/comparisons/adni1234_three_models_official_last_seed41_20261006/`。训练进度看 `queue_status.json`，最终结果看 `REPORT.md`。在报告完成前不得将部分折的数字当成完整十折结果。

输入为 798 张功能扫描、557 位受试者，AD 26 张、非 AD 772 张；CN/AD/SMC/EMCI/LMCI 顺序固定为 `[494, 26, 39, 42, 197]`。两张缺少原影像可靠预处理映射的 ADNI_2 扫描不纳入。MCI 统一并入 LMCI。T1 不另外计为分类扫描。

三模型共同训练参数：Adam；lr=5e-5；weight_decay=.005；batch=20；dropout=.5；疾病辅助损失系数1；脑图/群体图稀疏系数1e-4；不加权、不平滑、不裁剪。最多1000轮；验证batch均值总损失调度 factor=.5、patience=25；严格 lr<1e-6 或6小时/折停止。最终读取 last.pt，无best模型或按测试选择。

C固定使用 stable_v1（hidden=64、L=2、LayerNorm 等）；A/B 使用原有 hidden=116、L=4。网络结构属于实验对象差异，训练设置保持相同。C没有继承之前专用的类别加权、dropout=.25、辅助系数=.2或逐折选择流程。

扫描级官方十折会保留随访跨集合现象；本次360张测试扫描的受试者也出现在该折训练集中。全体模型使用相同扫描索引，metadata encoder仅从训练集拟合；训练shuffle使用独立seed41生成器以排除模型初始化消耗随机数造成的组批差异。

复运行（先检查GPU当前负载；使用全新的RUN_ID，不覆盖历史结果）：

```bash
cd /home/panyao/AD
PY=/home/panyao/miniconda3/envs/brainprompt/bin/python
# 已保存并由SHA256固定的输入可直接复用。
$PY -m BrainPromptPro.merged_cv prepare --run-id NEW_RUN_ID
CUDA_VISIBLE_DEVICES=5 $PY -m BrainPromptPro.merged_cv worker --run-id NEW_RUN_ID --model BrainPromptPro_new --fold 0 --smoke
$PY BrainPromptPro/scripts/run_merged_cv_queue.py --run-id NEW_RUN_ID --gpus 5 6 7 --jobs-per-gpu 2
$PY BrainPromptPro/scripts/report_merged_cv.py --run-id NEW_RUN_ID
```

需要从原始预处理文件重新构建输入时，可参考 `scripts/prepare_adni1234.py`；该入口有固定队列审计断言且拒绝覆盖已存在输入。请为新的数据版本显式更改输出版本、重新检查覆盖与来源冲突，不要删除旧数据以强行重建。

正式队列中三种模型都从全新初始化开始，`smoke/` 权重不参与正式实验。所有三种方法都监测测试但不据此停止或选择，以匹配官方外层流程；用于调度的唯一分数是验证损失。

最终报告独立复算逐折指标，检查每张输入恰好一次OOF覆盖、训练配置和元数据一致、checkpoint为最后一轮，重放所有验证损失调度和首次停止条件。均值标准差、每类指标、来源分层、配对差值、历史观察性对照和图表均随报告保存。
