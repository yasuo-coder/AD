# 005 十折末轮证据与复现

核验时间UTC：2026-10-12T11:22:05.566133+00:00。十折全部成功，controller2830424已完成分派，原始训练快照与旧004锁不变。本任务完成，无需重复训练。

## 版本和输入

- 实际训练本地commit：8e5d8c35810bc6f8e5d4bfdffe2b54cf61a8c1c6；公开等树归档commit：[f628bcda06d9abe7bcb62351aa988e934ae283ef](https://github.com/yasuo-coder/AD-/commit/f628bcda06d9abe7bcb62351aa988e934ae283ef)。两者tree=d0fdfb4fd333938bd158c4d556b8b2a93456d07f，API身份不同不代表另一份训练代码。
- 训练代码C/code/8e5d8c3；C=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012，完整来源C/source_version.json及C/results/source_sha256.json、additional_source_sha256.json。baseline_root=/home/panyao/AD/BrainPrompt，具体原始来源见仓库BrainPrompt/UPSTREAM.md；文件哈希在封存清单核验。
- 输入ADNI_Combined_1to7_v1，1323 FC/895人；CN850/AD65/SMC39/EMCI61/LMCI308，MCI→LMCI。AD65为FC样本数，非患者人数。输入SHA=217b662f09aa3c359b71b1d49cb51e1b77a37334c67f59663bed5aae5c6bd273。
- 新十折split SHA=f44a16439d0d5f651532ca571673b4588ecdf1f5874fbe64a4b09467ff5aa26a；seed41扫描分层十折，验证比例.111。TRAIN 1057/1058，VAL133，TEST133/132；每扫描恰测试一次。完整成员与身份不公开。

## 已执行命令与产物

[完整命令及恢复限制](../../REPRODUCIBILITY.md)。正式queue从C/code/8e5d8c3执行，--run-id adni1to7_brainprompt_cv10_last_seed41 --gpus 0,3,4,5,6,7。C/launcher.json、C/master.log保存启动命令与记录；C/results/queue_status.json为10/10退出0，无失败。逐折BrainPrompt/fold_0..9的history.jsonl、last.pt、result.json、predictions.csv和worker_logs在C/results。未经审查不能删除目录或重跑已完成折；旧worker不具备任意epoch完整RNG恢复功能。

只读报告直接运行仓库BrainPromptPro/scripts/adni_1to7/report_cv10_baseline.py --study-root C --output C/reports/verified_cv10_20261012 --device cuda:0；代码导入自动定位封存8e5d8c3。报告SHA见匿名JSON。报告目录已存在时用新目录，不覆盖。

## 核验与结果

3项官方协议契约、40TRAIN扫描反向检查通过（32参数张量更新、loss有限）。报告合成数据检查覆盖完美分类、不均衡全CN闭式指标及无效概率拒绝。正式10/10独立核验：源码/数据/旧锁/划分哈希、TEST成员标签顺序、仅TRAIN编码器、scheduler每步及首次停止、实际last.pt哈希、概率指标与混淆矩阵、OOF一次覆盖。十折CUDA重载预测均通过，最大概率差=2.384185791e-07。

停止轮数183–307；停止原因集合=['learning_rate_below_minimum']。平均±population SD(ddof0)：Accuracy=63.41±3.66%；Macro-F1=24.74±4.72%；Macro-Precision=25.19±6.38%；Macro-Recall/Balanced Accuracy=25.46±3.87%；ROC-AUC=70.87±3.23%。OOF逐类Recall：CN 86.82%，AD 4.62%，SMC 5.13%，EMCI 0.00%，LMCI 31.17%。全CN折数0/10，单类塌缩折数0/10。

[逐折结果](CV10_Test_Results.md)与[匿名JSON](cv10_summary.public.json)仅为聚合数值。原始概率/身份/混淆计数/划分/权重保留服务器，验证证据在C/reports/verified_cv10_20261012/verification.private.json与oof.private.pt，不能上传。

## 解释边界

宏P/R/F1固定五类，AUC为概率macro OvR；Balanced Accuracy等于macro Recall。十折均值与合并OOF不同，SD不是患者置信区间。扫描级允许同人关联；624/1323个OOF测试扫描的受试者也在对应TRAIN。这是此前开发数据上的固定基线描述性复现，不是新外部验证或未见受试者验证。原004 holdout/三seed/验证选checkpoint与本次十折末轮分开；不据本次TEST选优化模块、超参数或checkpoint。没有因结果重选方法或重新训练。

源码/报告在[PR2](https://github.com/yasuo-coder/AD-/pull/2)，叠加PR1研究分支；保留main共享discussion.md。发布版本与完整文件树以STATE/Git历史和远程核验为准，不预写当前提交SHA。下一步审阅结果；如开展新优化，先另行定义不使用本次TEST选型的研究范围。
