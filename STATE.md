# 当前状态
核对时间：2026-10-11T04:28:44.074324+00:00（UTC），2026-10-11。动态作业必须重新核对PID、command、日志和result。

## 活跃任务与完成边界
ADNI_1–7 / P2 Masked Connectome、P3 Task Token、P5 Reliability Population独立研究；[计划004](plans/004_adni1to7_p2p3p5.md)，分支`research/adni1to7-p2p3p5`，[草稿PR1](https://github.com/yasuo-coder/AD-/pull/1)。M0–M7实现与基础验证完成；M8/M9三seed比较正在运行；subject-disjoint基线已启动，最终选型和test未完成。

## 已核验
- 数据1323 FC/895人；CN850、AD65、SMC39、EMCI61、LMCI308。MCI→LMCI。统一经验Pearson，不与Nilearn收缩相关混用。
- 新sample-level冻结协议，split seed1729，训练seed42/3407/2026；存在同人跨划分与历史开发暴露，不能声称外部独立验证。完整协议/成员/关联审计保留N/dataset_v1。
- 20单测、CPU/CUDA八开关、真实TRAIN梯度、CPU断点恢复一致性通过；新增对照和微调入口共16份短程smoke；队列依赖等待、失败拒绝与签名检查通过。smoke不是性能证据。
- Phase1–4全部正常结束，31正式任务=21分类+10预训练；21分类的checkpoint/source/common-config/hash及私有概率指标独立复算通过。表见[探索结果](docs/research/adni_1to7/exploration_20261011/Validation_Results.md)。均为验证集选checkpoint；test未评估。
- A0三seed均值±样本SD：Accuracy57.89±6.51%、Macro-F1 35.63±1.59%、Macro-P38.28±2.03%、Macro-R/BalAcc39.00±2.95%、AUC67.71±1.45%。其他探索仅seed42。
- 同seed42：A0 F1 35.86%，A1/P2 35.31%，A2/P3 33.01%，A3/P5 35.59%；P5_fixed 38.86%，P3参数匹配37.63%、历史M2 37.32%。目前不支持三个完整模块均有效；不能用单seed宣布统计显著。
- no-dropout pair第40轮重建改善，但best/last微调F1 31.57/32.33%均低于A0；停止该预训练扩搜。[完整重建诊断](docs/research/adni_1to7/P2_Reconstruction_Diagnostics.md)保留全部十项。

## 实际运行
N=`/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010`。
Phase1–4为已完成记录，旧controller1153238/1165476/1173253/1180291已退出，不重启。
Phase5 `N/queues/phase5_three_seed`，controller PID1408894，启动2026-10-11T01:45:32Z；GPU0/3/4/5/6/7共享运行（其他作业未停止）。本次40项缺失实验，快照完成36、运行4、待启动0、failed=False。
目标A0–A7和11种关键对照各三seed，共57分类结果，其中17项已存在，复用且不覆盖；四项单seed预训练微调额外保留，不混入57项计数。
不可变源码`N/code/29f34f9`，本地训练commit29f34f9322d4b797329c8bd405fa64ea1bbbd582，公开等价tree commit251dd6c81ea44c116dfc3cdc92c349a086ef87cb；config SHA cd5a2194a5f03f5ba51a20108c03e2ca8589eb76c044e0cec28b9009fc93f0da。运行副本不可修改。
队列目录包含jobs.private.json、recipe_freeze.private.json、state.private.json、launcher.json、master.log和逐任务日志；checkpoint在N/experiments/实验名/seed种子。恢复见[RUNBOOK](docs/research/adni_1to7/RUNBOOK.md)。

## 下一步与限制
1. 继续检查Phase5，独立复算新完成结果；不重复训练已完成配置。故障只按原不可变代码/config/split/依赖/RNG恢复。
2. 已完成25项机制探针（7个方法三seed+4项seed42对照）；完整A0–A7三seed后计算四背景配对边际，不能把相关的12个差值当12位受试者。
3. 等待独立subject-disjoint A0基线，主验证矩阵完成后确定有证据的候选及补充比较；验证集决策冻结后才一次test。当前没有七批selection_lock，不运行test。
原五批12+72训练和selection_lock保留，旧test仍未运行。源码和匿名汇总可公开；身份、完整split、概率明细、权重不可公开。此前审批额度造成的文档滞后已在本次修复；新运行状态以实际队列为准。

## 本次新核验与后续作业
56份完成的主协议分类结果已独立复算（包含历史21项探索），最新[验证表](docs/research/adni_1to7/phase5_progress_20261011/Validation_Results.md)。完整矩阵仍有运行项，不补造缺失结果。25次checkpoint机制探针通过，详见[单seed](docs/research/adni_1to7/Mechanism_Diagnostics.md)与[三seed](docs/research/adni_1to7/Mechanism_ThreeSeed.md)。P5_no_reliability CPU概率差3.73e-5未通过，类别相同；原CUDA路径误差1.79e-7通过，失败记录保留，未调宽容差。
A0/A1/A2/A3已各3seed：Macro-F1均值35.63/36.24/34.34/34.80%。A1仅部分指标更好，不能宣称全面提升；P3/P5未证明比A0有效。P2 rec-only单seed表现未在三seed重复，clean Macro-F1均值33.57%。P5分支三seed残差比例约0.29%，边权接近均匀；细节见机制报告。
Phase6 `N/queues/phase6_subject_baseline`，controller1425910，2026-10-11T04:22:31Z启动，GPU1/2/5；冻结subject split、A0三seed，代码同29f34f9，独立输出N/subject_experiments。不混入主协议汇总、不用于改变主协议。
新增分析3单测及4项指标篡改拒绝检查通过；summarize_validation新增--protocol sample/subject明确分开两条协议。analyze_ablation只在24份完整A0–A7结果存在时计算配对贡献。observe_queue用于队列结束后的自动只读验证和报告，实际观察器启动情况以N/queues对应launcher与status为准。
审批额度已由用户恢复，报告接入继续进行；本次修改尚待提交时以git status为准。远程上一已核验提交251dd6c8，本次提交由后续Git历史定位，不能把未提交代码当训练版本。
