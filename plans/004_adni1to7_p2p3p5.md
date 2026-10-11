# 004 七数据集 P2/P3/P5 独立研究

建立日期：2026-10-10（Asia/Shanghai）。用户新任务以 ADNI_1–7 为独立研究；不是历史 M1–M5 重命名。分支 research/adni1to7-p2p3p5，接入基点 38c72b85f90ab0087556390d3bec97314cf10a3b。

## 当前问题与证据
历史阶段1共12次、阶段2共72次正式训练已完成核验。五批 selection_lock 和 frozen_decision 已在新任务到达前建立；最终test尚未运行。保持原工程、数据、锁、checkpoint与计划003，不在本研究中修改或复用其测试结论。
七目录均存在FC；初查ADNI6已有198份FC且预处理两例失败，ADNI7已有127份FC。这些仅为文件计数，纳入数量须通过逐扫描身份、标签、质量、ROI顺序和跨批去重审计。

## 里程碑与验收
- [x] M0 阅读上下文、Git状态，隔离新分支，核对历史锁。
- [x] M1 七批CSV/预处理/FC/标签/身份/ROI顺序审计；输出匿名汇总及受控逐扫描证据。
- [x] M2 ADNI_Combined_1to7_v1、独立80/10/10 sample-level划分、独立split seed；补充subject-disjoint划分。记录历史样本暴露和同人重叠。
- [x] M3 三篇论文/官方代码及历史M2/M3审查；明确复用、差异、排重、许可和有限参数搜索。
- [x] M4 P2对称mask、目标隔离、连续重建、双视图、TRAIN-only预训练和联合训练。
- [x] M5 P3复用M2查询计算，保留类别身份和逐类logit；参数匹配及语义对照。
- [x] M6 P5复用TRAIN参考/自排除契约；可靠性代理、连续边权；禁current-FC value捷径。
- [x] M7 三开关、统一CLI、CPU/CUDA/真实FC smoke、保存恢复一致性。
- [ ] M8 新协议原BrainPrompt、必要历史对照；同预算、seed、checkpoint规则。
- [ ] M9 单seed探索后A0–A7三seed完整消融及机制对照。不得用test选择。
- [ ] M10 验证集选型、冻结、一次test；失败方向如实保留。
- [ ] M11 技术/数据/指标/机制/失败报告。
- [ ] M12 公开审查、独立提交/分支推送、状态交接。

## 输出与恢复
代码仅位于BrainPromptPro，tests位于根tests；新增配置/运行脚本位于configs/adni_1to7和scripts/adni_1to7。
受控产物使用 /data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010，不纳入Git。数据版本与历史998FC独立；只从已有FC读取，不无故重跑fMRIPrep。任务不依赖的数据模块可先验证，数据审核通过前禁止正式训练。
长作业需真实PID、开始时间、配置、日志、checkpoint、恢复命令；先查进程和可恢复输出，禁止覆盖或重复运行。所有标签固定CN/AD/SMC/EMCI/LMCI，MCI合并LMCI。

## 2026-10-10 实际推进
数据1323FC/895人，独立数据版本和两个协议已封存。P2/P3/P5层与官方GCN整合已实现；16单测、CPU/CUDA各8组合和真实TRAIN反向测试通过。8份真实数据短程入口结果及CPU两轮断点恢复检查均已完成，正式消融未运行。详细私有协议留在服务器，公开仅总体匿名统计与hash。见docs/research/adni_1to7。

## 下一里程碑验收
M8先运行A0三seed；P2探索同时执行7个40轮TRAIN-only任务。冻结代码副本运行，保存commit/source/config/split/checkpoint hash；所有结果明确validation或TRAIN reconstruction，不访问test。P2若不优于TRAIN均值基线先诊断，不直接扩大到完整组合。M4–M6尚未勾选表示机制对照仍未完成，不表示缺少实现。

## Phase1 实际执行
初版实现已提交、推送，PR1草稿；M12首次发布完成，最终交接持续更新。controller1153238在GPU0/3/5/7运行A0三seed及P2七项预训练，产物路径见STATE。P3三个对照和P2两个基于饱和证据的数值候选已实现；20单测、CPU8组合以及新增5个CPU单轮smoke通过。下一步先核验完整phase1指标与重建对照，再固定下一批不可变代码。

Phase1的10项正式任务已完成；A0三seed独立指标核验通过。M8仍需Legacy/M2/M3对照，不能因A0完成就勾选全部基线。新一批将包含两个P2数值诊断、历史基线和P3/P5对照，先seed42；三seed最终消融和test未完成。

Phase2已启动，PID1165476，14项；截至本条两个归一化预训练完成，余项按queue运行。九项重建验证均未胜过TRAIN均值，公开负面结果表已生成。Phase3补齐P2四项分类机制比较及一个由实际Dropout差异支持的诊断候选；总范围固定，未经新证据不继续扩搜。

## 2026-10-11 当前里程碑
上文为历史日志；以STATE为最新快照。M4–M7标记完成指实现/接口/基础验证，非机制收益。Phase1–4全部结束，21分类指标独立复算与10重建探针已完成。D019保持完整登记配方并停止无效预训练扩搜。Phase5补齐40项缺失任务，完成后形成57项三seed主矩阵和关键对照；4种预训练微调仅探索保留。controller1408894，六卡，不可变29f34f9。
下一验收：57份结果的独立复核、A0–A7逐seed四背景边际贡献、机制反事实和subject-disjoint补充；收益不成立则否决模块。没有七批selection_lock，禁止提前test。源码、命令、恢复和真实状态见STATE/RUNBOOK；不修改运行副本。

2026-10-11推进：25项正式checkpoint机制探针完成（7方法三seed+4单seed），说明部分分支弱作用且未证明三项创新成功。Phase5后自动结果分析工具已实现，完整矩阵缺失则拒绝计算；3统计单测通过。Phase6补充subject-disjoint A0三seed正在GPU1/2/5运行；后续候选比较与test仍待验证集决策。STATE为动态快照，旧启动条目保留为历史。
