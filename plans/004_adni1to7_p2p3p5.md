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
- [ ] M4 P2对称mask、目标隔离、连续重建、双视图、TRAIN-only预训练和联合训练。
- [ ] M5 P3复用M2查询计算，保留类别身份和逐类logit；参数匹配及语义对照。
- [ ] M6 P5复用TRAIN参考/自排除契约；可靠性代理、连续边权；禁current-FC value捷径。
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
