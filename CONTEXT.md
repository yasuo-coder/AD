# 项目背景
更新：2026-10-11T11:27:30.526404+00:00

研究已在服务器开展，初始化文档不是实际进度。源码和匿名汇总现接入本仓库；历史运行仍在原目录，不迁移、不重训。

- 原工程：`/home/panyao/AD/BrainPrompt`（官方 c85304a 加既有本地修改）；`/home/panyao/AD/BrainPromptPro`（此前未纳入 Git）。
- 数据根：`/data2/panyao_data/ADNI`，历史五批研究使用 ADNI_1–5，FC 样本为一次 rs-fMRI acquisition；不同 visit 保留。
- 研究目录：`/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007`，候选阶段为其 `stage2_20261008`。
- 固定五类 CN/AD/SMC/EMCI/LMCI，MCI→LMCI；998 FC、685 biological subjects，固定798/100/100 holdout，seed42/3407/2026。详情见数据协议。
- 最新任务使用验证 macro-F1 选 checkpoint，与早期十折末轮协议分开；不执行初始化文档曾拟议的五折。
- 目标为五项指标整体提升，宏F1/P/R≥60%为研究目标而非已达成结果。不能强凑三个弱模块。
- BrainPromptPlus 全文及官方代码已审查，具体排重文件见 docs/research；不能照搬频谱/空间语义提示图创新。
- Source_Dataset 仅追踪；site 为历史metadata字段，未独立确认真实采集站点来源。
- 允许同人跨 split 关联，必须报告46/38/6共享受试者；ADNI_1–4有历史探索暴露，不是新外部验证队列。

当前活跃任务是文末及STATE所列七批独立研究；以上为五批历史背景。

下一步始终读取 STATE、EXPERIMENTS、活跃计划和实际队列；禁止重复启动已完成配置。公开仓库无受试者明细、FC、原始影像、权重、LLM模型或患者prompt。

2026-10-10：用户新增独立ADNI1–7/P2P3P5研究，活跃计划004。新数据1323FC/895人；历史998FC不代表新研究规模，两研究不能混合比较。

2026-10-11：七批研究完整主矩阵、subject A0/A1、63项一次冻结TEST及逐预测复算已完成。主要候选A1由VAL选择；主TEST A0/A1/A7 Macro-F1=28.33/28.25/27.84%。不支持三个有效模块或全面改善；本轮TEST已用，不再选型/调参。最终报告见docs/research/adni_1to7/Final_Report.md；STATE记录真实作业与证据。
