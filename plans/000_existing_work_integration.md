# 000 现有研究接入
更新：2026-10-09T21:21:08+08:00。范围：原BrainPrompt/BrainPromptPro代码、审查、冻结协议、匿名指标、作业与恢复证据。保留原运行工程及用户dirty修改。

## 里程碑
- [x] 克隆最新指定仓库，阅读AGENTS、所有上下文及计划，核对初始化与真实研究差异。
- [x] 盘点实际代码/数据/结果/作业。来源/home/panyao/AD/BrainPrompt与BrainPromptPro，R/S见复现文档。
- [x] 复制必要源码/配置、记录逐文件SHA，保留baseline来源和未知许可边界；无嵌套包、無数据移动。
- [x] 公开副本扫描级例外清单外置；不提交样本/FC/权重/缓存。
- [x] 核验两CLI --split与新runner help，公开隐私/标签检查3项通过。
- [x] 22项迁移合成回归通过；CLI接口通过。
- [x] 最终引用与暂存区逐文件/敏感信息审查通过：177项来源hash一致，15处历史断链已修复；见docs/provenance/public_integration_review.md。
- [ ] 提交并推送（受保护则独立branch/PR），记录实际commit。
- [ ] 转入003：完成独立汇审、冻结、test一次、最终报告并同步。

## 验收
source manifest可追溯；原项目未被移动/覆盖，原训练6项自然完成；公开tree无受试者数据、无二层Pro；数据协议不改，接口可用，文档不再称研究未开始。执行`python -B -m tests.run_integration_suite`、两个CLI help、stage2 verifier，检查git staged diff与远程HEAD。Git版本不倒填历史实验。
