# 当前状态
核对时间：2026-10-10T08:37:13+08:00

## 当前任务
先将真实服务器研究接入公开仓库，再完成已有限定范围的最终选型、冻结、测试与报告。活跃计划：[000 接入](plans/000_existing_work_integration.md)，后续 [003 最终验证](plans/003_experiment_validation.md)。

## 已核验事实
- 克隆默认分支 main，接入起点 `ed766de4749a66449948a08cbda9730c66032316`；原 BrainPrompt dirty 修改保留，原 Pro 无Git历史。
- 阶段1有12次正式训练（共同加权9次、未加权探索3次），阶段2有72次正式训练（24配置×3seed），另6次两轮smoke。数值只从 result.json 汇总。
- 最后队列 `last_triples_20261009` 六项全部退出0，运行中0、待启动0；队列主进程历史PID210880（不可仅凭旧PID判断活性）。未终止其他GPU作业，审计时 docker ps 无运行容器。
- 阶段2正式实验72/72独立核验PASSED；另6个实际smoke通过。--scope all中18项未计划smoke标pending，不是正式实验缺项。
- 公共副本保留nets/layers布局，CLI --split 和自定义路径；3项隐私/标签/布局检查通过；两项目CLI help与V3 runner help通过。合成图/梯度等22项回归全部通过（见docs/provenance/integration_validation.json）。
- 最后M1_M2_M5验证五项均值超过主BP；但M2四背景平均F1边际为负，不满足三个独立有效模块的预定标准。不能据此宣称Top-3已达成。

## 下一步
1. 公开暂存区审查、22项合成回归与文档引用核验已完成，提交并推送本次接入。
2. 72正式结果独立核验与完整消融/M1机制审查已完成；按已固定Final_Selection_Rule冻结M1保守配置和M1_M2_M5展示消融组，才创建两个selection_lock。
3. 使用冻结checkpoint对同一100例test各seed评估一次，核验导出、更新最终报告和仓库。当前没有最终test结果，勿填写目标值。

## 阻碍与边界
没有数据/GPU路径阻碍。公开克隆不包含受限数据/权重/历史split，直接在新机器训练需要合法恢复私有工件；冻结历史工作继续从原工程执行。没有实现部分epoch的自动恢复，队列仅跳过已完成且哈希通过的项，遇到未完成输出会拒绝覆盖。

本文件不预写本次提交SHA；接入提交以Git历史为准。历史实验不能归属于接入后提交。详细证据：[现有工作审计](docs/EXISTING_WORK_AUDIT.md)。
