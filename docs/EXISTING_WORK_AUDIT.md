# 现有工作审计与证据索引
核对：2026-10-10T08:37:13+08:00。仅检查用户项目目录、指定仓库及相关进程/容器；未扫描全服务器文件。公开记录仅汇总，不含样本明细。

## 来源和接入
目标https://github.com/yasuo-coder/AD-，main起点ed766de。官方BrainPrompt本地main c85304a，跟踪AngusMonroe/BrainPrompt但含未提交修改；原Pro无.git。旧`/data2/panyao_data/ADNI/BrainPromptPro`是过时镜像，不作运行来源。逐文件source/imported SHA：[导入清单](provenance/source_import_manifest.json)。
原模型/旧checkpoint/日志保留；必要baseline源码置根BrainPrompt，Pro源码置根BrainPromptPro，无第二层同名嵌套。保留原始nets/layers而非迁入空models/training。未导入数据目录、prompt权重/患者文本、索引或LLM缓存。
baseline来源和许可见BrainPrompt/UPSTREAM.md；未发现上游根LICENSE，不能虚构许可；既有额外baseline的license原文保留。

## 六类状态
| 状态 | 事项 | 可定位证据/下一步 |
|---|---|---|
| 已完成且有验证证据 | 数据审计/标签/固定split、阶段1基线、前期候选契约 | R/audits/data、protocol/config、R/verification；S/verification/candidate_contract_tests.json、m3v3_contract_tests.json |
| 已完成训练、汇审中 | 72项候选/对照/组合 | S/experiments各result，stage2正式72/72独立核验PASSED；额外6个实际smoke通过 |
| 已实现但尚未验证 | 本次公共副本真实数据端到端、最终test路径产物 | 未把受限数据搬到克隆；仅验证导入/CLI/合成测试，最终test待冻结 |
| 正在运行 | 接入与审计进程；模型训练0个 | last_triples_20261009_queue_status六finished/零running/零pending；不要按旧PID重启 |
| 失败/中断可诊断 | 旧stable分类塌缩；原M3V2捷径；文档代理中断；首次迁移梯度测试被只读沙箱拒绝写输出 | 已保存旧失败产物不重试架构；M3V3有明确新诊断再实现；原地代码未改；写权限回归重跑 |
| 尚未开始 | 选择冻结、正式test、最终20节结论 | plans/003；必须先独立汇审和机制审查 |
| 条件受阻 | 公共克隆不含受限数据/embedding/checkpoint；未确认独立真实acquisition site | 这是公开范围限制，不是服务器缺数据；服务器路径完整，继续原工程 |

## 数据和方法
ADNI1–5聚合998FC/685人/216多访视，训练798/val100/test100；MCI→LMCI，五类不变。每批图样本数和AD占比见数据协议。官方BP-G、旧legacy、历史stable、M1–M5和M3V3均实际存在；参数/梯度/原理见架构与排重文件。Source_Dataset不进入模型。原始CSV与label冲突取统一来源，未按预测修正。
已核查checkpoint、history、result、source归档、confusion、PCA/tSNE与诊断输出存在；未上传受试者行。已有BrainPromptPro_Optimization_Report.md是阶段性草稿，最终test前不能当最终结论；旧快照不删除。

## 运行和恢复
最后主队列历史PID210880，子进程210882–210887，六项已自然完成；对应日志S/queue_logs/last_triples_20261009_*.log及master.log。本轮不重复启动。docker ps当前为空。GPU上有其他任务，未中断、不认定为本实验。恢复详见REPRODUCIBILITY，partial checkpoint没有直接resume接口。

## 尚存科学限制
验证/测试各AD4例、小样本不平衡、同人关联、source/site高度关联、ADNI1–4历史暴露；三seed不是三份独立患者样本。M1 gate近常数、M3V3学习关系未稳定优于固定关系。M1_M2_M5虽验证五均值提升，M2平均消融贡献仍负；不先声称三创新成功或宏指标60%。

## 2026-10-10 接续核验
以上表格为五批接入时历史快照。现接入main已发布38c72b8，五批两个selection_lock已创建，test仍未运行。独立七批研究已完成1323FC/895人数据版本、模块实现及基础验证；详见STATE、计划004和docs/research/adni_1to7。原工程与历史锁不改。当前无七批正式结果，8份smoke和恢复测试证据保存在受控N目录，不能混入正式指标。
