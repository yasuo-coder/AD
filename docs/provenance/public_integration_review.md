# 公开接入独立审查（2026-10-10）

结论：本轮暂存内容未发现需阻止公开提交的隐私或导入结构问题。审查以 Git index blob 为准，不仅依赖 `.gitignore`；这不等同于对不可识别秘密的绝对保证。没有读取最终测试预测、创建冻结锁、执行训练或修改原服务器工程。

## 范围与证据

- 已读适用 `AGENTS.md`、CONTEXT、STATE、ROADMAP、DECISIONS、EXPERIMENTS、数据/架构/复现协议和接入计划。
- 首轮审查236个索引文件、约1.26 MB，包含226项变更；修复后复核238个索引文件（不计本报告）。无二进制、符号链接、超过300 KB的单文件、权重或数据工件。
- 对索引内容检查ADNI受试者格式、紧凑subject/BIDS格式、Image ID、常见token/密钥/带凭据URL。仅测试文件命中合成受试者格式；已逐处确认是明确声明的synthetic fixtures，未发现真实患者记录。作者公开联系信息保留在上游README；代码中的矩阵乘法不是邮箱信息。
- 8个CSV逐一检查表头和行单位，均为模型/seed/消融或按数据来源汇总，不含扫描级或受试者级记录。JSON汇总与公开源码未包含预测概率或split索引明细。
- `source_import_manifest.json`全部177项的source SHA与原文件、imported SHA与index blob一致；未将接入提交倒填为历史训练版本。`baseline_import_comparison.json`保留原基线commit及逐文件比较。

## 接口与私有例外

- 包布局保留 `BrainPromptPro/{nets,layers,data,cli.py}`，无`BrainPromptPro/BrainPromptPro/`。两个旧CLI保留`--split`；新研究runner维持固定holdout语义。server_research脚本的原服务器绝对路径在复现文档中明确标注，本轮不假称跨机器直接可运行。
- 22项合成回归和CLI help由接入主任务执行，结果见`integration_validation.json`；本独立审查未重复完整回归。
- 已复核`data/private_audit.py`：私有JSON逐字节SHA256固定、三项例外基数固定2/1/1、未知key拒绝、非空字符串/重复值检查均保留。没有“任意自填例外列表即可通过”的无约束路径。
- 仅在服务器读取私有JSON并比较，不输出患者标识：固定SHA与实际JSON一致；三个原prepare脚本的missing/replacement集合完全相等；原ADNI4人口学例外相等。9项布尔校验全部通过。私有JSON不在index。
- 原合成数据构建脚本仍保留原有metadata来源约定；公开接入没有顺带更改历史标签、划分或性别冲突处理协议。

## 已修复问题

15个失效相对Markdown链接已修复，复核全部已追踪Markdown的相对文件链接为0个失效：

- 历史Pro README链接改为`docs/historical/`内现有文件名。
- 未公开的历史报告、日志与候选契约JSON改为明确的服务器工件路径，不伪装为GitHub可点击文件。
- 候选规格的registry/protocol指向公开冻结配置镜像。
- 导入纯技术`BrainPrompt/docs/BASELINES.md`，补充历史来源与验证范围说明；其中46项旧测试并非本轮重跑结果。人口学CSV为接口示例，无实际患者标识。未导入其历史日志。
- 对修改后的导入文档刷新imported hash，原source hash不变；补充BASELINES来源记录及基线比较记录。

## 保留的边界

`git diff --cached --check`仍报告原样源码历史空白和匿名CSV的CRLF；未为格式统一修改训练快照。原BrainPrompt未找到根LICENSE，已在UPSTREAM中如实披露，第三方已知许可保留，未擅自声明新许可。公开clone缺少受限数据、prompt/LLM工件、权重及历史split；历史冻结实验继续在原工程执行。此审查不证明实际模型性能、最终Top-3资格或最终测试完成。
