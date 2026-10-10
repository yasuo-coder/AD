# 003 冻结与最终验证
当前问题：所有有限开发实验已完成，尚需独立汇审、固定最终选择并只评估一次test。
依据：R/研究协议，S/audits/Final_Selection_Rule_20261009.md；完整验证均值与消融，M1机制反事实。不得改成初始化拟议五折。

- [x] 五批固定holdout、三seed、共同训练/指标/验证checkpoint规则。
- [x] 72候选/对照/组合训练完成；最后队列6exit0，现运行0。
- [x] 正式scope experiments 72/72 PASSED；6个实际smoke也通过，all报告中18项未计划smoke不补跑。
- [x] build_ablation_evidence.py 输出四组三元的八配置、四背景平均边际和leave-one-out，两者不偷换。
- [ ] 根据验证与机制证据形成具体frozen_decision，较简候选/leading triple/三seedcheckpoint集合全部列出。
- [ ] 依次创建root/stage2 selection_lock，运行pretest preflight。
- [ ] root三基线+选定候选+leading triple完整八配置（去重）正式test，统一100例，不选bestseed。
- [ ] verify_final_test_artifacts.py 独立复算；完整指标、逐类、来源缺类AUC、预测分布、目标是否失败全部报告。
- [ ] 更新本地20节BrainPromptPro_Optimization_Report.md，公开匿名汇总及STATE/EXPERIMENTS，提交推送。

验收：锁先于test，数据/split/hash不变，所有被测模型三seed完整，未编造Top-3或60%，不看test后换配置。真实运行失败保留证据而不覆盖；当前无新训练要启动。
