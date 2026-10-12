# 005 BrainPrompt 七批扫描级十折、最后一轮

2026-10-12（Asia/Shanghai）。用户要求输出七批合并输入的BrainPrompt十折结果，并明确选择官方流程停止训练的最后一轮。现有七批研究只有holdout，不能冒充十折；新增独立描述性复现实验，保留004结果/锁，不做模型选型。

- [x] 核实没有同配置运行/完成结果；用户确认last规则。
- [ ] 冻结原1323FC输入、新十折索引、配置、来源和历史锁哈希；核验每扫描恰测试一次与同人关联。
- [ ] 现有官方协议测试及仅TRAIN反向smoke通过；不可变源码独立作业。
- [ ] BrainPrompt十折从头训练；每折磁盘last.pt重新加载后评估。
- [ ] 独立复算逐折五项指标、Balanced Accuracy、每类Recall与OOF；重放scheduler/首次停止规则，核验checkpoint/source/data/split。
- [ ] 匿名表与交接同步GitHub，不上传受试者身份、划分、权重或逐例预测。

复用BrainPromptPro.merged_cv.worker/official_cv.epoch_pass，无模型/优化代码修改；新cv10_baseline只提供七批prepare、来源核验与队列。扫描分层十折seed41，剩余数据按.111分验证；各折同seed41独立初始化与loader RNG。原baseline hidden116/L4/dropout.5，batch20、Adam5e-5/wd.005、两个无类别权重CE，无clip/平滑。最多1000轮、每折6小时、val-loss ReduceLROnPlateau(.5,25)，lr严格<1e-6停止，不恢复best。按旧官方worker每轮两次test监测，测试不参与任何决策。十折等权mean±SD(ddof0)，OOF另表，不混用近期三seed SD。

数据仍为ADNI_Combined_1to7_v1，1323FC/895人、CN850/AD65/SMC39/EMCI61/LMCI308，MCI→LMCI，Pearson。新十折存在同人关联，也全部属于历史开发过的数据；不能声称新外部验证。只是用户指定的固定baseline复现，不根据测试分数优化或筛选模块。

新受控根C=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012；源码C/code/<commit>，实际输出其BrainPromptPro/outputs/comparisons/adni1to7_brainprompt_cv10_last_seed41，通过C/results链接定位。单折入口cv10_baseline worker --run-id ... --split 0..9；队列launcher/日志记录真实PID。旧worker没有完整RNG恢复，不宣称任意epoch精确续训；若中断保留partial并审查，不能盲目覆盖或用其他折补齐。
