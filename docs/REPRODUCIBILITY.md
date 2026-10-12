# 复现与恢复
服务器Python：`/home/panyao/miniconda3/envs/brainprompt/bin/python`；Python3.10.21、torch2.4.0+cu121、DGL2.4.0+cu121、CUDA12.1、numpy1.26.4、scipy1.15.3、sklearn1.7.2、pandas2.3.3、transformers4.44.2。GPU为RTX4090；共享负载使耗时不适合作严格速度benchmark。

## 公开副本验证
在仓库根执行 `PYTHONDONTWRITEBYTECODE=1 python -B -m tests.run_integration_suite`（合成数据）；`python BrainPromptPro/llm-main.py --help`；`cd BrainPrompt && python llm-main.py --help`；`python -m BrainPromptPro.candidate_v3_initialized_runner --help`。不包含全量数据依赖测试成功的承诺。
两旧CLI保留`--config`和`--split`，baseline --split为0–9。新研究runner使用固定holdout，不接受--split；不要把历史CV接口强加新协议。

`BrainPromptPro/scripts/server_research/`明确保留原服务器绝对路径，是冻结历史运行的审计/恢复脚本，不是跨机器免配置入口。

## 当前实验命令与运行位置
工作目录`/home/panyao/AD`，始终用上述venv Python和原BrainPromptPro。R=`/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007`；S=R/stage2_20261008。
已执行：`python -m BrainPromptPro.candidate_v3_initialized_runner train --model MODEL --seed SEED --device cuda:0`；种子42/3407/2026，共同sqrt_inverse_frequency profile。单GPU由CUDA_VISIBLE_DEVICES映射；队列为S/audit_scripts/run_research_queue_v2.py，独立Popen start_new_session启动，状态/日志保存在S。最后phase last_triples_20261009已完成，不再执行train。
复核：`python "$S/audit_scripts/verify_stage2_development_v2.py" --scope all`；`python "$S/audit_scripts/summarize_stage2.py"`；`python "$S/audit_scripts/build_ablation_evidence.py"`。冻结后开发验证器按设计拒绝继续（防test污染），使用冻结前验证报告。
最终test需先按Final_Selection_Rule建立两个selection_lock，具体路径和checkpoint SHA固定；官方/旧Pro用research_runner test，V2用candidate_runner test，V3用candidate_v3_initialized_runner test；每项只允许导出一次。未冻结不运行test。导出后verify_final_test_artifacts.py独立复算。

## 协议
Adam lr5e-5 wd.005，batch20 dropout.5 clip5；400max、至少120且100轮验证选型无改善或LR<1e-6停止，LR loss patience25 factor.5。训练计数^-0.5均值归一化用于两个CE，辅助权重1、图项1e-4。checkpoint主验证macroF1，平局固定综合分再低loss。三seed报告mean±sample SD，不挑best seed；ROC-AUC=概率macro OvR，来源子集缺类则AUC缺失，P/R/F1仍固定五类macro。

## 路径与版本
新克隆提供源码而非复制受控产物。历史config中的路径与@pro保持原义；跨机器需在私有local配置映射data、baseline_root、prompt、bundle，恢复经哈希核验的私有split，不能直接重造划分。不要覆盖server frozen config。公开构建脚本的例外清单由BRAINPROMPT_PRIVATE_AUDIT_JSON指向服务器私有JSON，字段missing_original_images/excluded_historical_replacements/adni4_sex_conflicts；读取前与已裁决私有文件SHA256绑定，同时检查预定2/1/1基数；缺失/篡改/空/重复时拒绝。不要把它提交Git。
原Pro历史无Git版本；从每实验source_sha256、source/快照和checkpoint追溯。接入manifest列source/imported hash；接入修订不影响原训练源码。root阶段1全依赖封存为训练后补充证据，不能倒称完整训练前seal。

## 恢复限制
先读queue_status再用ps核对PID/command/starttime及result/checkpoint，不启动重复配置。queue可用新phase跳过已完成项且验证checkpoint SHA，遇到partial目录拒绝覆盖。last.pt保存optimizer/scheduler，但没有已实现的任意epoch续训入口，不能宣称自动恢复；先审计RNG/历史完整性，另设恢复计划。禁止删除partial目录或拿其他seed补齐。

## 独立七批研究入口
新入口 `python -m BrainPromptPro.scripts.adni_1to7.train --config BrainPromptPro/configs/adni_1to7/study_v1.json --experiment A0 --seed 42 --split 0 --device cuda:0 --output <N/new-output>`。`--split 0`为新sample-level协议，`--split 1`为独立subject-disjoint补充协议；不代表历史十折。恢复使用相同不可变代码目录、config、依赖checkpoint和输出，加`--resume`；完整保存optimizer/scheduler/模型/Python/NumPy/Torch/CUDA/loader/mask RNG，拒绝源码或配置hash改变。CPU两轮恢复已验证，旧五批runner的恢复限制不适用于此新入口。队列入口 `python -m BrainPromptPro.scripts.adni_1to7.run_queue --jobs <private-jobs.json> --state <private-state.json> --gpus 0,3,5,7`；启动前核实PID、输出和checkpoint，使用独立会话保活。新训练代码使用Git不可变副本，避免后续修改影响在跑源码封存。完整私有协议不复制到公开仓库。

当前七批作业、版本及恢复以[RUNBOOK](research/adni_1to7/RUNBOOK.md)为准。七批共同分类目标为两项TRAIN计数加权CE；旧config遗留graph sparse项在新runner未使用。P2按登记spec附加损失。elapsed_seconds包括训练循环和逐轮验证，不含初始数据准备/P5参考库构造，预训练40轮另计；共享GPU耗时不作严格速度benchmark。

七批最终：63项固定TEST已完成且两层复核通过；锁后不再运行模型测试。report_final从现有预测对齐冻结标签并直接计数复算，需新输出目录，命令及失败恢复历史见RUNBOOK。主/subject两套协议分开，三个seed不是患者置信区间。

## 005 七批 BrainPrompt 扫描十折末轮（独立于004 holdout）

计划[005](../plans/005_adni1to7_brainprompt_cv10_last.md)。C=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012，训练不可变代码C/code/8e5d8c3，版本见C/source_version.json；结果C/results为已核验源码输出目录的链接。1323FC、895人，扫描分层十折seed41；同人关联不隔离，624个OOF扫描的受试者也出现在对应TRAIN。官方末轮规则与配置见BrainPromptPro/configs/adni_1to7/brainprompt_cv10_last_v1.json。

正式命令在C/code/8e5d8c3工作目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/panyao/miniconda3/envs/brainprompt/bin/python -B -m BrainPromptPro.scripts.adni_1to7.cv10_baseline queue --run-id adni1to7_brainprompt_cv10_last_seed41 --gpus 0,3,4,5,6,7
```

已有队列时不能重复执行：先检查C/launcher.json、C/master.log、C/results/queue_status.json及实际PID。单折worker --run-id ... --split 0..9保留，但拒绝既有输出；旧worker并未实现完整RNG恢复，不得删目录后盲目重跑。原004锁不可修改。

完整十折成功后，独立报告入口直接运行文件，以C/source_version.json定位训练包；不从当前工作树导入可变模型，也不重做选型：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/panyao/miniconda3/envs/brainprompt/bin/python -B /home/panyao/AD/AD-/BrainPromptPro/scripts/adni_1to7/report_cv10_baseline.py --study-root /data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012 --output /data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_brainprompt_cv10_last_20261012/reports/verified_cv10_20261012 --device cuda:0
```

报告目录已存在时应使用新的报告目录，不覆盖旧证据。该入口要求10/10成功，核验输入/来源/封存锁、逐fold预测标签、TRAIN编码器、scheduler首次停止、last.pt与概率指标、OOF一次覆盖；--device额外重载同一末轮模型复核预测，属于结果审计，不是重新选择checkpoint。仅公开CV10_Test_Results.md和cv10_summary.public.json；private预测/混淆计数、完整划分及权重留服务器。十折mean±population SD(ddof0)与合并OOF分别报告，不能与004三seed holdout混用。
