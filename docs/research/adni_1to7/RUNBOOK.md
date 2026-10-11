# 七批研究运行与恢复

N=/data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010。
Python=/home/panyao/miniconda3/envs/brainprompt/bin/python；共同环境PYTHONDONTWRITEBYTECODE=1、OMP/MKL/OPENBLAS_NUM_THREADS=2。

## 先核实，再动作
读取N/queues/*/state.private.json和launcher.json，按PID执行`ps -p <PID> -o pid,ppid,etime,comm,args`并核对command/cwd及日志更新时间。PyTorch会将comm设成pt_main_thread，不能用comm=python作为唯一筛选。可用os.kill(pid,0)确认存活，但还需command防PID复用。不能只看Markdown或陈旧queue状态。

Phase1–4全部完成；Phase5 controller1408894，代码N/code/29f34f9，GPU0/3/4/5/6/7，40项新任务。读取N/queues/phase5_three_seed/state.private.json获取实际子进程和状态；旧PID不可直接用于恢复。

## 完成项复核
在审查仓库根执行：
```
/home/panyao/miniconda3/envs/brainprompt/bin/python -B -m BrainPromptPro.scripts.adni_1to7.summarize_validation --root /data2/panyao_data/ADNI/BrainPromptPro_runs/adni_1to7_p2p3p5_20261010 --output <N/reports/新的唯一目录>
```
验证result对应checkpoint、source、共同数据/划分/配置及私有概率指标；只导出匿名标量白名单。已有snapshot保留，不覆盖。P2 reconstruction_probe只评估VAL，均值基线只由TRAIN拟合；pretrain_mode_probe只使用TRAIN，不保存改变过BN的模型。

## 中断恢复
先确认controller和所有对应子进程均已停止；如果子进程仍在跑，等待它自然结束，不启动第二份。检查完整result、checkpoint SHA及history/status；有完整result者跳过，无result但有last.pt者才能恢复。
为待恢复项在N/queues/下新建唯一目录，复制相应jobs.private.json，只为已确认中断的项加`"resume": true`，保留code_root/config/seed/split/initial_checkpoint/reference_checkpoint原值。用同一不可变code_root运行`python -B -m BrainPromptPro.scripts.adni_1to7.run_queue --jobs <新jobs> --state <新state> --gpus <核实空余的GPU>`，通过subprocess.Popen(start_new_session=True)独立启动，stdout/stderr写全新master.log，并保存新launcher PID/command/cwd。不要覆盖旧日志，也不要在可编辑仓库用新源码恢复旧checkpoint。
若恢复单任务，复用原command/output并增加`--resume`；runner要求source/config/split/依赖hash一致，恢复optimizer/scheduler及Python/NumPy/Torch/CUDA/loader/mask RNG。改变代码需要新输出和新实验名，不能绕过签名检查。新结果完成后队列再验证身份、依赖、split与checkpoint。

## 协议与计算口径
主协议为新固定holdout，--split 0；补充subject-disjoint为--split 1，输出必须分开。不能混入历史十折末轮或五批结果。监督训练按验证MacroF1及固定平局规则选checkpoint。
共同分类目标为两项TRAIN sqrt-inverse-frequency加权CE，辅助权重1；新入口不使用旧配置遗留的graph sparse项。clip5、Adam5e-5/wd.005、400max/min120、patience100、LR factor.5/patience25/floor1e-6。P2额外重建/一致性按登记spec；预训练额外40轮及其成本单列。result.elapsed_seconds为本次训练循环（含逐轮验证），不是包含全部数据准备和P5参考编码的端到端计时；共享GPU不作严格速度benchmark。
P2无Dropout只针对预训练候选，监督阶段仍.5；best_training和固定last40都保留并比较，不能事后隐去较差checkpoint。P5引用同seed正式A0、TRAIN-only bank、固定编码器版本；不使用smoke或测试样本作参考。

## 发布与隐私
正式代码在独立Git分支/PR1，运行副本不可变；GitHub API写入commit与本地commit作者时间不同，但发布时精确比较tree。保留本地训练commit，远程同步不使用force/reset。只发布source、config路径模板、总体统计、hash和白名单标量；完整protocol/split、身份、FC、checkpoint及逐样本概率留在服务器。

## 受试者隔离与完成后分析
Phase6 controller1425910，GPU1/2/5，仍用N/code/29f34f9的训练入口和--split1；输出N/subject_experiments。汇总入口显式--protocol subject；主协议用默认sample，不能混合。
队列结束后可用新版summarize_validation与analyze_ablation。后者要求A0–A7各3seed完整且共同训练/划分hash相同；配对差先在每个seed内四背景平均，SD只跨3seed。observe_queue在不可变分析源码副本中等待指定state完成，然后生成新私有报告；不启动训练或test。恢复观察器须检查原PID，指定新的output/status，不覆盖已有报告。
机制探针直接运行脚本文件：`python -B BrainPromptPro/scripts/adni_1to7/mechanism_probe.py --run <正式实验seed目录> --output <新的私有JSON> --device cpu`，P2另加--mask-robustness。脚本自动使用该checkpoint原始训练源码；不要用-m绕过源码验证。重载偏差保存failure私有记录，需原CUDA复核时明确换设备，不调宽阈值。
