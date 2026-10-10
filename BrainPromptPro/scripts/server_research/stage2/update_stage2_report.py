import json,re,itertools,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
r=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007');s=r/'stage2_20261008';report=r/'BrainPromptPro_Optimization_Report.md';old=report.read_text();snapshot=r/'BrainPromptPro_Optimization_Report.stage2_single_snapshot_20261009.md'
if not snapshot.exists():shutil.copy2(report,snapshot)
parts=re.split(r'(?m)^(## \d+\. [^\n]+)\n',old);sections={int(re.search(r'## (\d+)',parts[i]).group(1)):(parts[i],parts[i+1]) for i in range(1,len(parts),2)}
registry=json.loads((s/'registry.json').read_text());keys=['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro'];seeds=[42,3407,2026];results={}
for p in (s/'experiments').glob('*/sqrt_inverse_frequency/seed_*/result.json'):
 x=json.loads(p.read_text());results.setdefault(x['model'],{})[x['seed']]=x
primary=[json.loads((r/f'experiments/BrainPrompt/sqrt_inverse_frequency/seed_{seed}/result.json').read_text()) for seed in seeds];bp=np.array([[x['val'][k] for k in keys] for x in primary]).mean(0)
def vals(name):
 rows=results.get(name,{})
 if set(rows)!=set(seeds):return None
 return np.array([[rows[z]['val'][k] for k in keys] for z in seeds])
def table(names):
 rows=['| 配置 | 完成seed | Accuracy % | Macro F1 % | Macro Precision % | Macro Recall % | ROC-AUC |','|---|---:|---:|---:|---:|---:|---:|']
 for name in names:
  v=vals(name)
  if v is None:cells=['未完成']*5
  else:
   m=v.mean(0);d=v.std(0,ddof=1);cells=[f'{100*m[i]:.2f}±{100*d[i]:.2f}' if i<4 else f'{m[i]:.4f}±{d[i]:.4f}' for i in range(5)]
  rows.append('| '+name+f' | {len(results.get(name,{}))}/3 | '+' | '.join(cells)+' |')
 return '\n'.join(rows)+'\n'
def setsection(n,title,text):sections[n]=(f'## {n}. {title}','\n'+text.strip()+'\n\n')
setsection(8,'Second-Generation Designs','''M1–M3 V2 已实现并完成三seed。M1 为 signed-FC 双路条件消息残差；M2 为中心化疾病查询、类别 token 保留的 ROI 读出；M3 为 TRAIN-only 798 参考库的三关系读出。实际公式见 [实现说明](stage2_20261008/audits/Implemented_Candidate_Specification.md)。

M3 V2 的均匀关系与直接 current-FC value 路径已被诊断：seed42 去掉直接 FC 项改变96/100验证预测，改为均匀attention只改变6/100；seed3407对应94/100与6/100。这是检查点敏感性探针，不是重训消融。训练F1近1与验证F1约.33–.36的差距进一步提示过拟合。详见各 M3 实验 diagnostics/population_interventions.json。

已据此实现 M3 V3：固定TRAIN-PCA32 key/value、age/sex/site独立有界metric与metadata score、top32稀疏关系、仅参考值聚合、rank8有界残差，删除直接FC value和高维可训练投影。配套fixed/no_metadata/direct三对照与全部三seed。方案见 [V3编码前审查](stage2_20261008/audits/M3_v3_Review_20261009.md) 与 [训练批次梯度支持的初始化修正](stage2_20261008/audits/M3_v3_Initialization_Amendment_20261009.md)。初始化不作为科学创新，未用验证结果选择尺度。

V3完整模型batch=1/7/20/100与逆序推理契约通过，参考自环为0，每关系32邻居；PCA仅TRAIN拟合。learned/no_metadata/fixed/direct可训练人口模块参数分别1418/1287/1185/1185，输出头相同而非全部训练容量严格相同。V2惩罚sigmoid(score)不保证softmax稀疏；V3返回实际simplex邻接，L1项为常量，稀疏性来自top32，均不冒称有效稀疏正则。''')
setsection(9,'New Candidate Directions','''总方向仍为M1–M5，其中M3有针对已观测失效的V3修订。额外M4为有序ROI slots、M5为患者内hidden-channel covariance。M4/M5属于通用读出对照，通过Plus排重不自动构成论文核心创新；没有把训练策略或初始化凑成额外核心模块。''')
setsection(10,'Candidate Rejection Log','''Plus spectral-text节点、spatial-text边语义和五提示语义图等复用路线被排除。M1–M5和M3 V3仅在具体审查边界内ALLOW；这不等同于性能KEEP。M2/M3 V2/M4单模块按既定五指标规则为Weak，组合继续运行仅为明确机制互补或失效的探索性检验；不据此自动晋级。M5为Mixed，M1为Acceptable；后续阶段仍可推翻单模块的入选资格。所有已运行结果均保留。''')
setsection(11,'Single-Module Results','''以下均为验证集，均值±样本标准差；每配置完整3seed才汇总。6个两轮smoke不计入正式结果。原BrainPrompt主基线见第4节；BP_control是独立复跑对照。\n\n'''+table(['BP_control','M1','M2','M3','M4','M5','M3_v3','M3_v3_fixed','M3_v3_no_metadata','M3_v3_direct']))
rank=['| 候选 | 对主BP的五项均值差（百分点；AUC也×100） | 预设等级 |','|---|---|---|']
for name in ['M1','M2','M3','M4','M5','M3_v3']:
 v=vals(name)
 if v is None:rank.append(f'| {name} | 未完成 | 待定 |');continue
 d=v.mean(0)-bp;positive=int((d>1e-12).sum());kind='Strong' if positive==5 else 'Acceptable' if positive>=4 and d.min()>=-.01 else 'Weak' if positive<=2 and d.min()<-.01 else 'Mixed';rank.append('| '+name+' | '+', '.join(f'{100*x:+.2f}' for x in d)+' | '+kind+' |')
setsection(12,'Module Ranking','差值顺序：Accuracy / Macro F1 / Macro Precision / Macro Recall / ROC-AUC。完整结果已独立核验；仅验证集用于研究选择。Strong=五项正，Acceptable=至少四项正且剩余不低于−1百分点，Weak=不超过两项正且有一项低于−1百分点，其余Mixed。\n\n'+'\n'.join(rank)+'\n\n原BP是固定主比较对象；[完整双基线差值CSV](stage2_20261008/summary/validation_three_seed.csv)同时保留与BP_control差值。三seed共享同一100例验证集，不表示独立生物学重复或统计显著提升。')
pairs=[k for k,v in registry.items() if v['kind']=='pair']
setsection(13,'Pairwise Compatibility','[组合范围及编码前审查](stage2_20261008/audits/Pairwise_Study_Plan_20261009.md)先于训练冻结。前四名单模块的六对，加M1+M2机制验证，共21组。不能只用F1上涨忽略Precision/AUC下降。\n\n'+table(pairs))
setsection(14,'Final Top-3 Selection','尚未冻结最终Top-3。单模块和两两结果不支持强行宣称三个模块全部独立有效。已登记两组三模块兼容性检验，并保留至多两组进一步检验额度；M3 V3必须同时与固定关系、无metadata及直接PCA对照比较后才讨论人口关系贡献。最终判断只基于训练/验证证据，test继续封存。')
abl=['[首批三模块计划](stage2_20261008/audits/First_Triple_Study_Plan_20261009.md)。下列是探索候选的八配置验证消融，不等同于已选定三个有效创新。']
for name,spec in registry.items():
 if spec['kind']!='triple':continue
 modules=spec['modules'];names=['BP_control']
 for size in [1,2,3]:names+=['_'.join(v) for v in itertools.combinations(modules,size)]
 abl+=['\n### '+name+'\n',table(names)]
setsection(15,'Complete Ablation','\n'.join(abl))
# Preserve stage1 representation evidence; append links without duplicating prior append.
base_repr=sections[16][1].split('### 阶段2补充')[0]
setsection(16,'Representation Analysis',base_repr+'''\n### 阶段2补充

已有固定seed42的 [M1表示](stage2_20261008/figures/M1_representation.png)、[M2表示](stage2_20261008/figures/M2_representation.png)、[M3 V2表示](stage2_20261008/figures/M3_representation.png)、[M2疾病注意力](stage2_20261008/figures/M2_disease_attention.png)。M2的不同查询注意力确有区别，但未自动带来宏F1提升。各模型train/val_payload.pt保存表示，val_representation.json保存silhouette；可视化是描述性证据，不是疾病因果区域证明。''')
count=sum(len(v) for v in results.values());registered=len(registry)*3
setsection(17,'Multi-Seed Results',f'''固定seed为42/3407/2026，所有结果均保留，不选最好seed。阶段2当前完成{count}/{registered}个登记正式实验，另有6个两轮smoke；阶段1为12个正式实验，分开统计。

[逐seed五指标、最佳轮、停止轮、耗时和结果路径](stage2_20261008/summary/validation_per_seed.csv)；[全部完整三seed均值与SD](stage2_20261008/summary/Validation_Results.md)。缺失seed不以中间最佳值补填。

[独立验证器](stage2_20261008/audit_scripts/verify_stage2_development_v2.py)不导入模型或创建loader：独立混淆矩阵/秩AUC复算、手工选模/调度/首次停止重放、checkpoint/source/bundle/split、train-only metadata/weights/参考/PCA检查。最近一次核验的完成数以 [验证报告](stage2_20261008/verification/development_verification.json) 为准，不能把尚未完成的队列任务当作已验证。''')
setsection(18,'Final Comparison','''最终新结构及正式test尚未冻结/评估。此前的五批固定holdout验证结果不可与早期四批十折末轮结果混用。

| 方法（最终test） | Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC |
|---|---|---|---|---|---|
| BrainPrompt | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| 旧版BrainPromptPro | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| 本轮最终BrainPromptPro | 未选定 | 未选定 | 未选定 | 未选定 | 未选定 |

AD FC样本比例由四批26/798=3.258%增至五批41/998=4.108%，增加0.850个百分点。数据、划分和选模协议也改变，没有同协议控制的AD比例干预，不能把性能变化因果归于AD增加。''')
setsection(19,'Final Target Check','''本轮尚不能宣称五项指标全部超过BrainPrompt，也未证明Macro F1/Precision/Recall均≥60%。三模块独立贡献、组合稳定性和最终test均必须据实检验。任何未完成项标“未完成/未验证”，不填目标值、不隐去下降指标。''')
setsection(20,'BrainPromptPlus Novelty Check','''全文审查完成，无需再次提供PDF。M1/M3/V3为Medium，M2/M4/M5为Low，具体机制均为NO-repeat/ALLOW；编码前证据与SHA绑定在 [registry](stage2_20261008/registry.json)。M3 V3及初始化修正另有明确前置审查，组合/三模块计划也分别登记。

### Comparison with BrainPromptPlus

| 方向 | BrainPrompt限制 | 与Plus区别 | 科学贡献状态 |
|---|---|---|---|
| M1 | 固定FC边强度/符号利用不足 | 数值正负关系条件残差，非LLM空间边提示 | 需独立消融与机制支持 |
| M2 | mean readout压缩类别条件信息 | 全类疾病查询与token身份保留，非Plus普通池化 | 单独较弱，检查组合贡献 |
| M3 V3 | 批内阈值图失效与V2直接FC捷径 | 固定训练参考值、独立关系、无query-query/直接query-value | 需优于固定关系和PCA-direct |
| M4/M5 | ROI顺序/二阶信息丢失 | 数值ordered/covariance readout，无Plus提示融合 | 通用对照，不自动算核心创新 |

Plus排重不等于全领域首次提出；Plus亦讨论learned graph未来方向与多metadata先前工作。最终三个模块尚未选定，不能把表中三个方向自动认定为三个成功创新。

原模型、历史输出、全部检查点和旧源码保留；新的训练文件均版本化。测试入口只有最终锁定后才允许使用；当前没有selection_lock或test_result。''')
now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
header=f'''# BrainPromptPro_Optimization_Report\n\n更新：{now}（Asia/Shanghai）。沿用研究目录 final_optimization_20261007。\n\n已完成论文/数据审计、阶段1基线12组、阶段2已完成{count}组正式实验。BrainPromptPlus全文阻塞已解除。以下新增结果均为验证集；组合与三模块的完成状态按实际文件列出，最终test仍封存。阶段1和早期阶段2快照均保留。\n\n'''
report.write_text(header+''.join(title+'\n'+body for _,(title,body) in sorted(sections.items())))
print('Updated report with',count,'completed stage2 experiments')
