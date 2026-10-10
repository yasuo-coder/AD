"""Generate factual first-stage report from completed train/validation artifacts."""
import csv,json,sys,shutil
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro.research_runner import RUN,dump,csv_out
MODELS=['BrainPrompt','CurrentPro_legacy','CurrentPro_stable'];SEEDS=[42,3407,2026]
NAMES={'BrainPrompt':'BrainPrompt','CurrentPro_legacy':'旧版 Pro（legacy）','CurrentPro_stable':'既有新版 Pro（stable_v1）'}
METRICS=['accuracy','f1_macro','precision_macro','recall_macro','roc_auc_macro'];CLASSES=['CN','AD','SMC','EMCI','LMCI']
def read(p):return json.loads(Path(p).read_text())
def result(m,p,s):return read(RUN/'experiments'/m/p/f'seed_{s}'/'result.json')
rows=[];summary={};perclass=[];collapse=[];diag={}
for model in MODELS:
 ds=[result(model,'sqrt_inverse_frequency',s) for s in SEEDS]
 summary[model]={role:{k:dict(mean=float(np.mean([d[role][k] for d in ds])),std=float(np.std([d[role][k] for d in ds],ddof=1))) for k in METRICS+['f1_weighted','balanced_accuracy']} for role in ['train','val']}
 for d in ds:
  path=RUN/'experiments'/model/'sqrt_inverse_frequency'/f"seed_{d['seed']}"
  hist=[json.loads(t) for t in (path/'history.jsonl').read_text().splitlines()]
  rows.append(dict(model=model,seed=d['seed'],profile=d['supporting_profile'],role='validation',best_epoch=d['best_epoch'],epochs=d['epochs_completed'],**{k:d['val'][k] for k in METRICS+['f1_weighted','balanced_accuracy']},parameters=d['parameter_count'],test='NOT_EVALUATED'))
  collapse.append(dict(model=model,seed=d['seed'],selected_prediction_counts=d['val']['prediction_distribution'],selected_true_counts=d['val']['true_distribution'],last_prediction_counts=hist[-1]['validation']['prediction_distribution'],all_cn_epochs=sum(t['validation']['prediction_distribution'][0]==100 for t in hist),epochs=len(hist)))
  for role in ['train','val']:
   for k,c in enumerate(CLASSES):perclass.append(dict(model=model,seed=d['seed'],split=role,class_name=c,**d[role]['per_class'][k]))
 diag[model]=read(RUN/'experiments'/model/'sqrt_inverse_frequency/seed_42/diagnostics/module_diagnostics.json')
summary_rows=[dict(model=m,split=role,metric=k,**v) for m,s in summary.items() for role,t in s.items() for k,v in t.items()]
csv_out(RUN/'validation_seed_metrics.csv',rows);csv_out(RUN/'validation_mean_std.csv',summary_rows);csv_out(RUN/'per_class_train_val.csv',perclass);dump(RUN/'collapse_diagnostics.json',collapse)
dump(RUN/'development_summary.json',dict(status='First-stage development complete; new architecture pending Plus full text',models=summary,aggregation='Mean and sample SD ddof=1 over seeds42,3407,2026 on the SAME validation split; not cross-validation SD',test='NOT_EVALUATED'))
audit=read(RUN/'audits/data/extended_audit.json');verification=read(RUN/'verification/development_verification.json');assert verification['passed'] and verification['experiments_verified']==12
protocol=read(RUN/'protocol.json')
heading='| 模型 | Accuracy | Macro F1 | Macro Precision | Macro Recall | ROC-AUC (macro OvR) |\n|---|---:|---:|---:|---:|---:|'
def table(role):
 lines=[heading]
 for m in MODELS:
  vals=[]
  for k in METRICS:
   x=summary[m][role][k];vals.append(f"{x['mean']:.4f} ± {x['std']:.4f}" if k=='roc_auc_macro' else f"{100*x['mean']:.2f}% ± {100*x['std']:.2f}%")
  lines.append('| '+NAMES[m]+' | '+' | '.join(vals)+' |')
 return '\n'.join(lines)
text=['# BrainPromptPro_Optimization_Report','',
'**第一阶段实际运行报告，更新日期：2026-10-08（Asia/Shanghai）。研究目录沿用启动时的 `final_optimization_20261007`。**',
'已完成 ADNI_1–5 合并、固定划分、原模型和两版既有 Pro 的 12 组训练、6 组详细模块探针、三 seed 对照与独立复算。**最终新结构、单模块/组合/完整消融和最终测试均未完成**。原因是 BrainPromptPlus 论文全文尚不可读；不能越过用户明确要求的 novelty firewall，也不能把旧 stable_v1 改名为本轮最终模型。','',
'## 1. ADNI_Combined Dataset Audit','',
'按 ADNI_1→5、各批规范化 subject/visit 排序拼接；一张 rs-fMRI 对应一张 AAL116 Pearson FC。保留同一 biological subject 的不同访视；T1 配对影像不另算一个 FC 样本。扫描 ID、Image ID、时间序列内容、FC 内容均检查；未发现重复 FC 或无效 FC。划分前不另行打乱，由固定分层划分器内部以 seed42 打乱。','',
'| 来源 | FC 样本 | 独立受试者 | 多访视受试者 | CN | AD | SMC | EMCI | LMCI | AD占比 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in audit['summary']:text.append('| '+str(r['Source'])+' | '+' | '.join(str(r[k]) for k in ['FC_Samples','Unique_Subjects','Multiple_Visit_Subjects','CN','AD','SMC','EMCI','LMCI'])+f" | {r['AD_Percent']:.3f}% |")
text += ['',
'合并 **998 FC / 685 位独立受试者 / 216 位有多访视的受试者**。AD 41（4.108%），非 AD 957（95.892%）。独立 AD 受试者41；分批受试者数存在交叉，不能直接相加。原 CSV 中 ADNI_1/2/3/4 的8/8/22/14条 AD 影像记录包含功能扫描与配对 T1，转换为4/4/11/7个 FC 样本；本任务计量单位是 FC，而非影像记录行数。',
'年龄74.88±8.08岁（51–96），女性537、男性461，历史 metadata site 编码48个；时间点45–976。元数据缺失0、非有限/非法FC为0。48.35%的非对角 FC 为负，若用正 sigmoid 邻接替换，不能声称保留了边符号。',
'下载清单原1000张功能扫描中，ADNI_2 的 两张经审计缺失的功能扫描 缺少可核验原扫描预处理映射，排除后998；均非 AD。历史替代 经审计的历史替代扫描 不在清单，不补入。ADNI_5 的 `AXIAL RS fMRI (EYES OPEN)` 已按空白规范化正确识别。此前798条 FC 和提示嵌入原值保留。',
'诊断一律采用各批下载 CSV 的 Group，MCI→LMCI；191条原始MCI已合并。固定顺序 CN=0、AD=1、SMC=2、EMCI=3、LMCI=4，Dataset label/glabel/疾病 prompt/分类器输出同序。存在79条原始诊断来源冲突，合并MCI后58条仍不同，另有1条性别来源冲突；均记录并遵循事先同一来源规则，未按预测效果改标签。这是实验来源约定，非独立临床裁决。','',
'| Split | FC | CN | AD | SMC | EMCI | LMCI |','|---|---:|---:|---:|---:|---:|---:|']
for role in ['train','val','test']:text.append(f"| {role} | {protocol['sizes'][role]} | "+' | '.join(str(protocol['distribution'][role][c]) for c in CLASSES)+' |')
text += ['',
'对应训练分布为62.66%/4.14%/3.88%/4.89%/24.44%，验证和测试均63%/4%/4%/5%/24%。Train–Val / Train–Test / Val–Test 共享 biological subjects 为46/38/6；按用户要求允许，扫描本身不交叉。本结果不能解释为未见受试者泛化。',
'Source_Dataset 不进入 model_inputs；仅追踪和诊断。source×site 的 Cramér’s V=0.969，site×diagnosis=0.340；稀疏列联格较多，仅作关联描述。site_id 来自历史 aggregate CSV 按 Image ID 连接，不等同于目录编号，但尚未用原始 acquisition site 表独立确认其编码来源。',
'完整年龄/性别/站点/访视分布：[distributions.json](audits/data/distributions.json)；[数据汇总](audits/data/dataset_summary.csv)；[标签映射](audits/data/label_mapping.csv)；[数据审计](audits/data/extended_audit.json)。','',
'## 2. BrainPromptPlus Novelty Exclusion Audit','',
'[完整代码证据排除清单](audits/methods/BrainPromptPlus_Novelty_Exclusion_List.md)已建立。官方仓库 commit `87932c1e6d03d0f4fb5c9aeaca00b043e51b6b14` 的代码和 notebook 已读；论文 DOI `10.1109/TMI.2026.3692958` 的官方入口与作者PDF链接目前返回验证页面，尚未取得全文。',
'据官方代码排除：①BOLD频谱统计→语言描述→冻结LLM的ROI频谱提示；②FC强弱/正负连接描述→LLM空间边提示；③该提示图中的边语义逐通道消息调制；④五类提示整体结构。通用 attention/GNN 本身不被全面禁止。代码出现但未接入 loss 的 `compute_dis_loss`，不能据此判断论文是否把该项作为贡献。',
'论文 loss、所有 representation 与训练框架边界未完整核对；新候选一律 PENDING，未把“代码里没看到”写成 ALLOW。','',
'## 3. BrainPrompt vs BrainPromptPlus vs Old BrainPromptPro','',
'| 方法 | ROI输入/图 | Readout | Population |','|---|---|---|---|',
'| BrainPrompt | 完整FC行+ROI提示；固定稀疏拓扑 | mean | subject/brain相似度阈值图；当前实现实际无边 |',
'| BrainPromptPlus（源码） | 频谱节点提示+空间边提示，边条件消息 | mean/sum/max | 源码仍为单相似度阈值图 |',
'| 旧 Pro legacy | FC与ROI语义生成正值稠密图 | disease-query注意力后再次混合成单向量 | age/sex/site 三关系，batch内 |',
'| 既有 stable_v1 | abs(FC)²连续先验、残差、LayerNorm | 归一化注意力+mean残差门 | metadata先验+关系融合门，batch内 |',
'',
'具体公式、信息流与代码位置见 [Architecture_Audit.md](audits/methods/Architecture_Audit.md)。Plus 这一行仅概括可读源码，不能替代全文贡献审查。','',
'## 4. Baseline Results','',
'**以下均为验证集，不是测试集；三 seed 在同一固定 split 上从头训练。** F1/P/R按五类macro平均，zero_division=0；AUC用概率macro OvR。均值±样本标准差（ddof=1），不是十折标准差。','',table('val'),'',
'训练协议采用本次最新附件要求：验证macro-F1选模；同分按预先固定综合分，再按较低验证loss。不会与此前十折末轮结果混用。Adam lr5e-5、weight_decay0.005、batch20、dropout0.5、梯度clip5；最多400轮，LR<1e-6或至少120轮且100轮未改善时停止。验证总loss按样本数加权汇总；ReduceLROnPlateau factor0.5/patience25。各模型架构差异保留：BP/legacy hidden116 L4，stable hidden64 L2。',
'主分类CE+1.0×疾病CE，已有脑图/群体图稀疏项1e-4，relation diversity=0。共同训练策略为训练集类别计数的平方根逆频率权重，均值归一化，同时用于两项CE。只据 BP seed42 验证改善固定此策略，然后补齐另外两seed；未按每个模型各挑最好策略或最好seed。它不是结构创新。','',
'以下 seed42 支持策略对照独立展示，不混入主表三seed均值：','',
'| 模型 | 配置 | Accuracy | Macro F1 | Macro P | Macro R | AUC |','|---|---|---:|---:|---:|---:|---:|']
for m in MODELS:
 for profile in ['unweighted','sqrt_inverse_frequency']:
  d=result(m,profile,42)['val'];text.append('| '+NAMES[m]+' | '+profile+' | '+' | '.join(f'{d[k]:.4f}' if k=='roc_auc_macro' else f'{100*d[k]:.2f}%' for k in METRICS)+' |')
text += ['',
'只有加权配置已完成全部三seed；unweighted目前仅seed42。加权对legacy的单seed macro-F1/P/R反而下降，不能宣传加权对所有模型都有效。','',
'## 5. Current BrainPromptPro Diagnosis','',
'1. 类别极不平衡，AD/SMC/EMCI各不到5%；验证AD仅4例，一个样本就使AD recall改变25个百分点。CN占63%，较高Accuracy可掩盖少数类失败。',
'2. Legacy脑图接近常数密图，ROI高度同质；readout疾病注意力近均匀，再压成单向量。节点和边都有梯度，但未必提供有效判别信息。',
'3. Legacy population关系高度相似，受batch组成影响，metadata幅度与脑表示不匹配。归一化与门控改变了stable的图结构，却尚未改善分类。',
'4. stable持续多数类塌缩；在当前统一训练配置下，辅助任务、门控、容量与优化的组合不奏效。由于还没做正式单模块消融，不能单独把原因归到某一个模块/超参数。',
'5. 数据存在短时间序列、年龄/站点分布差异与标签来源冲突。它们是需要后续核查的可能噪声来源，不是已经证明的单一失败原因。','',
'## 6. Classification Collapse Analysis','',
'验证真实分布固定为 `[63,4,4,5,24]`，顺序 CN/AD/SMC/EMCI/LMCI。','',
'| 模型 | seed | 验证选中轮预测数 | 末轮预测数 | 全CN轮数/总轮数 |','|---|---:|---|---|---|']
for r in collapse:text.append(f"| {NAMES[r['model']]} | {r['seed']} | {r['selected_prediction_counts']} | {r['last_prediction_counts']} | {r['all_cn_epochs']}/{r['epochs']} |")
text += ['',
'当前stable三个seed末轮均全CN。验证选择能保留部分早期非塌缩输出，但seed42的47次SMC预测不能解释为学会了SMC诊断。只看非CN预测数量也不够，必须看逐类正确率。','',
'![验证混淆矩阵](figures/validation_confusion_seed42.png)','',
'[逐类train/val P/R/F1、支持数](per_class_train_val.csv)；逐扫描softmax、entropy/confidence与margin汇总在各实验 `train_predictions.csv`、`val_predictions.csv`、`result.json`。测试集行为尚未评估。','',
'## 7. Existing Three Directions Audit','',
'下面使用共同加权配置的seed42检查点，在固定训练集每类4例的20例batch做局部探针。非零密度不等于有效密度；所有图统计排除对角线。','',
'| 模型 | 边均值 | 边跨样本std | ROI间cosine | 注意力归一化熵 |','|---|---:|---:|---:|---:|']
for m,d in diag.items():text.append(f"| {NAMES[m]} | {d['graph']['mean']:.6f} | {d['graph']['across_sample_edge_std']:.6f} | {d['roi_mean_cosine']:.6f} | "+(f"{d['attention']['normalized_entropy']:.6f}" if 'attention' in d else '不适用')+' |')
text += ['',
'**M1：** Legacy边均值约0.568，跨样本std约0.00104，100%非零，ROI cosine约0.995；支持“图和ROI同质化”的局部诊断。stable图差异提高、ROI cosine降至0.517，但分类仍差，说明修复图退化不自动等于预测提升。没有看到gate全0/全1；真正问题是legacy近均匀。类内/类间图距离见 [graph_class_diagnostics.csv](graph_class_diagnostics.csv)，只有每类4例，不做显著性或生物学结论。',
'**M2：** legacy注意力熵0.999516接近最大1；stable仍0.989364。疾病条件信息在加权混合后被进一步压缩。Top-10 ROI、疾病之间注意力图相似度与完整attention张量已保存。注意力图不同不证明该ROI为疾病因果区域。JSON里的 `disease_query_cosine` 实际是不同疾病attention map之间的cosine，并非原始Q向量cosine。',
'**M3：** legacy三关系图pair cosine为0.986–0.998，存在强冗余；stable降至0.740–0.865，但尚无性能收益。age/sex/site实际均有分类梯度；没有实现brain relation，记为“不适用”而非填0。',
'原BP的population阈值0.8高于 `sigmoid(1)^2≈0.53445` 理论上界，实际0条边。为保留基线未改其核心网络，未来修复也必须单独审查和公平比较。','',
'| 模型 | age/sex/site平均权重 | validation重排batch的概率L1 | 改变预测数/100 |','|---|---|---:|---:|']
for m,d in diag.items():
 weights=d['population'].get('relation_weights_mean');s=d['validation_batch_composition_sensitivity']
 text.append(f"| {NAMES[m]} | "+('/'.join(f'{x:.4f}' for x in weights) if weights else '不适用（阈值图为空）')+f" | {s['mean_probability_l1']:.6f} | {s['changed_predictions']} |")
text += ['',
'详细梯度探针为确定性eval模式下的**不加权CE**，用于横向看梯度通路；不能当作加权训练时随机dropout梯度的精确复现。疾病LLM embedding是冻结buffer，检查其projection/Q/K/V与attention梯度，而非要求冻结embedding本身有梯度。legacy未加权检查点的主/辅助ROI梯度cosine=-0.406，但加权训练检查点探针为+0.564；因此不能声称两损失始终冲突。stable此探针为-0.063，不能由这一数值单独解释全部退化。','',
'## 8. Second-Generation Designs','',
'**待完成，未放行。** 需要针对图差异保留、疾病条件信息保留、有效且稳定的患者关系提出M1/M2/M3第二代，但用户要求先完整核查Plus论文。当前不能以代码未出现某结构就批准V2/V3，也不会把现有stable再改名当成第二代创新。','',
'## 9. New Candidate Directions','',
'**尚未提出/未运行。** M4～M7等待全文审查与现有诊断后选择，额外最多2–4方向，总池5–7。没有预先默认prototype/contrastive/hierarchical graph可用。每个实际候选必须填写Problem/Core mechanism/Closest Plus component/Similarity/Does reproduce/Decision与证据。','',
'## 10. Candidate Rejection Log','',
'尚无已正式提出的新候选，因此没有伪造候选淘汰记录。Plus频谱/空间LLM图提示等机制已进入禁止复用清单。PENDING是审查证据不足，不能误写成High overlap，也不能误写ALLOW。','',
'## 11. Single-Module Results','',
'原BP验证结果见第4节。BP+M1 V2、BP+M2 V2、BP+M3 V2、额外候选均**未运行**。现有legacy/stable整体对照不等于单模块贡献。直接把Pro三个开关全关也不保证与官方GCN实现等价，后续必须核验公共基线实现。','',
'## 12. Module Ranking','',
'**缺失**：没有经过novelty审查的单模块实验，不虚构排名。','',
'## 13. Pairwise Compatibility','',
'**未运行**：不能用当前整体模型的分数代替两两互补/冗余/冲突证据。完整分阶段矩阵见 [Experiment_Matrix_Pending.md](audits/methods/Experiment_Matrix_Pending.md)。','',
'## 14. Final Top-3 Selection','',
'**尚未选择**：Final Module1/2/3均待定。不能把三个既有方向自动确定为最终Top3。','',
'## 15. Complete Ablation','',
'BP/+A/+B/+C/+AB/+AC/+BC/+ABC的最终三模块消融**未运行**；目前没有合法选定的A/B/C。','',
'## 16. Representation Analysis','',
'PCA仅在train拟合后投影val；t-SNE联合train/val可视化（seed42、perplexity30），仅作描述，不用于输入预处理或测试选型。距离以各自融合表示空间计算，hidden维不同，不直接比较跨模型绝对距离。','',
'| 模型 | split | 类内距离 | 类间距离 | 类间/类内 | Silhouette |','|---|---|---:|---:|---:|---:|']
for r in csv.DictReader((RUN/'representation_diagnostics.csv').open()):text.append(f"| {NAMES[r['model']]} | {r['split']} | {float(r['intra_class_distance']):.4f} | {float(r['inter_class_distance']):.4f} | {float(r['inter_intra_ratio']):.4f} | {float(r['silhouette']):.4f} |")
text += ['',
'Legacy训练分离度好于验证，符合泛化不足；三者验证silhouette均负，诊断类别没有形成清晰分离簇。stable类间/类内距离接近1，与弱判别性一致，但不构成因果证明。','',
'[BrainPrompt PCA/t-SNE](figures/BrainPrompt_representations.png) · [legacy PCA/t-SNE](figures/CurrentPro_legacy_representations.png) · [stable PCA/t-SNE](figures/CurrentPro_stable_representations.png) · [训练曲线](figures/validation_training_curves.png)','',
'## 17. Multi-Seed Results','',
'三个既有模型均完成42/3407/2026，不选最好seed。第4节为完整三seed验证均值；逐seed数值见 [validation_seed_metrics.csv](validation_seed_metrics.csv)，所有mean/SD及weighted-F1/balanced accuracy见 [validation_mean_std.csv](validation_mean_std.csv)。','',
'| 模型 | seed | 验证最佳轮 | 实际停止轮 |','|---|---:|---:|---:|']
for r in rows:text.append(f"| {NAMES[r['model']]} | {r['seed']} | {r['best_epoch']} | {r['epochs']} |")
text += ['',
'## 18. Final Comparison','',
'当前只是既有模型验证对照。legacy相对BP：Accuracy较高，但宏F1/P/R均值更低；AUC均值非常接近且seed波动更大。stable的宏F1/P/R/AUC明显低于BP；Accuracy接近多数类占比，不支持已取得整体提升。','',heading,
'| BrainPrompt（最终test） | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |',
'| 旧版BrainPromptPro（最终test） | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |',
'| 本轮最终新版BrainPromptPro | 尚未构建 | 尚未构建 | 尚未构建 | 尚未构建 | 尚未构建 |','',
'AD扫描比例从历史ADNI_1–4的26/798=3.258%提高到41/998=4.108%，增加0.850个百分点；仍属少数类。本轮改变了输入、划分和选模协议，历史十折末轮结果不能作为受控AD比例干预对照，不能把指标变化因果归于AD增加。也不能由三个seed宣称统计显著；它们共享同一100例验证集，且存在纵向受试者相关性。','',
'## 19. Final Target Check','',
'| 检查 | 本轮最终新版 | 当前stable三seed验证情况 |','|---|---|---|',
'| Accuracy > BP | 未验证 | 均值略高，但多数类塌缩，非整体提升 |',
'| Macro F1 > BP | 未验证 | NO |','| Macro Precision > BP | 未验证 | NO |','| Macro Recall > BP | 未验证 | NO |','| ROC-AUC > BP | 未验证 | NO |',
'| F1 ≥60% | 未验证 | NO |','| Precision ≥60% | 未验证 | NO |','| Recall ≥60% | 未验证 | NO |','',
'目前没有达成目标，不将未评估写成达标，也不把目标值填入结果表。','',
'## 20. BrainPromptPlus Novelty Check','','### Comparison with BrainPromptPlus','',
'| 最终模块 | BrainPrompt缺陷 | 与Plus的区别 | 创新/结论 |','|---|---|---|---|',
'| A | 待合法单模块实验确认 | 缺全文，不可签发NO-overlap | 未选择 |',
'| B | 待合法单模块实验确认 | 缺全文，不可签发NO-overlap | 未选择 |',
'| C | 待合法单模块实验确认 | 缺全文，不可签发NO-overlap | 未选择 |','',
'**下一必要输入：BrainPromptPlus论文全文的本地PDF路径或可读取链接。** 官方 DOI/作者主页访问验证尚未解决；已经读取官方源码、摘要并保留证据，但不能代替全文。该依赖来自用户附件“必须完整阅读论文与代码、编码前完成Overlap Check”的明确限制，不是另行添加的审批要求。','',
'### 可复现性与本轮实际代码变更','',
'新增 `scripts/prepare_adni_combined.py`、ADNI_Combined配置、`research_runner.py`，准备数据加载器支持五批scope；它们完成五批输入、固定split、统一metrics、验证选模、日志、源文件/检查点归档与test lock。既有网络和原BrainPrompt核心代码未改，历史checkpoint/log/result保留。审计/报告脚本归档于 `audit_scripts/`。',
'12组正式实验全部通过 [独立核验](verification/development_verification.json)：概率重新计算五项指标，重新拟合train metadata/class weights，逐epoch重放checkpoint选择、学习率调度与首次停止，确认split/bundle一致、原快照包含的源码哈希未变、test文件缺失且入口未解锁。3组两轮smoke只用于可运行检查，未计入正式结果。',
'源码复核发现最初按module类收集快照会遗漏继承父类/普通helper；已另封存全第一方依赖，并增加独立pre-test preflight核对实际样本ID/label与CSV、ck的bundle/split绑定和三seed完整性。补充封存是在训练后，不冒称为过去未记录文件的训练时哈希证明；其中可交叉找到既有实验哈希的证据也保留。见 `verification/full_dependency_seal/manifest.json`。',
'测试仍锁定；preflight脚本只检查，绝不执行测试推理。未来完成合法候选、选型冻结后，应先通过该preflight再统一评估。当前报告是进度交付，完整系统优化仍有明确未完成项。','']
(RUN/'BrainPromptPro_Optimization_Report.md').write_text('\n'.join(text))
# Keep the overview count exact.
p=RUN/'audits/methods/Architecture_Audit.md';p.write_text(p.read_text().replace('共 64 个 Python','共 65 个 Python'))
for name in ['build_research_report.py','plot_research_diagnostics.py','preflight_research_test.py','verify_research_development.py']:shutil.copy2(Path('/tmp')/name,RUN/'audit_scripts'/name)
print(json.dumps(dict(report=str(RUN/'BrainPromptPro_Optimization_Report.md'),validation={m:summary[m]['val'] for m in MODELS},formal_experiments=12,test='NOT_EVALUATED'),indent=2))
