import hashlib,json,os,subprocess
from pathlib import Path
s=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007/stage2_20261008')
p=s/'audits/Last_Triple_Study_Plan_20261009.md'
assert not p.exists()
p.write_text('''# Last two conditional triple experiments, before training or test access

The finite development matrix ends with these two triples; total four triples, five architectural directions with one evidence-driven M3 revision. No extra direction, loss, initialization or seed search follows. The completed pairs do not establish a strong system: M1+M3v3 validation macro F1 37.57%, Precision 36.41%, AUC .7252; M2+M3v3 F1 34.55%; M2+M5 F1 31.67%. These final tests examine whether the adaptive graph changes the representation enough to enable a conditional third-module interaction, not an assertion that weak singles have become innovations. All eight subsets of each triple will have the same three seeds, allowing rejection based on negative marginal contribution.

Candidate: M1_M2_M3v3
Problem addressed: conditional interaction of signed patient-conditioned ROI messages, disease-query readout and repaired train-reference population messages.
Core mechanism: exactly the already frozen M1 V2, M2 V2 and M3 V3 xavier_gain1 operators; no new operator or hyperparameter.
Closest BrainPromptPlus component: spatial semantic edge messages, ordinary readout, inherited threshold population graph.
Similarity: Medium
Does it reproduce a BrainPromptPlus novelty? NO
Decision: ALLOW
Difference: no spectral/spatial FC-to-language prompt construction or semantic edge-text message modulation; numerical signed edge gates, disease-conditioned ROI pooling, bounded train-only metric-neighbor residual respectively. Earlier individual overlap reviews and M3 initialization amendment remain binding. ALLOW means eligible for testing, not globally novel or effective.
Exact eight-way family: BP_control, M1, M2, M3_v3, M1_M2, M1_M3v3, M2_M3v3, M1_M2_M3v3. Never substitute the old M3 V2 in this family.

Candidate: M1_M2_M5
Problem addressed: whether adaptive ROI graph lets disease-aware first-order pooling and within-patient second-order hidden covariance complement each other.
Core mechanism: frozen M1 V2, M2 V2 and M5 V1, unchanged residual paths and losses.
Closest BrainPromptPlus component: spatial edge semantic messages and ordinary graph readout.
Similarity: Medium
Does it reproduce a BrainPromptPlus novelty? NO
Decision: ALLOW
Difference: numerical signed gates plus disease-query and hidden-covariance readouts; no Plus semantic edge prompts. M5 is a generic representation control, not an established scientific innovation.
Exact eight-way family: BP_control, M1, M2, M5, M1_M2, M1_M5, M2_M5, M1_M2_M5.

Same frozen ADNI_Combined, split, labels, seeds 42/3407/2026, weighted profile, common optimizer/stopping/checkpoint rule. No test inference exists. Selection follows Final_Selection_Rule_20261009.md even if no triple qualifies or the 60% targets fail.
''')
reg=json.loads((s/'registry.json').read_text())
for name,mods,v3 in [('M1_M2_M3v3',['M1','M2','M3'],True),('M1_M2_M5',['M1','M2','M5'],False)]:
 assert name not in reg
 spec=dict(kind='triple',modules=mods,version=3 if v3 else 2,novelty_claim='conditional complete ablation; no final innovation claim',overlap_check={'Decision':'ALLOW','Similarity':'Medium','Does it reproduce a BrainPromptPlus novelty?':'NO','evidence_path':str(p),'evidence_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 if v3:spec.update(population_version=3,population_initialization='xavier_gain1',control='learned')
 reg[name]=spec
(s/'registry.json').write_text(json.dumps(reg,indent=2)+'\n')
phase='last_triples_20261009'
cmd=['/home/panyao/miniconda3/envs/brainprompt/bin/python',str(s/'audit_scripts/run_research_queue_v2.py'),'--phase',phase,'--models','M1_M2_M3v3','M1_M2_M5','--gpus','0','1','2','3','5','7','--runner','BrainPromptPro.candidate_v3_initialized_runner']
with (s/'queue_logs'/f'{phase}_master.log').open('x') as f:
 proc=subprocess.Popen(cmd,cwd='/home/panyao/AD',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(s/f'{phase}_launcher.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd),indent=2)+'\n')
print('Started',proc.pid)
