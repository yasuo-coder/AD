import hashlib,json,itertools
from pathlib import Path
s=Path('/home/panyao/AD/BrainPromptPro/outputs/research/final_optimization_20261007/stage2_20261008');summary=json.loads((s/'summary/validation_summary.json').read_text());models={x['model'].split('/')[-1]:x for x in summary['models'] if x['model'].startswith('stage2/M')};order=sorted(models,key=lambda n:models[n]['f1_macro_mean'],reverse=True);assert order==['M5','M1','M3','M4','M2'],order
pairs=[['M1','M3'],['M1','M4'],['M1','M5'],['M3','M4'],['M3','M5'],['M4','M5'],['M1','M2']]
path=s/'audits/Pairwise_Study_Plan_20261009.md'
text='''# Pairwise development plan (frozen before pair training)

All 18 single/control runs have completed; independent verification passed. This is validation-only selection. The original BrainPrompt three-seed mean remains the primary comparator, and contemporary BP_control is retained as numerical sensitivity control. No test data are opened.

Single ranking by predeclared mean validation macro F1: M5, M1, M3, M4, M2. M1 is Acceptable under the original five-metric rule (four mean improvements, precision -0.43 percentage points). No single is Strong. M3 and M4 are Weak; M5 improves some macro metrics but loses Accuracy/AUC and has high seed variance. These are exploratory pairs, not promotion decisions.

Evaluate all six pairs among the four highest F1 singles. Additionally evaluate M1+M2 as one priority interaction probe: M1 changes signed ROI message passing, while M2 produces genuinely different disease queries but has poor standalone performance. This tests whether graph improvements make semantic pooling useful; it does not presume synergy. No other v2 pairs are planned.

M3 v2 is known to contain a strong directly supervised FC value route; its pair results cannot by themselves establish a population-graph contribution. A separately reviewed v3 may replace this mechanism, and will be evaluated first as a single with all three seeds. The v2 results remain visible.

All seeds are 42/3407/2026; all data/splits/optimizer/loss coefficients/stopping/checkpoint rules remain frozen. Pair scores and marginal changes versus both singles and both BP references will be reported. Final Top-3 is not assigned here.

## Combination-level BrainPromptPlus overlap check

Closest Plus components: numerical individual/population graph basics and ordinary pooling. These pairs add only the numerical candidate mechanisms already reviewed individually; they do not add spectral/spatial language prompts, LLM edge embeddings, or Plus semantic edge-message modulation. Generic composition is not automatically scientific novelty. M4/M5 remain generic controls unless independent evidence supports a narrower contribution.

| Candidate | Problem / mechanism | Similarity | Reproduces Plus novelty | Decision |
|---|---|---|---|---|
'''
for pair in pairs:
 n='_'.join(pair);similarity='Medium' if set(pair)&{'M1','M3'} else 'Low';text+=f'| {n} | Add approved {pair[0]} and {pair[1]} pathways around unchanged BP trunk; measure compatibility and redundancy | {similarity} | NO | ALLOW |\n'
text+='\nIndividual formulas and exclusions: BrainPromptPlus_Novelty_Exclusion_List.md. This review is relative to BP/Plus, not a universal novelty claim.\n'
assert not path.exists();path.write_text(text);digest=hashlib.sha256(path.read_bytes()).hexdigest();registry=json.loads((s/'registry.json').read_text())
for pair in pairs:
 n='_'.join(pair);assert n not in registry;registry[n]=dict(kind='pair',modules=pair,version=2,novelty_claim='provisional compatibility probe, no final innovation claim',overlap_check={'Decision':'ALLOW','Similarity':'Medium' if set(pair)&{'M1','M3'} else 'Low','Does it reproduce a BrainPromptPlus novelty?':'NO','evidence_path':str(path),'evidence_sha256':digest})
(s/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
(s/'single_queue_reconciled_20261009.json').write_text(json.dumps(dict(expected=18,completed=18,verification=str(s/'verification/development_verification.json'),note='Original queue status stale after interruption; each result independently verified; M4/M5 seed2026 recovered separately'),indent=2)+'\n')
print('Registered',list('_'.join(x) for x in pairs))
