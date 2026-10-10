"""Fail-closed supplementary preflight; this script NEVER evaluates the test set.

Use --seal-dependencies now to preserve the entire existing first-party training
code, including inherited/helper functions missing from per-module snapshots.
A later default invocation requires the final selection lock. No claim is made
that a post-training seal proves every source file at training start.
"""
import argparse,csv,json,sys,shutil
from pathlib import Path
import torch
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro.research_runner import RUN,load_context,allowed_spec,dump
from BrainPromptPro.data.brainnet_pro import sha256
from BrainPromptPro.data.metadata import MetadataEncoder,record_id
ROOT=Path('/home/panyao/AD/BrainPromptPro')

def dependencies():
 paths=[p for p in ROOT.rglob('*.py') if p.relative_to(ROOT).parts[0] not in ('outputs','scripts','tests')]
 paths += [p for d in ('nets','layers','norm') for p in (ROOT.parent/'BrainPrompt'/d).rglob('*.py')]
 return sorted(set(paths))

def seal():
 target=RUN/'verification/full_dependency_seal';target.mkdir(parents=True,exist_ok=False)
 evidence=[]
 # Recorded historical source hashes are additional evidence, not inferred claims.
 for root in [RUN/'experiments',ROOT/'outputs/comparisons']:
  for p in root.rglob('*sha256.json'):
   try:d=json.loads(p.read_text())
   except (ValueError,OSError):continue
   if isinstance(d,dict):
    for k,h in d.items():
     if isinstance(h,str) and len(h)==64:evidence.append((k,h,str(p)))
 rows=[]
 for p in dependencies():
  h=sha256(p);dst=target/'source'/p.relative_to(ROOT.parent);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
  matches=[src for key,v,src in evidence if v==h and (key==str(p) or key==str(p.relative_to(ROOT.parent)) or key==str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.relative_to(ROOT.parent/'BrainPrompt')))]
  rows.append(dict(path=str(p),sha256=h,archive=str(dst),matching_recorded_hash_evidence=matches))
 manifest=dict(scope='Full current first-party model/data/runner dependency superset',timing='POST_TRAINING supplemental seal; prior per-experiment snapshots remain authoritative for files they contain',limitation='A matching historical hash is corroboration; absent training-start hashes are not retroactively asserted',files=rows)
 dump(target/'manifest.json',manifest);print(json.dumps(dict(sealed=len(rows),without_matching_prior_hash=[r['path'] for r in rows if not r['matching_recorded_hash_evidence']],manifest=str(target/'manifest.json')),indent=2))

def preflight():
 lockfile=RUN/'selection_lock.json'
 if not lockfile.is_file():raise RuntimeError('Final selection has not been locked; test remains closed')
 lock=json.loads(lockfile.read_text());assert sha256(lock['evidence'])==lock['evidence_sha256']
 cfg,protocol,data,split=load_context()
 # The manifest binds each frozen CSV to actual loaded dataset identity and labels.
 for role,indices in split.items():
  rows=list(csv.DictReader((RUN/'splits/ADNI_Combined'/(role+'_samples.csv')).open()))
  assert len(rows)==len(indices)
  for row,i in zip(rows,indices):
   rec=data.records[i]
   assert int(row['scan_index'])==i and row['Sample_ID']==record_id(rec)
   assert row['Subject_ID']==rec['subject'] and row['acquisition_id']==rec['fmri_image_id']
   assert int(row['class_id'])==int(data.labels[i]) and row['diagnosis']==rec['diagnosis']
   assert float(row['age'])==float(rec['age']) and row['sex']==rec['sex'] and row['site']==str(rec['site_id'])
   assert row['Source_Dataset']==rec['source_batch']
 sealfile=RUN/'verification/full_dependency_seal/manifest.json';manifest=json.loads(sealfile.read_text())
 for row in manifest['files']:assert sha256(row['path'])==row['sha256']==sha256(row['archive']),row['path']
 expected=MetadataEncoder().fit([data.records[i] for i in split['train']],cfg['data']['site_column'],cfg['data']['missing_site']).state_dict()
 groups={}
 for entry in lock['experiments']:
  path=Path(entry['experiment']);ckpath=path/'best_validation.pt'
  assert path.is_relative_to(RUN/'experiments') and sha256(ckpath)==entry['checkpoint_sha256']
  ck=torch.load(ckpath,weights_only=True,map_location='cpu');res=json.loads((path/'result.json').read_text())
  assert ck['bundle_sha256']==cfg['data']['bundle_sha256'] and ck['split_sha256']==protocol['split_sha256']
  assert ck['training_indices_used']==split['train'] and ck['metadata_encoder']==expected
  assert ck['seed']==entry['seed'] and ck['model_name']==entry['model'] and ck['supporting_profile']==entry['supporting_profile']
  assert ck['spec']==allowed_spec(ck['model_name'])
  assert res['test']=='NOT_EVALUATED' and res['best_epoch']==ck['epoch']+1
  for p,h in ck['source_sha256'].items():assert sha256(p)==h
  groups.setdefault(ck['model_name'],[]).append(ck['seed'])
 for name,seeds in groups.items():assert sorted(seeds)==[42,2026,3407],(name,seeds)
 profiles={entry['supporting_profile'] for entry in lock['experiments']};assert len(profiles)==1
 dump(RUN/'verification/final_pretest_preflight.json',dict(passed=True,selection_lock_sha256=sha256(lockfile),dependency_seal_sha256=sha256(sealfile),test_inference_performed=False,models=groups,actual_samples_match_fixed_manifest=True,checkpoint_bundle_split_hashes_bound=True))
 print('PREFLIGHT_PASS; no test inference has been performed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal-dependencies',action='store_true');a=p.parse_args()
 if a.seal_dependencies:seal()
 else:preflight()
