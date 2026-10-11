"""Reload a sealed run for validation audit or explicitly locked one-time TEST.

Invoke this script directly. Imports come from the run's immutable training
snapshot. The default is validation audit; TEST requires a matching selection
lock, an exact checkpoint hash, and a new private output directory.
"""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check_lock(lock, run, result, protocol, expected_lock_hash):
    if not expected_lock_hash or sha(lock)!=expected_lock_hash:raise ValueError('Explicit selection lock hash required')
    doc=json.loads(Path(lock).read_text())
    if doc.get('evaluator_sha256')!=sha(__file__):raise ValueError('Evaluator code is not locked')
    if not doc.get('development_complete') or doc.get('test_selection_policy')!='NO_TEST_BASED_SELECTION':raise ValueError('Selection is not frozen')
    rows=[x for x in doc['evaluations'] if Path(x['run']).resolve()==run.resolve() and x['protocol']==protocol]
    if len(rows)!=1:raise ValueError('Run absent or duplicated in selection lock')
    entry=rows[0]
    for key in ['experiment','seed','checkpoint_sha256','dataset_hash','split_hash']:
        if entry[key]!=result[key]:raise ValueError('Locked run mismatch: '+key)
    if doc['protocol_hashes'][protocol]!=result['split_hash']:raise ValueError('Locked protocol mismatch')
    return doc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',choices=['sample','subject'],default='sample')
    p.add_argument('--role',choices=['validation','test'],default='validation')
    p.add_argument('--selection-lock',type=Path);p.add_argument('--selection-lock-sha256')
    p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Never overwrite or automatically repeat an evaluation')
    result=json.loads((a.run/'result.json').read_text())
    if result['stage']!='supervised' or result['smoke_epochs'] or result['test']!='NOT_EVALUATED':raise ValueError('Expected a formal supervised development run')
    ckpath=a.run/'best_validation.pt'
    if sha(ckpath)!=result['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    for path,expected in result['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('Training source changed')
    model_paths=[Path(x) for x in result['source_sha256'] if x.endswith('/BrainPromptPro/nets/connectome_p2p3p5.py')]
    if len(model_paths)!=1:raise ValueError('Ambiguous training package')
    code_root=model_paths[0].parents[2];sys.path.insert(0,str(code_root))
    import torch
    from torch.utils.data import DataLoader
    from BrainPromptPro.data.prepared_cohort import PreparedCohortDataset
    from BrainPromptPro.data.brainnet_pro import FoldDataset
    from BrainPromptPro.data.metadata import MetadataEncoder,record_id
    from BrainPromptPro.nets.connectome_p2p3p5 import ConnectomeBrainPrompt
    from BrainPromptPro.nets.adni1to7_controls import build_control
    from BrainPromptPro.research_runner import seed_all
    from BrainPromptPro.scripts.adni_1to7.train import evaluate
    if Path(inspect.getfile(ConnectomeBrainPrompt)).resolve()!=model_paths[0].resolve():raise ValueError('Invoke directly using the run package')
    ck=torch.load(ckpath,map_location='cpu',weights_only=True);cfg,spec=ck['config'],ck['spec']
    if ck['signature']!=json.loads((a.run/'signature.json').read_text()):raise ValueError('Checkpoint signature changed')
    study_root=Path(cfg['study']['output_root']).resolve()
    if not a.output.resolve().is_relative_to(study_root):raise ValueError('Evaluation output must remain in private study root')
    splitpath=Path(cfg['study'][a.protocol+'_split'])
    if sha(splitpath)!=result['split_hash']:raise ValueError('Wrong frozen protocol')
    lock_hash=None
    if a.role=='test':
        if a.selection_lock!=study_root/'selection_lock.json':raise ValueError('Use the study selection lock')
        check_lock(a.selection_lock,a.run,result,a.protocol,a.selection_lock_sha256)
        lock_hash=a.selection_lock_sha256
        expected=study_root/'final_test'/a.protocol/result['experiment']/f"seed{result['seed']}"
        if a.output.resolve()!=expected:raise ValueError('TEST output path is fixed by the lock')
    elif a.selection_lock or a.selection_lock_sha256:raise ValueError('Validation audits do not consume a test lock')
    torch.set_num_threads(2);seed_all(result['seed']);device=torch.device(a.device)
    dataset=PreparedCohortDataset(cfg['data']);split=json.loads(splitpath.read_text())
    metadata=MetadataEncoder().fit([dataset.records[i] for i in split['train']]);bank=teacher=None
    options=spec['modules'];state=ck['model']
    if options.get('use_p5'):
        extra=state['population._extra_state'];ids=[record_id(dataset.records[i]) for i in split['train']]
        if extra['reference_ids']!=ids or extra['encoder_version']!=result['reference_sha256']:raise ValueError('Reference membership/version changed')
        teacher=ConnectomeBrainPrompt(cfg,dataset.roi_prompt,dataset.disease_prompt,{}).to(device).eval()
        teacher.load_state_dict({k.removeprefix('reference_encoder.'):v for k,v in state.items() if k.startswith('reference_encoder.')})
        bank=dict(representation=state['population.reference'].to(device),reliability=state['population.reliability'].to(device),reference_ids=ids,train_ids=ids,encoder_version=extra['encoder_version'])
    seed_all(result['seed'])
    model=build_control(spec['architecture'],cfg,dataset,metadata,split['train']).to(device) if spec.get('architecture') else ConnectomeBrainPrompt(cfg,dataset.roi_prompt,dataset.disease_prompt,options,bank,teacher).to(device)
    model.load_state_dict(state);model.eval()
    counts=torch.bincount(dataset.labels[split['train']],minlength=5).float();weights=counts.pow(-.5);weights=(weights/weights.mean()).to(device)
    # Re-forward VAL before any TEST loader construction. A failed reload must
    # never be silently accepted just because it could produce test numbers.
    val_loader=DataLoader(FoldDataset(dataset,split['val'],metadata),batch_size=cfg['params']['batch_size'],shuffle=False)
    val,payload=evaluate(model,val_loader,device,weights)
    saved=torch.load(a.run/'val_payload.private.pt',map_location='cpu',weights_only=True)
    if not torch.equal(payload['truth'],saved['truth']) or payload['sample_ids']!=saved['sample_ids'] or not torch.equal(payload['probability'].argmax(-1),saved['probability'].argmax(-1)) or not torch.allclose(payload['probability'],saved['probability'],atol=2e-5,rtol=2e-5):raise ValueError('VAL reload failed; TEST remains inaccessible')
    base=dict(experiment=result['experiment'],seed=result['seed'],protocol=a.protocol,role=a.role,checkpoint_sha256=result['checkpoint_sha256'],dataset_hash=result['dataset_hash'],split_hash=result['split_hash'],training_source_verified=True,evaluator_sha256=sha(__file__),device=str(device),selection_lock_sha256=lock_hash,validation_reload_max_error=float((payload['probability']-saved['probability']).abs().max()))
    a.output.mkdir(parents=True,exist_ok=False)
    if a.role=='test':
        (a.output/'access_started.private.json').write_text(json.dumps(base,indent=2)+'\n')
        test_loader=DataLoader(FoldDataset(dataset,split['test'],metadata),batch_size=cfg['params']['batch_size'],shuffle=False)
        metrics,payload=evaluate(model,test_loader,device,weights)
    else:metrics=val
    torch.save(payload,a.output/'predictions.private.pt')
    report=dict(base,metrics=metrics,status='COMPLETE',test_accessed=a.role=='test')
    (a.output/'result.private.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if sha(ckpath)!=result['checkpoint_sha256']:raise RuntimeError('Checkpoint changed during evaluation')
    print(json.dumps({k:report[k] for k in ['experiment','seed','protocol','role','status','test_accessed','validation_reload_max_error']}))


if __name__=='__main__':main()
