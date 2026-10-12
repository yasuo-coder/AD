"""Seven-batch BrainPrompt-only official CV; reuse the audited legacy worker.

No model or optimization code is changed. Preparation freezes the seven-batch
input and fresh ten folds. The old holdout study and its locks remain untouched.
Run from an immutable Git snapshot in the private study directory.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from BrainPromptPro import merged_cv
from BrainPromptPro.data.data import LoadData
from BrainPromptPro.data.brainnet_pro import FoldDataset, sha256
from BrainPromptPro.data.metadata import MetadataEncoder
from BrainPromptPro.official_cv import audit_splits, make_official_splits, epoch_pass, dump
from BrainPromptPro.runtime import ROOT, output_path


def additional_sources(config):
    paths = {Path(__file__).resolve(), ROOT/'runtime.py', ROOT/'__init__.py'}
    baseline = Path(config['baseline_root'])
    for folder in ['nets', 'layers', 'utils']:
        paths.update((baseline/folder).rglob('*.py'))
    return {str(p): sha256(p) for p in sorted(paths)}


def validate_sources(root):
    config = json.loads((root/'config.json').read_text())
    if merged_cv.source_hashes() != json.loads((root/'source_sha256.json').read_text()):
        raise ValueError('Archived worker dependencies changed')
    if additional_sources(config) != json.loads((root/'additional_source_sha256.json').read_text()):
        raise ValueError('Baseline/wrapper source changed')
    seal = json.loads((root/'pretraining_seal.private.json').read_text())
    for name, digest in seal['files'].items():
        if sha256(name) != digest:
            raise ValueError('Frozen configuration, split, input or previous study lock changed')
    return config


def prepare(run_id, config_path):
    import dgl, sklearn
    config = json.loads(config_path.read_text())
    assert config['model'] == 'OriginalBrainPromptG'
    assert config['data']['required_batches'] == [f'ADNI_{i}' for i in range(1, 8)]
    assert config['params']['checkpoint_selection'] == 'last_epoch'
    assert config['params']['seed'] == 41
    torch.set_num_threads(2)
    data = LoadData(config['data'])
    if len(data) != 1323 or np.bincount(data.labels.numpy(), minlength=5).tolist() != [850,65,39,61,308]:
        raise ValueError('Unexpected seven-batch cohort')
    folds = make_official_splits(data.labels.numpy(), 41)
    audit = audit_splits(folds, data.records)
    root = output_path('comparisons', run_id)
    root.mkdir(exist_ok=False)
    dump(root/'config.json', config)
    dump(root/'splits.json', folds)
    dump(root/'split_audit.json', audit)
    dump(root/'source_sha256.json', merged_cv.source_hashes())
    dump(root/'additional_source_sha256.json', additional_sources(config))
    dump(root/'provenance.private.json', data.provenance)
    dump(root/'environment.json', dict(python=platform.python_version(), torch=str(torch.__version__),
        cuda=torch.version.cuda, dgl=dgl.__version__, numpy=np.__version__, sklearn=sklearn.__version__))
    protocol = dict(models=['BrainPrompt'], dataset_version=config['data']['cohort_name'],
        samples=len(data), biological_subjects=len({r['subject'] for r in data.records}),
        class_order=data.class_names, counts=np.bincount(data.labels.numpy(),minlength=5).tolist(),
        data_sha256=config['data']['bundle_sha256'], split_sha256=sha256(root/'splits.json'),
        split='Scan-stratified KFold(10, shuffle=True), validation fraction .111 of remaining; seed41',
        selection='Actual stopping last.pt, never validation-best',
        training='Same historical official merged_cv.worker, OriginalBrainPromptAdapter, no P2/P3/P5',
        training_seed=41, epochs_cap=1000, fold_time_cap_hours=6,
        test_monitoring='Twice per epoch as historical official worker; never used for stop, selection, or tuning',
        differences_from_upstream=['Seven local batches, not verified full paper cohort/order',
            'Fixed fallback split RNG and independent loader seed41',
            'Honor baseline.sh edge_ratio=.2', 'Correct multiclass macro metrics/probability OvR AUC',
            'Use existing I/O adapter and last state_dict checkpoint'],
        mean_sd='Ten fold equal-weight mean and population SD (ddof=0), following historical CV',
        interpretation='Descriptive replication on previously developed data, not a fresh external holdout',
        longitudinal_overlap='Allowed scan-level protocol; private split audit records subject overlap',
        prior_study='ADNI1to7 P2/P3/P5 holdout locked and completed; unchanged and reported separately')
    dump(root/'protocol.public.json', protocol)
    old_lock = Path(config['previous_study_selection_lock'])
    files=[root/x for x in ['config.json','splits.json','source_sha256.json','additional_source_sha256.json','protocol.public.json']]
    files += [old_lock, Path(config['data']['bundle'])]
    dump(root/'pretraining_seal.private.json',dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        files={str(p):sha256(p) for p in files}))
    print(json.dumps(dict(root=str(root),samples=len(data),folds=10,
        train_sizes=[len(s['train']) for s in folds],val_sizes=[len(s['val']) for s in folds],
        test_sizes=[len(s['test']) for s in folds],
        test_scans_with_train_subject=sum(x['test_scans_with_train_subject'] for x in audit))))


def check(run_id, device):
    from BrainPromptPro.nets.original_adapter import OriginalBrainPromptAdapter
    from torch.utils.data import DataLoader
    root=output_path('comparisons',run_id);config=validate_sources(root)
    data=LoadData(config['data']);split=json.loads((root/'splits.json').read_text())[0]
    enc=MetadataEncoder().fit([data.records[i] for i in split['train']],config['data']['site_column'],config['data']['missing_site'])
    torch.set_num_threads(2);torch.manual_seed(41)
    model=OriginalBrainPromptAdapter(config,data.roi_prompt,data.disease_prompt).to(device)
    batch=DataLoader(FoldDataset(data,split['train'][:40],enc),batch_size=20,shuffle=False)
    opt=torch.optim.Adam(model.parameters(),lr=config['params']['init_lr'],weight_decay=config['params']['weight_decay'])
    before=[p.detach().clone() for p in model.parameters()]
    train,_=epoch_pass(model,batch,device,config['loss'],opt)
    changed=sum(not torch.equal(x,p) for x,p in zip(before,model.parameters()))
    if not changed:raise ValueError('No parameter updates')
    val,_=epoch_pass(model,batch,device,config['loss'])
    result=dict(status='PASSED',scope='40 TRAIN scans, two optimizer steps; no VAL/TEST data',
        parameters_changed=changed,finite_train_loss=bool(np.isfinite(train['loss'])),finite_eval_loss=bool(np.isfinite(val['loss'])))
    dump(root/'train_only_check.json',result);print(json.dumps(result))


def queue(run_id,gpus):
    import fcntl
    root=output_path('comparisons',run_id);validate_sources(root)
    control=(root/'controller.lock').open('a');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'queue_status.json').exists():raise FileExistsError('Inspect existing queue before reviewed recovery')
    if len(set(gpus))!=len(gpus):raise ValueError('Duplicate GPU')
    if json.loads((root/'train_only_check.json').read_text())['status']!='PASSED':raise ValueError('TRAIN-only check missing')
    for gpu in gpus:
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<5000:raise RuntimeError('GPU insufficient; do not stop existing processes')
    logs=root/'worker_logs';logs.mkdir(exist_ok=False)
    pending=list(range(10));active={};completed=[];failed=[]
    while pending or active:
        for gpu,item in list(active.items()):
            rc=item['process'].poll()
            if rc is None:continue
            item['log'].close();row=dict(fold=item['fold'],gpu=gpu,returncode=rc)
            (completed if rc==0 else failed).append(row);del active[gpu]
            print(json.dumps(row),flush=True)
        if not failed:
            for gpu in gpus:
                if gpu in active or not pending:continue
                fold=pending.pop(0)
                log=(logs/f'BrainPrompt_fold_{fold}.log').open('x')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONHASHSEED='41',PYTHONDONTWRITEBYTECODE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                cmd=[sys.executable,'-B','-m','BrainPromptPro.scripts.adni_1to7.cv10_baseline','worker',
                     '--run-id',run_id,'--split',str(fold)]
                process=subprocess.Popen(cmd,cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=dict(process=process,log=log,fold=fold)
                print(json.dumps(dict(started_fold=fold,pid=process.pid,gpu=gpu)),flush=True)
        dump(root/'queue_status.json',dict(pid=os.getpid(),phase='FAILED' if failed and not active else 'COMPLETE' if not pending and not active else 'RUNNING',
            completed=completed,failed=failed,pending=pending,active=[dict(fold=x['fold'],gpu=g,pid=x['process'].pid) for g,x in active.items()]))
        if failed and not active:raise RuntimeError('Fold failed; preserve partial outputs')
        if active:time.sleep(10)
    validate_sources(root)
    print('TEN_FOLDS_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','check','worker','queue'])
    p.add_argument('--run-id',required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--split',type=int,choices=range(10))
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpus',default='0,3,4,5,6,7')
    a=p.parse_args()
    if Path(a.run_id).name!=a.run_id or a.run_id in ('.','..'):p.error('Invalid run ID')
    if a.command=='prepare':
        if a.config is None:p.error('--config required')
        prepare(a.run_id,a.config)
    elif a.command=='check':check(a.run_id,a.device)
    elif a.command=='queue':queue(a.run_id,a.gpus.split(','))
    else:
        if a.split is None:p.error('--split required')
        root=output_path('comparisons',a.run_id);validate_sources(root)
        merged_cv.worker(a.run_id,'BrainPrompt',a.split)
        validate_sources(root)


if __name__=='__main__':main()
