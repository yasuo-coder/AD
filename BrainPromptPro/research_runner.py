"""Fixed ADNI_Combined holdout: validation-only development, locked final test.

This runner deliberately does not reuse the historical test-monitoring CV loop.
Old models and outputs are never overwritten. Candidates require a documented
BrainPromptPlus overlap decision in the experiment registry before training.
"""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from .runtime import ROOT
from .data.data import LoadData
from .data.brainnet_pro import FoldDataset, sha256
from .data.metadata import MetadataEncoder, record_id
from .metrics import classification_metrics
from .train_TUs_graph_classification_llm import model_inputs

RUN = ROOT / 'outputs/research/final_optimization_20261007'
BASE_MODELS = ('BrainPrompt', 'CurrentPro_legacy', 'CurrentPro_stable')
SEEDS = (42, 3407, 2026)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def csv_out(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def validate_sample_split(split, records, labels):
    sets = {k: set(v) for k, v in split.items()}
    if set(sets) != {'train', 'val', 'test'}:
        raise ValueError('Exactly train/val/test are required')
    if any(not v or len(v) != len(split[k]) for k, v in sets.items()):
        raise ValueError('Empty partition or duplicate sample index')
    if set.union(*sets.values()) != set(range(len(records))):
        raise ValueError('Split must cover all FC samples exactly once')
    pairs = [('train','val'), ('train','test'), ('val','test')]
    if any(sets[a] & sets[b] for a,b in pairs):
        raise ValueError('FC sample overlap')
    expected = set(map(int, labels))
    if any({int(labels[i]) for i in values} != expected for values in sets.values()):
        raise ValueError('A partition lacks a class; do not search seeds to bypass this')
    subjects = {k: {records[i]['subject'] for i in v} for k,v in split.items()}
    return {a+'_'+b: len(subjects[a] & subjects[b]) for a,b in pairs}


def prepare(config_path):
    if (RUN/'protocol.json').exists():
        raise FileExistsError('Fixed protocol already exists; never replace splits')
    cfg = json.loads(Path(config_path).read_text())
    data = LoadData(cfg['data'])
    labels = data.labels.numpy()
    remain, test = train_test_split(np.arange(len(data)), test_size=.1, random_state=42, stratify=labels)
    train, val = train_test_split(remain, test_size=1/9, random_state=42, stratify=labels[remain])
    split = dict(train=sorted(train.tolist()), val=sorted(val.tolist()), test=sorted(test.tolist()))
    overlap = validate_sample_split(split, data.records, labels)
    split_dir = RUN/'splits/ADNI_Combined'; split_dir.mkdir(parents=True, exist_ok=False)
    cfg['params'].update(seed=42, epochs=400, batch_size=20, init_lr=5e-5, weight_decay=.005,
                         lr_reduce_factor=.5, lr_schedule_patience=25, min_lr=1e-6, threads=2,
                         checkpoint_selection='validation_macro_f1_then_composite_then_lower_loss',
                         early_stopping_patience=100, min_epochs=120, gradient_clip_norm=5.)
    cfg['net_params'].update(hidden_dim=116, L=4, dropout=.5, batch_norm=True)
    cfg['loss'].update(lambda_disease=1., lambda_brain_sparse=1e-4, lambda_pop_sparse=1e-4,
                       lambda_relation_div=0.)
    cfg['data']['split_mode']='fixed_sample_stratified_holdout'
    dump(RUN/'config.json', cfg); dump(split_dir/'indices.json', split)
    for role, values in split.items():
        rows=[]
        for i in values:
            row=data.records[i]
            rows.append(dict(scan_index=i, Subject_ID=row['subject'], Visit_ID=row.get('raw_visit',row['session']),
                Sample_ID=record_id(row), acquisition_id=row['fmri_image_id'], diagnosis=row['diagnosis'],
                class_id=int(labels[i]), age=row['age'], sex=row['sex'], site=row['site_id'],
                Source_Dataset=row['source_batch']))
        csv_out(split_dir/(role+'_samples.csv'),rows)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(), input='ADNI_Combined: ADNI_1 through ADNI_5',
        split_seed=42, training_seeds=list(SEEDS), split_ratios=[.8,.1,.1], split_sha256=sha256(split_dir/'indices.json'),
        split_csv_sha256={k:sha256(split_dir/(k+'_samples.csv')) for k in split},
        sample_definition='One FC per rsfMRI acquisition/visit; retain distinct visits of the same biological subject',
        source_is_feature=False, biological_subject_overlap_allowed=True, biological_subject_overlap=overlap,
        class_names=data.class_names, distribution={k:dict(Counter(data.records[i]['diagnosis'] for i in v)) for k,v in split.items()},
        sizes={k:len(v) for k,v in split.items()},
        selection='Highest validation macro-F1; ties use 0.20Acc+0.25F1+0.20P+0.20R+0.15AUC, then lower validation loss',
        stop='max 400 epochs; LR<1e-6; or >=120 epochs and 100 epochs without validation-selection improvement',
        common_training=cfg['params'], supporting_profiles=['unweighted','sqrt_inverse_frequency'],
        metrics='F1/Precision/Recall macro over all five fixed classes; zero_division=0; probability macro OvR AUC',
        test_access='No test loader in train(); evaluate_test requires frozen selection with checkpoint hashes',
        previous_exposure='ADNI_1--4 were used in historical exploratory experiments. This is a new fixed development protocol, not a wholly new external cohort.',
        source_partition='Only provenance and diagnostic grouping; never an input',
        baseline='Unchanged official BrainPrompt-G with common audited adapter; fixed sparse topology follows previously declared repository rule')
    dump(RUN/'protocol.json',protocol)
    dump(RUN/'registry.json',{name:dict(kind='existing',novelty_claim=False) for name in BASE_MODELS})
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


def load_context():
    cfg=json.loads((RUN/'config.json').read_text())
    protocol=json.loads((RUN/'protocol.json').read_text())
    path=RUN/'splits/ADNI_Combined/indices.json'
    if sha256(path)!=protocol['split_sha256']:
        raise ValueError('Frozen split changed')
    for role,h in protocol['split_csv_sha256'].items():
        if sha256(path.parent/(role+'_samples.csv'))!=h:
            raise ValueError('Frozen sample manifest changed')
    data=LoadData(cfg['data']);split=json.loads(path.read_text())
    validate_sample_split(split,data.records,data.labels)
    return cfg,protocol,data,split


def allowed_spec(name):
    registry=json.loads((RUN/'registry.json').read_text())
    if name not in registry:
        raise ValueError('Unregistered experiment')
    spec=registry[name]
    if name not in BASE_MODELS:
        check=spec.get('overlap_check',{})
        if check.get('Decision')!='ALLOW' or check.get('Similarity') not in ('Low','Medium') or check.get('Does it reproduce a BrainPromptPlus novelty?')!='NO':
            raise ValueError('Candidate has not passed the BrainPromptPlus novelty firewall')
        evidence=Path(check['evidence_path'])
        if not evidence.is_file() or sha256(evidence)!=check['evidence_sha256']:
            raise ValueError('Novelty evidence missing or changed')
    return spec


def build_model(name,cfg,data,encoder,spec):
    cfg=copy.deepcopy(cfg)
    if name=='BrainPrompt':
        from .nets.original_adapter import OriginalBrainPromptAdapter
        cfg['net_params'].update(use_adaptive_brain_graph=False,use_disease_aware_readout=False,
                                 use_multirelation_population_graph=False)
        return OriginalBrainPromptAdapter(cfg,data.roi_prompt,data.disease_prompt),cfg
    if name in BASE_MODELS:
        from .nets.load_net import gnn_model
        cfg['net_params']['stability_profile']='legacy' if name=='CurrentPro_legacy' else 'stable_v1'
        if name=='CurrentPro_stable':
            cfg['net_params'].update(hidden_dim=64,L=2,brain_edge_hidden_dim=32,disease_attention_dim=64,
                                     relation_hidden_dim=32,batch_norm=False)
        model=gnn_model(cfg['model'],cfg['net_params'],data.roi_prompt,data.disease_prompt,
                        data.subject_prompt.shape[-1],len(encoder.site_mapping)+1)
    else:
        from .nets.final_research_net import ResearchBrainPrompt
        model=ResearchBrainPrompt(cfg,data.roi_prompt,data.disease_prompt,spec,len(encoder.site_mapping)+1)
    return model,cfg


def source_snapshot(model,out):
    paths={Path(__file__).resolve(),ROOT/'metrics.py',ROOT/'data/brainnet_pro.py',ROOT/'data/prepared_cohort.py',
           ROOT/'data/metadata.py',ROOT/'data/labels.py',ROOT/'data/data.py',ROOT/'train_TUs_graph_classification_llm.py'}
    for module in model.modules():
        path=Path(inspect.getfile(type(module))).resolve()
        if path.is_relative_to(ROOT.parent):
            paths.add(path)
    hashes={str(p):sha256(p) for p in sorted(paths)}
    for p in paths:
        dest=out/'source'/p.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    dump(out/'source_sha256.json',hashes)
    return hashes


def loss_value(output,y,config,class_weights=None):
    loss=F.cross_entropy(output['logits'],y,weight=class_weights)
    loss=loss+config['loss']['lambda_disease']*F.cross_entropy(output['disease_logits'],y,weight=class_weights)
    a=output.get('A_brain')
    if a is not None and config['net_params']['use_adaptive_brain_graph']:
        mask=~torch.eye(a.shape[-1],dtype=torch.bool,device=a.device)
        loss=loss+config['loss']['lambda_brain_sparse']*a[...,mask].abs().mean()
    relations=output.get('population_adjacencies',{})
    for a in relations.values():
        if a.shape[-1]>1:
            mask=~torch.eye(a.shape[-1],dtype=torch.bool,device=a.device)
            loss=loss+config['loss']['lambda_pop_sparse']*a[...,mask].abs().mean()/len(relations)
    for term in output.get('extra_losses',{}).values():
        loss=loss+term
    return loss


def composite(m):
    return .2*m['accuracy']+.25*m['f1_macro']+.2*m['precision_macro']+.2*m['recall_macro']+.15*m['roc_auc_macro']


def selection_key(metrics):
    return (metrics['f1_macro'],composite(metrics),-metrics['loss'])


@torch.no_grad()
def evaluate(model,loader,device,config,class_weights=None,capture=False):
    model.eval();probs=[];ys=[];ids=[];sources=[];features=[];total=0.;count=0
    for batch in loader:
        output=model(**model_inputs(batch,device));y=batch['label'].to(device)
        loss=loss_value(output,y,config,class_weights)
        if not torch.isfinite(loss) or not torch.isfinite(output['logits']).all():
            raise FloatingPointError('Nonfinite evaluation')
        total+=float(loss)*len(y);count+=len(y)
        probs.append(output['logits'].softmax(-1).cpu());ys.append(y.cpu())
        ids.extend(batch['subject_id']);sources.extend(batch['source_batch'])
        if capture:
            features.append(output['fused_repr'].cpu())
    p=torch.cat(probs);y=torch.cat(ys)
    m=classification_metrics(y.numpy(),p.numpy(),model.num_classes)
    entropy=-(p*p.clamp_min(1e-12).log()).sum(-1);top=p.topk(2,-1).values
    m.update(loss=total/count,entropy_mean=float(entropy.mean()),confidence_mean=float(top[:,0].mean()),
             margin_mean=float((top[:,0]-top[:,1]).mean()),true_distribution=torch.bincount(y,minlength=model.num_classes).tolist(),
             prediction_distribution=torch.bincount(p.argmax(-1),minlength=model.num_classes).tolist())
    m['classification_collapse']=max(m['prediction_distribution'])==count
    payload=dict(probability=p,truth=y,sample_ids=ids,source_datasets=sources)
    if capture:payload['embeddings']=torch.cat(features)
    return m,payload


def export_payload(out,role,payload,classes):
    torch.save(payload,out/(role+'_payload.pt'))
    rows=[]
    for i,identity in enumerate(payload['sample_ids']):
        p=payload['probability'][i];y=int(payload['truth'][i])
        row=dict(Sample_ID=identity,Source_Dataset=payload['source_datasets'][i],true_label=y,predicted_label=int(p.argmax()),
                 confidence=float(p.max()),entropy=float(-(p*p.clamp_min(1e-12).log()).sum()))
        row.update({'p_'+name:float(p[k]) for k,name in enumerate(classes)});rows.append(row)
    csv_out(out/(role+'_predictions.csv'),rows)
    if 'embeddings' in payload:
        x=payload['embeddings'].numpy();y=payload['truth'].numpy()
        centered=x-x.mean(0);variance=float(np.square(centered).sum()/len(x))
        pca=PCA(n_components=2,random_state=42).fit_transform(x)
        diag=dict(samples=len(y),dimensions=x.shape[1],total_variance=variance,
                  silhouette=float(silhouette_score(x,y)) if variance>1e-12 and 1<len(np.unique(y))<len(y) else None,
                  class_centroid_distances=np.linalg.norm(np.stack([x[y==k].mean(0) for k in range(len(classes))])[:,None]-np.stack([x[y==k].mean(0) for k in range(len(classes))])[None,:],axis=-1).tolist())
        dump(out/(role+'_representation.json'),diag)
        csv_out(out/(role+'_pca.csv'),[dict(Sample_ID=identity,PC1=float(pca[i,0]),PC2=float(pca[i,1]),class_id=int(y[i])) for i,identity in enumerate(payload['sample_ids'])])


def seed_all(seed):
    import dgl
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    dgl.seed(seed);dgl.random.seed(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def experiment_path(name,profile,seed):
    if profile not in ('unweighted','sqrt_inverse_frequency') or seed not in SEEDS:
        raise ValueError('Unregistered supporting profile or seed')
    return RUN/'experiments'/name/profile/('seed_'+str(seed))


def train(name,profile,seed,device_name,smoke_epochs=0):
    if (RUN/'selection_lock.json').exists():
        raise RuntimeError('Final selection is locked; do not tune after test access')
    cfg,protocol,data,split=load_context();spec=allowed_spec(name)
    torch.set_num_threads(cfg['params']['threads']);seed_all(seed);device=torch.device(device_name)
    encoder=MetadataEncoder().fit([data.records[i] for i in split['train']],cfg['data']['site_column'],cfg['data']['missing_site'])
    model,cfg=build_model(name,cfg,data,encoder,spec);model=model.to(device)
    out=experiment_path(name,profile,seed)
    if smoke_epochs:out=RUN/'smoke'/name/profile/('seed_'+str(seed))
    out.mkdir(parents=True,exist_ok=False)
    hashes=source_snapshot(model,out)
    cfg['params']['seed']=seed
    dump(out/'config.json',cfg);dump(out/'spec.json',spec)
    params=cfg['params'];g=torch.Generator().manual_seed(seed)
    # The test partition is intentionally never instantiated here.
    loaders={k:DataLoader(FoldDataset(data,split[k],encoder),batch_size=params['batch_size'],shuffle=k=='train',
                         generator=g if k=='train' else None,num_workers=0,drop_last=False) for k in ('train','val')}
    train_eval=DataLoader(FoldDataset(data,split['train'],encoder),batch_size=params['batch_size'],shuffle=False)
    class_weights=None
    if profile=='sqrt_inverse_frequency':
        counts=torch.bincount(data.labels[split['train']],minlength=data.num_classes).float()
        class_weights=counts.pow(-.5);class_weights=(class_weights/class_weights.mean()).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=params['init_lr'],weight_decay=params['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=params['lr_reduce_factor'],patience=params['lr_schedule_patience'])
    best=None;best_epoch=-1;start=time.monotonic();gradient_norms={}
    epochs=smoke_epochs or params['epochs']
    for epoch in range(epochs):
        model.train();total=0.;count=0
        for batch in loaders['train']:
            opt.zero_grad(set_to_none=True)
            output=model(**model_inputs(batch,device));y=batch['label'].to(device)
            loss=loss_value(output,y,cfg,class_weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if epoch==0:
                for key,p in model.named_parameters():
                    if p.grad is not None:
                        gradient_norms[key]=max(gradient_norms.get(key,0.),float(p.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),params['gradient_clip_norm'],error_if_nonfinite=True)
            opt.step();total+=float(loss.detach())*len(y);count+=len(y)
        val,_=evaluate(model,loaders['val'],device,cfg,class_weights)
        scheduler.step(val['loss']);key=selection_key(val)
        ck=dict(model=model.state_dict(),config=cfg,spec=spec,model_name=name,supporting_profile=profile,seed=seed,
                epoch=epoch,metadata_encoder=encoder.state_dict(),split_sha256=protocol['split_sha256'],
                bundle_sha256=cfg['data']['bundle_sha256'],validation=val,source_sha256=hashes,
                training_indices_used=split['train'],optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                class_weights=None if class_weights is None else class_weights.cpu())
        if best is None or key>best:
            best=key;best_epoch=epoch;torch.save(ck,out/'best_validation.tmp.pt');(out/'best_validation.tmp.pt').replace(out/'best_validation.pt')
        reason=None
        if opt.param_groups[0]['lr']<params['min_lr']:reason='lr_floor'
        elif epoch+1>=params['min_epochs'] and epoch-best_epoch>=params['early_stopping_patience']:reason='validation_patience'
        elif epoch+1==epochs:reason='smoke_cap' if smoke_epochs else 'epoch_cap'
        entry=dict(epoch=epoch+1,train_loss=total/count,validation=val,best_epoch=best_epoch+1,
                   lr=opt.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-start,stop_reason=reason)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        dump(out/'status.json',entry)
        if epoch==0 or (epoch+1)%20==0 or reason:
            print(json.dumps(dict(model=name,profile=profile,seed=seed,epoch=epoch+1,best_epoch=best_epoch+1,
                   val_acc=val['accuracy'],val_f1=val['f1_macro'],val_auc=val['roc_auc_macro'],elapsed_seconds=entry['elapsed_seconds'],stop=reason)),flush=True)
        if reason:
            torch.save(ck,out/'last.pt');break
    last=torch.load(out/'best_validation.pt',map_location=device,weights_only=True);model.load_state_dict(last['model'])
    results={}
    for role,loader in [('train',train_eval),('val',loaders['val'])]:
        results[role],payload=evaluate(model,loader,device,cfg,class_weights,capture=True)
        export_payload(out,role,payload,data.class_names)
    assert all(sha256(p)==h for p,h in hashes.items()), 'Source changed during run'
    result=dict(model=name,supporting_profile=profile,seed=seed,selection='validation_macro_f1',best_epoch=best_epoch+1,
                epochs_completed=epoch+1,stop_reason=reason,elapsed_seconds=time.monotonic()-start,**results,
                test='NOT_EVALUATED',checkpoint_sha256=sha256(out/'best_validation.pt'),split_sha256=protocol['split_sha256'],
                parameter_count=sum(p.numel() for p in model.parameters()),gradient_norms_first_epoch=gradient_norms,
                source_sha256=hashes,device=device_name,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    dump(out/'result.json',result)
    print('COMPLETE '+json.dumps({k:result[k] for k in ('model','supporting_profile','seed','best_epoch','epochs_completed','elapsed_seconds','test')}),flush=True)


def lock_selection(paths,evidence):
    lock=RUN/'selection_lock.json'
    if lock.exists():raise FileExistsError(lock)
    evidence=Path(evidence).resolve()
    if not evidence.is_file():raise ValueError('Validation-based selection rationale required')
    entries=[]
    for path in paths:
        p=Path(path).resolve()
        if not p.is_relative_to(RUN/'experiments'):raise ValueError('Not a development experiment')
        result=json.loads((p/'result.json').read_text())
        if result['test']!='NOT_EVALUATED':raise ValueError('Test was accessed during development')
        entries.append(dict(experiment=str(p),checkpoint_sha256=sha256(p/'best_validation.pt'),
                            model=result['model'],seed=result['seed'],supporting_profile=result['supporting_profile']))
    dump(lock,dict(created_utc=datetime.now(timezone.utc).isoformat(),evidence=str(evidence),
                   evidence_sha256=sha256(evidence),experiments=entries,decision='No model changes permitted after this lock'))


def evaluate_test(path,device_name):
    lock=json.loads((RUN/'selection_lock.json').read_text());path=Path(path).resolve()
    entry=next((r for r in lock['experiments'] if r['experiment']==str(path)),None)
    if entry is None or sha256(path/'best_validation.pt')!=entry['checkpoint_sha256']:
        raise ValueError('Checkpoint is not in frozen final selection')
    if (path/'test_result.json').exists():raise FileExistsError('Final test already evaluated')
    cfg,protocol,data,split=load_context();torch.set_num_threads(2);device=torch.device(device_name)
    ck=torch.load(path/'best_validation.pt',map_location='cpu',weights_only=True)
    if any(sha256(p)!=h for p,h in ck['source_sha256'].items()):raise ValueError('Trained source changed')
    encoder=MetadataEncoder.from_state_dict(ck['metadata_encoder']);seed_all(ck['seed'])
    model,cfg=build_model(ck['model_name'],ck['config'],data,encoder,ck['spec']);model.to(device);model.load_state_dict(ck['model'])
    loader=DataLoader(FoldDataset(data,split['test'],encoder),batch_size=cfg['params']['batch_size'],shuffle=False)
    weights=None if ck['class_weights'] is None else ck['class_weights'].to(device)
    metrics,payload=evaluate(model,loader,device,cfg,weights,capture=True)
    export_payload(path,'test',payload,data.class_names)
    by_source={}
    for source in sorted(set(payload['source_datasets'])):
        mask=np.array([x==source for x in payload['source_datasets']])
        by_source[source]=classification_metrics(payload['truth'][mask].numpy(),payload['probability'][mask].numpy(),data.num_classes)
    result=dict(model=ck['model_name'],seed=ck['seed'],test=metrics,by_source=by_source,
                checkpoint_sha256=entry['checkpoint_sha256'],selection_lock_sha256=sha256(RUN/'selection_lock.json'))
    dump(path/'test_result.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--config',required=True)
    p=sub.add_parser('train');p.add_argument('--model',required=True);p.add_argument('--profile',default='unweighted')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-epochs',type=int,default=0)
    p=sub.add_parser('lock');p.add_argument('--experiments',nargs='+',required=True);p.add_argument('--evidence',required=True)
    p=sub.add_parser('test');p.add_argument('--experiment',required=True);p.add_argument('--device',default='cuda:0')
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.config)
    elif a.command=='train':train(a.model,a.profile,a.seed,a.device,a.smoke_epochs)
    elif a.command=='lock':lock_selection(a.experiments,a.evidence)
    else:evaluate_test(a.experiment,a.device)


if __name__=='__main__':main()
