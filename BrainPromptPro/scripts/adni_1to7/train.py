"""Seven-batch TRAIN/VAL runner with epoch-boundary resume; never creates a TEST loader."""
import argparse,copy,inspect,json,os,random,shutil,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from BrainPromptPro.data.prepared_cohort import PreparedCohortDataset
from BrainPromptPro.data.brainnet_pro import FoldDataset
from BrainPromptPro.data.metadata import MetadataEncoder,record_id
from BrainPromptPro.nets.connectome_p2p3p5 import ConnectomeBrainPrompt,reference_features
from BrainPromptPro.layers.masked_connectome import symmetric_fc_mask,reconstruction_loss,reconstruction_metrics,symmetric_kl
from BrainPromptPro.metrics import classification_metrics
from BrainPromptPro.research_runner import seed_all,selection_key
from BrainPromptPro.scripts.adni_1to7.audit_dataset import sha,write_json


def save(path,value):
    temp=path.with_suffix('.tmp.pt');torch.save(value,temp);temp.replace(path)


def inputs(batch,device):
    x={k:batch[k].to(device) for k in ['fc','subject_prompt','age','sex','site']};x['sample_ids']=batch['subject_id'];return x


def forward(model,x):
    if not getattr(model,'accepts_sample_ids',False):x={k:v for k,v in x.items() if k!='sample_ids'}
    return model(**x)


def ce(output,y,weights):
    return F.cross_entropy(output['logits'],y,weight=weights)+F.cross_entropy(output['disease_logits'],y,weight=weights)


@torch.no_grad()
def evaluate(model,loader,device,weights):
    model.eval();ps=[];ys=[];ids=[];loss=0.;n=0;t=time.monotonic()
    for batch in loader:
        x=inputs(batch,device);y=batch['label'].to(device);out=forward(model,x)
        value=ce(out,y,weights)
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite validation')
        loss+=float(value)*len(y);n+=len(y);ps.append(out['logits'].softmax(-1).cpu());ys.append(y.cpu());ids+=list(batch['subject_id'])
    prob=torch.cat(ps);truth=torch.cat(ys);m=classification_metrics(truth.numpy(),prob.numpy(),5)
    distribution=torch.bincount(prob.argmax(-1),minlength=5).tolist()
    m.update(loss=loss/n,true_class_distribution=torch.bincount(truth,minlength=5).tolist(),predicted_class_distribution=distribution,
        mean_class_probability=prob.mean(0).tolist(),prediction_entropy=float(-(prob*prob.clamp_min(1e-12).log()).sum(-1).mean()),
        majority_class_baseline=float(torch.bincount(truth,minlength=5).max()/n),classification_collapse='CLASSIFICATION_COLLAPSE' if max(distribution)==n else False,
        inference_seconds=time.monotonic()-t,auroc_unavailable_reason=None if m['roc_auc_macro'] is not None else 'One or more classes absent')
    return m,dict(probability=prob,truth=truth,sample_ids=ids)


def snapshot(model,out):
    root=Path(__file__).resolve().parents[3]
    paths=set(root.joinpath('BrainPromptPro/scripts/adni_1to7').glob('*.py'))
    paths.update(root.joinpath('BrainPromptPro/layers').glob('*.py'))
    paths.update(root.joinpath('BrainPromptPro/data').glob('*.py'))
    paths.update(root.joinpath('BrainPromptPro/nets').glob('*.py'))
    paths.update([root/'BrainPromptPro/metrics.py',root/'BrainPromptPro/research_runner.py',root/'BrainPromptPro/runtime.py'])
    for m in model.modules():
        f=Path(inspect.getfile(type(m))).resolve()
        if str(f).startswith('/home/panyao/AD/BrainPrompt/'):paths.add(f)
    hashes={str(p):sha(p) for p in sorted(paths)}
    for p in paths:
        relative=p.relative_to(root) if p.is_relative_to(root) else Path('original_baseline')/p.relative_to('/home/panyao/AD/BrainPrompt')
        dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write_json(out/'source_sha256.json',hashes);return hashes


def rng_state(loader,mask):
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],python=random.getstate(),
        numpy=dict(name=n[0],keys=torch.from_numpy(n[1].astype(np.int64)),pos=n[2],has_gauss=n[3],cached=n[4]),loader=loader.get_state(),mask=mask.get_state())


def restore_rng(s,loader,mask):
    torch.set_rng_state(s['torch']);random.setstate(s['python'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])
    n=s['numpy'];np.random.set_state((n['name'],n['keys'].numpy().astype(np.uint32),n['pos'],n['has_gauss'],n['cached']))
    loader.set_state(s['loader']);mask.set_state(s['mask'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--experiment',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--split',choices=['0','1'],default='0',help='0: frozen sample split; 1: independent subject split')
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke-epochs',type=int,default=0)
    ap.add_argument('--initial-checkpoint',type=Path);ap.add_argument('--reference-checkpoint',type=Path);ap.add_argument('--stop-after-epoch',type=int,default=0,help='Controlled checkpoint/resume verification; leaves no result.json')
    a=ap.parse_args();cfg=json.loads(a.config.read_text());study=cfg['study'];spec=copy.deepcopy(study['experiments'][a.experiment])
    root=Path(study['output_root']).resolve()
    if not a.output.resolve().is_relative_to(root):raise ValueError('Output must be inside isolated study namespace')
    if (root/'selection_lock.json').exists():raise RuntimeError('New study already locked for final test')
    if a.seed not in [42,3407,2026]:raise ValueError('Unregistered training seed')
    if spec.get('training_mode')=='finetune' and not a.initial_checkpoint:raise ValueError('Finetuning requires a registered pretraining checkpoint')
    if a.initial_checkpoint and spec.get('stage','supervised')!='supervised':raise ValueError('Initialization checkpoint is only for supervised training')
    device=torch.device(a.device);torch.set_num_threads(2);seed_all(a.seed)
    dataset=PreparedCohortDataset(cfg['data']);protocol=json.loads(Path(study['protocol']).read_text())
    splitpath=Path(study['sample_split'] if a.split=='0' else study['subject_split']);split=json.loads(splitpath.read_text())
    splithash=sha(splitpath);assert splithash==protocol['split_hash' if a.split=='0' else 'subject_split_hash']
    metadata=MetadataEncoder().fit([dataset.records[i] for i in split['train']])
    optconfig=cfg['params'];options=spec['modules'];bank=teacher=None
    if options.get('use_p5'):
        if not a.reference_checkpoint:raise ValueError('P5 requires same-seed frozen A0 encoder checkpoint')
        ck=torch.load(a.reference_checkpoint,map_location='cpu',weights_only=True)
        if ck['experiment']!='A0' or ck['seed']!=a.seed or ck['split_hash']!=splithash or ck['dataset_hash']!=cfg['data']['bundle_sha256']:raise ValueError('Reference baseline provenance mismatch')
        if not a.smoke_epochs and ck['signature']['smoke_epochs']:raise ValueError('Formal P5 cannot use a smoke reference checkpoint')
        teacher=ConnectomeBrainPrompt(cfg,dataset.roi_prompt,dataset.disease_prompt,{}).to(device);teacher.load_state_dict(ck['model']);teacher.eval()
        feats=[];reliability=[]
        for ids in torch.tensor(split['train']).split(20):
            q,r,_=reference_features(teacher,dataset.fc[ids].to(device),options.get('reliability_temperature',.05));feats.append(q);reliability.append(r)
        ids=[record_id(dataset.records[i]) for i in split['train']]
        bank=dict(representation=torch.cat(feats),reliability=torch.cat(reliability),reference_ids=ids,train_ids=ids,encoder_version=sha(a.reference_checkpoint))
    # Keep shared baseline initialization independent of optional reference work.
    seed_all(a.seed)
    if spec.get('architecture'):
        from BrainPromptPro.nets.adni1to7_controls import build_control
        model=build_control(spec['architecture'],cfg,dataset,metadata,split['train']).to(device)
    else:model=ConnectomeBrainPrompt(cfg,dataset.roi_prompt,dataset.disease_prompt,options,bank,teacher).to(device)
    if a.initial_checkpoint:
        ck=torch.load(a.initial_checkpoint,map_location='cpu',weights_only=True)
        if ck['seed']!=a.seed or ck['split_hash']!=splithash or ck['dataset_hash']!=cfg['data']['bundle_sha256'] or ck['stage']!='pretrain':raise ValueError('Pretraining provenance mismatch')
        if not a.smoke_epochs and ck['signature']['smoke_epochs']:raise ValueError('Formal finetuning cannot use a smoke pretraining checkpoint')
        model.base.original.load_state_dict({k.removeprefix('base.original.'):v for k,v in ck['model'].items() if k.startswith('base.original.')})
        if model.use_p2:
            for name in ['visibility_projection','decoder']:
                getattr(model,name).load_state_dict({k[len(name)+1:]:v for k,v in ck['model'].items() if k.startswith(name+'.')})
    stage=spec.get('stage','supervised');epochs=a.smoke_epochs or (spec.get('epochs',40) if stage=='pretrain' else optconfig['epochs'])
    if 'pretrain_dropout' in spec:
        if stage!='pretrain' or not 0<=spec['pretrain_dropout']<1:raise ValueError('Invalid pretraining-only dropout override')
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=spec['pretrain_dropout']
    best_path=a.output/('best_training.pt' if stage=='pretrain' else 'best_validation.pt')
    if stage=='pretrain' and not model.use_p2:raise ValueError('Pretraining needs P2')
    if a.output.exists() and not a.resume:raise FileExistsError('Output exists; do not overwrite')
    if a.resume and (a.output/'result.json').exists():raise ValueError('Already complete')
    if not a.resume:a.output.mkdir(parents=True,exist_ok=False)
    g=torch.Generator().manual_seed(a.seed+2000003);mg=torch.Generator(device=device).manual_seed(a.seed+1000003)
    loaders={r:DataLoader(FoldDataset(dataset,split[r],metadata),batch_size=optconfig['batch_size'],shuffle=r=='train',generator=g if r=='train' else None,num_workers=0) for r in ['train','val']}
    counts=torch.bincount(dataset.labels[split['train']],minlength=5).float();weights=counts.pow(-.5);weights=(weights/weights.mean()).to(device)
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=spec.get('lr',optconfig['init_lr']),weight_decay=optconfig['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=25)
    best=None;best_epoch=-1;first=0;elapsed=0.;gradients={}
    sources=snapshot(model,a.output) if not a.resume else json.loads((a.output/'source_sha256.json').read_text())
    signature=dict(config_sha256=sha(a.config),experiment=a.experiment,seed=a.seed,split_hash=splithash,dataset_hash=cfg['data']['bundle_sha256'],stage=stage,
        initial_sha256=sha(a.initial_checkpoint) if a.initial_checkpoint else None,reference_sha256=sha(a.reference_checkpoint) if a.reference_checkpoint else None,smoke_epochs=a.smoke_epochs)
    if a.resume:
        ck=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True)
        assert ck['signature']==signature and all(sha(p)==v for p,v in sources.items()),'Resume source/config mismatch'
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);restore_rng(ck['rng'],g,mg)
        best=tuple(ck['best_key']);best_epoch=ck['best_epoch'];first=ck['epoch']+1;elapsed=ck['elapsed_seconds']
    else:
        write_json(a.output/'git_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())))
        write_json(a.output/'config.json',cfg);write_json(a.output/'spec.json',spec);write_json(a.output/'signature.json',signature)
    start=time.monotonic()
    for epoch in range(first,epochs):
        model.train();totals=[];recstats=[]
        for batch in loaders['train']:
            x=inputs(batch,device);y=batch['label'].to(device);opt.zero_grad(set_to_none=True)
            p2=model.use_p2 and (stage=='pretrain' or spec.get('training_mode','joint')=='joint')
            if p2:
                masked,vis=symmetric_fc_mask(x['fc'],options.get('mask_ratio',.1),mg)
                if stage=='pretrain':
                    h,_=model.encode_roi(masked,vis);pred=model.decoder(h);loss=reconstruction_loss(pred,x['fc'],vis)
                    if options.get('consistency_weight',0):
                        m2,v2=symmetric_fc_mask(x['fc'],options.get('mask_ratio',.1),mg);h2,_=model.encode_roi(m2,v2)
                        loss=loss+options['consistency_weight']*F.mse_loss(F.normalize(h.mean(1),dim=-1),F.normalize(h2.mean(1),dim=-1))
                else:
                    clean=forward(model,x);out=model(**dict(x,fc=masked,visible=vis,reconstruct=True));pred=out['diagnostics']['reconstruction']
                    loss=ce(clean,y,weights)+options.get('reconstruction_weight',.1)*reconstruction_loss(pred,x['fc'],vis)
                    if options.get('consistency_weight',.1):
                        m2,v2=symmetric_fc_mask(x['fc'],options.get('mask_ratio',.1),mg);other=model(**dict(x,fc=m2,visible=v2))
                        loss=loss+options.get('consistency_weight',.1)*symmetric_kl(out['logits'],other['logits'])
                recstats.append(reconstruction_metrics(pred,x['fc'],vis))
            else:loss=ce(forward(model,x),y,weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train loss')
            loss.backward()
            if epoch==0:
                for name,p in model.named_parameters():
                    if p.grad is not None:gradients[name]=max(gradients.get(name,0),float(p.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();totals.append(float(loss))
        if stage=='pretrain':
            # Fixed epoch pretraining; all optimization examples are TRAIN.
            val=dict(loss=float(np.mean(totals)),masked_train_metrics={k:float(np.mean([x[k] for x in recstats if x[k] is not None])) for k in ['masked_mse','positive_mse','negative_mse','correlation'] if any(x[k] is not None for x in recstats)})
            key=(-val['loss'],);reason='epoch_cap' if epoch+1==epochs else None
        else:
            val,_=evaluate(model,loaders['val'],device,weights);key=selection_key(val);scheduler.step(val['loss'])
            reason='lr_floor' if opt.param_groups[0]['lr']<1e-6 else 'epoch_cap' if epoch+1==epochs else None
        improved=best is None or key>best
        if improved:best=key;best_epoch=epoch
        if stage!='pretrain' and not reason and epoch+1>=optconfig['min_epochs'] and epoch-best_epoch>=optconfig['early_stopping_patience']:reason='validation_patience'
        duration=elapsed+time.monotonic()-start
        ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(g,mg),signature=signature,config=cfg,spec=spec,
            experiment=a.experiment,seed=a.seed,stage=stage,epoch=epoch,best_key=list(best),best_epoch=best_epoch,elapsed_seconds=duration,
            split_hash=splithash,dataset_hash=cfg['data']['bundle_sha256'],validation=val,metadata_encoder=metadata.state_dict(),source_sha256=sources)
        save(a.output/'last.pt',ck)
        if improved:save(best_path,ck)
        entry=dict(epoch=epoch+1,train_loss=float(np.mean(totals)),validation=val,best_epoch=best_epoch+1,elapsed_seconds=duration,stop_reason=reason)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        write_json(a.output/'status.json',entry)
        if epoch==0 or (epoch+1)%20==0 or reason:print(json.dumps(dict(experiment=a.experiment,seed=a.seed,epoch=epoch+1,val_f1=val.get('f1_macro'),loss=val['loss'],seconds=duration,stop=reason)),flush=True)
        if a.stop_after_epoch and epoch+1==a.stop_after_epoch and not reason:
            print('PAUSED_AT_EPOCH_BOUNDARY',epoch+1,flush=True);return
        if reason:break
    assert all(sha(p)==h for p,h in sources.items()),'Source changed during run'
    ck=torch.load(best_path,map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
    result=dict(signature,best_epoch=best_epoch+1,epochs_completed=epoch+1,stop_reason=reason,elapsed_seconds=elapsed+time.monotonic()-start,
        parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint_sha256=sha(best_path),test='NOT_EVALUATED',gradients_first_epoch=gradients,source_sha256=sources,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if stage=='pretrain':result['reconstruction']=ck['validation']
    else:
        result['val'],payload=evaluate(model,loaders['val'],device,weights);save(a.output/'val_payload.private.pt',payload)
    write_json(a.output/'result.json',result);print('COMPLETE',a.experiment,a.seed,flush=True)

if __name__=='__main__':main()
