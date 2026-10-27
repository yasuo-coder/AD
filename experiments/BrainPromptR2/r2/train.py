"""One isolated inner fit. Does not contain an outer evaluator or historical paths."""
import argparse,copy,time,warnings,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .common import *
from .features import fit_features
from .models import BrainPromptG,NumericModel

def rank(m):return (round(m['macro_nll'],8),-round(m['macro_ap'],8),-round(m['macro_auroc'],8))
class MinimalG(BrainPromptG):
    def __init__(self,data,condition):
        super().__init__(dict(pop_fusion='off',layers=2,hidden_dim=64,dropout=.1,batch_norm=False),data)
        self.use_patient=condition not in ('no_prompt','roi_only')
        if condition=='no_prompt':self.core.pos_emb=torch.nn.Identity()
    def forward_batch(self,data,ids,device,**kwargs):
        h=self._encode(data['fc'][ids].to(device))
        if self.use_patient:h=h+self.core.linear_layer1(data['subject_prompt'][ids].to(device))
        return {'logits':self.core.MLP_layer(h)}

@torch.no_grad()
def predict(model,data,ids,device,bs=32):
    model.eval();parts=[model.forward_batch(data,ids[j:j+bs],device)['logits'].double().cpu() for j in range(0,len(ids),bs)]
    logits=torch.cat(parts).numpy();return logits,softmax(logits)

def package(data,ids,logits):
    p=softmax(logits);y,sp,sg=aggregate(data,ids,p)
    return dict(y=y,prob=sp,groups=sg,scan_logits=logits,scan_probability=p,scan_ids=np.array(ids),metrics=metric(y,sp),
      scale=dict(logit_min=float(logits.min()),logit_max=float(logits.max()),logit_std=float(logits.std()),
       probability_min=float(p.min()),probability_max=float(p.max()),mean_max_probability=float(sp.max(1).mean()),
       class_mean=sp.mean(0).tolist(),class_std=sp.std(0).tolist(),temperature=1.))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='/work/data');ap.add_argument('--out',default='/work/out')
    ap.add_argument('--task',default='/work/task.json');ap.add_argument('--smoke-epochs',type=int,default=0);args=ap.parse_args()
    start=time.time();folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    cfg=read(ROOT/'configs/study.json');task=read(args.task);tc=cfg['torch'];manifest=read(Path(args.data)/'manifest.json')
    seal=read(ROOT/'source_manifest.json')
    for rel,sha in seal['files'].items():
        if rel.startswith(('r2/','configs/','legacy/')):assert digest(ROOT/rel)==sha,f'Code changed: {rel}'
    assert digest(Path(args.data)/'data.private.pt')==manifest['data_sha256']
    assert digest(Path(args.data)/'plans.private.json')==manifest['plans_sha256']
    plans=read(Path(args.data)/'plans.private.json');plan=plans[task['plan']]
    data=torch.load(Path(args.data)/'data.private.pt',map_location='cpu',weights_only=True)
    group=np.asarray(data['groups']);parts=[set(group[plan[k]]) for k in ('train','stop','validation')]
    assert all(not(parts[i]&parts[j]) for i in range(3) for j in range(i))
    signature=objhash(dict(task=task,config=cfg,source=seal['source_sha256'],data=manifest['data_sha256'],splits=manifest['plans_sha256'],smoke=args.smoke_epochs))
    if (folder/'result.json').exists():assert read(folder/'result.json')['signature']==signature;return
    seed=plan['seed']+plan['fold']*101;seed_all(seed)
    train,stop,val=[plan[k] for k in ('train','stop','validation')]
    condition=task.get('condition','no_prompt');prior=subject_prior(data,train);train_groups=sorted(set(group[train]))
    bygroup={g:[i for i in train if group[i]==g] for g in train_groups}
    kind=task['kind'];trans=None;weights=None
    if task['stage']!='A':data,trans,weights=fit_features(data,train,condition)
    save(folder/'transforms.private.pt',trans)
    result=dict(signature=signature,task=task,repeat=plan['repeat'],fold=plan['fold'],seed=seed,
      unique_fit_prior=prior.tolist(),fit_subjects=len(train_groups),temperature=1.,diagnostic_only=bool(args.smoke_epochs))
    predictions={}
    if kind=='logistic':
        model=LogisticRegression(C=task['C'],solver='lbfgs',max_iter=cfg['logistic']['max_iter'],tol=cfg['logistic']['tol'],random_state=seed)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always');model.fit(data['x'][train].numpy(),data['y'][train].numpy(),sample_weight=weights)
        if any(issubclass(w.category,ConvergenceWarning) for w in captured):raise RuntimeError('Logistic did not converge; not a valid candidate')
        for split,ids in [('stop',stop),('validation',val)]:predictions[split+'_best']=package(data,ids,model.decision_function(data['x'][ids].numpy()))
        save(folder/'logistic.private.pt',dict(coef=model.coef_,intercept=model.intercept_,classes=model.classes_))
        result.update(epochs=None,selected_epoch=None,stop_reason='lbfgs_converged',iterations=model.n_iter_.tolist(),parameters=int(model.coef_.size+model.intercept_.size))
    else:
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if task['stage']=='A':model=BrainPromptG(dict(pop_fusion=kind,ordinal=False),data)
        elif kind=='mlp':model=NumericModel(data['x'].shape[1],'mlp')
        elif kind=='g_off':model=MinimalG(data,condition)
        else:raise ValueError(kind)
        model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=tc['lr_factor'],patience=tc['lr_patience'],threshold=tc['minimum_improvement'],threshold_mode='abs',min_lr=tc['lr_min'])
        history=[];best=None;control_best=float('inf');bad=0;floor=0;epoch=0;best_epoch=None
        if (folder/'last.private.pt').exists():
            state=torch.load(folder/'last.private.pt',map_location='cpu',weights_only=False);assert state['signature']==signature
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            history=state['history'];epoch=state['epoch'];best=state['best'];best_epoch=state['best_epoch'];control_best=state['control_best'];bad=state['bad'];floor=state['floor']
            random.setstate(state['random']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
            if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])
        while True:
            # A restored terminal checkpoint is finalized without an extra training epoch.
            if (floor>=tc['floor_epochs'] and bad>=tc['stop_bad_epochs']) or (args.smoke_epochs and epoch>=args.smoke_epochs):break
            if time.time()-start>tc['wall_guard_hours']*3600:raise RuntimeError('Wall guard: incomplete fit retained, no ranking')
            epoch+=1;model.train();total=0.;seen=0
            selected=np.array([np.random.choice(bygroup[g]) for g in train_groups]);np.random.shuffle(selected)
            for j in range(0,len(selected),tc['batch_size']):
                ids=selected[j:j+tc['batch_size']].tolist();optimizer.zero_grad(set_to_none=True)
                logits=model.forward_batch(data,ids,device)['logits'];loss=F.cross_entropy(logits,data['y'][ids].to(device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
            if hasattr(model,'refresh_reference'):model.refresh_reference(data,train,device,tc['batch_size'])
            stop_logits,_=predict(model,data,stop,device,tc['batch_size']);sm=package(data,stop,stop_logits)['metrics']
            if best is None or rank(sm)<rank(best):
                best=sm;best_epoch=epoch;save(folder/'best.private.pt',{'model':model.state_dict(),'epoch':epoch,'signature':signature})
            if sm['macro_nll']<control_best-tc['minimum_improvement']:control_best=sm['macro_nll'];bad=0
            else:bad+=1
            lr_used=float(optimizer.param_groups[0]['lr']);scheduler.step(sm['macro_nll']);lr_next=float(optimizer.param_groups[0]['lr'])
            floor=floor+1 if lr_used<=tc['lr_min']*(1+1e-8) else 0
            history.append(dict(epoch=epoch,train_ce=total/seen,stop=sm,lr_used=lr_used,lr_next=lr_next,bad_epochs=bad,floor_epochs=floor))
            state=dict(signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,epoch=epoch,best=best,best_epoch=best_epoch,control_best=control_best,bad=bad,floor=floor,
              random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            save(folder/'last.private.pt',state);write(folder/'history.json',history)
            print(json.dumps(dict(epoch=epoch,train_ce=total/seen,stop_macro_nll=sm['macro_nll'],lr=lr_used,best_epoch=best_epoch)),flush=True)
        for checkpoint in ['last','best']:
            state=torch.load(folder/(checkpoint+'.private.pt'),map_location=device,weights_only=False);model.load_state_dict(state['model'])
            if hasattr(model,'refresh_reference'):model.refresh_reference(data,train,device,tc['batch_size'])
            for split,ids in [('stop',stop),('validation',val)]:
                logits,_=predict(model,data,ids,device,tc['batch_size']);predictions[split+'_'+checkpoint]=package(data,ids,logits)
        result.update(epochs=epoch,selected_epoch=best_epoch,stop_reason='SMOKE_ONLY' if args.smoke_epochs else 'lr_floor_and_no_macro_nll_improvement',parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),history_sha256=digest(folder/'history.json'))
    result['metrics']={k:v['metrics'] for k,v in predictions.items()};result['scale']={k:v['scale'] for k,v in predictions.items()}
    result['seconds']=time.time()-start;result['status']='SMOKE_COMPLETE' if args.smoke_epochs else 'COMPLETE'
    save(folder/'predictions.private.pt',predictions);result['predictions_sha256']=digest(folder/'predictions.private.pt');write(folder/'result.json',result)
    print('COMPLETED',task,flush=True)
if __name__=='__main__':main()
