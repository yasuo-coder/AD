"""Train/validation-only optimization; freeze per-fold choices before test.

Keeps the existing official scan splits and last-epoch/LR-floor schedule.
Candidate selection uses validation macro F1, then validation macro OvR AUC.
The previous Pro last checkpoint is retained as a validation-only fallback.
"""
import argparse,copy,csv,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data.data import LoadData
from .data.brainnet_pro import FoldDataset,sha256
from .data.metadata import MetadataEncoder,record_id
from .losses import compute_losses
from .metrics import classification_metrics
from .nets.load_net import gnn_model
from .official_cv import audit_splits,dump,source_hashes,stop_reason
from .runtime import ROOT,output_path
from .train_TUs_graph_classification_llm import model_inputs

REFERENCE=ROOT/'outputs/comparisons/adni123_official_scan10_last_seed41_20261005_v2'
CANDIDATES=('stable','stable_weighted')


def experiment(run_id):
    if Path(run_id).name!=run_id or run_id in ('.','..'):raise ValueError('Invalid run ID')
    return output_path('optimizations',run_id)


def hashes():
    return dict(source_hashes(),**{str(ROOT/'optimization_cv.py'):sha256(ROOT/'optimization_cv.py')})


def class_weights(train_labels,num_classes,power):
    counts=torch.bincount(torch.as_tensor(train_labels),minlength=num_classes).float()
    if (counts==0).any():raise ValueError('All classes must be present in TRAIN')
    weights=counts.pow(-power)
    return weights/(weights*counts/counts.sum()).sum()


def optimization_losses(output,labels,weights,model_config,class_weight=None,smoothing=0.):
    values=compute_losses(output,labels,weights,model_config)
    cls=F.cross_entropy(output['logits'],labels,weight=class_weight,label_smoothing=smoothing)
    disease=F.cross_entropy(output['disease_logits'],labels,weight=class_weight,label_smoothing=smoothing)
    values['total']=values['total']-values['cls']-weights['lambda_disease']*values['disease']+cls+weights['lambda_disease']*disease
    values.update(cls=cls,disease=disease)
    return values


def pass_epoch(model,loader,device,loss_config,class_weight=None,smoothing=0.,optimizer=None,capture=False):
    model.train(optimizer is not None);totals={};correct=count=nb=0;ps=[];ys=[];ids=[]
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            y=batch['label'].to(device)
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            output=model(**model_inputs(batch,device))
            losses=optimization_losses(output,y,loss_config,model.config,class_weight,smoothing)
            if not torch.isfinite(losses['total']):raise FloatingPointError('Nonfinite loss')
            if optimizer is not None:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
            for k,v in losses.items():totals[k]=totals.get(k,0.)+v.detach().item()
            correct+=int((output['logits'].argmax(-1)==y).sum());count+=len(y);nb+=1
            if capture:ps.append(output['logits'].detach().softmax(-1).cpu());ys.append(y.cpu());ids.extend(batch['subject_id'])
    if not nb:raise ValueError('Empty loader')
    result={k:v/nb for k,v in totals.items()};result.update(loss=result['total'],accuracy=correct/count,samples=count)
    payload=dict(probability=torch.cat(ps),truth=torch.cat(ys),ids=ids) if capture else None
    if capture:result.update(classification_metrics(payload['truth'].numpy(),payload['probability'].numpy(),model.num_classes))
    return result,payload


def prepare(run_id):
    root=experiment(run_id);root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():raise FileExistsError('Plan already frozen')
    base=json.loads((REFERENCE/'config.json').read_text())
    config=copy.deepcopy(base)
    config['net_params'].update(stability_profile='stable_v1',hidden_dim=64,L=2,dropout=.25,
        brain_edge_hidden_dim=32,disease_attention_dim=64,relation_hidden_dim=32,batch_norm=False)
    config['loss']['lambda_disease']=.2
    config['params'].update(label_smoothing=.05,gradient_clip_norm=5.)
    dump(root/'config.json',config)
    (root/'splits.json').write_bytes((REFERENCE/'splits.json').read_bytes())
    (root/'split_audit.json').write_bytes((REFERENCE/'split_audit.json').read_bytes())
    dump(root/'source_sha256.json',hashes())
    dump(root/'plan.json',dict(reference=str(REFERENCE),candidates=list(CANDIDATES),
        selection='Per fold: maximum validation macro F1; tie-break validation macro OvR AUC, then deterministic candidate name.',
        fallback='Previous BrainPromptPro last checkpoint; evaluated on validation only for selection.',
        class_weight_power=dict(stable=0.,stable_weighted=.5),class_weights_fit='TRAIN only',
        epochs='Last epoch, no restoration of a validation-best epoch; cap 1000, LR < 1e-6, 6 hours/fold.',
        outer_split='Same scan-level ten folds as reference; same known subject overlap.',
        test_policy='No test loader or test monitoring during fitting. Select all 10 candidates before any new test evaluation.',
        limitations='Previously inspected cohort; exploratory comparison, not a new independent clinical validation.',
        lr=5e-5,weight_decay=.005,batch_size=20,seed=41,dropout=.25,hidden_dim=64,L=2,
        lambda_disease=.2,label_smoothing=.05,prior='continuous |FC|^2*(0.25+0.75*learned_edge)',
        architecture='LayerNorm; FC residual; disease readout residual; normalized/gated metadata and population branches; relation-specific priors',
        original_brainprompt_modified=False))
    print(root,flush=True)


def save_predictions(path,payload,indices,data):
    if payload['ids']!=[record_id(data.records[i]) for i in indices]:raise ValueError('Prediction ID mismatch')
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['scan_index','subject_id','true_label','predicted_label']+['p_'+c for c in data.class_names])
        for j,i in enumerate(indices):
            p=payload['probability'][j];writer.writerow([i,payload['ids'][j],int(payload['truth'][j]),int(p.argmax())]+p.tolist())


def train(run_id,candidate,fold,smoke=False):
    root=experiment(run_id)
    if hashes()!=json.loads((root/'source_sha256.json').read_text()):raise ValueError('Source changed after plan freeze')
    config=json.loads((root/'config.json').read_text());params=config['params']
    if smoke:params['epochs']=3
    torch.set_num_threads(params['threads']);data=LoadData(config['data'])
    folds=json.loads((root/'splits.json').read_text());audit_splits(folds,data.records);split=folds[fold]
    out=(root/'smoke' if smoke else root)/candidate/f'fold_{fold}';out.mkdir(parents=True,exist_ok=False)
    seed=params['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(False);device=torch.device('cuda:0')
    encoder=MetadataEncoder().fit([data.records[i] for i in split['train']],config['data']['site_column'],config['data']['missing_site'])
    model=gnn_model('BrainPromptPro',config['net_params'],data.roi_prompt,data.disease_prompt,2048,len(encoder.site_mapping)+1).to(device)
    power=0. if candidate=='stable' else .5
    weights=class_weights(data.labels[split['train']],5,power).to(device) if power else None
    config['params'].update(class_weight_power=power,class_weights=None if weights is None else weights.tolist())
    dump(out/'config.json',config)
    # Deliberately instantiate TRAIN and VAL only here.
    loaders={k:DataLoader(FoldDataset(data,split[k],encoder),batch_size=20,shuffle=k=='train',num_workers=0) for k in ('train','val')}
    opt=torch.optim.Adam(model.parameters(),lr=params['init_lr'],weight_decay=params['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=params['lr_reduce_factor'],patience=params['lr_schedule_patience'])
    start=time.monotonic();history=[]
    for epoch in range(params['epochs']):
        lr=opt.param_groups[0]['lr']
        training,_=pass_epoch(model,loaders['train'],device,config['loss'],weights,params['label_smoothing'],opt)
        validation,_=pass_epoch(model,loaders['val'],device,config['loss'],weights,params['label_smoothing'])
        scheduler.step(validation['loss']);elapsed=time.monotonic()-start
        reason=stop_reason(epoch,opt.param_groups[0]['lr'],elapsed,params)
        entry=dict(epoch=epoch,epoch_number=epoch+1,train=training,validation=validation,lr=lr,next_lr=opt.param_groups[0]['lr'],elapsed_seconds=elapsed,stop_reason=reason)
        history.append(entry)
        with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
        ck=dict(model=model.state_dict(),config=config,metadata_encoder=encoder.state_dict(),split=split,fold=fold,
            last_epoch=epoch,epochs_completed=epoch+1,checkpoint_selection='last_epoch',optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),stop_reason=reason,provenance=data.provenance,
            training_indices_used=split['train'],split_sha256=sha256(root/'splits.json'))
        torch.save(ck,out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt');dump(out/'status.json',entry)
        if epoch==0 or (epoch+1)%40==0 or reason:print(json.dumps(dict(candidate=candidate,fold=fold,epoch=epoch+1,lr=lr,train_loss=training['loss'],val_loss=validation['loss'],elapsed=elapsed,stop_reason=reason)),flush=True)
        if reason:break
    checkpoint=torch.load(out/'last.pt',weights_only=True,map_location=device);model.load_state_dict(checkpoint['model'])
    if checkpoint['last_epoch']!=history[-1]['epoch']:raise ValueError('Not last checkpoint')
    final_train,_=pass_epoch(model,loaders['train'],device,config['loss'],weights,params['label_smoothing'],capture=True)
    validation,payload=pass_epoch(model,loaders['val'],device,config['loss'],weights,params['label_smoothing'],capture=True)
    save_predictions(out/'validation_predictions.csv',payload,split['val'],data)
    dump(out/'validation_result.json',dict(candidate=candidate,fold=fold,validation=validation,train=final_train,
         epochs_completed=epoch+1,last_epoch=epoch,checkpoint_selection='last_epoch',checkpoint=str(out/'last.pt'),
         checkpoint_sha256=sha256(out/'last.pt'),stop_reason=reason,final_lr=opt.param_groups[0]['lr'],
         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
         elapsed_seconds=time.monotonic()-start,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),smoke=smoke))


def legacy_validation(run_id):
    root=experiment(run_id);torch.set_num_threads(2)
    config=json.loads((REFERENCE/'config.json').read_text());data=LoadData(config['data'])
    for fold in range(10):
        path=REFERENCE/'BrainPromptPro'/f'fold_{fold}'/'last.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
        encoder=MetadataEncoder.from_state_dict(ck['metadata_encoder'])
        model=gnn_model('BrainPromptPro',ck['config']['net_params'],data.roi_prompt,data.disease_prompt,2048,len(encoder.site_mapping)+1).to('cuda:0');model.load_state_dict(ck['model'])
        loader=DataLoader(FoldDataset(data,ck['split']['val'],encoder),batch_size=20,shuffle=False)
        val,payload=pass_epoch(model,loader,'cuda:0',ck['config']['loss'],capture=True)
        out=root/'legacy'/f'fold_{fold}';out.mkdir(parents=True,exist_ok=False)
        save_predictions(out/'validation_predictions.csv',payload,ck['split']['val'],data)
        dump(out/'validation_result.json',dict(candidate='legacy',fold=fold,validation=val,checkpoint=str(path),
             checkpoint_sha256=sha256(path),last_epoch=ck['last_epoch'],epochs_completed=ck['epochs_completed'],checkpoint_selection='last_epoch'))


def select(run_id):
    root=experiment(run_id)
    if (root/'selection.json').exists():raise FileExistsError('Selection already locked')
    selected=[]
    for fold in range(10):
        options=[json.loads((root/c/f'fold_{fold}'/'validation_result.json').read_text()) for c in (*CANDIDATES,'legacy')]
        for o in options:
            if sha256(o['checkpoint'])!=o['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        choice=max(options,key=lambda r:(r['validation']['f1_macro'],r['validation']['roc_auc_macro'],r['candidate']))
        selected.append(dict(fold=fold,candidate=choice['candidate'],checkpoint=choice['checkpoint'],checkpoint_sha256=choice['checkpoint_sha256'],
            validation_f1=choice['validation']['f1_macro'],validation_auc=choice['validation']['roc_auc_macro'],
            last_epoch=choice['last_epoch'],options=[dict(candidate=o['candidate'],f1=o['validation']['f1_macro'],auc=o['validation']['roc_auc_macro']) for o in options]))
    dump(root/'selection.json',dict(rule=json.loads((root/'plan.json').read_text())['selection'],folds=selected,locked_before_new_test_evaluation=True))
    print(json.dumps(selected,indent=2),flush=True)


def evaluate_selected(run_id):
    root=experiment(run_id);torch.set_num_threads(2)
    selection=json.loads((root/'selection.json').read_text());selection_hash=sha256(root/'selection.json')
    config=json.loads((root/'config.json').read_text());data=LoadData(config['data'])
    if hashes()!=json.loads((root/'source_sha256.json').read_text()):raise ValueError('Source changed before final test')
    for row in selection['folds']:
        if sha256(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Selected model changed')
        ck=torch.load(row['checkpoint'],weights_only=True,map_location='cpu');encoder=MetadataEncoder.from_state_dict(ck['metadata_encoder'])
        model=gnn_model('BrainPromptPro',ck['config']['net_params'],data.roi_prompt,data.disease_prompt,2048,len(encoder.site_mapping)+1).to('cuda:0');model.load_state_dict(ck['model'])
        loader=DataLoader(FoldDataset(data,ck['split']['test'],encoder),batch_size=20,shuffle=False)
        weights=ck['config']['params'].get('class_weights');weights=torch.tensor(weights,device='cuda:0') if weights else None
        test,payload=pass_epoch(model,loader,'cuda:0',ck['config']['loss'],weights,ck['config']['params'].get('label_smoothing',0.),capture=True)
        out=root/'selected'/f"fold_{row['fold']}";out.mkdir(parents=True,exist_ok=False)
        save_predictions(out/'predictions.csv',payload,ck['split']['test'],data)
        dump(out/'result.json',dict(fold=row['fold'],candidate=row['candidate'],test=test,selection_sha256=selection_hash,
             checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],last_epoch=ck['last_epoch'],
             checkpoint_selection='last_epoch',epochs_completed=ck['epochs_completed']))
    print('All ten selected last checkpoints evaluated.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','legacy-validation','select','evaluate'])
    p.add_argument('--run-id',required=True);p.add_argument('--candidate',choices=CANDIDATES)
    p.add_argument('--fold',type=int,choices=range(10));p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare(a.run_id)
    elif a.command=='train':
        if a.candidate is None or a.fold is None:p.error('train requires candidate and fold')
        train(a.run_id,a.candidate,a.fold,a.smoke)
    elif a.command=='legacy-validation':legacy_validation(a.run_id)
    elif a.command=='select':select(a.run_id)
    else:evaluate_selected(a.run_id)


if __name__=='__main__':main()
