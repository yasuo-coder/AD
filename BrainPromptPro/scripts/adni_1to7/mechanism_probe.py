"""Read-only TRAIN/VAL probes using each run's immutable training imports.

Invoke this file directly, not with -m. No checkpoint is modified and no TEST
loader is constructed. Counterfactuals diagnose a fitted model, not a retrained
ablation. The output is private; export aggregate fields through a whitelist.
"""
import argparse
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--mask-robustness', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = json.loads((a.run / 'result.json').read_text())
    if result['stage'] != 'supervised' or result['smoke_epochs'] or result['test'] != 'NOT_EVALUATED':
        raise ValueError('Only completed formal TRAIN/VAL classification runs')
    paths = result['source_sha256']
    for path, expected in paths.items():
        if digest(path) != expected:
            raise ValueError('Changed immutable training source')
    model_paths = [Path(x) for x in paths if x.endswith('/BrainPromptPro/nets/connectome_p2p3p5.py')]
    if len(model_paths) != 1:
        raise ValueError('Ambiguous model source')
    sys.path.insert(0, str(model_paths[0].parents[2]))
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from BrainPromptPro.data.prepared_cohort import PreparedCohortDataset
    from BrainPromptPro.data.brainnet_pro import FoldDataset
    from BrainPromptPro.data.metadata import MetadataEncoder, record_id
    from BrainPromptPro.layers.masked_connectome import symmetric_fc_mask
    from BrainPromptPro.nets.connectome_p2p3p5 import ConnectomeBrainPrompt, reference_features
    from BrainPromptPro.metrics import classification_metrics
    from BrainPromptPro.scripts.adni_1to7.train import inputs, forward, ce
    if Path(inspect.getfile(ConnectomeBrainPrompt)).resolve() != model_paths[0].resolve():
        raise ValueError('Invoke directly to load the original immutable package')
    torch.set_num_threads(2)
    device = torch.device(a.device)
    cp = a.run / 'best_validation.pt'
    if digest(cp) != result['checkpoint_sha256']:
        raise ValueError('Changed selected checkpoint')
    ck = torch.load(cp, map_location='cpu', weights_only=True)
    cfg, spec = ck['config'], ck['spec']
    if spec.get('architecture'):
        raise ValueError('This probe is for A0/P2/P3/P5; historical controls use separate diagnostics')
    split_path = Path(cfg['study']['sample_split'])
    if digest(split_path) != result['split_hash']:
        raise ValueError('Probe currently supports only the registered sample split')
    split = json.loads(split_path.read_text())
    dataset = PreparedCohortDataset(cfg['data'])
    if ck['dataset_hash'] != result['dataset_hash'] or ck['signature'] != json.loads((a.run/'signature.json').read_text()):
        raise ValueError('Checkpoint signature mismatch')
    metadata = MetadataEncoder().fit([dataset.records[i] for i in split['train']])
    loader = DataLoader(FoldDataset(dataset, split['val'], metadata), batch_size=20, shuffle=False)
    bank = teacher = None
    options = spec['modules']
    if options.get('use_p5'):
        state = ck['model']; extra = state['population._extra_state']
        ids = [record_id(dataset.records[i]) for i in split['train']]
        if extra['reference_ids'] != ids or extra['encoder_version'] != result['reference_sha256']:
            raise ValueError('Saved bank membership/order/version mismatch')
        teacher = ConnectomeBrainPrompt(cfg, dataset.roi_prompt, dataset.disease_prompt, {}).to(device).eval()
        teacher.load_state_dict({k.removeprefix('reference_encoder.'): v for k,v in state.items() if k.startswith('reference_encoder.')})
        bank = dict(representation=state['population.reference'].to(device), reliability=state['population.reliability'].to(device),
                    reference_ids=ids, train_ids=ids, encoder_version=extra['encoder_version'])
    model = ConnectomeBrainPrompt(cfg, dataset.roi_prompt, dataset.disease_prompt, options, bank, teacher).to(device).eval()
    model.load_state_dict(ck['model'])
    report = dict(experiment=result['experiment'], seed=result['seed'], evaluation='TRAIN gradients; VAL diagnostics only',
                  test_accessed=False, checkpoint_sha256=digest(cp), dataset_hash=result['dataset_hash'], split_hash=result['split_hash'],
                  training_source_verified=True, device=str(device), probe_source_sha256=digest(__file__), counterfactual_warning='No retraining; not a causal efficacy estimate')
    def stats(value):
        value = value.detach().double().cpu().flatten()
        if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite diagnostic')
        return dict(mean=float(value.mean()), std=float(value.std(unbiased=False)), minimum=float(value.min()), maximum=float(value.max()))
    def class_stats(value, labels):
        return {name: stats(value[labels==i]) for i,name in enumerate(cfg['data']['class_names']) if (labels==i).any()}
    probability=[];truth=[];att=[];tokens=[];pops=[];reps=[];queries=[];qrels=[];qvars=[];qids=[]
    with torch.no_grad():
        for batch in loader:
            x=inputs(batch,device);out=forward(model,x);probability.append(out['logits'].softmax(-1).cpu());truth.append(batch['label'])
            reps.append(out['patient_embedding'].cpu())
            if model.use_p3 and out['diagnostics']['disease_tokens'] is not None:
                att.append(out['diagnostics']['roi_attention'].cpu());tokens.append(out['diagnostics']['disease_tokens'].cpu())
            if model.use_p5:
                pops.append({k:v.cpu() for k,v in out['diagnostics']['population'].items()})
                q,rel,var=reference_features(model.reference_encoder,x['fc'],options.get('reliability_temperature',.05))
                queries.append(q);qrels.append(rel);qvars.append(var);qids+=list(batch['subject_id'])
    prob=torch.cat(probability);labels=torch.cat(truth);rep=torch.cat(reps)
    saved=torch.load(a.run/'val_payload.private.pt',map_location='cpu',weights_only=True)
    if not torch.equal(labels,saved['truth']) or not torch.equal(prob.argmax(-1),saved['probability'].argmax(-1)) or not torch.allclose(prob,saved['probability'],atol=2e-5,rtol=2e-5):
        diagnostic=dict(experiment=result['experiment'], device=str(device), probability_max_error=float((prob-saved['probability']).abs().max()), prediction_disagreements=int((prob.argmax(-1)!=saved['probability'].argmax(-1)).sum()), labels_match=bool(torch.equal(labels,saved['truth'])), checkpoint_sha256=digest(cp), probe_source_sha256=digest(__file__), status='REFORWARD_MISMATCH_NOT_ACCEPTED')
        failure=a.output.with_suffix('.failure.private.json')
        with failure.open('x') as stream: json.dump(diagnostic, stream, indent=2)
        raise ValueError('Selected checkpoint re-forward differs: '+json.dumps(diagnostic))
    report['reloaded_probability_max_error']=float((prob-saved['probability']).abs().max())
    report['metrics']=classification_metrics(labels.numpy(),prob.numpy(),5)
    report['predicted_distribution']=torch.bincount(prob.argmax(-1),minlength=5).tolist()
    report['mean_class_probability']=prob.mean(0).tolist()
    report['classification_collapse']=len(torch.unique(prob.argmax(-1)))==1
    z=F.normalize(rep,dim=-1);cos=z@z.T
    report['representation_pairwise_cosine']=stats(cos[~torch.eye(len(z),dtype=torch.bool)])
    if att:
        attention=torch.cat(att);token=torch.cat(tokens);off=~torch.eye(5,dtype=torch.bool)
        entropy=-(attention*attention.clamp_min(1e-12).log()).sum(-1)/math.log(attention.shape[-1])
        distance=(attention[:,:,None]-attention[:,None,:]).abs().sum(-1)
        t=F.normalize(token,dim=-1);pair=t@t.transpose(-1,-2)
        report['p3']=dict(normalized_attention_entropy=stats(entropy), class_mean_entropy=entropy.mean(0).tolist(),
                          class_attention_pair_l1=stats(distance[:,off]), token_pair_cosine=stats(pair[:,off]), token_norm=stats(token.norm(dim=-1)),
                          by_diagnosis_attention_entropy=class_stats(entropy.mean(-1),labels))
    if model.use_p5:
        query=torch.cat(queries);qr=torch.cat(qrels);qv=torch.cat(qvars)
        pa=torch.cat([x['attention'] for x in pops]);res=torch.cat([x['residual'] for x in pops]);sim=torch.cat([x['similarity'] for x in pops])
        # Recompute TRAIN references from the saved fixed encoder, without mutating the saved bank.
        max_rep=max_rel=0.
        with torch.no_grad():
            for offset in range(0,len(split['train']),20):
                index=split['train'][offset:offset+20];r,rel,_=reference_features(model.reference_encoder,dataset.fc[index].to(device),options.get('reliability_temperature',.05))
                max_rep=max(max_rep,float((r-model.population.reference[offset:offset+len(index)]).abs().max()))
                max_rel=max(max_rel,float((rel-model.population.reliability[offset:offset+len(index)]).abs().max()))
        if max_rep>2e-4 or max_rel>2e-5: raise ValueError('TRAIN reference recomputation mismatch')
        with torch.no_grad():
            alone=model.population(query[:1],qr[:1],qids[:1],model.encoder_version)['residual']
            together=model.population(query[:20],qr[:20],qids[:20],model.encoder_version)['residual'][:1]
            batch_error=float((alone-together).abs().max())
        if batch_error>1e-5: raise ValueError('Population batch dependence')
        counts=torch.bincount(pa.argmax(-1),minlength=pa.shape[1]);original=model.population.reliability.clone()
        counter={}
        variants=['ones','permuted','disable_population']
        # The same reference permutation is shared by all queries; no label-based edits.
        order=torch.randperm(len(original),generator=torch.Generator().manual_seed(45119)).to(device)
        with torch.no_grad():
            for variant in variants:
                if variant=='ones': model.population.reliability.fill_(1)
                elif variant=='permuted':model.population.reliability.copy_(original[order])
                else:model.population.reliability.copy_(original)
                values=[]
                for batch in loader:
                    x=inputs(batch,device);out=forward(model,x)
                    if variant=='disable_population':
                        fused=out['fused_repr']-out['diagnostics']['population']['residual'];logits=model.base.original.MLP_layer(fused)
                        if model.use_p3:logits=logits+out['diagnostics']['task_logits']
                    elif variant=='ones':
                        q,_,_=reference_features(model.reference_encoder,x['fc'],options.get('reliability_temperature',.05))
                        replacement=model.population(q,torch.ones(len(q),device=device),x['sample_ids'],model.encoder_version)['residual']
                        fused=out['fused_repr']-out['diagnostics']['population']['residual']+replacement;logits=model.base.original.MLP_layer(fused)
                        if model.use_p3:logits=logits+out['diagnostics']['task_logits']
                    else:logits=out['logits']
                    values.append(logits.softmax(-1).cpu())
                changed=torch.cat(values);counter[variant]=dict(probability_l1_mean=float((changed-prob).abs().sum(-1).mean()),prediction_change_fraction=float((changed.argmax(-1)!=prob.argmax(-1)).float().mean()),metrics=classification_metrics(labels.numpy(),changed.numpy(),5))
            model.population.reliability.copy_(original)
        report['p5']=dict(reference_recompute_max_error=max_rep,reliability_recompute_max_error=max_rel,batch_residual_max_error=batch_error,
            train_reliability=stats(original),query_reliability=stats(qr),query_variance=stats(qv),
            reliability_by_diagnosis=class_stats(qr.cpu(),labels),train_reliability_by_diagnosis=class_stats(original.cpu(),dataset.labels[split['train']]),
            normalized_edge_entropy=stats(torch.cat([x['entropy'] for x in pops])/math.log(model.population.topk)),
            residual_to_fused_norm=stats(res.norm(dim=-1)/rep.norm(dim=-1).clamp_min(1e-8)),similarity=stats(sim),
            most_frequent_top1_reference_fraction=float(counts.max()/len(labels)),distinct_top1_references=int((counts>0).sum()),
            learned_metric=stats(1+.5*model.population.metric.tanh()),gain=float(model.population.gain.sigmoid()),counterfactuals=counter)
    if a.mask_robustness:
        robust=[]
        with torch.no_grad():
            for ratio in [.05,.10,.15]:
                per_seed=[]
                for seed in [91201,91202,91203]:
                    generator=torch.Generator(device=device).manual_seed(seed);values=[]
                    for batch in loader:
                        x=inputs(batch,device);masked,visible=symmetric_fc_mask(x['fc'],ratio,generator)
                        out=model(**dict(x,fc=masked,visible=visible));values.append(out['logits'].softmax(-1).cpu())
                    masked_prob=torch.cat(values);m=classification_metrics(labels.numpy(),masked_prob.numpy(),5)
                    per_seed.append(dict(mask_seed=seed,metrics=m,probability_l1_mean=float((masked_prob-prob).abs().sum(-1).mean())))
                robust.append(dict(mask_ratio=ratio,mean_macro_f1=float(np.mean([v['metrics']['f1_macro'] for v in per_seed])),mean_accuracy=float(np.mean([v['metrics']['accuracy'] for v in per_seed])),mean_probability_l1=float(np.mean([v['probability_l1_mean'] for v in per_seed])),runs=per_seed))
        report['masked_robustness']=robust
    # Gradient diagnostic uses a TRAIN batch in eval mode, so BN and dropout
    # state are not modified. This differs from optimizer-time gradients.
    train_loader=DataLoader(FoldDataset(dataset,split['train'],metadata),batch_size=20,shuffle=False)
    batch=next(iter(train_loader));counts=torch.bincount(dataset.labels[split['train']],minlength=5).float();w=counts.pow(-.5);w=(w/w.mean()).to(device)
    model.zero_grad(set_to_none=True);out=forward(model,inputs(batch,device));ce(out,batch['label'].to(device),w).backward()
    report['checkpoint_train_batch_gradients']={name:float(param.grad.norm()) for name,param in model.named_parameters() if param.grad is not None and name.startswith(('task_fusion.','population.'))}
    report['first_epoch_module_gradients']={name:value for name,value in result['gradients_first_epoch'].items() if name.startswith(('task_fusion.','population.'))}
    if model.use_p3 and hasattr(model.task_fusion,'class_weight'):
        report['class_weight_gradient_by_class']=model.task_fusion.class_weight.grad.norm(dim=-1).tolist()
    if digest(cp)!=report['checkpoint_sha256']:raise RuntimeError('Probe altered checkpoint')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(experiment=report['experiment'],reloaded_probability_max_error=report['reloaded_probability_max_error'],p3='p3' in report,p5='p5' in report,robustness='masked_robustness' in report,test_accessed=False)))


if __name__=='__main__':
    main()
