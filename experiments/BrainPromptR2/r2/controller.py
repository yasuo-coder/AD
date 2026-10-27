"""Persistent A -> B -> C -> D executor; terminal state requires external review."""
import argparse,subprocess,time,os,fcntl,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .common import *
from .isolation import command

def task_id(task):
    return f"{task['stage']}_{task['name']}_p{task['plan']:02d}"

def execute_tasks(tasks,gpus,run):
    def lane(gpu,lane_tasks):
        for task in lane_tasks:
            name=task_id(task);out=run/'units'/name;out.mkdir(parents=True,exist_ok=True)
            taskfile=run/'tasks'/(name+'.json');write(taskfile,task)
            if (out/'result.json').exists():
                r=read(out/'result.json');assert r['status']=='COMPLETE' and r['task']==task;continue
            with (out/'worker.log').open('a') as log:
                proc=subprocess.Popen(command(taskfile,out,gpu),stdout=log,stderr=subprocess.STDOUT)
                write(out/'process.json',dict(pid=proc.pid,gpu=gpu,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
                code=proc.wait()
            if code:raise RuntimeError(f'{name} failed with {code}; see {out}/worker.log')
            assert read(out/'result.json')['status']=='COMPLETE'
            print('COMPLETED',name,flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(lane,g,tasks[j::len(gpus)]) for j,g in enumerate(gpus)]
        for f in futures:f.result()

SCALARS=['macro_nll','nll','macro_auroc','macro_ap','macro_f1','balanced_accuracy','accuracy','qwk']
def summarize(tasks,run,checkpoints=('best',),tau=0.):
    candidates={}
    data=torch.load(ROOT/'prepared/data.private.pt',weights_only=True,map_location='cpu')
    for task in tasks:
        folder=run/'units'/task_id(task);r=read(folder/'result.json')
        assert digest(folder/'predictions.private.pt')==r['predictions_sha256']
        pred=torch.load(folder/'predictions.private.pt',weights_only=False,map_location='cpu')
        for cp in checkpoints:
            name=task['name']+('_'+cp if len(checkpoints)>1 else '')
            x=pred['validation_'+cp]
            # Apply the requested logit shift per scan, then average within person.
            # This deliberately does not shift already-averaged patient probabilities.
            shifted=corrected(x['scan_probability'],np.asarray(r['unique_fit_prior']),tau)
            y,p,g=aggregate(data,x['scan_ids'],shifted)
            m=metric(y,p);candidates.setdefault(name,[]).append(dict(repeat=r['repeat'],fold=r['fold'],metrics=m,y=y,p=p,groups=g,scale=r['scale']['validation_'+cp],selected_epoch=r['selected_epoch'],epochs=r['epochs']))
    output={}
    for name,folds in candidates.items():
        repeat=[]
        for rep in range(3):
            ff=[f for f in folds if f['repeat']==rep];assert len(ff)==3
            groups=np.concatenate([f['groups'] for f in ff]);assert len(groups)==len(set(groups))==715
            repeat.append(dict(repeat=rep,mean_fold={k:float(np.mean([f['metrics'][k] for f in ff])) for k in SCALARS},
              pooled_oof=metric(np.concatenate([f['y'] for f in ff]),np.concatenate([f['p'] for f in ff]))))
        output[name]=dict(mean_fold={k:float(np.mean([f['metrics'][k] for f in folds])) for k in SCALARS},repeats=repeat,
          folds=[{k:v for k,v in f.items() if k not in ['y','p','groups']} for f in folds])
    return output

def ranking(summaries):
    return sorted(summaries,key=lambda name:(round(summaries[name]['mean_fold']['macro_nll'],8),-round(summaries[name]['mean_fold']['macro_ap'],8),-round(summaries[name]['mean_fold']['macro_auroc'],8),name))

def stable_gain(candidate,baseline):
    wins=[]
    for c,b in zip(candidate['repeats'],baseline['repeats']):
        c,b=c['mean_fold'],b['mean_fold']
        wins.append(c['macro_nll']<b['macro_nll']-1e-8 and c['macro_ap']>=b['macro_ap']-1e-8 and c['macro_auroc']>=b['macro_auroc']-1e-8)
    return sum(wins)>=2 and candidate['mean_fold']['macro_nll']<baseline['mean_fold']['macro_nll']-1e-8,wins

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',default='4');ap.add_argument('--run',default=str(ROOT/'runs/fixed_fit_v1'));a=ap.parse_args()
    run=Path(a.run);run.mkdir(parents=True,exist_ok=True);lock=(run/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(ROOT/'configs/study.json');gpus=a.gpus.split(',');public=run/'public';public.mkdir(exist_ok=True)
    seal=read(ROOT/'public/source_manifest.json')
    for rel,sha in seal['files'].items():assert digest(ROOT/rel)==sha,f'Code changed: {rel}'
    assert (ROOT/'public/publication.json').exists(),'Publish plan before formal training'
    def status(stage,**kw):write(run/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw));print(stage,kw,flush=True)
    try:
        status('A')
        at=[dict(stage='A',name=kind,kind=kind,plan=p) for p in range(9) for kind in cfg['A']]
        execute_tasks(at,gpus,run);s=summarize(at,run,('last','best'));write(public/'A_2x2.json',s)
        soft_gain,wins=stable_gain(s['soft_topk_best'],s['original_best'])
        write(public/'A_gate.json',dict(soft_topk_stable=soft_gain,repeat_wins=wins,downstream_population='off',reason='B minimum baseline is always population off; A does not expand the candidate list'))
        status('B')
        specs=[dict(name='logistic_C'+str(c).rstrip('0').rstrip('.') if c<1 else 'logistic_C'+str(int(c)),kind='logistic',C=c) for c in cfg['logistic']['C']]+[dict(name='mlp',kind='mlp'),dict(name='g_off',kind='g_off')]
        bt=[dict(stage='B',plan=p,**spec) for p in range(9) for spec in specs]
        execute_tasks(bt,gpus,run);b=summarize(bt,run);write(public/'B_baselines.json',b)
        numeric={k:v for k,v in b.items() if k!='g_off'};best_numeric=ranking(numeric)[0]
        gpass,gwins=stable_gain(b['g_off'],b[best_numeric]);chosen='g_off' if gpass else best_numeric
        chosen_spec=next(x for x in specs if x['name']==chosen)
        write(public/'B_selection.json',dict(selected=chosen,best_numeric=best_numeric,g_stable_gain=gpass,g_repeat_wins=gwins,ranking=ranking(b),T='STOP_T_ROUTE'))
        status('C',selected=chosen)
        chosen_tasks=[t for t in bt if t['name']==chosen];gate=cfg['C_ranking_gate']
        signal=[r['mean_fold']['macro_auroc']>gate['repeat_mean_auroc_min'] and r['mean_fold']['macro_ap']>gate['repeat_mean_ap_min'] for r in b[chosen]['repeats']]
        tau_results={};tau=0.
        if sum(signal)>=gate['passing_repeats']:
            for value in cfg['C_tau']:tau_results[str(value)]=summarize(chosen_tasks,run,tau=value)[chosen]
            tau=float(ranking(tau_results)[0])
        write(public/'C_prior_correction.json',dict(status='COMPLETE' if tau_results else 'SKIPPED_WEAK_RANKING',selected=chosen,signal_repeats=signal,tau=tau,candidates=tau_results,prior_source='optimization FIT unique subjects; not scan frequency',application='per-scan logit shift before subject probability mean'))
        chosen_corrected=summarize(chosen_tasks,run,tau=tau)[chosen]
        collapsed=[sum(x>0 for x in f['metrics']['predicted_counts'])==1 for f in chosen_corrected['folds']]
        if all(collapsed):
            write(public/'internal_summary.json',dict(status='STOPPED_COLLAPSE_REQUIRES_REVIEW',selected_backbone=chosen,tau=tau,completed_fits=len(at)+len(bt),D='not started: all nine corrected validation folds predict one class',outer_evaluation='NOT_RUN'))
            status('STOPPED_COLLAPSE_REQUIRES_REVIEW',selected=chosen,tau=tau);return
        # D diagnoses prompt information with the already chosen backbone; no G/T complexity escalation.
        status('D',selected=chosen,tau=tau)
        dt=[dict(stage='D',name=condition,kind=chosen_spec['kind'],**({'C':chosen_spec['C']} if 'C' in chosen_spec else {}),condition=condition,plan=p) for p in range(9) for condition in cfg['D']]
        execute_tasks(dt,gpus,run);d=summarize(dt,run,tau=tau);raw=summarize(dt,run,tau=0.)
        write(public/'D_prompt_ablation.json',dict(backbone=chosen,tau_fixed_from_C=tau,corrected=d,uncorrected=raw,ranking=ranking(d),semantic_claim='not established; require stable true-vs-both-control gains'))
        controls={c:stable_gain(d['roi_age_sex'],d[c])[0] for c in ['roi_age_sex_shuffled','roi_age_sex_random']}
        write(public/'internal_summary.json',dict(status='AWAIT_KIMI_REVIEW',selected_backbone=chosen,tau=tau,D_ranking=ranking(d),true_vs_control_gates=controls,completed_fits=len(at)+len(bt)+len(dt),outer_evaluation='NOT_RUN; not implemented',scope='exploratory repeated internal validation of historically exposed subjects'))
        status('AWAIT_KIMI_REVIEW',completed_fits=len(at)+len(bt)+len(dt),selected=chosen,tau=tau)
    except BaseException as e:
        status('FAILED_REQUIRES_INSPECTION',error=str(e));traceback.print_exc();raise
if __name__=='__main__':main()
