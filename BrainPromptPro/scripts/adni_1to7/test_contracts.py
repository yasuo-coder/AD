"""CPU/CUDA full-model contracts then a TRAIN-only real-FC smoke check."""
import argparse,copy,json,itertools,time
from pathlib import Path
import torch
from BrainPromptPro.data.prepared_cohort import PreparedCohortDataset
from BrainPromptPro.data.metadata import MetadataEncoder,record_id
from BrainPromptPro.nets.connectome_p2p3p5 import ConnectomeBrainPrompt,reference_features
from BrainPromptPro.nets.original_adapter import OriginalBrainPromptAdapter
from BrainPromptPro.layers.masked_connectome import symmetric_fc_mask,reconstruction_loss,symmetric_kl
from BrainPromptPro.research_runner import seed_all
from BrainPromptPro.scripts.adni_1to7.audit_dataset import write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=json.loads(a.config.read_text());data=PreparedCohortDataset(cfg['data']);split=json.loads(Path(cfg['new_research']['split_path']).read_text())
    torch.set_num_threads(2);seed_all(42);device=torch.device(a.device)
    x=torch.randn(6,116,116,device=device);x=((x+x.transpose(-1,-2))/6).clamp(-1,1);x.diagonal(dim1=-2,dim2=-1).fill_(1)
    prompt=torch.randn(6,2048,device=device);age=torch.zeros(6,device=device);sex=site=torch.zeros(6,dtype=torch.long,device=device)
    teacher=ConnectomeBrainPrompt(cfg,data.roi_prompt,data.disease_prompt,{}).to(device).eval()
    ref,rel,_=reference_features(teacher,x)
    bank=dict(representation=ref,reliability=rel,reference_ids=[f'synthetic-{i}' for i in range(6)],train_ids=[f'synthetic-{i}' for i in range(6)],encoder_version='synthetic-v1')
    results=[]
    for flags in itertools.product([False,True],repeat=3):
        opt=dict(zip(['use_p2','use_p3','use_p5'],flags));opt['reference_topk']=2
        seed_all(42);model=ConnectomeBrainPrompt(cfg,data.roi_prompt,data.disease_prompt,opt,bank if flags[2] else None,copy.deepcopy(teacher) if flags[2] else None).to(device)
        ids=['synthetic-0','query-one','query-two']
        args=dict(fc=x[:3],subject_prompt=prompt[:3],age=age[:3],sex=sex[:3],site=site[:3],sample_ids=ids)
        model.eval();out=model(**args);assert out['logits'].shape==(3,5) and torch.isfinite(out['logits']).all()
        if not any(flags):
            seed_all(42);official=OriginalBrainPromptAdapter(cfg,data.roi_prompt,data.disease_prompt).to(device).eval()
            refout=official(**{k:v for k,v in args.items() if k!='sample_ids'})
            torch.testing.assert_close(out['logits'],refout['logits'],rtol=0 if device.type=='cpu' else 1e-6,atol=0 if device.type=='cpu' else 1e-6)
        singles=[]
        for i in range(3):
            one={k:([ids[i]] if k=='sample_ids' else v[i:i+1]) for k,v in args.items()}
            singles.append(model(**one)['logits'])
        torch.testing.assert_close(out['logits'],torch.cat(singles),atol=2e-5,rtol=2e-5)
        model.train();out=model(**args);loss=torch.nn.functional.cross_entropy(out['logits'],torch.tensor([0,1,4],device=device))
        if flags[0]:
            g=torch.Generator(device=device).manual_seed(99);masked,vis=symmetric_fc_mask(x[:3],.1,g)
            augmented=model(**dict(args,fc=masked,visible=vis,reconstruct=True))
            loss=loss+reconstruction_loss(augmented['diagnostics']['reconstruction'],x[:3],vis)+symmetric_kl(out['logits'],augmented['logits'])
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        expected=['base.original.embedding_h']+(['decoder','visibility_projection'] if flags[0] else [])+(['task_fusion.readout.q','task_fusion.class_weight'] if flags[1] else [])+(['population.scorer','population.value'] if flags[2] else [])
        gradients={key:sum(float(p.grad.abs().sum()) for name,p in model.named_parameters() if name.startswith(key) and p.grad is not None) for key in expected}
        assert all(v>0 for v in gradients.values()),gradients
        state=copy.deepcopy(model.state_dict());model.load_state_dict(state)
        results.append(dict(flags=opt,status='PASSED',gradient_sums=gradients))
    # True TRAIN examples only, after synthetic full-model checks.
    train=split['train'][:4];metadata=MetadataEncoder().fit([data.records[i] for i in split['train']]);m=metadata.transform([data.records[i] for i in train])
    seed_all(42);realmodel=ConnectomeBrainPrompt(cfg,data.roi_prompt,data.disease_prompt,dict(use_p2=True,use_p3=True)).to(device).train()
    fc=data.fc[train].to(device);masked,vis=symmetric_fc_mask(fc,.1,torch.Generator(device=device).manual_seed(33))
    out=realmodel(masked,data.subject_prompt[train].to(device),**{k:v.to(device) for k,v in m.items()},visible=vis,reconstruct=True)
    loss=torch.nn.functional.cross_entropy(out['logits'],data.labels[train].to(device))+reconstruction_loss(out['diagnostics']['reconstruction'],fc,vis)
    loss.backward();assert torch.isfinite(loss)
    report=dict(status='PASSED',device=a.device,module_switches=results,real_train_samples=len(train),real_train_loss=float(loss),test_accessed=False,source_sha256={str(p):sha(p) for p in [Path(__file__),Path('BrainPromptPro/nets/connectome_p2p3p5.py')]})
    write_json(a.output,report);print(json.dumps(dict(status='PASSED',device=a.device,switches=len(results),real_train_samples=len(train))))

if __name__=='__main__':main()
