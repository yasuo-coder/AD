"""Every estimated transform is fit on optimization subjects, never STOP/validation."""
import numpy as np
import torch
from .common import subject_prior

def fit_features(data, train, condition='no_prompt'):
    out=dict(data)
    g=np.asarray(data['groups']); weights=np.array([1/sum(g[train]==g[i]) for i in train])
    meta=data['metadata'].numpy(); known=np.unique(meta[train,2])
    age_mean=np.average(meta[train,0],weights=weights)
    age_std=max(np.sqrt(np.average((meta[train,0]-age_mean)**2,weights=weights)),1.)
    patient=np.zeros((len(g),2048),np.float32)
    if condition not in ('no_prompt','roi_only'):
        patient[:,0]=(meta[:,0]-age_mean)/age_std; patient[:,1]=meta[:,1]
        if condition=='full':
            for j,site in enumerate(known):patient[:,j+2]=(meta[:,2]==site)
    out['subject_prompt']=torch.tensor(patient)
    semantic=data['semantic'].clone(); rng=np.random.default_rng(26027)
    if condition.endswith('shuffled'):semantic=semantic[rng.permutation(116)]
    if condition.endswith('random'):
        random=torch.tensor(rng.normal(size=semantic.shape),dtype=torch.float32)
        semantic=random/random.norm(dim=1,keepdim=True)*semantic.norm(dim=1,keepdim=True)
    out['semantic']=semantic
    row,col=torch.triu_indices(116,116,1); raw=data['fc'][:,row,col].numpy()
    if condition!='no_prompt':
        # Same number of downstream fitted coefficients for true/shuffled/random.
        projection=torch.tensor(rng.normal(size=(2048,8)),dtype=torch.float32)/np.sqrt(2048)
        b=semantic@projection;b=b/b.norm(dim=0,keepdim=True).clamp_min(1e-8)
        projected=torch.einsum('ri,brs,sj->bij',b,data['fc'],b)
        rr,cc=torch.triu_indices(8,8,0)
        raw=np.concatenate([raw,projected[:,rr,cc].numpy()],axis=1)
    if condition not in ('no_prompt','roi_only'):
        nmeta=2+len(known) if condition=='full' else 2
        raw=np.concatenate([raw,patient[:,:nmeta]],axis=1)
    mean=np.average(raw[train],axis=0,weights=weights)
    std=np.sqrt(np.average((raw[train]-mean)**2,axis=0,weights=weights));std=np.maximum(std,1e-6)
    out['x']=torch.tensor((raw-mean)/std,dtype=torch.float32)
    transform=dict(mean=mean,std=std,known_sites=known,age_mean=age_mean,age_std=age_std,condition=condition)
    return out,transform,weights
