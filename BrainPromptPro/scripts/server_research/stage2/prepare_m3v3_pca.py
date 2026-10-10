import hashlib,json,sys
from pathlib import Path
import torch
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro import candidate_runner as r
from BrainPromptPro.data.metadata import record_id
from BrainPromptPro.data.brainnet_pro import sha256
cfg,protocol,data,split=r.load_context();torch.set_num_threads(2);dest=r.RUN/'preprocessing';dest.mkdir(exist_ok=True);path=dest/'m3v3_train_pca32.pt'
if path.exists():raise FileExistsError(path)
upper=torch.triu_indices(data.node_num,data.node_num,offset=1);raw=data.fc[split['train']][:,upper[0],upper[1]].contiguous();x=raw.double();mu=x.mean(0);center=x-mu
values,left=torch.linalg.eigh(center@center.T/(len(x)-1));idx=torch.argsort(values,descending=True)[:32];eigen=values[idx];left=left[:,idx]
assert torch.isfinite(eigen).all() and float(eigen[-1])>1e-8
components=center.T@left/((len(x)-1)*eigen).sqrt()[None]
pivots=components.abs().argmax(0);sign=components[pivots,torch.arange(32)].sign();components=components*sign[None]
epsilon=max(1e-8,1e-4*float(eigen[0]));scale=(eigen+epsilon).rsqrt();keys=torch.nn.functional.normalize((center@components)*scale,dim=-1,eps=1e-6)
ids=[record_id(data.records[i]) for i in split['train']]
artifact=dict(upper=upper,pca_mean=mu.float(),pca_components=components.float(),pca_scale=scale.float(),reference_keys=keys.float(),eigenvalues=eigen,epsilon=epsilon,train_indices=split['train'],reference_ids=ids,bundle_sha256=cfg['data']['bundle_sha256'],split_sha256=protocol['split_sha256'],raw_train_fc_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),train_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
torch.save(artifact,path)
record=dict(artifact_sha256=sha256(path),train_samples=len(x),features=x.shape[1],components=32,fit_dtype='float64',storage_dtype='float32',algorithm='deterministic TRAIN Gram eigh; stable signs; no random sampling',training_indices=split['train'],raw_train_fc_sha256=artifact['raw_train_fc_sha256'],train_ids_sha256=artifact['train_ids_sha256'],split_sha256=protocol['split_sha256'],bundle_sha256=cfg['data']['bundle_sha256'],orthogonality_error=float((components.T@components-torch.eye(32)).abs().max()),epsilon=epsilon,labels_or_source_used=False,test_loader_created=False)
r.dump(dest/'m3v3_train_pca32.json',record);print(json.dumps({k:v for k,v in record.items() if k!='training_indices'},indent=2))
