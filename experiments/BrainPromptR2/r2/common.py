import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,cohen_kappa_score
CLASSES=['CN','AD','SMC','EMCI','LMCI']
ORDINAL=np.array([0,4,1,2,3])
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/panyao/AD/BrainPromptNComms2025')
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def objhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');torch.save(x,tmp);os.replace(tmp,p)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(2)
 torch.backends.cudnn.benchmark=False
def aggregate(data,ids,p):
 ids=np.asarray(ids);g=np.asarray(data['groups'])[ids];y=data['y'].numpy()[ids]
 names=np.unique(g);pp=[];yy=[]
 for name in names:
  mask=g==name;labels=np.unique(y[mask]);assert len(labels)==1
  yy.append(labels[0]);pp.append(p[mask].mean(0))
 return np.array(yy),np.array(pp),names
def metric(y,p):
 p=np.asarray(p,float);p=p/p.sum(1,keepdims=True);y=np.asarray(y);pred=p.argmax(1)
 cm=confusion_matrix(y,pred,labels=np.arange(5));support=cm.sum(1);counts=cm.sum(0);tp=np.diag(cm)
 recall=np.divide(tp,support,out=np.zeros(5,dtype=float),where=support>0)
 precision=np.divide(tp,counts,out=np.zeros(5,dtype=float),where=counts>0)
 f1=np.divide(2*tp,support+counts,out=np.zeros(5,dtype=float),where=support+counts>0)
 auc=[float(roc_auc_score(y==c,p[:,c])) if 0<support[c]<len(y) else None for c in range(5)]
 ap=[float(average_precision_score(y==c,p[:,c])) if support[c]>0 else None for c in range(5)]
 losses=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12));macro_nll=[float(losses[y==c].mean()) if support[c] else None for c in range(5)]
 avg=lambda x:float(np.mean(x)) if all(v is not None for v in x) else None
 qwk=float(cohen_kappa_score(ORDINAL[y],ORDINAL[pred],labels=np.arange(5),weights='quadratic'))
 return dict(accuracy=float((pred==y).mean()),macro_f1=float(f1.mean()),macro_precision=float(precision.mean()),
  balanced_accuracy=float(recall.mean()),macro_auroc=avg(auc),macro_ap=avg(ap),nll=float(losses.mean()),macro_nll=avg(macro_nll),
  qwk=qwk if np.isfinite(qwk) else None,recall=recall.tolist(),precision=precision.tolist(),f1=f1.tolist(),
  auroc=auc,ap=ap,confusion_matrix=cm.tolist(),support=support.tolist(),predicted_counts=counts.tolist(),subjects=len(y))
def softmax(x):
 x=np.asarray(x,float);e=np.exp(x-x.max(1,keepdims=True));return e/e.sum(1,keepdims=True)
def subject_prior(data,ids):
 groups=np.asarray(data['groups']);labels={groups[i]:int(data['y'][i]) for i in ids}
 counts=np.bincount(list(labels.values()),minlength=5);assert (counts>0).all()
 return counts/counts.sum()
def corrected(prob,prior,tau):
 return softmax(np.log(np.maximum(prob,1e-300))-float(tau)*np.log(prior))
