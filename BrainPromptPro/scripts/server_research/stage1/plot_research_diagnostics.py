"""Exploratory representation and graph figures from TRAIN/VAL only."""
import csv,json,os,sys,itertools
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances,silhouette_score
sys.dont_write_bytecode=True;sys.path.insert(0,'/home/panyao/AD')
from BrainPromptPro.research_runner import RUN,dump,csv_out
os.environ['MPLCONFIGDIR']='/tmp/brainprompt_research_matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(2)
models=['BrainPrompt','CurrentPro_legacy','CurrentPro_stable'];classes=['CN','AD','SMC','EMCI','LMCI']
out=RUN/'figures';out.mkdir(exist_ok=True)
representations=[];graph_rows=[]
for model in models:
 exp=RUN/'experiments'/model/'sqrt_inverse_frequency/seed_42'
 data={role:torch.load(exp/(role+'_payload.pt'),weights_only=True,map_location='cpu') for role in ['train','val']}
 train=data['train']['embeddings'].numpy();val=data['val']['embeddings'].numpy()
 pca=PCA(n_components=2,random_state=42).fit(train)
 pc={'train':pca.transform(train),'val':pca.transform(val)}
 ts=TSNE(n_components=2,random_state=42,perplexity=30,init='pca',learning_rate='auto').fit_transform(np.concatenate([train,val]))
 te={'train':ts[:len(train)],'val':ts[len(train):]}
 fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
 for j,role in enumerate(['train','val']):
  x=data[role]['embeddings'].numpy();y=data[role]['truth'].numpy();d=pairwise_distances(x)
  tri=np.triu(np.ones((len(y),len(y)),bool),1);same=y[:,None]==y[None,:]
  intra=float(d[tri&same].mean());inter=float(d[tri&~same].mean())
  row=dict(model=model,seed=42,profile='sqrt_inverse_frequency',split=role,intra_class_distance=intra,inter_class_distance=inter,inter_intra_ratio=inter/intra if intra else None,silhouette=float(silhouette_score(x,y)),dimensions=x.shape[1])
  representations.append(row)
  for i,(name,arr) in enumerate([('PCA fitted on training only',pc[role]),('t-SNE joint train/val; visual only',te[role])]):
   ax=axes[i,j]
   for k,c in enumerate(classes):ax.scatter(arr[y==k,0],arr[y==k,1],s=12 if role=='train' else 24,alpha=.65,label=c)
   ax.set_title(f'{role}: {name}');ax.legend(fontsize=8)
  csv_out(exp/(role+'_tsne.csv'),[dict(Sample_ID=s,TSNE1=float(te[role][i,0]),TSNE2=float(te[role][i,1]),class_id=int(y[i])) for i,s in enumerate(data[role]['sample_ids'])])
 fig.suptitle(model+' | seed 42 | exploratory TRAIN/VAL representations',fontsize=12)
 fig.savefig(out/(model+'_representations.png'),dpi=150);plt.close(fig)
 st=torch.load(exp/'diagnostics/training_structure.pt',map_location='cpu',weights_only=True)
 a=st['A_brain'].numpy();mask=~np.eye(116,dtype=bool);v=a[:,mask];y=st['diagnoses'].numpy();d=pairwise_distances(v);tri=np.triu(np.ones(d.shape,bool),1);same=y[:,None]==y[None,:]
 graph_rows.append(dict(model=model,balanced_training_examples=20,mean_intra_class_graph_distance=float(d[tri&same].mean()),mean_inter_class_graph_distance=float(d[tri&~same].mean()),nonzero_density=float((v!=0).mean()),effective_density_above_point05=float((v>.05).mean())))
 fig,axes=plt.subplots(1,5,figsize=(15,3),layout='constrained')
 for k,ax in enumerate(axes):
  z=a[y==k].mean(0).copy();np.fill_diagonal(z,0);im=ax.imshow(z,cmap='viridis',vmin=0,vmax=max(float(v.max()),1e-6));ax.set_title(classes[k]);ax.set_xticks([]);ax.set_yticks([])
 fig.colorbar(im,ax=axes,shrink=.8);fig.suptitle(model+' | mean graph by TRAIN label (4 samples/class; exploratory)')
 fig.savefig(out/(model+'_brain_graphs.png'),dpi=150);plt.close(fig)
 att=st['roi_attention']
 if att is not None:
  fig,ax=plt.subplots(figsize=(14,3),layout='constrained');im=ax.imshow(att.mean(0).numpy(),aspect='auto',cmap='viridis');ax.set(yticks=range(5),yticklabels=classes,xlabel='AAL116 ROI index',title=model+' | average disease attention, 20 TRAIN examples');fig.colorbar(im,ax=ax);fig.savefig(out/(model+'_attention.png'),dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
for model,ax in zip(models,axes):
 r=json.loads((RUN/'experiments'/model/'sqrt_inverse_frequency/seed_42/result.json').read_text());cm=np.array(r['val']['confusion_matrix']);ax.imshow(cm,cmap='Blues');ax.set(xticks=range(5),yticks=range(5),xticklabels=classes,yticklabels=classes,title=model,xlabel='Predicted',ylabel='True')
 for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
fig.suptitle('Validation only | seed 42 | identical 100 scans')
fig.savefig(out/'validation_confusion_seed42.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
for j,model in enumerate(models):
 for seed in [42,3407,2026]:
  p=RUN/'experiments'/model/'sqrt_inverse_frequency'/f'seed_{seed}'/'history.jsonl'
  if not p.exists():continue
  h=[json.loads(s) for s in p.read_text().splitlines()];e=[t['epoch'] for t in h]
  axes[0,j].plot(e,[t['validation']['f1_macro'] for t in h],label=str(seed));axes[1,j].plot(e,[t['validation']['loss'] for t in h],label=str(seed))
 axes[0,j].set(title=model,ylabel='Validation macro F1',xlabel='Epoch',ylim=(0,1));axes[1,j].set(ylabel='Validation total loss',xlabel='Epoch');axes[0,j].legend();axes[1,j].legend()
fig.savefig(out/'validation_training_curves.png',dpi=150);plt.close(fig)
csv_out(RUN/'representation_diagnostics.csv',representations);csv_out(RUN/'graph_class_diagnostics.csv',graph_rows)
dump(RUN/'figures/visualization_protocol.json',dict(used_test=False,selection_use='Exploratory development only; no choices made from test',pca='Fit train, transform train and validation',tsne='Joint train/val embedding for visualization only, seed42 perplexity30',graph='20 training examples; four/class; no significance inference',gradient_probe='module_diagnostics uses deterministic eval mode and unweighted CE, including for weighted-trained checkpoints; local probe not the stochastic weighted training gradient',disease_query_cosine_key='Legacy diagnostic key denotes cosine between attention MAPS, not raw query embeddings'))
print(json.dumps(dict(figures=str(out),representation=representations,graph=graph_rows),indent=2))
