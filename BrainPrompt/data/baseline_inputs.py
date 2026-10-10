"""Optional baseline inputs. Dataset rows stay in the original graph/split order."""
import csv
import re
from pathlib import Path

import torch


def activation_text(timeseries, roi_names, activation_fraction=0.6, correlation_count=5):
    """fTSPL Eq.(1) ABTG, using only one subject's ROI time series [N,T].

    The paper does not specify H_c numerically; correlation_count is explicit.
    Self correlations are excluded from the count of other correlated regions.
    """
    x = timeseries.float()
    if x.ndim != 2 or x.shape[0] != len(roi_names) or x.shape[1] < 2:
        raise ValueError('ABTG needs [N,T] time series aligned with N ROI names')
    if not torch.isfinite(x).all():
        raise ValueError('Non-finite fTSPL time series')
    active = (x > activation_fraction*x.max()).sum(-1) >= x.shape[1]/20
    centered = x-x.mean(-1,keepdim=True)
    centered = torch.nn.functional.normalize(centered,dim=-1)
    fc = centered @ centered.T
    connected = fc>0.5
    connected.fill_diagonal_(False)
    counts = connected.sum(-1)
    sentences = []
    for i in active.nonzero().flatten().tolist():
        degree = 'strong' if counts[i]>3*correlation_count else 'moderate' if counts[i]>=correlation_count else 'weak'
        sentences.append(f'A {degree} brain activity of {roi_names[i]}.')
    return ' '.join(sentences) or 'No brain region exceeds the activation threshold.'


def read_demographics(path, count):
    """CSV graph_index,age,sex,site; graph_index is zero-based original .bin order."""
    with open(path,newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'graph_index','age','sex','site'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Metadata CSV must contain {sorted(required)}')
    result = torch.empty(count,3)
    seen = set()
    sites = {s:i for i,s in enumerate(sorted({r['site'].strip() for r in rows}))}
    for row in rows:
        idx = int(row['graph_index'])
        if idx in seen or not 0 <= idx < count:
            raise ValueError('Metadata graph_index must be a unique index into the original .bin')
        sex = row['sex'].strip().lower()
        if sex not in ('m','f','male','female','0','1'):
            raise ValueError('sex must be M/F, male/female, or 0/1')
        result[idx] = torch.tensor([float(row['age']),float(sex in ('m','male','1')),sites[row['site'].strip()]])
        seen.add(idx)
    if len(seen)!=count or not torch.isfinite(result).all():
        raise ValueError('Metadata must cover every original graph exactly once with finite values')
    return result


def attach_baseline_inputs(dataset, model_name, p):
    if model_name == 'LG-GNN':
        path = p.get('baseline_metadata_csv')
        if not path:
            raise ValueError('LG-GNN requires --baseline_metadata_csv (graph_index,age,sex,site)')
        values = read_demographics(path,len(dataset.all))
        for graph,row in zip(dataset.all.graph_lists,values):
            graph.ndata['demographics'] = row.expand(graph.num_nodes(),-1).clone()
    if model_name == 'fTSPL':
        from layers.ftspl_backbone import load_ftspl_tokenizer
        tokenizer = load_ftspl_tokenizer(p)
        path = Path(p.get('ftspl_roi_names','data/prompts/ROI_prompts/AAL116_short.txt'))
        # Reuse ROI terminology only, not disease/subject prompts.
        names = [line.split(':',1)[0].strip() for line in path.read_text().splitlines() if line.strip()]
        for graph in dataset.all.graph_lists:
            raw = graph.ndata['timeseries']
            text = activation_text(raw,names,p.get('ftspl_activation_fraction',0.6),p.get('ftspl_correlation_count',5))
            tokens = tokenizer([text]).cpu().long()[0]
            graph.ndata['ftspl_tokens'] = tokens.expand(graph.num_nodes(),-1).clone()
        # Tokens are attached once; all split datasets reference these same graphs.
