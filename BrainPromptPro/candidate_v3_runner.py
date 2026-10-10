"""Same frozen trainer, with the specifically reviewed M3 V3 factory."""
import copy
import torch
from . import candidate_runner as runner
from .data.metadata import record_id
from .data.brainnet_pro import sha256
from .nets.research_population_v3 import SparseReferencePopulation

_original_build=runner.build_model

def build_model(name,cfg,data,encoder,spec):
    model,cfg=_original_build(name,cfg,data,encoder,spec)
    if spec.get('population_version')!=3:return model,cfg
    path=runner.RUN/'preprocessing/m3v3_train_pca32.pt';declaration=runner.RUN/'preprocessing/m3v3_train_pca32.json'
    import json
    meta=json.loads(declaration.read_text())
    if sha256(path)!=meta['artifact_sha256']:raise ValueError('Frozen PCA artifact changed')
    pca=torch.load(path,map_location='cpu',weights_only=True)
    split=json.loads((runner.BASE_RUN/'splits/ADNI_Combined/indices.json').read_text());records=[data.records[i] for i in split['train']]
    ids=[record_id(r) for r in records]
    if pca['train_indices']!=split['train'] or pca['reference_ids']!=ids:raise ValueError('PCA not fitted on exact TRAIN')
    if pca['bundle_sha256']!=cfg['data']['bundle_sha256']:raise ValueError('PCA belongs to different data')
    with torch.random.fork_rng(devices=[]):
        model.population=SparseReferencePopulation(cfg['net_params']['hidden_dim'],pca,encoder.transform(records),ids,meta['artifact_sha256'],spec.get('control','learned'))
    return model,cfg

runner.build_model=build_model

if __name__=='__main__':runner.main()
