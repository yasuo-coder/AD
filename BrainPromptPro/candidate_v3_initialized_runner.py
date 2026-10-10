"""Train-only gradient-audited initialization of the unchanged reviewed V3 operator."""
import torch
from torch import nn
from . import candidate_v3_runner
from . import candidate_runner as runner
_previous_build=runner.build_model

def build_model(name,cfg,data,encoder,spec):
    model,cfg=_previous_build(name,cfg,data,encoder,spec)
    if spec.get('population_version')!=3:return model,cfg
    if spec.get('population_initialization')!='xavier_gain1':raise ValueError('Unreviewed V3 initialization')
    with torch.random.fork_rng(devices=[]):
        for scorer in model.population.scorers.values():
            for layer in scorer:
                if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(model.population.bottleneck.weight)
        nn.init.xavier_uniform_(model.population.out.weight)
        nn.init.zeros_(model.population.gain_raw)
        if spec.get('control') in ('fixed','no_metadata','direct'):
            for parameter in model.population.scorers.parameters():nn.init.zeros_(parameter)
    return model,cfg

runner.build_model=build_model
if __name__=='__main__':runner.main()
