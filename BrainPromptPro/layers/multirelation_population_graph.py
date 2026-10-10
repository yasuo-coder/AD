"""Independent age/sex/site edge learners and relation-specific GCNs."""
import torch
from torch import nn
from .gcn_layer import WeightedGCNLayer, unit_diagonal


def symmetric_pair(x):
    return torch.cat(((x[:, None] - x[None, :]).abs(), x[:, None] * x[None, :]), -1)


class MultiRelationPopulationGraph(nn.Module):
    def __init__(self, hidden_dim, relation_hidden_dim=64, num_sexes=3,
                 num_sites=1, categorical_dim=8, relation_gate='subject_conditioned',
                 relations=('age', 'sex', 'site')):
        super().__init__()
        if len(set(relations)) != len(relations) or not relations or not set(relations) <= {'age', 'sex', 'site'}:
            raise ValueError('relations must be a nonempty unique subset of age/sex/site')
        if relation_gate not in ('global', 'subject_conditioned'):
            raise ValueError('Unknown relation_gate')
        self.relations = tuple(relations)
        self.gate_mode = relation_gate
        self.sex_embedding = nn.Embedding(num_sexes, categorical_dim, padding_idx=0)
        self.site_embedding = nn.Embedding(num_sites, categorical_dim, padding_idx=0)
        dims = {'age': 3, 'sex': 2 * categorical_dim + 1, 'site': 2 * categorical_dim + 1}
        self.scorers = nn.ModuleDict({r: nn.Sequential(
            nn.Linear(2 * hidden_dim + dims[r], relation_hidden_dim), nn.GELU(),
            nn.Linear(relation_hidden_dim, 1)) for r in self.relations})
        self.gnns = nn.ModuleDict({r: WeightedGCNLayer(hidden_dim, hidden_dim,
                                                     batch_norm=False, residual=True)
                                   for r in self.relations})
        if relation_gate == 'global':
            self.relation_gate = nn.Parameter(torch.zeros(len(self.relations)))
        else:
            self.relation_gate = nn.Sequential(
                nn.Linear((1 + len(self.relations)) * hidden_dim, relation_hidden_dim),
                nn.GELU(), nn.Linear(relation_hidden_dim, len(self.relations)))

    def forward(self, subject_repr, age, sex, site):
        # subject_repr: [B,D]; age/sex/site: [B]; raw A_relation: [B,B].
        subject_pair = symmetric_pair(subject_repr)
        metadata_pairs = {
            'age': torch.stack(((age[:, None] - age[None, :]).abs(),
                                age[:, None] * age[None, :],
                                (age[:, None] + age[None, :]) * .5), dim=-1)
        }
        for name, ids, embedding in [('sex', sex, self.sex_embedding), ('site', site, self.site_embedding)]:
            if name in self.relations:
                known_same = (ids[:, None] == ids[None, :]) & (ids[:, None] != 0) & (ids[None, :] != 0)
                metadata_pairs[name] = torch.cat((symmetric_pair(embedding(ids)),
                                                  known_same.to(subject_repr.dtype).unsqueeze(-1)), -1)
        adjacencies, outputs = {}, []
        for relation in self.relations:
            pair = torch.cat((subject_pair, metadata_pairs[relation]), dim=-1)
            a = self.scorers[relation](pair).squeeze(-1).sigmoid()
            a = unit_diagonal((a + a.T) * .5)
            adjacencies[relation] = a
            outputs.append(self.gnns[relation](subject_repr, a))
        if self.gate_mode == 'global':
            weights = self.relation_gate.softmax(-1).expand(subject_repr.shape[0], -1)
        else:
            weights = self.relation_gate(torch.cat((subject_repr, *outputs), dim=-1)).softmax(-1)
        population = (torch.stack(outputs, 1) * weights.unsqueeze(-1)).sum(1)
        return population, adjacencies, weights
