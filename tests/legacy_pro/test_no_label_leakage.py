import inspect
import torch
from .common import fixture
from BrainPromptPro.losses import compute_losses
from BrainPromptPro.data.splits import make_splits
from BrainPromptPro.train_TUs_graph_classification_llm import evaluate


def test_labels_only_enter_loss_and_evaluation_does_not_learn():
    m,x,c=fixture();m.eval()
    for module in (m,m.adaptive_brain,m.disease_readout,m.population):
        assert not {'y','label','labels','y_true'}&set(inspect.signature(module.forward).parameters)
    before=m(**x)
    for labels in (torch.tensor([0,1,2,3]),torch.tensor([4,4,0,0])):
        compute_losses(before,labels,c['loss'],m.config)
        after=m(**x)
        for key in ('logits','A_brain','roi_attention','relation_weights'):
            torch.testing.assert_close(before[key],after[key])
    state={k:v.clone() for k,v in m.state_dict().items()}
    batch={**x,'label':torch.arange(4),'subject_id':[f'S{i}/init' for i in range(4)]}
    evaluate(m,[batch],torch.device('cpu'),c['loss'])
    for k,v in m.state_dict().items():torch.testing.assert_close(v,state[k])
    assert all(p.grad is None for p in m.parameters())


def test_grouped_cv_has_no_patient_overlap():
    records=[dict(subject=f'S{i//2}',session=str(i%2)) for i in range(80)]
    labels=[(i//2)%2 for i in range(80)]
    folds=make_splits(records,labels,n_splits=4)
    for fold in folds:
        groups={k:{records[i]['subject'] for i in indices} for k,indices in fold.items()}
        assert not groups['train']&groups['test']
        assert not groups['train']&groups['val']
