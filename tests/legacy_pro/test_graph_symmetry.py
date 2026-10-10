import torch
from .common import fixture


def test_symmetry_range_and_finite_training_and_evaluation():
    m,x,_=fixture()
    m.adaptive_brain.edge_mlp[2].p=.3
    for training in (True,False):
        m.train(training)
        out=m(**x)
        for a in [out['A_brain'],*out['population_adjacencies'].values()]:
            torch.testing.assert_close(a,a.transpose(-1,-2),atol=1e-6,rtol=1e-6)
            assert torch.isfinite(a).all() and (a>=0).all() and (a<=1).all()
            torch.testing.assert_close(a.diagonal(dim1=-2,dim2=-1),torch.ones_like(a.diagonal(dim1=-2,dim2=-1)))
        torch.testing.assert_close(out['roi_attention'].sum(-1),torch.ones(4,5))
        torch.testing.assert_close(out['relation_weights'].sum(-1),torch.ones(4))


def test_roi_subject_isolation_and_batch_permutation():
    m,x,_=fixture();m.eval()
    h,a=m.encode_brain(x['fc'])
    changed=x['fc'].clone();changed[0]=torch.eye(116)
    h2,a2=m.encode_brain(changed)
    torch.testing.assert_close(h[1:],h2[1:]);torch.testing.assert_close(a[1:],a2[1:])
    order=torch.tensor([2,0,3,1]); permuted={k:v[order] for k,v in x.items()}
    out=m(**x);out2=m(**permuted)
    torch.testing.assert_close(out['logits'][order],out2['logits'],atol=1e-5,rtol=1e-5)


def test_dgl_batch_layout():
    import dgl
    graphs=[dgl.graph(([0,1],[1,0]),num_nodes=116) for _ in range(4)]
    for i,g in enumerate(graphs):g.ndata['id']=torch.full((116,),i)
    batch=dgl.batch(graphs);src,dst=batch.edges()
    assert torch.equal(src//116,dst//116)
    assert batch.ndata['id'].reshape(4,116)[:,0].tolist()==[0,1,2,3]
