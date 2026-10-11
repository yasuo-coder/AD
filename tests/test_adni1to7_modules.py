"""Synthetic contracts, including counterfactual hidden targets and query batches."""
import copy
import unittest
import torch
from BrainPromptPro.layers.masked_connectome import (symmetric_fc_mask,masked_topology,
    SymmetricConnectionDecoder,reconstruction_loss,symmetric_kl)
from BrainPromptPro.layers.task_token_fusion import TaskSpecificROIFusion,ParameterMatchedROIFusion
from BrainPromptPro.layers.reliability_population import ReliabilityReferencePopulation,stability_reliability


class Modules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4);torch.set_num_threads(2)
        x=torch.randn(4,16,16);self.fc=((x+x.transpose(-1,-2))/5).clamp(-1,1)
        self.fc.diagonal(dim1=-2,dim2=-1).fill_(1)
    def mask(self,ratio=.1):
        return symmetric_fc_mask(self.fc,ratio,torch.Generator().manual_seed(17))
    def population(self):
        return ReliabilityReferencePopulation(torch.randn(12,8),torch.linspace(.1,1,12),
            [f'synthetic-{i}' for i in range(12)],[f'synthetic-{i}' for i in range(12)],'synthetic-v1',16,topk=4)
    def test_fc_mask_symmetry(self):
        for r in [.05,.10,.15]:
            x,m=self.mask(r);self.assertTrue(torch.equal(x,x.transpose(-1,-2)));self.assertTrue(torch.equal(m,m.transpose(-1,-2)))
            self.assertTrue(torch.equal(x[m],self.fc[m]));self.assertTrue(m.diagonal(dim1=-1,dim2=-2).all());self.assertTrue((x[~m]==0).all())
    def test_masked_edge_no_leakage(self):
        x,m=self.mask(.4);other=self.fc.clone();other[~m]=-other[~m]
        y,_=symmetric_fc_mask(other,.4,torch.Generator().manual_seed(17))
        self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(masked_topology(x,m),masked_topology(y,m)))
        self.assertTrue((masked_topology(x,m)[~m]==0).all())
    def test_double_view_and_resume_generator(self):
        g=torch.Generator().manual_seed(5);_,a=symmetric_fc_mask(self.fc,.15,g);state=g.get_state()
        _,b=symmetric_fc_mask(self.fc,.15,g);self.assertFalse(torch.equal(a,b))
        g.set_state(state);_,c=symmetric_fc_mask(self.fc,.15,g);self.assertTrue(torch.equal(b,c))
    def test_reconstruction_loss(self):
        pred=torch.zeros(1,2,2,requires_grad=True);target=torch.tensor([[[1.,-.5],[-.5,1.]]],requires_grad=True)
        vis=torch.eye(2,dtype=torch.bool)[None];loss=reconstruction_loss(pred,target,vis)
        self.assertAlmostEqual(float(loss),.25);loss.backward();self.assertIsNone(target.grad)
        self.assertEqual(float(pred.grad[0,1,0]),0);self.assertEqual(float(pred.grad[0,0,1]),1)
    def test_decoder_gradient_symmetry(self):
        for mode in ['dot','pair']:
            h=torch.randn(4,16,20,requires_grad=True);decoder=SymmetricConnectionDecoder(20,mode,8)
            p=decoder(h);self.assertTrue(torch.allclose(p,p.transpose(-1,-2),atol=1e-6))
            _,m=self.mask();reconstruction_loss(p,self.fc,m).backward();self.assertGreater(float(h.grad.abs().sum()),0)
    def test_normalized_decoder_large_common_component(self):
        h=(100*torch.randn(2,1,20)+torch.randn(2,16,20)).requires_grad_()
        for mode in ['dot','pair']:
            decoder=SymmetricConnectionDecoder(20,mode,8,normalization='centered_layernorm')
            prediction=decoder(h)
            self.assertTrue(torch.isfinite(prediction).all())
            torch.testing.assert_close(prediction,prediction.transpose(-1,-2))
            self.assertLess(float((prediction.abs()>.99).float().mean()),.01)
            prediction.square().mean().backward(retain_graph=True)
            self.assertGreater(float(decoder.project.weight.grad.abs().sum()),0)
    def test_no_mask_loss(self):
        p=self.fc.clone().requires_grad_();loss=reconstruction_loss(p,self.fc,torch.ones_like(p,dtype=torch.bool));loss.backward();self.assertEqual(float(loss),0)
    def test_consistency_stability(self):
        a=torch.randn(8,5)*100;b=torch.randn(8,5)*100
        self.assertTrue(torch.isfinite(symmetric_kl(a,b)));self.assertAlmostEqual(float(symmetric_kl(a,a)),0)
        self.assertTrue(torch.allclose(symmetric_kl(a,b),symmetric_kl(b,a)))
    def test_roi_token_shape_and_gradients(self):
        m=TaskSpecificROIFusion(20,30);h=torch.randn(3,16,20,requires_grad=True);p=torch.randn(5,30)
        o=m(h,p);self.assertEqual(o['tokens'].shape,(3,5,20));self.assertEqual(o['attention'].shape,(3,5,16));self.assertEqual(o['logits'].shape,(3,5))
        torch.nn.functional.cross_entropy(o['logits'],torch.tensor([0,1,4])).backward()
        self.assertGreater(float(m.readout.q.weight.grad.abs().sum()),0);self.assertGreater(float(h.grad.abs().sum()),0)
    def test_disease_token_order(self):
        with self.assertRaises(ValueError):TaskSpecificROIFusion(20,30,['AD','CN','SMC','EMCI','LMCI'])
    def test_no_label_query_leakage(self):
        import inspect
        self.assertEqual(list(inspect.signature(TaskSpecificROIFusion.forward).parameters),['self','roi','disease_prompt'])
    def test_class_head_uses_corresponding_token(self):
        m=TaskSpecificROIFusion(20,30);x=torch.randn(2,5,20);y=x.clone();y[:,2]+=1
        delta=m.class_logits(y)-m.class_logits(x)
        self.assertTrue(torch.equal(delta[:,[0,1,3,4]],torch.zeros(2,4)));self.assertTrue((delta[:,2]!=0).all())
    def test_random_semantic_control(self):
        m=TaskSpecificROIFusion(116,2048,mode='random');h=torch.randn(3,16,116);a=torch.randn(5,2048)
        torch.testing.assert_close(m(h,a)['logits'],m(h,a.flip(0))['logits'])
        baseline=TaskSpecificROIFusion(116,2048)
        self.assertEqual(sum(p.numel() for p in m.parameters()),sum(p.numel() for p in baseline.parameters()))
    def test_learned_queries_without_disease_input(self):
        m=TaskSpecificROIFusion(116,2048,mode='learned_query');h=torch.randn(3,16,116);a=torch.randn(5,2048)
        o=m(h,a);torch.testing.assert_close(o['logits'],m(h,torch.zeros_like(a))['logits'])
        o['logits'].square().sum().backward();self.assertGreater(float(m.queries.grad.abs().sum()),0)
    def test_parameter_matched_pool_control(self):
        m=ParameterMatchedROIFusion(116,2048);baseline=TaskSpecificROIFusion(116,2048)
        b=sum(p.numel() for p in baseline.parameters());c=sum(p.numel() for p in m.parameters())
        self.assertLess(abs(c-b)/b,.001)
        x=m(torch.randn(3,16,116),torch.randn(5,2048));self.assertIsNone(x['attention'])
        self.assertEqual(x['logits'].shape,(3,5));x['logits'].square().sum().backward()
        self.assertGreater(float(m.hidden.weight.grad.abs().sum()),0)
    def test_population_reference_isolation(self):
        with self.assertRaises(ValueError):ReliabilityReferencePopulation(torch.randn(3,8),torch.ones(3),['a','b','test'],['a','b','c'],'v',16,topk=2)
        m=self.population();out=m(torch.randn(2,8),torch.ones(2),['synthetic-0','unseen'],'synthetic-v1')
        self.assertEqual(float(out['attention'][0,0]),0)
        with self.assertRaises(ValueError):m(torch.randn(2,8),torch.ones(2),['a','b'],'changed')
    def test_population_batch_independence(self):
        m=self.population().eval();q=torch.randn(7,8);r=torch.rand(7);ids=['unseen']*7
        full=m(q,r,ids,'synthetic-v1')['residual'];parts=torch.cat([m(q[i:i+1],r[i:i+1],ids[i:i+1],'synthetic-v1')['residual'] for i in range(7)])
        torch.testing.assert_close(full,parts,atol=2e-7,rtol=1e-5)
        reverse=m(q.flip(0),r.flip(0),ids,'synthetic-v1')['residual'].flip(0);torch.testing.assert_close(full,reverse)
    def test_reliability_score(self):
        x=torch.randn(3,1,8).expand(-1,4,-1).clone();r,u=stability_reliability(x);torch.testing.assert_close(r,torch.ones(3))
        x[:,0]*=-1;s,_=stability_reliability(x);self.assertTrue((s<r).all())
    def test_edge_gradient_and_reference_use(self):
        m=self.population();q=torch.randn(3,8);r=torch.rand(3);o=m(q,r,['unseen']*3,'synthetic-v1');o['residual'].square().sum().backward()
        self.assertGreater(float(m.scorer[0].weight.grad.abs().sum()),0)
        original=o['residual'].detach();m.reference.zero_();changed=m(q,r,['unseen']*3,'synthetic-v1')['residual'];self.assertGreater(float((original-changed).abs().sum()),0)
    def test_checkpoint_reload(self):
        m=self.population().eval();c=copy.deepcopy(m);c.load_state_dict(m.state_dict());q=torch.randn(2,8);r=torch.ones(2)
        torch.testing.assert_close(m(q,r,['a','b'],'synthetic-v1')['residual'],c(q,r,['a','b'],'synthetic-v1')['residual'])
        state=m.state_dict();state['_extra_state']['encoder_version']='changed'
        with self.assertRaises(ValueError):c.load_state_dict(state)


if __name__=='__main__':unittest.main()
