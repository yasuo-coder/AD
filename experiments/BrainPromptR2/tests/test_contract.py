import unittest
import numpy as np
import torch
from r2.common import *
from r2.features import fit_features

class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=torch.load(ROOT/'prepared/data.private.pt',weights_only=True,map_location='cpu')
        cls.plans=read(ROOT/'prepared/plans.private.json')
    def test_subject_partition_and_repeated_coverage(self):
        groups=np.array(self.data['groups'])
        for repeat in range(3):
            val=[]
            for p in self.plans[3*repeat:3*repeat+3]:
                parts=[set(groups[p[k]]) for k in ['train','stop','validation']]
                for i in range(3):
                    for j in range(i):self.assertFalse(parts[i]&parts[j])
                self.assertEqual(set.union(*parts),set(groups));val+=list(parts[2])
            self.assertEqual(len(val),len(set(val)));self.assertEqual(len(val),715)
    def test_preprocessing_ignores_validation(self):
        p=self.plans[0];altered=dict(self.data)
        altered['fc']=self.data['fc'].clone();altered['fc'][p['validation']]+=99
        altered['metadata']=self.data['metadata'].clone();altered['metadata'][p['validation']]+=99
        x,t,w=fit_features(self.data,p['train'],'full');xx,tt,ww=fit_features(altered,p['train'],'full')
        for k in ['mean','std','known_sites']:np.testing.assert_array_equal(t[k],tt[k])
        np.testing.assert_array_equal(x['x'][p['train']],xx['x'][p['train']])
        self.assertEqual(x['x'].shape[1],xx['x'].shape[1])
    def test_prior_unique_subject_and_logit_identity(self):
        d={'groups':[0,0,1,2,3,4],'y':torch.tensor([0,0,1,2,3,4])}
        np.testing.assert_allclose(subject_prior(d,list(range(6))),np.ones(5)/5)
        logits=np.array([[1.,2.,-1.,0.,3.]])
        prior=np.array([.4,.2,.1,.2,.1])
        for tau in [0,.25,1]:np.testing.assert_allclose(corrected(softmax(logits),prior,tau),softmax(logits-tau*np.log(prior)))
    def test_fc_is_6670(self):
        d,_,_=fit_features(self.data,self.plans[0]['train']);self.assertEqual(d['x'].shape,(1051,6670))

if __name__=='__main__':unittest.main()
