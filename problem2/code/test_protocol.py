import unittest
import numpy as np
import torch
from protocol import CASES,VARIANTS,content_mask,make_mask
from models import LocalMissingModel,task_loss,LowRankTemporal

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(31)
        self.d=torch.ones(2,6,dtype=torch.bool);self.o=torch.ones(2,6,3,dtype=torch.bool)
        self.f={k:torch.randn(2,6,n) for k,n in zip(('text','audio','vision'),(768,74,35))}
    def test_conditions(self):
        self.assertEqual(len(CASES),113);self.assertEqual(len({c['name'] for c in CASES}),113)
    def test_interior_text_missing_not_padding(self):
        tb=np.zeros((1,3,50));tb[0,0,:5]=[101,9,0,8,102];tb[0,1,[0,1,3,4]]=1
        d,o=content_mask(tb);self.assertEqual(d.sum(),3);self.assertFalse(o[0,2]);self.assertTrue(d[0,2])
    def test_local_masks_keep_observation_and_padding(self):
        d=np.zeros((2,50),bool);d[:,1:11]=True;o=np.repeat(d[:,:,None],3,2);o[1,:,2]=False;o[1,5,2]=True
        for c in CASES:
            m,interval=make_mask(d,o,['a','b'],c)
            self.assertFalse(np.any(m&~o));self.assertTrue(m.any(1).all())
            for i in range(2):
                for k in c['mods']:
                    ix=np.flatnonzero(o[i,:,k]&~m[i,:,k])
                    a,b=interval[i,k]
                    if len(ix):self.assertTrue(((ix>=a)&(ix<b)).all())
    def test_missing_payload_invariance_and_gradients(self):
        self.o[:,2:4]=False
        bad={k:v.clone() for k,v in self.f.items()}
        for v in bad.values():v[:,2:4]=float('nan')
        for spec in VARIANTS.values():
            m=LocalMissingModel(spec).eval();a=m(self.f,self.o,self.d);b=m(bad,self.o,self.d)
            self.assertTrue(torch.allclose(a['logits'],b['logits']))
            (task_loss(a,torch.tensor([0,2]),torch.tensor([-1.,1.])).mean()+.05*a['aux']).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    def test_covariance_grows_without_observation(self):
        m=LocalMissingModel(VARIANTS['function_full']).eval();self.o[:,2:5]=False
        z=m(self.f,self.o,self.d)
        self.assertTrue((z['variance'][:,3]>z['variance'][:,2]).all())
        self.assertTrue(torch.allclose(z['states'][:,3],z['states'][:,2]))
    def test_completion_preserves_observed(self):
        z=torch.randn(2,6,3,32).tanh();self.o[:,2:4,0]=False
        r,aux=LowRankTemporal()(z*self.o[...,None],self.o,self.d)
        mask=self.o[...,None].expand_as(z)
        self.assertTrue(torch.equal(r[mask],z[mask]));self.assertTrue(torch.isfinite(aux))
    def test_no_complete_missing_nan(self):
        for spec in VARIANTS.values():
            m=LocalMissingModel(spec).eval()
            z=m(self.f,torch.zeros_like(self.o),self.d)
            self.assertTrue(torch.isfinite(z['raw']).all())

if __name__=='__main__':unittest.main()
