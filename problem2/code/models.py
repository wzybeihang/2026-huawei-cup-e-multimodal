"""Trainable local-missing sentiment models. No direct-observation bypass.
The estimator is a robust, diagonal state-estimation network, not a certified H-infinity controller.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from protocol import KEYS,DIMS

class LowRankTemporal(nn.Module):
    def __init__(self,h=32,rank=8,temporal=.05,ridge=.03,iterations=20):
        super().__init__();self.h=h;self.rank=rank;self.temporal=temporal;self.ridge=ridge;self.iterations=iterations
        self.basis=nn.Parameter(torch.randn(3*h,rank)/math.sqrt(rank))

    def forward(self,z,observed,domain):
        x=z.flatten(2);mask=observed[...,None].expand(*observed.shape,self.h).flatten(2)
        weight=mask.to(x.dtype)/self.h
        adjacency=domain[:,1:]&domain[:,:-1]
        degree=F.pad(adjacency.to(x.dtype),(1,0))+F.pad(adjacency.to(x.dtype),(0,1))
        a=torch.einsum('btd,dr,ds->btrs',weight,self.basis,self.basis)
        a=a+(self.ridge+self.temporal*degree)[...,None,None]*torch.eye(self.rank,device=x.device)
        rhs=torch.einsum('btd,dr->btr',x*weight,self.basis)
        inverse=torch.cholesky_inverse(torch.linalg.cholesky(a))
        u=torch.einsum('btrs,bts->btr',inverse,rhs)
        for _ in range(self.iterations if self.temporal else 0):
            neighbors=F.pad(u[:,:-1]*adjacency[...,None],(0,0,1,0))+F.pad(u[:,1:]*adjacency[...,None],(0,0,0,1))
            u=torch.einsum('btrs,bts->btr',inverse,rhs+self.temporal*neighbors)
        reconstructed=torch.einsum('btr,dr->btd',u,self.basis)
        repaired=torch.where(mask,x,reconstructed.clamp(-1,1))
        repaired=torch.where(domain[...,None],repaired,0)
        auxiliary=((reconstructed-x.detach()).square()*weight).sum()/observed.sum().clamp_min(1)
        return repaired.reshape(*z.shape),auxiliary

class LocalMissingModel(nn.Module):
    def __init__(self,spec,hidden=32,rank=8):
        super().__init__();self.spec=dict(spec);self.h=hidden
        self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(d,hidden),nn.Tanh()) for d in DIMS])
        self.attention=nn.Linear(hidden,1)
        self.head=nn.Sequential(nn.Linear(2*hidden+3,64),nn.GELU(),nn.Dropout(.1))
        self.classifier=nn.Linear(64,3);self.regressor=nn.Linear(64,1)
        if spec['method']=='matrix':self.completion=LowRankTemporal(hidden,rank,spec.get('temporal',.05))
        if spec['method']=='function':
            self.r_logits=nn.Parameter(torch.full((3,hidden),math.log(.25/.70)))
            self.q_logits=nn.Parameter(torch.full((hidden,),math.log(.015/.185)))

    def forward(self,features,observed,domain):
        o=observed&domain[...,None];zs=[]
        for m,key in enumerate(KEYS):
            x=torch.where(o[...,m,None],features[key],0)
            zs.append(torch.where(o[...,m,None],self.encoders[m](x),0))
        z=torch.stack(zs,2);aux=z.new_zeros(());variances=None
        if self.spec['method']=='matrix':
            repaired,aux=self.completion(z,o,domain);states=repaired.mean(2)
        elif self.spec['method']=='baseline':
            states=z.sum(2)/o.sum(2,keepdim=True).clamp_min(1)
        else:
            r=.05+.95*torch.sigmoid(self.r_logits);q=.005+.20*torch.sigmoid(self.q_logits)
            state=z.new_zeros((z.shape[0],self.h));p=z.new_ones((z.shape[0],self.h));ss=[];pp=[]
            for t in range(z.shape[1]):
                if not self.spec.get('memory',True):state=state*0;p=p*0+1
                prior=p+q
                innovation=z[:,t]-state[:,None]
                quality=torch.ones_like(innovation)
                if self.spec.get('adaptive',True):
                    standardized=innovation.abs()/torch.sqrt(prior[:,None]+r[None])
                    quality=torch.clamp(.75/(standardized+1e-6),max=1.)
                precision=o[:,t,:,None].to(z.dtype)*quality/r[None]
                posterior=1/(1/prior+precision.sum(1))
                updated=state+posterior*(precision*innovation).sum(1)
                real=domain[:,t,None]
                state=torch.where(real,updated,state);p=torch.where(real,posterior,p)
                ss.append(torch.where(real,state,0));pp.append(p)
            states=torch.stack(ss,1);variances=torch.stack(pp,1)
        states=torch.where(domain[...,None],states,0)
        n=domain.sum(1,keepdim=True).clamp_min(1)
        weights=torch.softmax(self.attention(states).squeeze(-1).masked_fill(~domain,-1e4),1)
        weights=weights*domain;weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        pool=torch.cat([(states*weights[...,None]).sum(1),states.sum(1)/n,o.sum(1)/n],-1)
        hidden=self.head(pool)
        return {'logits':self.classifier(hidden),'raw':3*torch.tanh(self.regressor(hidden)).squeeze(-1),
                'aux':aux,'states':states,'variance':variances,'attention':weights}

def task_loss(output,classes,intensities):
    return F.cross_entropy(output['logits'],classes,reduction='none')+F.smooth_l1_loss(output['raw'],intensities,reduction='none')
