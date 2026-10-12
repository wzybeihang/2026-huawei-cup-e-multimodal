"""Competition Q2 protocol. Conditions are study choices, not prescribed numeric rates."""
import hashlib
import numpy as np

KEYS=('text','audio','vision')
DIMS=(768,74,35)
RATES=(.1,.3,.5,.7)
POSITIONS=('start','middle','end','hashed')
COMBINATIONS=((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))
INIT=2026  # Fixed initialization for reproducibility; NOT an experimental factor.

def cases():
    result=[dict(name='clean',mods=(),rate=0.,position='none',text_key='clean')]
    for mods in COMBINATIONS:
        modal='+'.join('TAV'[m] for m in mods)
        for rate in RATES:
            for position in POSITIONS:
                result.append(dict(name=f'{modal}_{rate:.1f}_{position}',mods=mods,rate=rate,
                                   position=position,text_key=f'{rate:.1f}_{position}' if 0 in mods else 'clean'))
    return result

CASES=cases()
SELECT_CASES=[0]+[i for i,c in enumerate(CASES) if c['mods'] in ((0,),(1,),(2,),(0,1,2))
                 and c['rate']==.3 and c['position']=='middle']
VARIANTS={
 'matrix_full':dict(method='matrix',augment=True,temporal=.05),
 'function_full':dict(method='function',augment=True,adaptive=True,memory=True),
 'baseline_clean':dict(method='baseline',augment=False),
 'baseline_aug':dict(method='baseline',augment=True),
 'matrix_no_temporal':dict(method='matrix',augment=True,temporal=0.),
 'matrix_no_aug':dict(method='matrix',augment=False,temporal=.05),
 'function_no_adaptive':dict(method='function',augment=True,adaptive=False,memory=True),
 'function_no_memory':dict(method='function',augment=True,adaptive=True,memory=False),
 'function_no_aug':dict(method='function',augment=False,adaptive=True,memory=True),
}

def content_mask(tb):
    """Use visible SEP as end marker; interior zero tokens remain real, missing positions."""
    ids=np.asarray(tb)[:,0].astype(np.int64); att=np.asarray(tb)[:,1]>0
    domain=np.zeros_like(att)
    for i in range(len(ids)):
        sep=np.where((ids[i]==102)&att[i])[0]
        if not len(sep):
            raise ValueError('No visible SEP: effective endpoint is ambiguous; supply verified sequence lengths')
        domain[i,1:int(sep[-1])]=True
    observed=domain & att & (ids!=0) & (ids!=103)
    return domain,observed

def make_mask(domain,observed,ids,case):
    mask=observed.copy();intervals=np.full((len(domain),3,2),-1,dtype=np.int16)
    if not case['mods']:return mask,intervals
    for i in range(len(domain)):
        positions=np.flatnonzero(domain[i]);n=len(positions)
        if n<2:continue
        width=min(n-1,max(1,int(np.floor(n*case['rate']+.5))))
        for m in case['mods']:
            if case['position']=='start':a=0
            elif case['position']=='middle':a=(n-width)//2
            elif case['position']=='end':a=n-width
            elif case['position']=='hashed':
                code=hashlib.sha256(f'{ids[i]}|{case["rate"]}|{m}|q2-local-v2'.encode()).digest()
                a=int.from_bytes(code[:8],'little')%(n-width+1)
            else:raise ValueError(case)
            b=a+width
            visible=np.flatnonzero(observed[i,positions,m])
            # Keep at least one pre-existing observation. Never create whole-modality dropout.
            if len(visible) and ((visible>=a)&(visible<b)).all():
                if visible[-1]-a >= b-visible[0]-1:b=int(visible[-1])
                else:a=int(visible[0])+1
            if b>a:
                mask[i,positions[a:b],m]=False
                intervals[i,m]=(int(positions[a]),int(positions[b-1])+1)
    return mask,intervals
