import numpy as np

LABELS=('Negative','Neutral','Positive')
RULES=[{'kind':'classification_projected'}]+[{'kind':'threshold','delta':v} for v in (0,.05,.1,.15,.2,.3,.4,.5,.6,.8)]

def decode(logits,raw,rule):
    if rule['kind']=='classification_projected':
        c=logits.argmax(-1);mag=np.maximum(np.abs(raw),1e-6)
        r=np.where(c==0,-mag,np.where(c==2,mag,0))
    else:
        c=np.where(raw < -rule['delta'],0,np.where(raw>rule['delta'],2,1));r=np.where(c==1,0,raw)
    return c.astype(np.int64),r.astype(np.float32)

def metrics(y,r,c,p):
    cm=np.zeros((3,3),np.int64);np.add.at(cm,(y,c),1)
    den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den>0)
    return dict(accuracy=float((y==c).mean()),macro_f1=float(f.mean()),weighted_f1=float((f*cm.sum(1)).sum()/len(y)),
                mae=float(np.abs(r-p).mean()),pearson=float(np.corrcoef(r,p)[0,1]) if np.std(p)>0 and np.std(r)>0 else None,
                f1_negative=float(f[0]),f1_neutral=float(f[1]),f1_positive=float(f[2]),confusion=cm.tolist())

def choose_rule(y,r,outputs):
    best=None
    for rule in RULES:
        values=[metrics(y,r,*decode(*p,rule)) for p in outputs]
        def score(m):return m['mae']+.5*(1-m['macro_f1'])
        loss=.5*score(values[0])+.5*np.mean([score(m) for m in values[1:]])
        row=dict(rule=rule,score=float(loss),clean=values[0],stress=values[1:])
        if best is None or loss<best['score']:best=row
    return best
