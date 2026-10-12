"""Run prescribed TASK outputs using a fixed, explicitly documented condition grid.
Random initialization is fixed, never used as a substitute for missingness/ablation experiments.
"""
from pathlib import Path
import argparse,csv,json,time,platform,shutil
import numpy as np
import torch
from protocol import KEYS,CASES,SELECT_CASES,VARIANTS,INIT,make_mask
from prepare_data import sha,write_json,status
from models import LocalMissingModel,task_loss
from evaluation import metrics,decode,choose_rule

def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Data:
    def __init__(self,root,split):
        self.root=Path(root)/split
        with np.load(self.root/'meta.npz',allow_pickle=False) as z:self.meta={k:z[k] for k in z.files}
        self.ids=self.meta['ids'];self.domain=self.meta['domain'];self.observed=self.meta['observed']
        self.text={key:np.load(self.root/f'text_{key}.npy',mmap_mode='r') for key in {c['text_key'] for c in CASES}}
        self.masks=[];self.intervals=[]
        for case in CASES:
            mask,spans=make_mask(self.domain,self.observed,self.ids,case)
            if np.any(mask&~self.observed):raise ValueError('A mask created new observations')
            if np.any(self.observed.any(1)&~mask.any(1)):raise ValueError('Artificial mask removed an entire modality')
            self.masks.append(mask);self.intervals.append(spans)
    def batch(self,ix,case,device):
        text=np.asarray(self.text[CASES[case]['text_key']][ix],dtype=np.float32)
        f={'text':torch.as_tensor(text,device=device)}
        for k in KEYS[1:]:f[k]=torch.as_tensor(self.meta[k][ix],device=device)
        o=torch.as_tensor(self.masks[case][ix],device=device,dtype=torch.bool)
        d=torch.as_tensor(self.domain[ix],device=device,dtype=torch.bool)
        # The mask is an explicit input and all unavailable vectors are zero.
        f={k:torch.where(o[...,m,None],f[k],0) for m,k in enumerate(KEYS)}
        return f,o,d

@torch.inference_mode()
def predict(model,data,case,device='cuda',batch_size=128):
    model.eval();logits=[];raw=[]
    for at in range(0,len(data.ids),batch_size):
        output=model(*data.batch(slice(at,at+batch_size),case,device))
        logits.append(output['logits'].cpu().numpy());raw.append(output['raw'].cpu().numpy())
    result=np.concatenate(logits),np.concatenate(raw)
    assert all(np.isfinite(v).all() for v in result)
    return result

def initialize(spec,device):
    torch.manual_seed(INIT)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(INIT)
    return LocalMissingModel(spec).to(device)

def train(name,spec,data,valid,args,identity):
    dest=Path(args.output)/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        marker=json.loads((dest/'complete.json').read_text(encoding='utf-8'))
        if marker['identity']!=identity:raise ValueError('Model output belongs to another protocol')
        status('MODEL_REUSED',model=name);return
    model=initialize(spec,args.device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    history=[];best=float('inf');wait=0;clock=time.perf_counter();stress=SELECT_CASES[-1]
    for epoch in range(1,args.epochs+1):
        model.train();order=np.random.default_rng(INIT+epoch).permutation(len(data.ids))
        loss_sum=0.;aux_sum=0.;n=0
        for bi,at in enumerate(range(0,len(order),args.batch_size)):
            ix=order[at:at+args.batch_size]
            cls=torch.as_tensor(data.meta['classification'][ix],device=args.device)
            reg=torch.as_tensor(data.meta['regression'][ix],device=args.device)
            clean=model(*data.batch(ix,0,args.device));primary=task_loss(clean,cls,reg).mean()
            if spec['augment']:
                # Balanced deterministic cycle over the experimental conditions, not a seed sweep.
                step=(epoch-1)*int(np.ceil(len(order)/args.batch_size))+bi
                c=1+(step*37)%(len(CASES)-1)
                losses=[task_loss(model(*data.batch(ix,k,args.device)),cls,reg) for k in (c,stress)]
                primary=primary+.5*torch.stack(losses).max(0).values.mean()
            loss=primary+.05*clean['aux']
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {name} epoch {epoch}')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            loss_sum+=primary.detach().item()*len(ix);aux_sum+=clean['aux'].detach().item()*len(ix);n+=len(ix)
        outputs=[predict(model,valid,c,args.device,args.batch_size) for c in SELECT_CASES]
        chosen=choose_rule(valid.meta['classification'],valid.meta['regression'],outputs)
        row=dict(epoch=epoch,task_loss=loss_sum/n,auxiliary_loss=aux_sum/n,selection=chosen)
        history.append(row)
        if chosen['score']<best-1e-8:
            best=chosen['score'];wait=0
            torch.save(dict(state_dict=model.state_dict(),spec=spec,epoch=epoch,selection=chosen,identity=identity),dest/'best.pt')
        else:wait+=1
        write_json(dest/'history.json',history)
        status('EPOCH',model=name,epoch=epoch,validation_selection=chosen['score'])
        if epoch>=8 and wait>=args.patience:break
        model.train()
    write_json(dest/'complete.json',dict(identity=identity,epochs_run=len(history),best_score=best,
                 train_seconds=time.perf_counter()-clock,parameters=sum(x.numel() for x in model.parameters())))

def evaluate(name,data,args,identity):
    dest=Path(args.output)/name
    if (dest/'grid_complete.json').exists():
        marker=json.loads((dest/'grid_complete.json').read_text(encoding='utf-8'))
        if marker.get('best_sha256')==sha(dest/'best.pt'):
            status('GRID_REUSED',model=name);return
    saved=torch.load(dest/'best.pt',map_location=args.device,weights_only=False)
    assert saved['identity']==identity
    model=initialize(saved['spec'],args.device);model.load_state_dict(saved['state_dict'])
    rows=[];classes=[];values=[];all_logits=[];all_raw=[]
    y=data.meta['classification'];r=data.meta['regression']
    for index,case in enumerate(CASES):
        logits,raw=predict(model,data,index,args.device,args.batch_size);c,p=decode(logits,raw,saved['selection']['rule'])
        m=metrics(y,r,c,p);m.pop('confusion')
        mask=data.masks[index];selected=list(case['mods'] or (0,1,2))
        before=int(data.observed[:,:,selected].sum());after=int(mask[:,:,selected].sum())
        denominator=int(data.domain.sum())*len(selected)
        rows.append(dict(model=name,condition=case['name'],modalities='+'.join('TAV'[m] for m in case['mods']) or 'none',
                         rate=case['rate'],position=case['position'],samples=len(y),removed_observations=before-after,
                         actual_new_missing_rate=(before-after)/max(before,1),total_missing_rate=1-after/max(denominator,1),**m))
        classes.append(c);values.append(p);all_logits.append(logits);all_raw.append(raw)
        if index%28==0:status('CONDITION',model=name,completed=index+1,total=len(CASES))
    csv_write(dest/'conditions.csv',rows)
    np.savez_compressed(dest/'predictions.npz',classes=np.stack(classes),intensities=np.stack(values),
                       logits=np.stack(all_logits),raw=np.stack(all_raw),ids=data.ids,
                       true_class=y,true_intensity=r,conditions=np.asarray([c['name'] for c in CASES]))
    write_json(dest/'grid_complete.json',dict(best_sha256=sha(dest/'best.pt'),conditions=len(CASES),samples=len(y)))

def export_masks(valid,out):
    np.savez_compressed(out/'validation_masks.npz',ids=valid.ids,domain=valid.domain,original=valid.observed,
                        observed=np.stack(valid.masks),intervals=np.stack(valid.intervals),
                        conditions=np.asarray([c['name'] for c in CASES]))
    summary=[]
    for i,c in enumerate(CASES):
        for m in range(3):
            selected=valid.intervals[i][:,m,0]>=0
            lengths=valid.intervals[i][selected,m,1]-valid.intervals[i][selected,m,0]
            summary.append(dict(condition=c['name'],modality=KEYS[m],requested_rate=c['rate'],position=c['position'],
                                affected_samples=int(selected.sum()),min_interval_positions=int(lengths.min()) if len(lengths) else 0,
                                max_interval_positions=int(lengths.max()) if len(lengths) else 0,
                                removed_observations=int((valid.observed[:,:,m]&~valid.masks[i][:,:,m]).sum()),
                                natural_whole_missing_samples=int((valid.observed[:,:,m].sum(1)==0).sum())))
    csv_write(out/'missingness_audit.csv',summary)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--models',nargs='*',default=list(VARIANTS))
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config=dict(version='official-requirements-v2',grid=CASES,selection_cases=SELECT_CASES,variants=VARIANTS,
                fixed_initialization=INIT,initialization_is_experiment_factor=False,epochs=args.epochs,patience=args.patience,
                batch_size=args.batch_size,lr=args.lr,device=args.device,hidden=32,rank=8,
                code_sha256={name:sha(Path(__file__).with_name(name)) for name in ('protocol.py','models.py','evaluation.py','prepare_data.py','run_experiments.py')},
                cache_identity=json.loads((Path(args.cache)/'complete.json').read_text(encoding='utf-8')),
                environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                task_requirements='local continuous zero blocks; type/rate/location/duration; retrained ablations; attachment3 inference; valid metrics/error analysis',
                study_choices='rates and grid are team-defined, not numerical mandates from the question')
    # Canonical JSON makes tuple-valued modality combinations comparable on a rerun.
    config=json.loads(json.dumps(config,ensure_ascii=False))
    configfile=out/'experiment_config.json'
    if configfile.exists() and json.loads(configfile.read_text(encoding='utf-8'))!=config:
        raise ValueError('Existing experiment configuration differs; use another output')
    write_json(configfile,config);identity=sha(configfile)
    train_data=Data(args.cache,'train');valid=Data(args.cache,'valid')
    assert len(train_data.ids)==3395 and len(valid.ids)==728
    assert not(set(train_data.ids)&set(valid.ids))
    export_masks(valid,out);shutil.copy2(Path(args.cache)/'scaler.npz',out/'scaler.npz')
    for name in args.models:
        if name not in VARIANTS:raise ValueError(name)
        train(name,VARIANTS[name],train_data,valid,args,identity);evaluate(name,valid,args,identity)
    status('REQUESTED_EXPERIMENTS_COMPLETE',models=args.models,conditions_each=len(CASES))

if __name__=='__main__':main()
