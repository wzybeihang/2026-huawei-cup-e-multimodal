"""Same frozen BERT for all inputs; remove text BEFORE contextual encoding.
Only official train fits means/scales. No external sentiment data.
"""
from pathlib import Path
import argparse,hashlib,json,pickle,time
import numpy as np
import torch
from protocol import KEYS,DIMS,CASES,content_mask,make_mask

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def read_pickle(path,split=None):
    with Path(path).open('rb') as f:d=pickle.load(f)
    if split:d=d[split]
    elif set(d)=={'test'}:d=d['test']
    return d

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def status(event,**kw):print(json.dumps(dict(event=event,**kw),ensure_ascii=False),flush=True)

def source_fields(fields,ids):
    tb=np.asarray(fields['text_bert'],dtype=np.int64)
    if tb.shape!=(len(ids),3,50):raise ValueError('Expected aligned_50 interface')
    domain,to=content_mask(tb)
    observed=[to]
    for k,dim in zip(KEYS[1:],DIMS[1:]):
        a=np.asarray(fields[k]);assert a.shape==(len(ids),50,dim)
        observed.append(domain & np.isfinite(a).all(-1)&np.any(a!=0,-1))
    o=np.stack(observed,-1)
    return tb,domain,o,np.asarray(ids,dtype=str)

@torch.inference_mode()
def encode(model,tb,observed_text,device,batch_size=64):
    result=np.empty((len(tb),50,768),np.float32)
    for at in range(0,len(tb),batch_size):
        sl=slice(at,at+batch_size);x=torch.tensor(tb[sl].copy(),device=device,dtype=torch.long)
        is_special=(x[:,0]==101)|(x[:,0]==102)
        available=torch.tensor(observed_text[sl],device=device)|is_special
        x[:,0]=torch.where(available,x[:,0],0)
        x[:,1]=x[:,1]*available.long()
        z=model(input_ids=x[:,0],attention_mask=x[:,1],token_type_ids=x[:,2]).last_hidden_state
        result[sl]=z.cpu().numpy()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--bert',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--old-cache');p.add_argument('--only-base',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    root=Path(a.cache);root.mkdir(parents=True,exist_ok=True)
    identity={'dataset_sha256':sha(a.data),'bert_sha256':sha(Path(a.bert)/'model.safetensors'),
              'version':'q2-official-local-v2','protocol_sha256':sha(Path(__file__).with_name('protocol.py')),
              'prepare_code_sha256':sha(__file__),'text_storage':'standardized float16; restored to float32 at model input'}
    meta=root/'identity.json'
    if meta.exists() and json.loads(meta.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Cache identity differs; use another cache directory')
    write_json(meta,identity)
    from transformers import BertModel
    model=BertModel.from_pretrained(a.bert,local_files_only=True).to(a.device).eval();model.requires_grad_(False)
    all_data=read_pickle(a.data);prepared={};stats={}
    for split in ('train','valid'):
        fields=all_data[split];tb,domain,obs,ids=source_fields(fields,fields['id'])
        y=np.asarray(fields['regression_labels'],np.float32).reshape(-1)
        cls=np.asarray(fields['classification_labels'],np.int64).reshape(-1)
        assert np.array_equal(cls,np.where(y<0,0,np.where(y>0,2,1)))
        arrays={k:np.asarray(fields[k],np.float32) for k in ('audio','vision')}
        dest=root/split;dest.mkdir(exist_ok=True)
        old=Path(a.old_cache)/f'{split}.npz' if a.old_cache else None
        # A read-only acceleration: verify original data/BERT hashes before using prior CLEAN representations.
        if old and old.exists():
            oldid=json.loads((old.parent/'complete.json').read_text(encoding='utf-8'))
            assert oldid['data_sha256']==identity['dataset_sha256'] and oldid['bert_weights_sha256']==identity['bert_sha256']
            with np.load(old,allow_pickle=False) as z:
                assert np.array_equal(z['domain'],domain) and np.array_equal(z['observed'],obs)
                assert np.array_equal(z['ids'],ids)
                clean_text=z['feature_text']
            if split=='train':
                with np.load(old.parent/'scaler.npz') as s:
                    stats={k:{v:s[k+'_'+v] for v in ('mean','scale')} for k in KEYS}
        else:
            clean_text=encode(model,tb,obs[:,:,0],a.device)
            arrays['text']=clean_text
            if split=='train':
                for m,k in enumerate(KEYS):
                    values=arrays[k][obs[:,:,m]]
                    stats[k]={'mean':values.mean(0,dtype=np.float64).astype(np.float32),
                              'scale':np.maximum(values.std(0,dtype=np.float64),1e-4).astype(np.float32)}
            clean_text=(clean_text-stats['text']['mean'])/stats['text']['scale']
        clean_text=np.where(obs[:,:,0,None],np.clip(clean_text,-8,8),0).astype(np.float16)
        for m,k in enumerate(KEYS[1:],1):
            arrays[k]=np.where(obs[:,:,m,None],np.clip((arrays[k]-stats[k]['mean'])/stats[k]['scale'],-8,8),0)
        np.save(dest/'text_clean.npy',clean_text)
        np.savez_compressed(dest/'meta.npz',domain=domain,observed=obs,ids=ids,regression=y,classification=cls,
                            tb=tb,raw_text=np.asarray(fields['raw_text'],dtype=str),audio=arrays['audio'],vision=arrays['vision'])
        prepared[split]=(tb,domain,obs,ids)
        status('BASE_READY',split=split,samples=len(ids),natural_missing=[int((domain&~obs[:,:,m]).sum()) for m in range(3)])
    np.savez(root/'scaler.npz',**{k+'_'+v:stats[k][v] for k in KEYS for v in ('mean','scale')})
    if a.only_base:return
    for split,(tb,domain,obs,ids) in prepared.items():
        dest=root/split
        for case in [c for c in CASES if c['mods']==(0,)]:
            target=dest/f'text_{case["text_key"]}.npy'
            if target.exists():
                status('TEXT_CACHE_REUSED',split=split,case=case['text_key']);continue
            masked,_=make_mask(domain,obs,ids,case)
            clock=time.perf_counter()
            z=encode(model,tb,masked[:,:,0],a.device)
            z=np.where(masked[:,:,0,None],np.clip((z-stats['text']['mean'])/stats['text']['scale'],-8,8),0)
            np.save(target,z.astype(np.float16))
            status('TEXT_REENCODED',split=split,case=case['text_key'],samples=len(ids),seconds=time.perf_counter()-clock)
    write_json(root/'complete.json',identity);status('CACHE_COMPLETED',train=3395,valid=728,text_conditions=17)

if __name__=='__main__':main()
