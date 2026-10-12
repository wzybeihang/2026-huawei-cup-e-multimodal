"""Minimal safe reader and dynamic zero padding, independent of ML frameworks."""
from pathlib import Path
import os,sys
if os.environ.get('Q1_APPROVED_SITE_PACKAGES'):sys.path.insert(0,os.environ['Q1_APPROVED_SITE_PACKAGES'])
import numpy as np

def load_sample(root,sample_id):
    with np.load(Path(root)/'features'/(sample_id+'.npz'),allow_pickle=False) as z:
        return {key:z[key] for key in z.files}

def padded_batch(root,sample_ids):
    values=[load_sample(root,sid) for sid in sample_ids]
    lengths=np.array([len(v['word_intervals']) for v in values],dtype=np.int32)
    maximum=int(max(lengths,default=0))
    output={'sample_ids':np.array(sample_ids),'lengths':lengths,
            'seq_mask':np.arange(maximum)[None,:]<lengths[:,None]}
    for key in ['text_aligned','audio_aligned','visual_aligned','word_intervals','text_mask','audio_mask','visual_mask']:
        if not values:continue
        batch=np.zeros((len(values),maximum)+values[0][key].shape[1:],dtype=values[0][key].dtype)
        for i,v in enumerate(values):batch[i,:lengths[i]]=v[key]
        output[key]=batch
    return output

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('results');a=p.parse_args()
    ids=[x['sample_id'] for x in json.loads((Path(a.results)/'manifest.json').read_text(encoding='utf-8'))]
    batch=padded_batch(a.results,ids)
    print({k:tuple(v.shape) for k,v in batch.items()})
