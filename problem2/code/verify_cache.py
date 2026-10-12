from pathlib import Path
import argparse,json
import numpy as np
from prepare_data import sha,write_json
from protocol import CASES

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--data',required=True)
    p.add_argument('--bert',required=True);p.add_argument('--record-checksums',action='store_true');a=p.parse_args()
    root=Path(a.cache);identity=json.loads((root/'complete.json').read_text(encoding='utf-8'))
    assert sha(a.data)==identity['dataset_sha256']
    assert sha(Path(a.bert)/'model.safetensors')==identity['bert_sha256']
    assert sha(Path(__file__).with_name('prepare_data.py'))==identity['prepare_code_sha256']
    assert sha(Path(__file__).with_name('protocol.py'))==identity['protocol_sha256']
    paths=[root/'scaler.npz']
    for split,n in [('train',3395),('valid',728)]:
        paths.append(root/split/'meta.npz')
        with np.load(root/split/'meta.npz',allow_pickle=False) as z:assert len(z['ids'])==n
        for key in {c['text_key'] for c in CASES}:
            path=root/split/f'text_{key}.npy';assert np.load(path,mmap_mode='r').shape==(n,50,768);paths.append(path)
    actual={str(path.relative_to(root)).replace('\\','/'):sha(path) for path in paths}
    manifest=root/'cache_checksums.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf-8'))==actual,'Cache contents changed'
    elif a.record_checksums:write_json(manifest,actual)
    else:raise ValueError('No cache checksum record; audit or rebuild the cache first')
    print(json.dumps(dict(status='passed',cache_files=len(paths),original_hashes_match=True)))
if __name__=='__main__':main()
