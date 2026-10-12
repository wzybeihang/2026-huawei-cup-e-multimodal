"""Verify packaged result provenance and file checksums. Run after extracting the ZIP."""
from pathlib import Path
import argparse, hashlib, json

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parent))
    a=p.parse_args();root=Path(a.root)
    manifest=json.loads((root/'交付文件SHA256.json').read_text(encoding='utf-8'))
    for rel,expected in manifest['files'].items():
        assert sha(root/rel)==expected, f'File changed: {rel}'
    result=root/'实验结果'
    config=json.loads((result/'experiment_config.json').read_text(encoding='utf-8'))
    for name,expected in config['code_sha256'].items():
        assert sha(root/name)==expected, f'Training source changed: {name}'
    report=json.loads((result/'核验报告.json').read_text(encoding='utf-8'))
    assert report['status']=='passed'
    assert report['models']==9 and report['total_metric_rows']==1017
    assert report['attachment3_samples']==30 and report['figures']==13
    print(json.dumps({'status':'passed','checked_files':len(manifest['files']),
                      'model_training_source_matches':True,'scope':'file integrity; not predictive accuracy'}))

if __name__=='__main__':main()
