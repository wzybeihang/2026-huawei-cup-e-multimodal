"""Download frozen public weights over verified HTTPS; verify official SHA256."""
from pathlib import Path
import argparse, hashlib, json, time
import requests

MODELS = {
 'bert': {'repo':'google-bert/bert-base-uncased','revision':'0a6aa9128b6194f4f3c4db429b6cb4891cdb421b','sha256':'68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3','files':['config.json','vocab.txt','tokenizer.json','tokenizer_config.json','model.safetensors']},
 'wav2vec2': {'repo':'facebook/wav2vec2-base-960h','revision':'22aad52d435eb6dbaf354bdad9b0da84ce7d6156','sha256':'8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a','files':['config.json','vocab.json','preprocessor_config.json','tokenizer_config.json','special_tokens_map.json','model.safetensors']}
}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),''): 
   if not b:break
   h.update(b)
 return h.hexdigest()
def download(url,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():return
 print('download',url,flush=True)
 with requests.get(url,stream=True,timeout=(30,180)) as r:
  r.raise_for_status()
  with open(path.with_suffix(path.suffix+'.partial'),'wb') as f:
   for b in r.iter_content(1024*1024): f.write(b)
 path.with_suffix(path.suffix+'.partial').replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--endpoint',default='https://huggingface.co');ap.add_argument('--modelscope-weights',action='store_true');ap.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));args=ap.parse_args();root=Path(args.root);records=[]
 for name,m in MODELS.items():
  for fn in m['files']:
   path=root/'models'/name/fn;url=f"{args.endpoint}/{m['repo']}/resolve/{m['revision']}/{fn}"
   if fn=='model.safetensors' and args.modelscope_weights:
    repo,rev=('AI-ModelScope/bert-base-uncased','e655d956b91d226a698bac3dee0acba528f76eaf') if name=='bert' else ('facebook/wav2vec2-base-960h','e9c7a4d3fe3b840a95bfcdf1c38e9959cc8bb1a2')
    url=f'https://modelscope.cn/api/v1/models/{repo}/repo?Revision={rev}&FilePath=model.safetensors'
   download(url,path);digest=sha(path)
   if fn=='model.safetensors' and digest!=m['sha256']:raise RuntimeError('Official weight checksum mismatch')
   records.append(dict(model=name,repository=m['repo'],revision=m['revision'],file=str(path.relative_to(root)),download_url=url,sha256=digest,bytes=path.stat().st_size))
 url='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';path=root/'models'/'face_landmarker.task';download(url,path)
 records.append(dict(model='face_landmarker',revision='float16/1',download_url=url,file=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size))
 (root/'config'/'model_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 print('Model downloads and checksums complete.',flush=True)
if __name__=='__main__':main()
