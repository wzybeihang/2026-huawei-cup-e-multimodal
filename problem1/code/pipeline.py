"""Problem 1: frozen three-modality extraction with complete timestamp provenance."""
from pathlib import Path
import argparse, hashlib, json, re, sys, time, traceback, platform, importlib, os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC
from num2words import num2words
from alignment import ctc_forced_align, overlap_pool, diagonal_kalman_filter

ROOT=Path(__file__).resolve().parents[1]
def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def normalize_words(text):
 excluded=[dict(start=m.start(),end=m.end(),text=m.group(),reason='bracketed_nonspoken_annotation') for m in re.finditer(r'\[[^\]]*\]',text)]
 words=[]
 for m in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)?",text):
  if any(m.start()<e['end'] and m.end()>e['start'] for e in excluded):continue
  val=m.group().replace('’',"'")
  if val[0].isdigit():val=num2words(float(val) if '.' in val else int(val.replace(',','')))
  norm=re.sub(r"[^A-Z']+",'|',val.upper()).strip('|')
  words.append(dict(index=len(words),text=m.group(),char_start=m.start(),char_end=m.end(),normalized=norm))
 if not words:raise ValueError('No spoken words')
 return words,excluded

class Extractor:
 def __init__(self,config,device):
  self.cfg=config;self.device=device
  self.bt=AutoTokenizer.from_pretrained(ROOT/'models'/'bert',local_files_only=True,use_fast=True)
  self.bm=AutoModel.from_pretrained(ROOT/'models'/'bert',local_files_only=True).to(device).eval()
  self.wp=Wav2Vec2Processor.from_pretrained(ROOT/'models'/'wav2vec2',local_files_only=True)
  self.wm=Wav2Vec2ForCTC.from_pretrained(ROOT/'models'/'wav2vec2',local_files_only=True).to(device).eval()
  from native_media import NativeMediaExtractor
  self.media=NativeMediaExtractor(ROOT/'models'/'face_landmarker.task',config)
 def text_and_times(self,text,wave,audio_start,audio_end):
  words,excluded=normalize_words(text)
  enc=self.bt(text,return_tensors='pt',return_offsets_mapping=True,truncation=False)
  offsets=enc.pop('offset_mapping')[0].numpy().astype(np.int32)
  if enc['input_ids'].shape[1]>512:raise ValueError('BERT length >512: explicit windows required')
  ids=enc['input_ids'][0].numpy().astype(np.int32)
  with torch.inference_mode():emb=self.bm(**{k:v.to(self.device) for k,v in enc.items()}).last_hidden_state[0].float().cpu().numpy()
  indptr=[0];indices=[];aligned=[]
  for w in words:
   ii=np.where((offsets[:,0]<w['char_end']) & (offsets[:,1]>w['char_start']))[0]
   if not len(ii):raise ValueError('Word without WordPiece')
   w['bert_token_indices']=ii.tolist();indices.extend(ii.tolist());indptr.append(len(indices));aligned.append(emb[ii].mean(0))
  vocab=self.wp.tokenizer.get_vocab();target=[];owner=[]
  for j,w in enumerate(words):
   if j:target.append(vocab['|']);owner.append(-1)
   for char in w['normalized']:
    if char not in vocab:raise ValueError('CTC unknown normalized character '+char)
    target.append(vocab[char]);owner.append(j if char!='|' else -1)
  processed=self.wp(wave,sampling_rate=16000,return_tensors='pt')
  with torch.inference_mode():lp=torch.log_softmax(self.wm(processed.input_values.to(self.device)).logits[0].float(),dim=-1).cpu().numpy()
  spans=ctc_forced_align(lp,target,blank_id=self.wp.tokenizer.pad_token_id)
  intervals=[];scores=[]
  for j,w in enumerate(words):
   chars=[x for x,o in zip(spans,owner) if o==j]
   a=chars[0]['start_frame'];b=chars[-1]['end_frame'];start=audio_start+a*320/16000;end=min(audio_end,audio_start+((b-1)*320+400)/16000)
   posterior=float(np.mean([x['posterior'] for x in chars]));intervals.append([start,end]);scores.append(posterior)
   w.update(ctc_start_frame=a,ctc_end_frame=b,ctc_receptive_end=end,ctc_posterior=posterior,ctc_low_quality=posterior<self.cfg['ctc_low_posterior_threshold'])
  for j in range(len(intervals)-1):intervals[j][1]=min(intervals[j][1],intervals[j+1][0])
  for w,(a,b) in zip(words,intervals):
   if b<=a:raise ValueError('Degenerate forced-aligned word interval')
   w.update(start_seconds=float(a),end_seconds=float(b))
  arr=dict(text_native=emb.astype(np.float32),text_token_ids=ids,text_token_offsets=offsets,text_aligned=np.asarray(aligned,dtype=np.float32),text_source_indptr=np.asarray(indptr,np.int32),text_source_indices=np.asarray(indices,np.int32),word_intervals=np.asarray(intervals,np.float64),word_ctc_posterior=np.asarray(scores,np.float32),text_mask=np.ones(len(words),bool),ctc_token_ids=np.asarray(target,np.int32),ctc_token_owner=np.asarray(owner,np.int32),ctc_token_intervals_frames=np.array([[x['start_frame'],x['end_frame']] for x in spans],np.int32),ctc_token_posterior=np.array([x['posterior'] for x in spans],np.float32))
  # Retain per-frame posterior matrix for independent CTC-path verification.
  arr['ctc_log_probabilities']=lp.astype(np.float32)
  meta=dict(original_text=text,words=words,excluded_annotations=excluded,bert_tokens=self.bt.convert_ids_to_tokens(ids.tolist()),ctc_vocabulary=vocab,ctc_frame_count=len(lp),ctc_time_convention='frame start=i*320/16000; receptive end=(i*320+400)/16000; word end capped at next word start; audio PTS offset added',normalization='bracket annotations excluded; letters/apostrophes uppercased; decimal/integer number names via num2words; punctuation becomes boundaries; all original character offsets retained')
  return arr,meta

def add_pool(arr,prefix,intervals,nearest=0.0):
 p=overlap_pool(arr[prefix+'_native'],arr[prefix+'_intervals'],intervals,quality=arr[prefix+'_quality'],valid=arr[prefix+'_native_mask'])
 # Preserve strict physical overlap pool and source CSR. Nearest estimates are separate.
 for key,value in p.items():arr[prefix+'_'+('aligned' if key=='features' else key)]=value
 arr[prefix+'_nearest']=np.zeros_like(p['features']);arr[prefix+'_nearest_mask']=np.zeros(len(intervals),bool);arr[prefix+'_nearest_source']=np.full(len(intervals),-1,np.int32);arr[prefix+'_nearest_distance']=np.zeros(len(intervals),np.float32)
 if nearest:
  valid=np.flatnonzero(arr[prefix+'_native_mask']);native=arr[prefix+'_intervals']
  for j,(s,e) in enumerate(intervals):
   if p['mask'][j] or not len(valid):continue
   centers=native[valid].mean(1);dist=np.maximum(np.maximum(s-centers,centers-e),0);pick=int(np.argmin(dist));k=int(valid[pick])
   if dist[pick]<=nearest:
    arr[prefix+'_nearest'][j]=arr[prefix+'_native'][k];arr[prefix+'_nearest_mask'][j]=True;arr[prefix+'_nearest_source'][j]=k;arr[prefix+'_nearest_distance'][j]=dist[pick]
 return p

def add_filter(arr,prefix,cfg):
 y=arr[prefix+'_aligned'];mask=arr[prefix+'_mask'];valid=y[mask]
 center=np.median(valid,axis=0) if len(valid) else np.zeros(y.shape[1]);scale=np.maximum(1.4826*np.median(abs(valid-center),axis=0),1e-4) if len(valid) else np.ones(y.shape[1])
 z=(y-center)/scale;z[~mask]=0
 # Pool source quality in the same CSR weights. This is a proxy, not calibrated confidence.
 q=np.zeros(len(y),np.float32);ptr=arr[prefix+'_source_indptr'];idx=arr[prefix+'_source_indices'];weight=arr[prefix+'_source_weights']
 for i in range(len(q)):
  k=slice(ptr[i],ptr[i+1]);q[i]=np.dot(weight[k],arr[prefix+'_quality'][idx[k]]) if ptr[i+1]>ptr[i] else 0
 dt=np.maximum(np.diff(arr['word_intervals'].mean(1),prepend=arr['word_intervals'][0].mean()),0.0)
 k=diagonal_kalman_filter(z,mask,quality=q,process_variance=cfg['kalman_process_variance'],observation_variance=cfg['kalman_observation_variance'],huber_delta=cfg['kalman_huber_delta'],quality_floor=cfg['kalman_quality_floor'],dt=dt)
 arr[prefix+'_filter_center']=center.astype(np.float32);arr[prefix+'_filter_scale']=scale.astype(np.float32);arr[prefix+'_filtered']=(k['mean']*scale+center).astype(np.float32);arr[prefix+'_filter_variance']=(k['variance']*scale**2).astype(np.float32);arr[prefix+'_predicted_only']=~mask;arr[prefix+'_aligned_quality']=q
 arr[prefix+'_filter_initialized']=np.maximum.accumulate(mask)

def environment(device):
 versions={};local={}
 for name in ['numpy','pandas','torch','transformers','tokenizers','av','opensmile','mediapipe','cv2','scipy','num2words']:
  m=importlib.import_module(name);versions[name]=getattr(m,'__version__',None);local[name]=getattr(m,'__file__',None)
 return dict(python=sys.version,platform=platform.platform(),effective_versions=versions,device=device,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),local

def aggregate(result,rows,manifest):
 # A padded convenience file duplicates only word-level arrays; native archives remain authoritative.
 ids=[r['sample_id'] for r in rows if r['status']!='failed'];data=[np.load(result/'features'/(sid+'.npz'),allow_pickle=False) for sid in ids]
 if not data:return
 lengths=np.array([len(x['word_intervals']) for x in data],np.int32);M=int(max(lengths));out=dict(sample_ids=np.array(ids),lengths=lengths,seq_mask=np.arange(M)[None,:]<lengths[:,None])
 for key in ['text_aligned','audio_aligned','visual_aligned','word_intervals','text_mask','audio_mask','visual_mask','visual_nearest_mask']:
  a=data[0][key];p=np.zeros((len(data),M)+a.shape[1:],dtype=a.dtype)
  for j,d in enumerate(data):p[j,:len(d[key])]=d[key]
  out[key]=p
 np.savez_compressed(result/'aligned_padded.npz',**out)
 pd.DataFrame(rows).to_csv(result/'sample_summary.csv',index=False,encoding='utf-8-sig')
 modalities=[]
 for r in rows:
  if r['status']=='failed':continue
  for m,d in [('text',768),('audio',25),('visual',52)]:
   start=r.get(m+'_start',0);end=r.get(m+'_end',r['duration_seconds'])
   modalities.append(dict(sample_id=r['sample_id'],modality=m,original_effective_duration=end-start,modality_start=start,modality_end=end,native_length=r[m+'_native_length'],word_length=r['word_length'],aligned_valid_count=r[m+'_valid_words'],feature_dimension=d,alignment_granularity='CTC official spoken word interval',padding_length=M,status=r['status']))
 pd.DataFrame(modalities).to_csv(result/'modality_summary.csv',index=False,encoding='utf-8-sig');dump(result/'manifest.json',manifest)
 dump(result/'padding.json',dict(batch_size=len(ids),max_word_length=M,padding_value=0,truncation=False,modality_masks_independent=True,seq_mask_definition='position < original valid word length',sample_order=ids))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--output-dir',default='results',help='Absolute path or path relative to package root');ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');args=ap.parse_args()
 cfg=json.loads((ROOT/'config'/'pipeline.json').read_text());torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cudnn.benchmark=False;torch.set_num_threads(4)
 source=Path(args.data_root);result=(ROOT/args.output_dir).resolve()
 for d in ['features','metadata','words']: (result/d).mkdir(parents=True,exist_ok=True)
 table=pd.read_excel(source/'label-100.xlsx');expected={f"{r.video_id}/{int(r.clip_id)}.mp4" for r in table.itertuples()};actual={p.relative_to(source).as_posix() for p in source.rglob('*.mp4')}
 if len(table)!=100 or len(expected)!=100 or actual!=expected:raise ValueError('100-row label/video bijection failed')
 # Labels are exported independently, never passed to an extractor or parameter selection.
 pd.DataFrame([dict(sample_id=f'{r.video_id}$_${int(r.clip_id)}',label=r.label,annotation=r.annotation) for r in table.itertuples()]).to_csv(result/'labels_separate.csv',index=False,encoding='utf-8-sig')
 env,paths=environment(args.device);dump(ROOT/'config'/'environment.json',env);dump(ROOT/'logs'/'local_import_paths.json',paths)
 extractor=Extractor(cfg,args.device);rows=[];manifest=[]
 code_hashes={name:sha(ROOT/'src'/name) for name in ['pipeline.py','native_media.py','alignment.py']};model_hash=sha(ROOT/'config'/'model_manifest.json')
 selection=table.iloc[:args.limit] if args.limit else table
 for n,r in enumerate(selection.itertuples()):
  sid=f'{r.video_id}$_${int(r.clip_id)}';path=source/str(r.video_id)/(str(int(r.clip_id))+'.mp4');start=time.time();print(f'[{n+1}/{len(selection)}] {sid}',flush=True)
  dest=result/'features'/(sid+'.npz');metap=result/'metadata'/(sid+'.json')
  if args.resume and dest.exists() and metap.exists():
   m=json.loads(metap.read_text(encoding='utf-8'))
   if m['config_sha256']!=sha(ROOT/'config'/'pipeline.json') or m['manifest']['source_sha256']!=sha(path) or m['manifest']['feature_sha256']!=sha(dest) or m.get('extraction_code_sha256')!=code_hashes or m.get('model_manifest_sha256')!=model_hash or json.loads((result/'words'/(sid+'.json')).read_text(encoding='utf-8'))['original_text']!=str(r.text):raise ValueError('Resume source/config/model/code/transcript fingerprint mismatch; back up results and rerun without --resume')
   rows.append(m['summary']);manifest.append(m['manifest']);continue
  try:
   # native_media returns (arrays, metadata, mono_float_waveform); all times share media origin.
   native,media,wave=extractor.media.extract(path)
   arr,wordmeta=extractor.text_and_times(str(r.text),wave,media['audio_start'],media['audio_end']);arr.update(native)
   add_pool(arr,'audio',arr['word_intervals']);add_pool(arr,'visual',arr['word_intervals'],cfg['visual_nearest_tolerance_seconds'])
   if cfg['kalman_enabled']:
    for m in ['audio','visual']:add_filter(arr,m,cfg)
   for key,value in arr.items():
    if np.issubdtype(value.dtype,np.floating) and not np.isfinite(value).all():raise ValueError('Nonfinite '+key)
   np.savez_compressed(dest,**arr);dump(result/'words'/(sid+'.json'),wordmeta)
   flags=[]
   if np.any(arr['word_ctc_posterior']<cfg['ctc_low_posterior_threshold']):flags.append('low_ctc_posterior')
   if not arr['visual_mask'].all():flags.append('visual_word_gaps')
   if not arr['audio_mask'].all():flags.append('audio_word_gaps')
   row=dict(sample_id=sid,video_id=str(r.video_id),clip_id=int(r.clip_id),duration_seconds=media['duration'],audio_start=media['audio_start'],audio_end=media['audio_end'],visual_start=media['video_start'],visual_end=media['video_end'],word_length=len(wordmeta['words']),status='ok' if not flags else 'completed_with_flags',flags=';'.join(flags),elapsed_seconds=round(time.time()-start,3),ctc_low_quality_words=int(sum(arr['word_ctc_posterior']<cfg['ctc_low_posterior_threshold'])),visual_nearest_words=int(arr['visual_nearest_mask'].sum()))
   row['text_start']=float(arr['word_intervals'][0,0]);row['text_end']=float(arr['word_intervals'][-1,1])
   for mod in ['text','audio','visual']:row[mod+'_native_length']=len(arr[mod+'_native']);row[mod+'_valid_words']=int(arr[mod+'_mask'].sum())
   entry=dict(sample_id=sid,source_video=path.relative_to(source).as_posix(),source_sha256=sha(path),feature_file=dest.relative_to(result).as_posix(),feature_sha256=sha(dest),words_file=f'words/{sid}.json',words_sha256=sha(result/'words'/(sid+'.json')))
   meta=dict(sample_id=sid,summary=row,media=media,manifest=entry,config_sha256=sha(ROOT/'config'/'pipeline.json'),quality_flags=flags,filter_notes='offline per-clip observed-word median/MAD scaling; audio/visual only; raw missing stays zero; prediction arrays separate; no emotion labels used',arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arr.items()})
   meta['extraction_code_sha256']=code_hashes;meta['model_manifest_sha256']=model_hash
   dump(metap,meta);rows.append(row);manifest.append(entry)
  except Exception as e:
   row=dict(sample_id=sid,status='failed',error=repr(e));rows.append(row);dump(ROOT/'logs'/(sid+'_error.json'),dict(error=repr(e),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
  with open(ROOT/'logs'/'events.jsonl','a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  aggregate(result,rows,manifest)
  print(json.dumps(row,ensure_ascii=False),flush=True)
 aggregate(result,rows,manifest)
 if any(r['status']=='failed' for r in rows):raise SystemExit('At least one sample failed; inspect explicit error logs')
 print('Completed',len(rows),'samples',flush=True)
if __name__=='__main__':main()
