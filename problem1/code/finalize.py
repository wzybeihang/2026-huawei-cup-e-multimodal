"""Deterministic metadata-only final audit. Does not change feature arrays."""
from pathlib import Path
import argparse,hashlib,json,sys,os,importlib.metadata
if os.environ.get('Q1_APPROVED_SITE_PACKAGES'):sys.path.insert(0,os.environ['Q1_APPROVED_SITE_PACKAGES'])
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def edit_distance(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a):
  e=[i+1]
  for j,y in enumerate(b):e.append(min(e[-1]+1,d[j+1]+1,d[j]+(x!=y)))
  d=e
 return d[-1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--output-dir',default='results',help='Absolute path or path relative to package root');ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args();source=Path(args.data_root);out=(ROOT/args.output_dir).resolve();manifest=[];rows=[];mods=[];diagnostics=[]
 if not (out/'padding.json').is_file():
  ap.error('No completed extraction found (padding.json missing). Run src/run.py successfully first, or use src/run_checked.py. See logs/*_error.json for the extraction error.')
 table=pd.read_excel(source/'label-100.xlsx');pad=json.loads((out/'padding.json').read_text(encoding='utf-8'))
 for item in table.itertuples():
  sid=f'{item.video_id}$_${int(item.clip_id)}';mp=out/'metadata'/(sid+'.json');wp=out/'words'/(sid+'.json')
  if args.allow_partial and not mp.exists():continue
  m=json.loads(mp.read_text(encoding='utf-8'));w=json.loads(wp.read_text(encoding='utf-8'));a=np.load(out/'features'/(sid+'.npz'),allow_pickle=False);r=m['summary'];intervals=a['word_intervals'];media=m['media'];vocabulary={i:c for c,i in w['ctc_vocabulary'].items()};greedy=np.argmax(a['ctc_log_probabilities'],axis=1);collapsed=[int(k) for j,k in enumerate(greedy) if j==0 or greedy[j-1]!=k];blank=w['ctc_vocabulary']['<pad>'];greedy_text=''.join(vocabulary[k] for k in collapsed if k!=blank).replace('|',' ').strip();reference=' '.join(x['normalized'].replace('|',' ') for x in w['words']);distance=edit_distance(reference.split(),greedy_text.split());wer=distance/max(1,len(reference.split()))
  uncovered=[]
  last=media['audio_start']
  for start,end in intervals:
   if start>last:uncovered.append([float(last),float(start)])
   last=max(last,end)
  if last<media['audio_end']:uncovered.append([float(last),media['audio_end']])
  centers=media['audio_start']+(np.arange(len(greedy))*320+200)/16000
  outside_extent=(centers<intervals[0,0])|(centers>=intervals[-1,1]);extra_nonblank=int(np.sum(outside_extent&(greedy!=blank)))
  diagnosis=dict(greedy_asr_text=greedy_text,normalized_official_reference=reference,greedy_reference_word_edit_rate=float(wer),word_edit_rate_definition='Levenshtein(greedy words, official normalized words)/official count; diagnostic disagreement, not measured transcription accuracy',transcript_uncovered_intervals=uncovered,suspected_extra_speech_outside_reference_extent=extra_nonblank>0,outside_reference_extent_nonblank_frames=extra_nonblank,notes='Greedy ASR is a diagnostic only and never replaces official text; uncovered intervals can include silence, pauses, or speech absent from the official excerpt')
  w['diagnostics']=diagnosis;dump(wp,w)
  r['text_start']=float(intervals[0,0]);r['text_end']=float(intervals[-1,1]);r['word_interval_union_duration']=float(np.sum(intervals[:,1]-intervals[:,0]));r['greedy_reference_word_edit_rate']=float(wer);r['suspected_extra_speech_outside_reference_extent']=extra_nonblank>0
  if extra_nonblank>0:
   flags=r['flags'].split(';') if r['flags'] else []
   if 'suspected_extra_speech_outside_reference_extent' not in flags:flags.append('suspected_extra_speech_outside_reference_extent')
   r['flags']=';'.join(flags);r['status']='completed_with_flags';m['quality_flags']=flags
  if 'duration_audit' not in media:
   am=media['audio'];vm=media['video'];origin=media['media_origin_absolute_seconds'];vd=(vm['start_pts']+vm['duration_pts'])*vm['time_base'][0]/vm['time_base'][1] if vm['start_pts'] is not None and vm['duration_pts'] is not None else None
   media['duration_audit']=dict(container_declared_seconds=am.get('container_duration_seconds'),observed_presentation_span_seconds=media['duration'],container_minus_observed_seconds=am['container_duration_seconds']-media['duration'] if am.get('container_duration_seconds') is not None else None,audio_declared_minus_observed_end_seconds=am['end_limit_seconds']-(origin+media['audio_end']) if am.get('end_limit_seconds') is not None else None,video_declared_minus_observed_end_seconds=vd-(origin+media['video_end']) if vd is not None else None,audio_video_end_difference_seconds=media['audio_end']-media['video_end'])
   media['timing_flags']=[]
   if am.get('discarded_preroll_seconds',0)>1/16000:media['timing_flags'].append('decoded_preroll_cropped_to_declared_presentation_start')
   if am.get('gap_samples',0):media['timing_flags'].append('audio_pts_gap_zero_filled_with_false_mask')
   if am.get('overlapping_samples',0):media['timing_flags'].append('audio_pts_overlap_first_observation_retained')
   if abs(media['duration_audit']['container_minus_observed_seconds'] or 0)>.001:media['timing_flags'].append('container_and_observed_duration_differ')
  m['manifest']['words_sha256']=sha(wp);m['diagnostic_summary']={k:diagnosis[k] for k in ['greedy_reference_word_edit_rate','suspected_extra_speech_outside_reference_extent','outside_reference_extent_nonblank_frames']}
  # Backfill the first pilot/batch only: metadata-only and resume guard changes
  # did not change the extraction math. Never overwrite an existing fingerprint.
  m.setdefault('extraction_code_sha256',{name:sha(ROOT/'src'/name) for name in ['pipeline.py','native_media.py','alignment.py']});m.setdefault('model_manifest_sha256',sha(ROOT/'config'/'model_manifest.json'))
  dump(mp,m);manifest.append(m['manifest']);rows.append(r);diagnostics.append(dict(sample_id=sid,**diagnosis))
  for mod,dim in [('text',768),('audio',25),('visual',52)]:
   start=r[mod+'_start'];end=r[mod+'_end'];mods.append(dict(sample_id=sid,modality=mod,clip_duration_seconds=media['duration'],original_effective_duration=end-start,duration_definition='official_word_extent_on_audio_clock' if mod=='text' else 'decoded_presentation_end_minus_start',modality_start=start,modality_end=end,word_interval_union_duration=r['word_interval_union_duration'],native_length=r[mod+'_native_length'],word_length=r['word_length'],aligned_valid_count=r[mod+'_valid_words'],feature_dimension=dim,alignment_granularity='CTC official spoken word interval',padding_length=pad['max_word_length'],status=r['status']))
 pd.DataFrame(rows).to_csv(out/'sample_summary.csv',index=False,encoding='utf-8-sig');pd.DataFrame(mods).to_csv(out/'modality_summary.csv',index=False,encoding='utf-8-sig');dump(out/'manifest.json',manifest);dump(out/'alignment_diagnostics.json',diagnostics)
 dump(ROOT/'config'/'data_manifest.json',dict(label_file='label-100.xlsx',label_file_sha256=sha(source/'label-100.xlsx'),expected_samples=100,completed_samples=len(rows),partial_run=len(rows)!=100,source_mp4_count=len(list(source.rglob('*.mp4'))),label_is_not_an_extractor_input=True,transcript_source='label-100.xlsx:text',sample_order=[r['sample_id'] for r in rows]))
 env=json.loads((ROOT/'config'/'environment.json').read_text());env['effective_versions']['num2words']=importlib.metadata.version('num2words');env['notes']='Preinstalled approved numpy/pandas used on this host; all other listed versions are actual imports. Clean environments can install locked versions without the optional host path.';dump(ROOT/'config'/'environment.json',env)
 dump(ROOT/'config'/'code_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p)) for p in sorted((ROOT/'src').glob('*.py'))],run_note='During the completed batch only metadata summaries were extended (modality duration, text extent, duration_audit); feature extraction math was unchanged. finalize.py deterministically applies metadata additions to all samples. This manifest records the final reproducible code.'))
 print(json.dumps(dict(samples=len(rows),modalities=len(mods),suspected_extra_speech_count=sum(r['suspected_extra_speech_outside_reference_extent'] for r in rows)),indent=2))
if __name__=='__main__':main()
