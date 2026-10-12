"""Independent archive/provenance audit; failure exits nonzero, never fills data.

Run: python src/validate.py --data-root PATH
Use --expected-count 1 only for a clearly labelled partial pilot.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time
import os
approved=os.environ.get('Q1_APPROVED_SITE_PACKAGES')
if approved:
    if not Path(approved).is_dir():raise RuntimeError('Configured preinstalled site-packages does not exist')
    sys.path.insert(0,approved)
import numpy as np
from openpyxl import load_workbook
from alignment import ctc_forced_align

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def require(condition,message):
    if not condition:raise AssertionError(message)

def equal(a,b,name,atol=2e-5):
    require(np.shape(a)==np.shape(b),name+' shape differs')
    np.testing.assert_allclose(a,b,rtol=2e-5,atol=atol,err_msg=name)

def table(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def union_length(intervals):
    if not len(intervals):return 0.
    ordered=sorted(map(tuple,intervals));start,end=ordered[0];total=0.
    for a,b in ordered[1:]:
        if a>end:total+=end-start;start,end=a,b
        else:end=max(end,b)
    return total+end-start

def audit_pool(a,prefix,words):
    native=a[prefix+'_native'];span=a[prefix+'_intervals'];q=a[prefix+'_quality'];mask=a[prefix+'_native_mask']
    require(len(native)==len(span)==len(q)==len(mask),prefix+' native row counts')
    require(np.all(span[:,1]>=span[:,0]),prefix+' invalid native intervals')
    require(np.all((q>=0)&(q<=1)),prefix+' quality range')
    require(np.all(native[~mask]==0),prefix+' missing native values must be zero')
    ptr=a[prefix+'_source_indptr'];idx=a[prefix+'_source_indices'];weights=a[prefix+'_source_weights'];saved_overlap=a[prefix+'_source_overlap']
    require(ptr.shape==(len(words)+1,) and ptr[0]==0 and ptr[-1]==len(idx)==len(weights)==len(saved_overlap),prefix+' CSR size')
    require(np.all(np.diff(ptr)>=0),prefix+' CSR monotonic pointers')
    for j,(start,end) in enumerate(words):
        left=np.maximum(span[:,0],start);right=np.minimum(span[:,1],end)
        overlap=np.maximum(right-left,0)
        expected=np.flatnonzero(mask&(q>0)&(overlap>0));sl=slice(ptr[j],ptr[j+1])
        require(np.array_equal(idx[sl],expected),prefix+f' source set at word {j}')
        observed=len(expected)>0
        require(bool(a[prefix+'_mask'][j])==observed,prefix+' pooled mask')
        if observed:
            w=overlap[expected]*q[expected];w=w/w.sum()
            equal(weights[sl],w,prefix+' weights',1e-10)
            equal(saved_overlap[sl],overlap[expected],prefix+' overlap',1e-10)
            equal(a[prefix+'_aligned'][j],np.sum(native[expected]*w[:,None],axis=0),prefix+' reconstructed feature')
            cover=union_length(np.c_[left[expected],right[expected]])/(end-start)
            equal(a[prefix+'_coverage'][j],min(1.,cover),prefix+' coverage',1e-9)
        else:
            require(np.all(a[prefix+'_aligned'][j]==0),prefix+' absent aligned zero')
            require(a[prefix+'_coverage'][j]==0,prefix+' absent coverage zero')
    if prefix+'_predicted_only' in a:
        require(np.array_equal(a[prefix+'_predicted_only'],~a[prefix+'_mask']),prefix+' filter must preserve raw missingness')
        require(np.all(a[prefix+'_filter_variance']>=0),prefix+' nonnegative filter variance')
    near=a[prefix+'_nearest_mask'];require(not np.any(near&a[prefix+'_mask']),prefix+' nearest overwrites actual observation')
    require(np.all(a[prefix+'_nearest'][~near]==0),prefix+' unavailable nearest zero')
    for j in np.flatnonzero(near):
        k=a[prefix+'_nearest_source'][j]
        require(0<=k<len(native) and mask[k],prefix+' nearest source validity')
        equal(a[prefix+'_nearest'][j],native[k],prefix+' nearest feature')
        center=span[k].mean();distance=max(words[j,0]-center,center-words[j,1],0)
        equal(a[prefix+'_nearest_distance'][j],distance,prefix+' nearest time distance')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',required=True);parser.add_argument('--root',default=str(ROOT));parser.add_argument('--output-dir',default='results');parser.add_argument('--expected-count',type=int,default=100);args=parser.parse_args()
    root=Path(args.root);result=Path(args.output_dir)
    if not result.is_absolute():result=root/result
    # Always save a diagnostic report, including when extraction never started.
    result.mkdir(parents=True,exist_ok=True)
    source=Path(args.data_root);cfg=json.loads((root/'config'/'pipeline.json').read_text(encoding='utf-8'))
    start=time.time();report=dict(status='running',expected_count=args.expected_count,partial_pilot=args.expected_count!=100,checks=[],failures=[])
    try:
        wb=load_workbook(source/'label-100.xlsx',read_only=True,data_only=True);raw=list(wb.active.values);wb.close();header=list(raw[0]);labels=[dict(zip(header,r)) for r in raw[1:] if any(x is not None for x in r)]
        expected={f"{r['video_id']}$_${int(r['clip_id'])}":r for r in labels}
        require(len(expected)==100 and len(labels)==100,'official 100 unique label rows')
        actual_video={p.relative_to(source).as_posix() for p in source.rglob('*.mp4')}
        expected_video={f"{r['video_id']}/{int(r['clip_id'])}.mp4" for r in labels}
        require(actual_video==expected_video,'official label/video bijection')
        report['source_labels_sha256']=sha(source/'label-100.xlsx');report['checks'].append('100 official rows <-> 100 MP4 paths bijection')
        require((result/'manifest.json').is_file(),'No completed extraction manifest. Run src/run.py successfully and then src/finalize.py, or use src/run_checked.py.')
        manifest=json.loads((result/'manifest.json').read_text(encoding='utf-8'));summary=table(result/'sample_summary.csv');modality=table(result/'modality_summary.csv')
        ids=[m['sample_id'] for m in manifest]
        require(len(ids)==args.expected_count and len(set(ids))==len(ids),'expected unique manifest count')
        require(set(ids).issubset(expected),'unknown sample ID')
        if args.expected_count==100:require(set(ids)==set(expected),'full source/output bijection')
        require({x['sample_id'] for x in summary}==set(ids) and len(summary)==len(ids),'summary set/count mismatch')
        require(all(x['status']!='failed' for x in summary),'failed sample in summary')
        require(len(modality)==3*len(ids),'modality summary must have 3 rows per sample')
        require({(r['sample_id'],r['modality'])for r in modality}=={(i,m)for i in ids for m in ['text','audio','visual']},'modality summary coverage')
        fileids={p.stem for p in (result/'features').glob('*.npz')};require(fileids==set(ids),'stray or missing feature archive')
        report['checks'].append('output manifest, summary, modality rows and NPZ file sets agree')
        arrays={};all_words=0;native_counts={'audio':0,'visual':0,'text':0};missing_counts={'audio':0,'visual':0};ctc_low=0
        schema={}
        for m in manifest:
            sid=m['sample_id'];prefix=sid+': ';path=result/m['feature_file'];words_path=result/m['words_file']
            require(sha(path)==m['feature_sha256'],prefix+'feature SHA256')
            require(sha(source/m['source_video'])==m['source_sha256'],prefix+'source MP4 SHA256')
            require(sha(words_path)==m['words_sha256'],prefix+'word metadata SHA256')
            a=dict(np.load(path,allow_pickle=False));arrays[sid]=a
            meta=json.loads((result/'metadata'/(sid+'.json')).read_text(encoding='utf-8'));wordmeta=json.loads(words_path.read_text(encoding='utf-8'))
            require(meta['sample_id']==sid and meta['manifest']==m,prefix+'metadata identity')
            require(meta['config_sha256']==sha(root/'config'/'pipeline.json'),prefix+'config SHA256')
            require(set(meta['arrays'])==set(a),prefix+'array schema keys')
            for key,value in a.items():
                require(value.dtype!=object,prefix+key+' unsafe object dtype')
                require(list(value.shape)==meta['arrays'][key]['shape'] and str(value.dtype)==meta['arrays'][key]['dtype'],prefix+key+' declared shape/dtype')
                if np.issubdtype(value.dtype,np.floating):require(np.isfinite(value).all(),prefix+key+' nonfinite')
                schema.setdefault(key,dict(dtype=str(value.dtype),rank=value.ndim,example_shape=list(value.shape)))
            intervals=a['word_intervals'];words=wordmeta['words'];n=len(words);all_words+=n
            require(n==len(intervals)==len(a['text_aligned']),prefix+'word lengths')
            require(np.all(intervals[:,1]>intervals[:,0]),prefix+'nonpositive word span')
            require(np.all(intervals[1:,0]>=intervals[:-1,1]-1e-9),prefix+'word order/overlap')
            media=meta['media'];origin=media['media_origin_absolute_seconds']
            require(intervals[0,0]>=media['audio_start']-1e-9 and intervals[-1,1]<=media['audio_end']+1e-9,prefix+'words exceed audio')
            require(wordmeta['original_text']==str(expected[sid]['text']),prefix+'official transcript provenance')
            for j,w in enumerate(words):
                require(w['index']==j and wordmeta['original_text'][w['char_start']:w['char_end']]==w['text'],prefix+'original character mapping')
                equal([w['start_seconds'],w['end_seconds']],intervals[j],prefix+'word metadata time',1e-10)
                b=a['text_token_offsets'];idx=np.flatnonzero((b[:,0]<w['char_end'])&(b[:,1]>w['char_start']))
                require(idx.tolist()==w['bert_token_indices'],prefix+'WordPiece source index')
                ptr=a['text_source_indptr'];saved=a['text_source_indices'][ptr[j]:ptr[j+1]]
                require(np.array_equal(saved,idx),prefix+'WordPiece CSR')
                equal(a['text_aligned'][j],a['text_native'][idx].mean(0),prefix+'WordPiece mean')
            require(np.all(a['text_mask']),prefix+'all transcript words must remain present')
            lp=a['ctc_log_probabilities'];equal(np.exp(lp).sum(1),np.ones(len(lp)),prefix+'CTC posterior normalization',1e-5)
            blank=wordmeta['ctc_vocabulary']['<pad>'];spans=ctc_forced_align(lp,a['ctc_token_ids'],blank)
            equal([[s['start_frame'],s['end_frame']]for s in spans],a['ctc_token_intervals_frames'],prefix+'CTC Viterbi reconstruction',0)
            equal([s['posterior']for s in spans],a['ctc_token_posterior'],prefix+'CTC emission score')
            for j,w in enumerate(words):
                t=a['ctc_token_intervals_frames'][a['ctc_token_owner']==j]
                expected_start=media['audio_start']+t[0,0]*cfg['ctc_hop_samples']/cfg['sample_rate']
                expected_end=min(media['audio_end'],media['audio_start']+((t[-1,1]-1)*cfg['ctc_hop_samples']+cfg['ctc_receptive_field_samples'])/cfg['sample_rate'])
                if j+1<n:expected_end=min(expected_end,intervals[j+1,0])
                equal(intervals[j],[expected_start,expected_end],prefix+'CTC sample clock',1e-10)
            v=a['visual_all_frame_records'];ii=a['visual_source_frame_index']
            require(np.all(np.diff(v[:,4])>0),prefix+'video PTS order')
            equal(v[:,4],v[:,1]*v[:,2]/v[:,3],prefix+'raw video PTS to seconds',1e-10)
            equal(a['visual_intervals'],np.c_[v[ii,4],v[ii,4]+v[ii,5]]-origin,prefix+'visual display support',1e-10)
            require(np.array_equal(a['visual_source_pts'],v[ii,1].astype(np.int64)),prefix+'visual PTS identifiers')
            if cfg['video_sample_fps']==0:require(np.array_equal(ii,np.arange(len(v))),prefix+'full-frame coverage')
            source_audio=a['audio_source_frame_records'];equal(source_audio[:,6],source_audio[:,1]*source_audio[:,2]/source_audio[:,3],prefix+'raw audio PTS to seconds',1e-10)
            equal(source_audio[:,7]-source_audio[:,6],source_audio[:,4]/source_audio[:,5],prefix+'decoded audio samples duration',1e-10)
            equal(len(a['audio_sample_observed_mask'])/cfg['sample_rate'],media['audio_end']-media['audio_start'],prefix+'waveform clock',1e-10)
            require(np.sum(~a['audio_sample_observed_mask'])==media['audio']['gap_samples'],prefix+'gap mask count')
            for p in ['audio','visual']:
                audit_pool(a,p,intervals);missing_counts[p]+=int((~a[p+'_mask']).sum())
            for p,d in [('text',768),('audio',25),('visual',52)]:
                require(a[p+'_native'].shape[1]==d,prefix+p+' feature dimension');native_counts[p]+=len(a[p+'_native'])
            ctc_low+=int((a['word_ctc_posterior']<cfg['ctc_low_posterior_threshold']).sum())
        report['checks']+=['SHA256 for source videos, NPZ archives and word JSON files','schema, finite values, independent modality masks and zero missing observations','official transcript character offsets and WordPiece mean reconstruction','CTC Viterbi spans, posterior normalization and physical-time conversion','all native video PTS/display supports and audio sample/PTS clocks','independent CSR source-set, overlap, quality-weight and coverage reconstruction']
        padded=dict(np.load(result/'aligned_padded.npz',allow_pickle=False));pids=padded['sample_ids'].tolist();require(pids==ids,'padded sample order')
        lengths=np.array([len(arrays[s]['word_intervals'])for s in ids]);require(np.array_equal(padded['lengths'],lengths),'padded valid lengths')
        width=max(lengths);require(padded['seq_mask'].shape==(len(ids),width),'observed max length no truncation')
        require(np.array_equal(padded['seq_mask'],np.arange(width)[None,:]<lengths[:,None]),'sequence padding mask')
        for key,value in padded.items():
            if key in ['sample_ids','lengths','seq_mask']:continue
            for i,sid in enumerate(ids):
                equal(value[i,:lengths[i]],arrays[sid][key],key+' padded prefix')
                require(np.all(value[i,lengths[i]:]==0),key+' padding must be zero/False')
        report['checks'].append('padded arrays: unchanged valid prefixes, max observed length, no truncation, zero/False tails')
        report.update(status='passed',validated_count=len(ids),word_count=all_words,native_counts=native_counts,missing_word_counts=missing_counts,low_ctc_posterior_words=ctc_low,max_word_length=int(width))
        (root/'config'/'feature_schema.json').write_text(json.dumps(dict(format='NPZ; numpy.load(path,allow_pickle=False)',array_schema=schema,note='example_shape is from first validated sample; row lengths vary by sample; each metadata JSON stores exact shapes'),ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception as exc:
        report.update(status='failed');report['failures'].append(repr(exc))
        import traceback;report['traceback']=traceback.format_exc()
    report['elapsed_seconds']=time.time()-start
    (result/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':
    # This optional path uses an existing approved runtime installation; it
    # changes dependency search only and never modifies OS policy.
    main()
