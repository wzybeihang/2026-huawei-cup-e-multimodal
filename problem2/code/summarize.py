from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from protocol import KEYS,CASES,VARIANTS
from prepare_data import read_pickle,source_fields,encode,write_json,sha,status
from run_experiments import csv_write,initialize
from evaluation import metrics,decode,LABELS

def summaries(result,cache):
    frames=[];summary=[];errors=[];groups=[];confusions={}
    selected_name=min(VARIANTS,key=lambda name:json.loads((result/name/'complete.json').read_text(encoding='utf-8'))['best_score'])
    with np.load(Path(cache)/'valid'/'meta.npz',allow_pickle=False) as d:
        text=d['raw_text'];domain=d['domain'];original=d['observed'];ids=d['ids']
    for name in VARIANTS:
        dest=result/name
        table=pd.read_csv(dest/'conditions.csv');frames.append(table)
        complete=json.loads((dest/'complete.json').read_text(encoding='utf-8'))
        saved=torch.load(dest/'best.pt',map_location='cpu',weights_only=False)
        row=dict(model=name,epochs_run=complete['epochs_run'],best_epoch=saved['epoch'],parameters=complete['parameters'],
                 training_seconds=complete['train_seconds'],selection_score=saved['selection']['score'])
        for col in ('accuracy','macro_f1','weighted_f1','mae','pearson'):
            row['clean_'+col]=float(table.iloc[0][col]);row['missing_mean_'+col]=float(table.iloc[1:][col].mean())
        summary.append(row)
        if name not in ('matrix_full','function_full',selected_name):continue
        with np.load(dest/'predictions.npz',allow_pickle=False) as z:
            y=z['true_class'];truth=z['true_intensity'];c=z['classes'][0];p=z['intensities'][0];raw=z['raw'][0]
            assert np.array_equal(z['ids'],ids)
            m=metrics(y,truth,c,p);confusions[name]=m['confusion']
            for i,sid in enumerate(ids):
                errors.append(dict(model=name,sample_id=sid,text=text[i],true_polarity=LABELS[y[i]],predicted_polarity=LABELS[c[i]],
                                   true_intensity=float(truth[i]),predicted_intensity=float(p[i]),raw_intensity=float(raw[i]),
                                   absolute_error=float(abs(truth[i]-p[i])),polarity_correct=bool(y[i]==c[i]),
                                   valid_positions=int(domain[i].sum()),audio_missing_rate=float(1-original[i,:,1].sum()/domain[i].sum()),
                                   vision_missing_rate=float(1-original[i,:,2].sum()/domain[i].sum())))
            selections={
                '强负向 y<-1':truth < -1, '弱负向 -1<=y<0':(truth>=-1)&(truth<0),
                '中性 y=0':truth==0, '弱正向 0<y<=1':(truth>0)&(truth<=1), '强正向 y>1':truth>1,
                '序列不超过15位置':domain.sum(1)<=15,'序列超过15位置':domain.sum(1)>15,
                '视觉原生缺失不超过20%':1-original[:,:,2].sum(1)/domain.sum(1)<=.2,
                '视觉原生缺失超过20%':1-original[:,:,2].sum(1)/domain.sum(1)>.2,
            }
            for group,ix in selections.items():
                mm=metrics(y[ix],truth[ix],c[ix],p[ix]);mm.pop('confusion')
                groups.append(dict(model=name,group=group,samples=int(ix.sum()),mean_predicted_intensity=float(p[ix].mean()),
                                   mean_true_intensity=float(truth[ix].mean()),**mm))
    allrows=pd.concat(frames,ignore_index=True);allrows.to_csv(result/'全部条件结果.csv',encoding='utf-8-sig',index=False)
    pd.DataFrame(summary).to_csv(result/'模型与消融总表.csv',encoding='utf-8-sig',index=False)
    error_table=pd.DataFrame(errors);error_table.to_csv(result/'验证集逐样本预测与错误.csv',encoding='utf-8-sig',index=False)
    error_table.sort_values(['model','absolute_error'],ascending=[True,False]).groupby('model').head(10).to_csv(result/'强度误差最大的样本.csv',encoding='utf-8-sig',index=False)
    pd.DataFrame(groups).to_csv(result/'错误分组统计.csv',encoding='utf-8-sig',index=False)
    write_json(result/'混淆矩阵.json',confusions)
    for key,title in [('rate','缺失率'),('modalities','缺失模态'),('position','缺失位置')]:
        g=allrows[allrows.condition!='clean'].groupby(['model',key])[[*('accuracy','macro_f1','mae','pearson'),'actual_new_missing_rate']].mean().reset_index()
        g.to_csv(result/f'{title}影响汇总.csv',encoding='utf-8-sig',index=False)
    selected=min(summary,key=lambda x:x['selection_score'])
    write_json(result/'方案选择.json',dict(selected=selected['model'],criterion='predeclared validation clean plus T/A/V/TAV middle30 weighted score',
               labels_from_attachment3_used=False,candidates=list(VARIANTS),score=selected['selection_score'],
               scope_note='After ablations, compare all nine structures using the same previously fixed validation score. Validation is used for model selection, not independent final testing.'))
    pair={}
    for model in ('matrix_full','function_full'):
        with np.load(result/model/'predictions.npz',allow_pickle=False) as z:pair[model]=(z['classes'],z['intensities'])
    write_json(result/'两方案差异核验.json',dict(validation_samples=728,conditions=len(CASES),
        clean_polarity_disagreements=int((pair['matrix_full'][0][0]!=pair['function_full'][0][0]).sum()),
        clean_mean_abs_intensity_difference=float(np.abs(pair['matrix_full'][1][0]-pair['function_full'][1][0]).mean()),
        all_condition_polarity_disagreements=int((pair['matrix_full'][0]!=pair['function_full'][0]).sum()),
        all_condition_mean_abs_intensity_difference=float(np.abs(pair['matrix_full'][1]-pair['function_full'][1]).mean())))

@torch.inference_mode()
def inference(result,attachment,bert,device):
    from transformers import BertModel
    files=sorted(Path(attachment).glob('*.pkl'));assert len(files)==30
    rows=[read_pickle(p) for p in files]
    if any(set(x)!={'text_bert','audio','vision'} for x in rows):raise ValueError('Special-test schema changed')
    fields={k:np.concatenate([np.asarray(r[k]) for r in rows]) for k in ('text_bert','audio','vision')}
    tb,domain,obs,ids=source_fields(fields,[p.stem for p in files])
    enc=BertModel.from_pretrained(bert,local_files_only=True).to(device).eval();enc.requires_grad_(False)
    arrays={'text':encode(enc,tb,obs[:,:,0],device)};del enc
    arrays.update(audio=fields['audio'],vision=fields['vision'])
    with np.load(result/'scaler.npz') as scale:
        for m,k in enumerate(KEYS):
            z=(arrays[k]-scale[k+'_mean'])/scale[k+'_scale']
            arrays[k]=np.where(obs[:,:,m,None],np.clip(z,-8,8),0).astype(np.float32)
    arrays['text']=arrays['text'].astype(np.float16).astype(np.float32)
    f={k:torch.tensor(v,device=device) for k,v in arrays.items()}
    o=torch.tensor(obs,device=device);d=torch.tensor(domain,device=device)
    outputs={};dest=result/'附件3预测';dest.mkdir(exist_ok=True)
    selected=json.loads((result/'方案选择.json').read_text(encoding='utf-8'))['selected']
    candidates=list(dict.fromkeys(['matrix_full','function_full',selected]))
    for name in candidates:
        saved=torch.load(result/name/'best.pt',map_location=device,weights_only=False)
        model=initialize(saved['spec'],device);model.load_state_dict(saved['state_dict']);model.eval()
        out=model(f,o,d);c,p=decode(out['logits'].cpu().numpy(),out['raw'].cpu().numpy(),saved['selection']['rule'])
        outputs[name]=(c,p)
        csv_write(dest/f'问题2_附件3_{name}.csv',[dict(sample_id=sid,sentiment_polarity=LABELS[c[i]],sentiment_intensity=float(p[i])) for i,sid in enumerate(ids)])
    joined=[]
    for i,sid in enumerate(ids):
        row=dict(sample_id=sid,valid_positions=int(domain[i].sum()))
        for m,k in enumerate(KEYS):row[k+'_missing_rate']=float(1-obs[i,:,m].sum()/domain[i].sum())
        for name,(c,p) in outputs.items():row[name+'_polarity']=LABELS[c[i]];row[name+'_intensity']=float(p[i])
        joined.append(row)
    csv_write(dest/'附件3全量预测及缺失情况.csv',joined)
    import shutil
    shutil.copy2(dest/f'问题2_附件3_{selected}.csv',dest/'问题2_附件3预测结果.csv')
    write_json(dest/'预测来源清单.json',dict(samples=30,labels_available=False,selected_model=selected,
              files=[dict(sample_id=p.stem,file=p.name,sha256=sha(p)) for p in files],
              text_missing_samples=int((domain&~obs[:,:,0]).any(1).sum()),
              modality_missing_fraction=[float((domain&~obs[:,:,m]).sum()/domain.sum()) for m in range(3)]))
    np.savez_compressed(dest/'原始缺失掩码.npz',domain=domain,observed=obs,ids=ids)
    status('ATTACHMENT3_INFERENCE_COMPLETE',samples=30,models=len(candidates),selected=selected)

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--attachment3',required=True);p.add_argument('--bert',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4)
    summaries(Path(a.result),a.cache);inference(Path(a.result),a.attachment3,a.bert,a.device)

if __name__=='__main__':main()
