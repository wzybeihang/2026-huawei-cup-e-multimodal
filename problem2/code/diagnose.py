"""Independent output checks and one transparent joint-gap mechanism example."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from protocol import CASES,VARIANTS,SELECT_CASES,INIT
from prepare_data import write_json,encode
from run_experiments import Data,initialize,csv_write
from evaluation import decode

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--bert',required=True);a=p.parse_args();root=Path(a.result);torch.set_num_threads(4)
    tables=pd.read_csv(root/'全部条件结果.csv');assert len(tables)==len(VARIANTS)*len(CASES)==1017
    checks=[]
    for name in VARIANTS:
        with np.load(root/name/'predictions.npz',allow_pickle=False) as z:
            assert z['classes'].shape==(113,728) and len(set(z['ids']))==728
            saved=torch.load(root/name/'best.pt',map_location='cpu',weights_only=False)
            cls,val=decode(z['logits'],z['raw'],saved['selection']['rule'])
            assert np.array_equal(cls,z['classes']) and np.allclose(val,z['intensities'])
            for j,case in enumerate(CASES):
                c=cls[j];v=val[j];y=z['true_class'];r=z['true_intensity']
                f1=[]
                for k in range(3):
                    numerator=2*int(((y==k)&(c==k)).sum());denominator=int((y==k).sum()+(c==k).sum())
                    f1.append(numerator/denominator if denominator else 0.)
                row=tables[(tables.model==name)&(tables.condition==case['name'])].iloc[0]
                assert abs(row.accuracy-float((y==c).mean()))<1e-7
                assert abs(row.macro_f1-float(np.mean(f1)))<1e-7
                assert abs(row.mae-float(np.abs(r.astype(float)-v.astype(float)).mean()))<1e-6
    checks.append('All 1017 metric rows independently reconstructed from saved per-sample predictions')
    with np.load(root/'validation_masks.npz',allow_pickle=False) as z:
        assert not np.any(z['observed']&~z['original'][None])
        assert not np.any(z['original'].any(1)[None]&~z['observed'].any(2))
        assert not np.any(z['observed']&~z['domain'][None,:,:,None])
    checks.append('No artificial whole-modality dropout, no padding counted as observation, no fabricated observations')
    pred=root/'附件3预测'
    for f in pred.glob('问题2_附件3*.csv'):
        t=pd.read_csv(f);assert len(t)==30 and not t.sample_id.duplicated().any()
        y=t.sentiment_intensity.to_numpy();assert np.isfinite(y).all() and (np.abs(y)<=3).all()
        assert np.array_equal(t.sentiment_polarity.to_numpy(),np.where(y<0,'Negative',np.where(y>0,'Positive','Neutral')))
    checks.append('All attachment3 prediction files cover 30 unique samples with finite coherent outputs')

    # Empirical leakage test: changing erased token IDs must not change visible encoded values.
    from transformers import BertModel
    valid=Data(a.cache,'valid');tb=valid.meta['tb'][:1].copy();mask=valid.masks[SELECT_CASES[1]][:1,:,0]
    other=tb.copy();where=valid.domain[:1]&~mask;other[:,0][where]=2023
    bert=BertModel.from_pretrained(a.bert,local_files_only=True).cuda().eval();bert.requires_grad_(False)
    e1=encode(bert,tb,mask,'cuda');e2=encode(bert,other,mask,'cuda');del bert
    assert np.array_equal(e1,e2)
    checks.append('Changing the hidden token payload leaves the BERT output exactly unchanged')

    ix=int(np.flatnonzero((valid.domain.sum(1)>=20)&((valid.observed.sum((1,2)))==3*valid.domain.sum(1)))[0])
    condition=next(i for i,c in enumerate(CASES) if c['name']=='T+A+V_0.5_middle')
    sample={}
    for name in ('matrix_full','function_full'):
        saved=torch.load(root/name/'best.pt',map_location='cuda',weights_only=False)
        model=initialize(saved['spec'],'cuda');model.load_state_dict(saved['state_dict']);model.eval()
        with torch.no_grad():output=model(*valid.batch([ix],condition,'cuda'))
        cls,val=decode(output['logits'].cpu().numpy(),output['raw'].cpu().numpy(),saved['selection']['rule'])
        sample[name]=dict(polarity=int(cls[0]),intensity=float(val[0]),state_norm=output['states'][0].norm(dim=-1).cpu().tolist())
        if output['variance'] is not None:sample[name]['mean_variance']=output['variance'][0].mean(-1).cpu().tolist()
    evidence=dict(sample_id=str(valid.ids[ix]),selection_rule='first validation sample with at least20 content positions and three fully observed modalities',
                 condition=CASES[condition]['name'],true_class=int(valid.meta['classification'][ix]),
                 true_intensity=float(valid.meta['regression'][ix]),text=str(valid.meta['raw_text'][ix]),
                 domain=valid.domain[ix].tolist(),observed=valid.masks[condition][ix].tolist(),
                 intervals=valid.intervals[condition][ix].tolist(),predictions=sample)
    write_json(root/'共同缺失区间示例.json',evidence)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'svg.fonttype':'path'})
    n=int(np.flatnonzero(valid.domain[ix])[-1])+1;x=np.arange(n)
    fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[1,1.3,1.3]})
    img=np.where(valid.domain[ix,:n,None],valid.masks[condition][ix,:n].astype(float),np.nan).T
    axes[0].imshow(img,aspect='auto',cmap='Greys',vmin=0,vmax=1,extent=(-.5,n-.5,2.5,-.5));axes[0].set_yticks([0,1,2],['文本','语音','视觉']);axes[0].set_title('黑色为可见观测 白色为连续缺失')
    for name,color in [('matrix_full','#197A91'),('function_full','#D27B48')]:
        axes[1].plot(x,sample[name]['state_norm'][:n],label=name,color=color)
    axes[1].set_ylabel('潜在表示范数');axes[1].legend(fontsize=9)
    axes[2].plot(x,sample['function_full']['mean_variance'][:n],color='#D27B48');axes[2].set_ylabel('状态方差代理');axes[2].set_xlabel('对齐序列位置 不是秒')
    left,right=valid.intervals[condition][ix,0]
    for ax in axes[1:]:ax.axvspan(left-.5,right-.5,color='#F6D98C',alpha=.35);ax.grid(alpha=.15)
    fig.suptitle('图13 三模态同一局部区间全缺失的处理示例',fontsize=16)
    fig.text(.5,.012,'示例按固定规则选取；潜在表示不是逐帧情感真值；方差代理不是已校准置信区间。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(root/'图表'/'图13_共同缺失区间.png',dpi=210);fig.savefig(root/'图表'/'图13_共同缺失区间.svg');plt.close(fig)
    captions_path=root/'图表'/'图注.json';captions=json.loads(captions_path.read_text(encoding='utf-8'))
    captions['图13_共同缺失区间']='固定选择第一个不少于20位置、三模态原本完整的验证样本，遮挡共同中段50%。方差随无观测步骤增长；图示为内部状态诊断，不是逐位置情感标注。'
    write_json(captions_path,captions)
    (root/'图表'/'图注.md').write_text('\n\n'.join(f'## {k}\n\n{v}' for k,v in captions.items()),encoding='utf-8')
    schedule=[dict(epoch=e+1,batch_index=b,condition=CASES[1+((e*54+b)*37)%112]['name'],second_candidate=CASES[SELECT_CASES[-1]]['name']) for e in range(20) for b in range(54)]
    csv_write(root/'训练遮挡计划.csv',schedule)
    write_json(root/'核验报告.json',dict(status='passed',models=9,train_samples=3395,valid_samples=728,conditions_per_model=113,
                                       total_metric_rows=1017,attachment3_samples=30,figures=13,checks=checks,
                                       initialization_sweep=False,external_sentiment_data=False,
                                       warning='This verifies the experiment implementation and artifacts, not universal robustness or a statistical significance claim.'))
    print(json.dumps({'status':'passed','checks':checks},ensure_ascii=False))

if __name__=='__main__':main()
