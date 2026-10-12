from pathlib import Path
import argparse,json,os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAMES={'matrix_full':'矩阵补全','function_full':'函数状态估计','baseline_clean':'基础融合 无增强',
       'baseline_aug':'基础融合 有增强','matrix_no_temporal':'矩阵 去时间正则','matrix_no_aug':'矩阵 去缺失训练',
       'function_no_adaptive':'函数 去自适应权重','function_no_memory':'函数 去状态记忆','function_no_aug':'函数 去缺失训练'}
MAIN=('matrix_full','function_full');COLORS={'matrix_full':'#197A91','function_full':'#D27B48','baseline_clean':'#777777','baseline_aug':'#7B68A6'}
METRICS=[('accuracy','Accuracy ↑'),('macro_f1','Macro-F1 ↑'),('mae','MAE ↓'),('pearson','Pearson ↑')]

def generate(root):
    root=Path(root);dest=root/'图表';dest.mkdir(exist_ok=True)
    selected=json.loads((root/'方案选择.json').read_text(encoding='utf-8'))['selected']
    COLORS.setdefault(selected,'#418856')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    table=pd.read_csv(root/'全部条件结果.csv');summary=pd.read_csv(root/'模型与消融总表.csv').set_index('model')
    missing=table[table.condition!='clean'];captions={}
    def save(fig,name,caption):
        footer='附件2验证集 728条  |  固定初始化  |  真实运行结果  |  条件均值不是独立重复次数'
        if name.startswith('图10_'):footer='附件3专项测试集 30条  |  无标签  |  仅展示预测 不报告准确率'
        if name.startswith('图1_'):footer='同一aligned_50输入接口  |  三分类极性与连续强度双任务  |  各路线独立训练'
        fig.text(.5,.014,footer,ha='center',fontsize=8,color='#666666')
        fig.tight_layout(rect=(0,.055,1,.94))
        fig.savefig(dest/(name+'.png'),dpi=210);fig.savefig(dest/(name+'.svg'));plt.close(fig);captions[name]=caption
    fig,ax=plt.subplots(figsize=(11,4.5));ax.set_xlim(0,11);ax.set_ylim(0,5);ax.axis('off')
    def box(x,y,w,h,text,color='#E6F0F3'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#A8BAC0'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    box(.2,1.7,2,1.5,'同一对齐版输入\n文本 768  语音 74  视觉 35\n有效序列掩码 + 缺失掩码')
    box(2.8,1.7,1.7,1.5,'三路 Linear + tanh\n各投影至32维\n缺失值不进入编码')
    box(5.1,3.05,2.2,1.2,'矩阵路线\n低秩补全 + 时间正则')
    box(5.1,.65,2.2,1.2,'函数路线\n状态递推 + 自适应精度')
    box(8,1.7,2.6,1.5,'注意力池化 + 均值池化\n三模态覆盖率\n极性分类头 + 强度回归头')
    for a,b in [((2.25,2.45),(2.72,2.45)),((4.58,2.45),(5.02,3.65)),((4.58,2.45),(5.02,1.25)),((7.38,3.65),(7.92,2.45)),((7.38,1.25),(7.92,2.45))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#445566','lw':1.5})
    fig.suptitle('图1 两条建模路线与共同输入输出',fontsize=17)
    save(fig,'图1_网络结构','两路线均没有绕过专用融合模块的直接观测残差分支。三分类标签与连续强度同时学习。')

    fig,axes=plt.subplots(2,2,figsize=(10,6.8))
    for ax,(metric,title) in zip(axes.flat,METRICS):
        vals=[summary.loc[m,'clean_'+metric] for m in MAIN]
        ax.bar([0,1],vals,color=[COLORS[m] for m in MAIN],width=.55)
        ax.set_xticks([0,1],[NAMES[m] for m in MAIN]);ax.set_title(title);ax.set_ylim(0,max(vals)*1.25)
        for i,v in enumerate(vals):ax.text(i,v+.012,f'{v:.4f}',ha='center')
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('图2 未新增缺失时的基础性能',fontsize=17)
    save(fig,'图2_验证基础性能','clean仍保留官方输入中已有的缺失。模型与决策规则均在验证集上选择，因此属于内部验证指标。')

    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(metric,title) in zip(axes.flat,METRICS):
        for m in dict.fromkeys([*MAIN,selected,'baseline_aug']):
            g=missing[missing.model==m].groupby('rate')[metric].mean()
            ax.plot(g.index*100,g.values,'o-',label=NAMES[m],color=COLORS[m],lw=1.8,ms=4)
        ax.set_title(title);ax.set_xlabel('目标连续缺失区间 / 有效位置数 (%)');ax.set_xticks([10,30,50,70]);ax.grid(alpha=.2)
        ax.margins(y=.20)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.925),ncol=4,fontsize=9)
    fig.suptitle('图3 缺失率变化与预测性能',fontsize=17)
    save(fig,'图3_缺失率影响','每点为7种模态组合×4种位置的28个条件等权平均。短序列取整、原生缺失和保留至少一个观测会使实际缺失率偏离目标比例；真实比例另存CSV。')

    mods=['T','A','V','T+A','T+V','A+V','T+A+V']
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for ax,(metric,title) in zip(axes,[('macro_f1','Macro-F1 ↑'),('mae','MAE ↓')]):
        values=np.array([missing[missing.model==m].groupby('modalities')[metric].mean().reindex(mods).values for m in MAIN])
        im=ax.imshow(values,aspect='auto',cmap='YlGnBu' if metric=='macro_f1' else 'YlOrRd')
        ax.set_xticks(range(7),mods,rotation=30);ax.set_yticks(range(2),[NAMES[m] for m in MAIN]);ax.set_title(title)
        for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='black',fontsize=9,
                                                     bbox=dict(facecolor='white',alpha=.6,edgecolor='none',pad=1))
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('图4 缺失模态类型的影响',fontsize=17)
    save(fig,'图4_缺失模态影响','T/A/V分别表示文本/语音/视觉。每格平均4档缺失比例×4种位置共16个条件，颜色不是显著性检验。')

    fig,axes=plt.subplots(1,2,figsize=(10,4.6));values=[]
    for m in MAIN:
        values.append(missing[missing.model==m].pivot_table(index='position',columns='rate',values='mae').reindex(['start','middle','end','hashed']).values)
    lo=min(v.min() for v in values);hi=max(v.max() for v in values)
    for ax,m,v in zip(axes,MAIN,values):
        im=ax.imshow(v,cmap='YlOrRd',aspect='auto',vmin=lo,vmax=hi)
        ax.set_xticks(range(4),['10%','30%','50%','70%']);ax.set_yticks(range(4),['前段','中段','后段','散列随机位置'])
        ax.set_title(NAMES[m]+' MAE ↓');ax.set_xlabel('目标连续区间比例')
        for (i,j),x in np.ndenumerate(v):ax.text(j,i,f'{x:.3f}',ha='center',va='center',fontsize=10,
                                               bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=1))
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('图5 缺失位置与长度的联合影响',fontsize=17)
    save(fig,'图5_位置与长度','每格平均7种模态组合。散列随机位置由样本编号、比例和模态计算，固定且可复核；多模态可在不同位置缺失。横轴不是秒。')

    fig,axes=plt.subplots(1,2,figsize=(11,6.2));names=list(NAMES)
    for ax,metric,title in zip(axes,['macro_f1','mae'],['112种缺失条件平均 Macro-F1 ↑','112种缺失条件平均 MAE ↓']):
        vals=[summary.loc[m,'missing_mean_'+metric] for m in names]
        ax.barh(range(len(names)),vals,color=[COLORS.get(m,'#AAC1C8') for m in names])
        ax.set_yticks(range(len(names)),[NAMES[m] for m in names]);ax.invert_yaxis();ax.set_title(title,fontsize=11)
        ax.set_xlim(0,max(vals)*1.21);ax.grid(axis='x',alpha=.15)
        for i,v in enumerate(vals):ax.text(v+.007,i,f'{v:.4f}',va='center',fontsize=9)
    fig.suptitle('图6 重新训练的消融实验',fontsize=17)
    save(fig,'图6_消融实验','9种配置均独立重新训练，并使用相同数据、遮挡计划和验证选择规则。基础融合有增强作为去专用模块的参照，去时间正则、去自适应权重、去状态记忆和去缺失训练用于更具体的对照。')

    conf=json.loads((root/'混淆矩阵.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(10,4.7))
    for ax,m in zip(axes,MAIN):
        cm=np.asarray(conf[m]);normalized=cm/cm.sum(1,keepdims=True)
        ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
        ax.set_xticks(range(3),['负向','中性','正向']);ax.set_yticks(range(3),[f'{t} (n={n})' for t,n in zip(['负向','中性','正向'],cm.sum(1))])
        ax.set_xlabel('预测极性');ax.set_ylabel('真实极性');ax.set_title(NAMES[m])
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v}\n{normalized[i,j]:.1%}',ha='center',va='center',color='white' if normalized[i,j]>.55 else 'black',fontsize=11)
    fig.suptitle('图7 验证集混淆矩阵',fontsize=17)
    save(fig,'图7_混淆矩阵','格内为数量与同一真实类别中的比例。连续标签0是中性；分类编号为0负向、1中性、2正向。')

    errors=pd.read_csv(root/'验证集逐样本预测与错误.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for ax,m in zip(axes,MAIN):
        e=errors[errors.model==m];ax.scatter(e.true_intensity,e.predicted_intensity,s=10,alpha=.28,color=COLORS[m])
        ax.plot([-3,3],[-3,3],'--',color='#666666');ax.set(xlim=(-3.2,3.2),ylim=(-3.2,3.2),xlabel='真实情感强度',ylabel='预测情感强度',title=NAMES[m])
        ax.grid(alpha=.15)
    fig.suptitle('图8 强度预测与真实值',fontsize=17)
    save(fig,'图8_强度散点','每点为一条有标签验证样本。使用已冻结的输出决策规则；偏离对角线表示回归误差。')

    groups=pd.read_csv(root/'错误分组统计.csv');grouplist=list(groups.group.unique())[:5]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,col,title in zip(axes,['mae','accuracy'],['按真实强度分组的 MAE ↓','按真实强度分组的 Accuracy ↑']):
        for i,m in enumerate(MAIN):
            vals=groups[groups.model==m].set_index('group').loc[grouplist]
            ax.bar(np.arange(5)+(i-.5)*.34,vals[col],width=.34,label=NAMES[m],color=COLORS[m])
        ax.set_xticks(range(5),[f'{g.split()[0]}\nn={n}' for g,n in zip(grouplist,vals.samples)],fontsize=9);ax.set_title(title);ax.grid(axis='y',alpha=.15)
        ax.margins(y=.2);ax.legend(fontsize=9)
    fig.suptitle('图9 错误分组与强度收缩',fontsize=17)
    save(fig,'图9_错误分组','分组依据真实强度。相关性分组只提供诊断线索，不能替代对原始视频的因果判断。')

    pred=pd.read_csv(root/'附件3预测'/'附件3全量预测及缺失情况.csv')
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    x=np.arange(1,31)
    for m in dict.fromkeys([*MAIN,selected]):axes[0].plot(x,pred[m+'_intensity'],'o-',ms=4,lw=1,label=NAMES[m],color=COLORS[m])
    axes[0].axhline(0,color='#777777',lw=.7);axes[0].set_ylabel('预测强度');axes[0].set_ylim(-3.1,3.1);axes[0].legend(ncol=2)
    for key,color in [('audio','#658F5B'),('vision','#8B78A8')]:axes[1].plot(x,pred[key+'_missing_rate']*100,'o-',ms=3,label=key,color=color)
    axes[1].set_ylabel('原始缺失率 (%)');axes[1].legend(ncol=2);axes[1].set_xlabel('附件3样本序号');axes[1].set_xticks(x);axes[1].tick_params(labelsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('图10 附件3全部30条预测与输入缺失情况',fontsize=17)
    save(fig,'图10_附件3全量展示','附件3无标签，图中仅呈现预测与原始缺失情况，不计算或暗示测试准确率。文本缺失率另存全量表。')

    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for m in MAIN:
        h=json.loads((root/m/'history.json').read_text(encoding='utf-8'))
        axes[0].plot([x['epoch'] for x in h],[x['task_loss'] for x in h],'o-',label=NAMES[m],color=COLORS[m])
        axes[1].plot([x['epoch'] for x in h],[x['selection']['score'] for x in h],'o-',label=NAMES[m],color=COLORS[m])
    for ax,title in zip(axes,['共同任务损失','验证选择分数 ↓']):ax.set_title(title);ax.set_xlabel('轮次');ax.grid(alpha=.15);ax.legend(fontsize=9)
    fig.suptitle('图11 训练过程与检查点选择',fontsize=17)
    save(fig,'图11_训练过程','共同任务损失不含矩阵专有辅助重构项；检查点由预设clean及4种中段30%缺失条件的验证分数选择，未使用附件3标签。')
    fig,axes=plt.subplots(1,2,figsize=(10,4.7));cm=np.asarray(conf[selected]);norm=cm/cm.sum(1,keepdims=True)
    axes[0].imshow(norm,cmap='Greens',vmin=0,vmax=1)
    axes[0].set_xticks(range(3),['负向','中性','正向']);axes[0].set_yticks(range(3),['负向','中性','正向'])
    axes[0].set_xlabel('预测极性');axes[0].set_ylabel('真实极性');axes[0].set_title('最终选型 混淆矩阵')
    for (i,j),v in np.ndenumerate(cm):axes[0].text(j,i,f'{v}\n{norm[i,j]:.1%}',ha='center',va='center',color='white' if norm[i,j]>.55 else 'black')
    e=errors[errors.model==selected];axes[1].scatter(e.true_intensity,e.predicted_intensity,s=10,alpha=.3,color=COLORS[selected])
    axes[1].plot([-3,3],[-3,3],'--',color='#777777');axes[1].set(xlim=(-3.2,3.2),ylim=(-3.2,3.2),xlabel='真实强度',ylabel='预测强度',title='最终选型 强度预测')
    fig.suptitle('图12 根据消融结果确定的最终模型 '+NAMES[selected],fontsize=15)
    save(fig,'图12_最终模型验证','在所有9种配置中按统一验证分数选型。最终保留的模块以实测为依据，并非所有新增模块都有收益。验证集用于选型，指标不作为独立测试成绩。')
    (dest/'图注.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'图注.md').write_text('\n\n'.join(f'## {k}\n\n{v}' for k,v in captions.items()),encoding='utf-8')
    print(json.dumps({'figures':len(captions),'destination':str(dest)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('result');a=p.parse_args();generate(a.result)
