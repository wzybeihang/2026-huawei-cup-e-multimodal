"""Generate a readable PDF and editable Markdown from the actual experiment tables."""
from pathlib import Path
import json, html, os, hashlib, tempfile
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image

ROOT=Path(__file__).resolve().parent
RESULT=ROOT/'实验结果'
S=pd.read_csv(RESULT/'模型与消融总表.csv').set_index('model')
P=pd.read_csv(RESULT/'附件3预测'/'附件3全量预测及缺失情况.csv')
E=pd.read_csv(RESULT/'验证集逐样本预测与错误.csv')
G=pd.read_csv(RESULT/'错误分组统计.csv')
NAMES={'matrix_full':'矩阵完整配置','function_full':'函数完整配置','baseline_clean':'基础融合 无增强',
       'baseline_aug':'基础融合 有增强','matrix_no_temporal':'矩阵 去时间正则','matrix_no_aug':'矩阵 去缺失增强',
       'function_no_adaptive':'函数 去自适应权重','function_no_memory':'函数 去状态记忆','function_no_aug':'函数 去缺失增强'}
selected=json.loads((RESULT/'方案选择.json').read_text(encoding='utf-8'))['selected']
assert selected=='matrix_no_aug','The written interpretation must be revised for a different selection'

pdfmetrics.registerFont(TTFont('Body',r'C:\Windows\Fonts\msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('Bold',r'C:\Windows\Fonts\msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
styles={
 'title':ParagraphStyle('title',fontName='Bold',fontSize=21,leading=29,spaceAfter=15,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName='Bold',fontSize=15,leading=21,spaceAfter=11,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='Bold',fontSize=11,leading=16,spaceBefore=8,spaceAfter=6,wordWrap='CJK'),
 'p':ParagraphStyle('p',fontName='Body',fontSize=9.7,leading=15.2,spaceAfter=7,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8.6,leading=13,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.2,leading=12,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='Bold',fontSize=8.2,leading=12,wordWrap='CJK'),
 'formula':ParagraphStyle('formula',fontName='Body',fontSize=9.4,leading=16,spaceBefore=4,spaceAfter=7,wordWrap='CJK'),
}
story=[];md=[];page_titles=[]
def paragraph(text,kind='p'):
    safe=str(text)
    for source,target in [('Ẑ','Z_hat'),('λ₀','lambda_0'),('λₜ','lambda_T'),('s₀','s_0'),('P₀','P_0'),('ᵀ','^T')]:
        safe=safe.replace(source,target)
    story.append(Paragraph(html.escape(safe).replace('\n','<br/>'),styles[kind]))
    md.append(str(text)+'\n')
def page(title,first=False):
    if not first:story.append(PageBreak())
    page_titles.append(title);paragraph(title,'title' if first else 'h1')
    md[-1]=('# ' if first else '## ')+title+'\n'
def sub(title):paragraph(title,'h2');md[-1]='### '+title+'\n'
def formula(display,latex):
    from PIL import Image as PILImage
    asset_dir=ROOT/'公式素材'
    for piece in latex.split(r'\quad '):
        expression=piece.strip(' ,;')
        path=asset_dir/(hashlib.sha256(expression.encode()).hexdigest()[:16]+'.png')
        if not path.exists():raise FileNotFoundError('Run render_equations.py before building the PDF')
        with PILImage.open(path) as im:w,h=im.size
        width=min(w/230*72,16.6*cm);height=width*h/w
        picture=Image(str(path),width=width,height=height);picture.hAlign='LEFT'
        story.extend([picture,Spacer(1,5)])
    md.append('$$\n'+latex+'\n$$\n')
def table(headers,rows,widths=None,small=False):
    data=[[Paragraph(html.escape(str(x)),styles['head']) for x in headers]]
    data += [[Paragraph(html.escape(str(x)),styles['cell']) for x in row] for row in rows]
    widths=widths or [16.8/len(headers)]*len(headers)
    t=Table(data,colWidths=[w*cm for w in widths],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFEAF0')),
        ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),3 if small else 5),('BOTTOMPADDING',(0,0),(-1,-1),3 if small else 5)]))
    story.extend([t,Spacer(1,8)])
    md.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
              '\n'.join('| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows)+'\n')
def figure(name,width=16.8):
    path=RESULT/'图表'/(name+'.png')
    from PIL import Image as PILImage
    with PILImage.open(path) as im:w,h=im.size
    story.extend([Image(str(path),width=width*cm,height=width*cm*h/w),Spacer(1,6)])
    md.append(f'![{name}](实验结果/图表/{name}.png)\n')
def f(value):return f'{value:.4f}'
def metric_rows(names,prefix='clean_'):
    return [[NAMES[m],f(S.loc[m,prefix+'accuracy']),f(S.loc[m,prefix+'macro_f1']),
             f(S.loc[m,prefix+'mae']),f(S.loc[m,prefix+'pearson'])] for m in names]

page('问题2 局部模态缺失预测建模与实验报告',True)
paragraph('以赛题原文的问题2及第四部分四项正文要求为依据，使用附件2的训练和验证划分，完成两条建模路线、局部缺失条件实验、重新训练的消融实验，以及附件3全量预测。所有指标来自实际运行。')
paragraph('主要结论：矩阵补全与函数状态估计产生不同预测，完整配置在728条验证样本上有113条极性判断不同。消融实验显示，并非所有创新模块都有效。按统一验证分数选择后，最终保留低秩补全和时间正则，去掉本次设置的缺失增强，采用matrix_no_aug配置。')
table(['题目正文要求','本报告及对应材料'],[
 ['建模原理 结构 目标 训练 参数','第1部分；models.py、experiment_config.json'],
 ['缺失规律与消融','第2部分；1017行条件结果、9种重新训练配置'],
 ['附件3全量预测与展示','第3部分；30条完整预测表与CSV'],
 ['验证评价 可视化 错误归因','第4部分；混淆矩阵、散点图、错误样本与分组统计']], [5.4,11.4])
table(['数据','数量','作用'],[
 ['附件2 train','3395','唯一用于参数学习和标准化拟合'],
 ['附件2 valid','728','模型结构 检查点 决策规则选择与内部验证'],
 ['附件2 test','727','本次不使用'],
 ['附件3 对齐版本','30','无标签专项推理 不训练 不调参']], [5.1,2,9.7])
paragraph('依据文件为所提供的《复杂场景下多模态情感识别的数学建模与算法设计.docx》，其正文标题使用“情感预测”。题目没有规定具体缺失率档位；本报告中的10%、30%、50%、70%及位置网格是为覆盖任务要求而设定的实验方案，不冒充题目给定参数。','small')
paragraph('本次没有做随机种子扫描。固定初始化仅用于复现；实验自变量是缺失模态、缺失比例、缺失位置和模型模块。旧版结果作为探索记录保留，不作为本报告的正式实验表。','small')

page('1 建模原理与网络结构')
sub('1.1 输入接口与缺失语义')
paragraph('全流程使用aligned_50版本。文本选择text_bert三路整数输入，通过冻结的通用BERT编码为50×768；语音为50×74，视觉为50×35。text与text_bert二选一，不构成四个模态。三路编码使用相同模型与处理规则，附件3不混入另一版本特征。')
paragraph('每个样本有有效序列掩码D和三模态观测掩码O。CLS、SEP和padding不计入有效内容；内容区间内全零语音或视觉向量标为缺失，单个数值0仍是合法特征值。内部缺失词元保留其序列位置，不当作padding。标准化均值和标准差仅在train原本可见的位置拟合，标准化后裁切至[-8,8]。')
paragraph('文本人工遮挡先将目标位置的词元编号和注意力可见性置零，再重新执行BERT，并把缺失位置的输出置零。因此剩余位置不能通过注意力读到被遮挡词元；核验实验已确认，改变隐藏词元的原始编号不会改变编码输出。原始文件不被修改。')
formula('z[t,m] = O[t,m] · tanh(W[m] x[t,m] + b[m])，每个模态投影为32维。',
        r'z_{t,m}=O_{t,m}\tanh(W_m x_{t,m}+b_m),\quad z_{t,m}\in\mathbb{R}^{32}.')
figure('图1_网络结构')
paragraph('两路线共享网络层的结构和初始值，但各自独立训练。融合输出只经注意力池化与均值池化、再拼接三模态覆盖率进入预测头，没有直接观测表示绕过专用模块的残差分支。预测头为67→64、GELU、Dropout 0.1，随后输出3类logits和一个范围[-3,3]的连续强度。','small')

page('1.2 带时间正则的低秩矩阵补全')
paragraph('将三路32维表示拼成Z∈R^(T×96)，对应观测掩码复制到各模态的32个维度。假定跨模态与相邻位置存在可学习的低维相关结构，用训练集学习共享字典B∈R^(96×8)，对每个输入样本单独估计时间因子U∈R^(T×8)。')
formula('固定B时求解 J(U) = Σ[t,d] w[t,d](Z[t,d] − b[d]ᵀu[t])² + λ₀Σ[t]||u[t]||² + λₜΣ[t]||u[t] − u[t−1]||²。',
        r'\min_U\ \sum_{t,d}w_{td}(Z_{td}-b_d^\top u_t)^2+\lambda_0\sum_t\|u_t\|_2^2+\lambda_T\sum_{t\sim t-1}\|u_t-u_{t-1}\|_2^2.')
paragraph('其中w为观测掩码除以32；λ₀=0.03，λₜ=0.05。相邻差分只连接相邻有效内容位置，不连接padding。λ₀>0使固定字典时的二次问题具有正定正规方程，但不能由此保证缺失特征可被真实恢复。')
formula('A[t] = Bᵀdiag(w[t])B + (λ₀ + λₜdeg[t])I；b[t] = Bᵀdiag(w[t])Z[t]。\nu[t]^(k+1) = A[t]⁻¹ { b[t] + λₜΣ[j邻接t]u[j]^(k) }。',
        r'A_t=B^\top\operatorname{diag}(w_t)B+(\lambda_0+\lambda_T\deg t)I,\quad b_t=B^\top\operatorname{diag}(w_t)Z_t,\quad u_t^{k+1}=A_t^{-1}(b_t+\lambda_T\sum_{j\sim t}u_j^k).')
paragraph('实现中通过Cholesky求逆后做20次块Jacobi更新。这是固定迭代次数的近似求解，不声称每个样本都达到精确最优。重建Ẑ=UBᵀ；已观测位置保留原来的投影表示，仅在缺失位置采用裁切至[-1,1]的重建值。随后对三个模态表示取均值，得到每个位置的融合状态。')
sub('全模态在同一局部区间缺失时')
paragraph('该区间内部没有直接观测约束，估计主要由相邻时刻与训练字典决定。连续区间越长，先验依赖通常越强。补值仍是模型估计，O保持为0；它不会被改记为真实观测。')
sub('辅助重构目标')
formula('L_rec = Σ[t,d] w[t,d](Ẑ[t,d] − stopgrad(Z[t,d]))² / max(Σ[t,m]O[t,m], 1)。',
        r'L_{\mathrm{rec}}=\frac{\sum_{t,d}w_{td}(\widehat Z_{td}-\operatorname{sg}(Z_{td}))^2}{\max(\sum_{t,m}O_{tm},1)}.')
paragraph('重构辅助项只以原本可见的输入表示为目标，缺失真实值不用于专项推理。字典和预测头仅从train更新。矩阵方案的最终选型使用该辅助项和时间正则，未保留缺失增强损失。')

page('1.3 函数状态估计与共同学习目标')
paragraph('函数路线以32维潜在表示作为递推状态，采用随机游走先验。各模态投影为同一状态的观测；协方差以对角向量近似。所有除法、乘法及开方按分量进行，P是内部不确定性代理。')
formula('s[t]⁻ = s[t−1]；P[t]⁻ = P[t−1] + Q；e[t,m] = z[t,m] − s[t]⁻。\nω[t,m] = min{1, 0.75 / (|e[t,m]| / √(P[t]⁻ + R[m]) + 10⁻⁶)}。',
        r's_t^-=s_{t-1},\ P_t^-=P_{t-1}+Q,\ e_{tm}=z_{tm}-s_t^-;\quad \omega_{tm}=\min\{1,\frac{0.75}{|e_{tm}|/\sqrt{P_t^-+R_m}+10^{-6}}\}.')
formula('J[t,m] = O[t,m]ω[t,m] / R[m]；P[t] = {1/P[t]⁻ + Σ[m]J[t,m]}⁻¹。\ns[t] = s[t]⁻ + P[t]Σ[m]J[t,m]e[t,m]。',
        r'J_{tm}=\frac{O_{tm}\omega_{tm}}{R_m},\quad P_t=((P_t^-)^{-1}+\sum_mJ_{tm})^{-1},\quad s_t=s_t^-+P_t\sum_mJ_{tm}e_{tm}.')
paragraph('初值s₀=0、P₀=1。Q=0.005+0.20·sigmoid(q)，初值0.02；R=0.05+0.95·sigmoid(r)，初值0.30。Q和R在训练集学习，保持严格为正。缺失模态精度为0；三模态都缺失时仅做预测，状态维持、P增长。padding不更新状态。')
paragraph('自适应权重借鉴稳健估计的创新降权思想，但模态间不一致也可能是有效情感线索，不能直接等同传感器故障。该实现不包含控制输入、滚动优化或H∞性能界证明，因此不能命名为已实现的鲁棒MPC或声称理论最坏误差保证。')
sub('共同任务目标与缺失训练')
formula('ℓ = CE(c, logits) + SmoothL1(y, y_raw)。\nL = E[ℓ_clean] + β E[max(ℓ_mask1, ℓ_mask2)] + γ L_rec。',
        r'\ell=\mathrm{CE}(c,\mathrm{logits})+\mathrm{SmoothL1}(y,\widehat y_{\mathrm{raw}}),\quad L=\mathbb{E}\ell_{\mathrm{clean}}+\beta\mathbb{E}\max(\ell_{M_1},\ell_{M_2})+\gamma L_{\mathrm{rec}}.')
paragraph('增强配置β=0.5；无增强消融β=0。矩阵配置γ=0.05，其他模型γ=0。mask1按确定性平衡计划轮换112种条件，mask2固定为T+A+V中段30%。这里的max只作用于两个训练候选，不等于对全部可能扰动取全局最坏情况。')

page('1.4 训练方案与关键参数')
table(['参数或规则','本次实际设置'],[
 ['训练数据','附件2 train 3395条；不混入valid或无标签专项集'],
 ['文本编码','通用BERT base uncased；12层、768维；冻结、不做情感微调'],
 ['输入与模态投影','50位置；维度768/74/35；各模态Linear+tanh到32维'],
 ['优化器','AdamW；学习率0.001；weight decay 0.0001'],
 ['批量与梯度','batch size 64；梯度范数裁切1.0'],
 ['停止规则','最多20轮；至少8轮；连续5轮验证分数未改善时停止'],
 ['检查点选择','同一验证分数；两完整配置最佳均为第3轮；最终矩阵选型为第4轮'],
 ['矩阵参数','秩8；岭正则0.03；时间正则0.05；20次块Jacobi'],
 ['函数参数','状态32维；Q初值0.02；R初值0.30；创新阈值0.75'],
 ['固定初始化','2026；仅用于复现，不是实验自变量'],
 ['运行环境','Python 3.12.14；PyTorch 2.11.0+cu128；transformers 4.57.6'],
 ['存储与精度','文本缓存float16、模型输入float32；音视频float32']], [4.4,12.4])
sub('验证分数与输出规则')
formula('S = 0.5·g(clean) + 0.5·mean{g(T), g(A), g(V), g(T+A+V)}；g = MAE + 0.5(1 − Macro-F1)。',
        r'S=\frac{1}{2}g_{\mathrm{clean}}+\frac{1}{8}\sum_{m\in\{T,A,V,TAV\}}g_{m,\rho=.3,\mathrm{middle}},\quad g=\mathrm{MAE}+\frac{1}{2}(1-\mathrm{MacroF1}).')
paragraph('每轮仅按上述clean及4种中段30%条件选检查点和决策规则。候选规则为分类头决定极性并把回归值投影至相应符号，或按回归值的中性区间分类。中性阈值候选为0、0.05、0.1、0.15、0.2、0.3、0.4、0.5、0.6、0.8。连续真实标签0始终只属于中性；阈值只是预测规则。','small')
paragraph('实测矩阵完整配置选择阈值0.2规则；函数完整配置与最终矩阵配置选择分类投影规则。全套消融后使用同一个S在9种配置中选型；因此这些指标属于用于选择结构的内部验证，不是独立测试成绩。','small')

page('2 缺失实验条件与影响规律')
paragraph('每个模型在同一728条验证样本上评价：clean加7种模态组合×4档比例×4种位置，共113条件；9种模型配置共1017行指标。7组合为T、A、V、T+A、T+V、A+V、T+A+V。位置为前段、中段、后段及由样本编号、比例、模态散列得到的可复核随机位置。')
paragraph('遮挡沿有效内容序列连续选取。目标长度为round(ρT)，限制在1至T−1；若遮挡将删除某模态仅剩的全部观测，则缩短区间以保留至少一个观测。原生缺失保留、不伪造。validation_masks.npz存储原始与处理后掩码、每个样本每个模态的起止位置；missingness_audit.csv记录实际数量。')
paragraph('ρ为目标连续区间占有效位置的比例，不是秒数。实际新增缺失率=本次移除的原有观测数/受影响模态的原有观测数；总缺失率则以有效位置×受影响模态数为分母。题目没有逐词物理时间戳，不能把50位置解释成50秒。')
figure('图3_缺失率影响')
paragraph('随目标缺失率由10%增至70%，最终矩阵配置的平均Macro-F1从0.5816下降至0.5209，MAE从0.5984上升至0.6609。函数完整配置对应Macro-F1从0.5722下降至0.5163、MAE从0.6042上升至0.6780。这些是跨28个固定条件的描述性均值，不是28次独立重复实验。','small')

page('2.2 缺失模态与缺失位置')
figure('图4_缺失模态影响')
paragraph('本次文本缺失的影响较大。最终矩阵配置仅缺语音、仅缺视觉的平均MAE分别为0.5912、0.5927；仅缺文本为0.6506，文本与视觉同时缺失为0.6590。单模态编码后的维度相同，因此这一差异不能简单归结为文本原始维度更高。')
figure('图5_位置与长度')
paragraph('位置影响具有模型差异。函数完整配置的前段缺失平均MAE为0.6509，高于后段的0.6332；单向状态递推对早期信息的依赖是一种机制解释，但不能单凭该相关性视为因果证明。最终矩阵配置四类位置的平均MAE在0.6247至0.6297之间，位置差异较小。','small')
paragraph('不能声称“缺失模态越多必然越差”。例如最终矩阵配置在A+V缺失条件的平均MAE为0.5880，略低于clean的0.5965。这可能与移除非关键信息、条件组成和模型偏差有关，不能据此认定语音或视觉没有用。','small')

page('2.3 消融实验与最终选型')
rows=[]
for name in NAMES:
    rows.append([NAMES[name],str(int(S.loc[name,'best_epoch'])),f(S.loc[name,'clean_macro_f1']),
                 f(S.loc[name,'missing_mean_macro_f1']),f(S.loc[name,'missing_mean_mae'])])
table(['模型配置','最佳轮','clean F1','缺失均值F1','缺失均值MAE'],rows,[5.8,1.4,3.2,3.2,3.2],small=True)
paragraph('以上9种配置均重新训练，不是只在推理时临时关闭模块。去时间正则只将矩阵λₜ置0；去自适应权重令函数ω=1；去状态记忆在每个位置重置s=0、P=1；去缺失增强只将β置0。基础融合使用可见模态均值，分别保留和去掉缺失训练，作为专用模块的参照。','small')
paragraph('矩阵时间正则的增益很小：完整配置与去时间正则的缺失平均MAE为0.6312与0.6317，不能夸大。函数去自适应权重后MAE反而由0.6398降至0.6342；去状态记忆后为0.6382，当前实验没有证明这两个模块带来整体收益。','small')
paragraph('矩阵去缺失增强后，平均Macro-F1由0.5246升至0.5560，MAE由0.6312降至0.6265。故最终采用matrix_no_aug。与同样无增强的基础融合相比，其平均MAE降低3.69%，Macro-F1提高0.0067；这是本轮固定条件的实测差异，不是显著性保证。','small')
figure('图6_消融实验',width=14.4)

page('3 附件3全量预测结果')
paragraph('附件3对齐版本共30条，全部保留并完成推理。下表列出两套完整配置的强度，以及验证选型后的最终结果；矩阵与函数完整配置的极性、原始缺失率及来源信息另存全量CSV。最终文件为“问题2_附件3预测结果.csv”，来自matrix_no_aug。','small')
translate={'Negative':'负向','Neutral':'中性','Positive':'正向'}
rows=[]
for _,r in P.iterrows():
    rows.append([r.sample_id,f(r.matrix_full_intensity),f(r.function_full_intensity),
                 translate[r.matrix_no_aug_polarity],f(r.matrix_no_aug_intensity)])
table(['样本编号','矩阵完整强度','函数完整强度','最终极性','最终强度'],rows,[3.6,3.7,3.7,2.3,3.5],small=True)
paragraph('表中数值保留4位小数，CSV保留计算精度。中性输出强度0，负向输出为负，正向输出为正。附件3没有标签，因此本表不报告Accuracy、F1、MAE或Pearson。','small')

page('3.2 全量展示与专项输入核查')
figure('图10_附件3全量展示')
table(['项目','实际结果'],[
 ['预测覆盖','30/30，无缺失编号，无重复编号'],
 ['最终极性分布','负向3条，中性9条，正向18条'],
 ['文本原生缺失','本批30条样本均未发现内容位置文本缺失'],
 ['音视频缺失','按有效位置加权，语音21.65%，视觉22.31%'],
 ['padding处理','不计入缺失率；全零填充不作为局部故障'],
 ['无缺失样本','01、05、10在本接口中未检出音视频内容缺失，原样保留']], [4.4,12.4])
paragraph('训练、验证和专项推理均通过text_bert接口重新提取冻结BERT特征。附件3未用于拟合字典、标准化参数、预测头或中性阈值；矩阵推理求解的是单个样本的时间因子，不会用无标签专项样本重新训练共享参数。')
paragraph('CSV列为sample_id、sentiment_polarity、sentiment_intensity。编号使用专项文件名去掉扩展名；来源清单记录每个原始文件SHA256，保证预测与输入一一对应。')

page('4 验证集基础性能与可视化')
table(['配置','Accuracy','Macro-F1','MAE','Pearson'],
      metric_rows(['matrix_full','function_full','matrix_no_aug']),[5.2,2.9,2.9,2.9,2.9])
paragraph('以上为未新增人工缺失的验证性能，仍保留官方输入中已有的缺失。F1主报三类等权Macro-F1，同时在CSV提供Weighted-F1与各类F1。Accuracy高不代表中性识别好；本次函数完整配置Accuracy高于矩阵完整配置，但Macro-F1更低。')
figure('图12_最终模型验证')
table(['真实类别','样本数','正确数','召回率'],[
 ['负向','206','137','66.50%'],['中性','184','66','35.87%'],['正向','338','257','76.04%']], [5.4,3.8,3.8,3.8])
paragraph('最终模型的主要分类短板是中性：184条中性中81条被判正向、37条被判负向。训练集中正向占49.19%，类别比例是可能的影响因素，但这里没有通过重采样对照证明其因果贡献。','small')
paragraph('两完整模型在728条样本中有113条极性不同，预测强度平均绝对差为0.1170，排除了“读取同一份预测”的解释。旧版的9条差异主要反映共同分支占比大；新结果仍不能保证任意两种算法都必然相差很大。','small')

page('4.2 错误归因与具体样本')
sub('强情感幅度被低估')
table(['真实强度组','样本数','真实均值','预测均值','MAE'],[
 ['强负向 y<-1','85','-1.7961','-0.8925','0.9698'],
 ['弱负向 -1≤y<0','121','-0.5675','-0.2767','0.5808'],
 ['中性 y=0','184','0.0000','0.0867','0.3963'],
 ['弱正向 0<y≤1','224','0.6198','0.4430','0.4709'],
 ['强正向 y>1','114','1.7515','0.8867','0.9049']], [5.2,2.0,3.2,3.2,3.2])
paragraph('极端强度预测向0收缩较明显；弱负向也容易与中性混淆。MAE的改善与类别识别不总同步，应同时检查分类与回归指标，不把预测曲线“平滑”直接等同鲁棒性。')
sub('案例一 比较对象与情感词作用范围')
paragraph('f_ZJ7L14oYQ$_$21：真实+2.0000，最终预测-1.6844。文本为“They just so much more convenient than stupid boots that take forever to put on and just aren’t comfortable!”。句子在赞扬当前对象，同时批评比较对象boots；stupid和aren’t comfortable可能造成目标作用范围混淆。这是基于文本的诊断假设，尚非模型内部因果证明。','small')
sub('案例二 隐含评价')
paragraph('266791$_$23：真实-2.6667，预测+0.2570。文本评价演员像第一次看着提示卡念台词，负向态度通过描述表达，没有简单的否定情感词。该例说明词面与语境关系仍可能被误判；不能仅凭预测错误断定某个模态发生故障。','small')
sub('案例三 官方50位置截断')
paragraph('102858$_$4：真实+3.0000，预测+0.3354，极性正确但强度明显偏低。原始转写重新分词为83个含特殊词元的token，存储接口只保留48个内容位置，末尾停在“but it also has”。原文后面的“pretty funny”和“absolutely awesome”未进入给定文本输入。此例支持“关键信息被截断”这一可核验的数据原因。','small')
paragraph('没有扩展或替换题目提供的特征序列，也没有使用raw_text重新生成更长的训练输入。raw_text仅用于本页核查。视觉原生缺失多的分组MAE并未一致更差，因此不能直接把全部错误归因于视觉缺失。','small')

page('4.3 三模态共同局部缺失的机制验证')
figure('图13_共同缺失区间')
paragraph('示例按固定规则选择：验证集中第一个不少于20个有效内容位置、三模态原本全部可见的样本-UacrmKiTn4$_$7，然后遮挡共同中段50%。这不是整条样本的三模态全缺失，前后仍有可用观测。')
paragraph('在全缺局部区间，函数路线停止观测更新，方差代理随递推增长；矩阵路线由前后位置和训练字典给出估计。图中的潜在表示范数不是逐时刻情感强度，方差代理也不是已经校准的置信区间。该图验证计算行为，不能取代分类与回归评价。')
sub('核验结果')
paragraph('7项核心测试通过，覆盖内部文本缺失与padding区分、局部掩码保留观测、隐藏NaN值不影响输出、梯度有限、无观测时方差增长、矩阵保留已观测表示等。全部1017行Accuracy、Macro-F1及MAE已从存储的逐样本预测独立重算并一致。')
paragraph('另外核对了30条专项预测的编号唯一性、覆盖率、强度范围及极性符号一致性；改变被遮挡词元的原始编号后BERT输出完全相同，证明该输入路径没有读取被遮挡内容。核验通过表示实现和文件检查通过，不表示预测无误。')

page('复现材料与结果使用')
table(['材料','用途'],[
 ['protocol.py','113条件、有效位置、局部缺失规则与9种模型配置'],
 ['prepare_data.py','同一BERT处理、train标准化、先遮挡后编码'],
 ['models.py 与 evaluation.py','两路线、基础融合、损失、输出规则与指标'],
 ['run_experiments.py','训练、验证选择、消融、全条件评价'],
 ['summarize.py 与 diagnose.py','全量推理、错误统计、独立核验'],
 ['plot_results.py 与 build_report.py','由实测表格生成图与报告'],
 ['实验结果 各配置 best.pt','9种配置的模型参数与所选输出规则'],
 ['scaler.npz 配置与SHA256','训练预处理参数、模型环境及输入版本核验']], [7.0,9.8])
paragraph('一键入口为Run_Required_Experiments.ps1，默认执行数据准备、核心测试、9配置训练、113条件评价、附件3推理、绘图和独立核验。具体命令和迁移路径见《运行说明.md》。已完成的配置可复用；未完成配置会从头重训，不声称支持未实现的逐轮恢复。')
paragraph('BERT来自google-bert/bert-base-uncased通用模型，模型卡地址：https://huggingface.co/google-bert/bert-base-uncased 。本次使用其通用语言预训练参数，未引入外部情感数据集或情感专用预测头。缓存身份文件保存实际权重SHA256；复现须使用相同权重。','small')
paragraph('依赖版本另附requirements-lock.txt。原始官方数据和通用BERT权重不重复放入轻量代码包；二者的来源与哈希已记录，文本缓存可通过脚本重建。最终训练参数体积很小。实验结果中不包含参赛单位、姓名或队号；整套比赛附件仍须与问题1、问题3合并后核对≤50MB限制。','small')
sub('可以支持的结论')
paragraph('本次实验证明两条算法路线实现不同，并提供了模态、缺失率和位置的完整实测对照。验证结果支持保留低秩补全方案作为当前专项预测模型；时间正则增益有限，缺失增强和自适应权重未在所有指标上带来收益。所有结论限定于本次数据划分、输入版本、训练预算及固定实验条件，不宣称普遍优势或理论鲁棒性能界。')
paragraph('正文图表、全量CSV、逐样本预测与掩码提供从报告数字回溯到实验条件和输入样本的路径。附件3结果没有真实标签，不能用于报告分类或回归准确率。')

def footer(canvas,doc):
    canvas.setFont('Body',8);canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawCentredString(A4[0]/2,1.0*cm,f'问题2 局部模态缺失建模与实验  |  {doc.page}')
output=ROOT/'问题2四项要求完整报告.pdf'
doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=2.1*cm,rightMargin=2.1*cm,
                      topMargin=1.65*cm,bottomMargin=1.65*cm,title='问题2局部模态缺失预测建模与实验报告',author='')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'问题2四项要求完整报告.md').write_text('\n'.join(md),encoding='utf-8')
from pypdf import PdfReader
reader=PdfReader(output)
print(json.dumps({'pages':len(reader.pages),'planned_sections':len(page_titles),'pdf':str(output)},ensure_ascii=False))
for i,p in enumerate(reader.pages):
    text=p.extract_text()
    print(i+1,len(text),text[:60].replace('\n',' '))
