"""Generate all visualization figures for Q2-model-on-Q3 experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import ROOT, KEYS, read

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.dpi': 210})
COL = ['#24648c', '#c47b25', '#4f8b73']
NAMES = ['文本', '语音', '视觉']
CLASS = ['负向', '中性', '正向']
R = ROOT / '实验结果'
F = ROOT / '图表'
F.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(F / (name + '.png'), bbox_inches='tight', facecolor='white')
    fig.savefig(F / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    met = json.loads((R / 'valid_metrics.json').read_text(encoding='utf-8'))
    audit = json.loads((R / 'attribution_audit.json').read_text(encoding='utf-8'))
    v = pd.read_csv(R / '验证集逐样本预测.csv')
    va = np.load(R / 'valid_attributions.npz')
    ta = np.load(R / 'attachment4_attributions.npz')
    p = pd.read_csv(ROOT / '提交结果/问题3_附件4预测与解释结果_Q2模型.csv', dtype={'sample_id': str})

    # Load 附件4 raw texts
    a4_dir = Path(r'<project_root>\附件4-可解释专项视频样本与特征文件\附件4-可解释专项视频样本与特征文件\对齐版本')
    a4_items = {read(p_)['id']: read(p_) for p_ in sorted(a4_dir.glob('*.pkl'))}
    a4_raw = {sid: a4_items[sid]['raw_text'] for sid in a4_items}

    # ---- 图1: 模型结构 ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.5); ax.axis('off')
    for j, (name, d) in enumerate(zip(NAMES, [768, 74, 35])):
        y = 3.7 - j * 1.25
        for x, label, w in [(.1, f'{name}\n50 × {d}', 1.5),
                            (2., '训练集标准化\n掩码排除填充', 1.6),
                            (4.2, 'Linear+Tanh\n→ 32维', 1.6)]:
            ax.text(x + w / 2, y, label, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=.5', facecolor='#eef4f7', edgecolor=COL[j]), fontsize=10)
        ax.annotate('', (2, y), (1.6, y), arrowprops=dict(arrowstyle='->', color=COL[j]))
        ax.annotate('', (4.2, y), (3.7, y), arrowprops=dict(arrowstyle='->', color=COL[j]))
    # Matrix completion box
    ax.text(6.6, 2.45, '低秩矩阵补全\n(秩8 + 时间正则)\n+ 三模态均值融合', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=.7', facecolor='#fff3e0', edgecolor='#c47b25'), fontsize=10)
    for j in range(3):
        y = 3.7 - j * 1.25
        ax.annotate('', (5.9, 2.45), (5.9, y), arrowprops=dict(arrowstyle='->', color=COL[j]))
    ax.text(8.7, 2.45, '注意力池化 + 均值池化\n+ 模态覆盖率(3)\n→ 67维 → 64\n极性三分类 ＋ 强度回归',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=.7', facecolor='#f3f3f3', edgecolor='#777'))
    ax.annotate('', (7.6, 2.45), (7.3, 2.45), arrowprops=dict(arrowstyle='->', color='#333'))
    fig.tight_layout()
    save(fig, '图1_问题2模型结构')

    # ---- 图2: Q2 vs Q3 验证集对比 ----
    q3_met = dict(accuracy=0.6484, macro_f1=0.6299, mae=0.5656, pearson=0.6851)
    labels = ['Accuracy', 'Macro F1', '1 - MAE', 'Pearson']
    q2_vals = [met['accuracy'], met['macro_f1'], 1 - met['mae'], met['pearson']]
    q3_vals = [q3_met['accuracy'], q3_met['macro_f1'], 1 - q3_met['mae'], q3_met['pearson']]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - .18, q2_vals, .36, label='问题2模型 (matrix_no_aug)', color=COL[0])
    ax.bar(x + .18, q3_vals, .36, label='问题3模型 (fusion_h32)', color=COL[2])
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel('指标值 (越高越好, MAE取1-MAE)')
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    for i, (a, b) in enumerate(zip(q2_vals, q3_vals)):
        ax.text(i - .18, a + .015, f'{a:.3f}', ha='center', fontsize=8)
        ax.text(i + .18, b + .015, f'{b:.3f}', ha='center', fontsize=8)
    fig.tight_layout()
    save(fig, '图2_问题2与问题3模型验证对比')

    # ---- 图3: 混淆矩阵 + 回归散点 ----
    cm = np.array(met['confusion_matrix'])
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
    axs[0].imshow(cm, cmap='Blues')
    axs[0].set_xticks(range(3), CLASS); axs[0].set_yticks(range(3), CLASS)
    axs[0].set_xlabel('预测类别'); axs[0].set_ylabel('真实类别')
    for i in range(3):
        for j in range(3):
            axs[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() * .55 else 'black')
    axs[1].scatter(v.true_intensity, v.predicted_intensity, s=12, alpha=.4, color=COL[0])
    axs[1].plot([-3, 3], [-3, 3], '--', color='#777')
    axs[1].set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), xlabel='真实情感强度', ylabel='预测情感强度',
               title=f"MAE {met['mae']:.4f}    Pearson {met['pearson']:.4f}")
    fig.tight_layout()
    save(fig, '图3_混淆矩阵与回归散点')

    # ---- 图4: 模态作用差异 ----
    signed = va['shap'][:, :, 1]
    weights = np.abs(signed) / np.maximum(np.abs(signed).sum(1, keepdims=True), 1e-12)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axs[0].boxplot([weights[:, m] for m in range(3)], tick_labels=NAMES, showfliers=False)
    axs[0].set_ylabel('当前预测类别的绝对贡献占比')
    counts = audit['valid']['principal_modality_counts']
    axs[1].bar(NAMES, counts, color=COL)
    for i, c in enumerate(counts):
        axs[1].text(i, c + 10, f'{c}条\n{c / len(v):.1%}', ha='center')
    axs[1].set_ylim(0, 740); axs[1].set_ylabel('主要参考模态样本数')
    fig.tight_layout()
    save(fig, '图4_验证集模态作用差异')

    # ---- 图5: 忠实性删除实验 ----
    faith = pd.read_csv(R / '解释忠实性删除实验.csv')
    fig, ax = plt.subplots(figsize=(7.7, 4.2))
    for method, label, color in [('integrated_gradients', '积分梯度关键位置', COL[0]),
                                  ('attention', '注意力权重位置', COL[1]),
                                  ('hashed_uniform', '均匀伪随机位置 5组', COL[2])]:
        z = faith[faith.method == method].groupby('rate').probability_drop_mean.mean()
        ax.plot(z.index, z.values, 'o-', label=label, color=color)
    ax.set(xlabel='替换为训练均值的位置比例', ylabel='原预测类别概率平均下降',
           title='验证集 728条  固定掩码的特征级删除实验')
    ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, '图5_解释忠实性验证')

    # ---- 图6: 附件4全量预测 ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.3))
    y = np.arange(20)
    left = np.zeros(20)
    for m, k in enumerate(KEYS):
        axs[0].barh(y, p[k + '_effect'], left=left, color=COL[m], label=NAMES[m])
        left += p[k + '_effect']
    axs[0].set_yticks(y, p.sample_id); axs[0].invert_yaxis()
    axs[0].set_xlabel('模态绝对贡献占比')
    axs[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    colors = [COL[1] if z < 0 else COL[0] for z in p.intensity]
    axs[1].barh(y, p.intensity, color=colors)
    axs[1].set_yticks(y, [f'{sid} {CLASS[c]}' for sid, c in zip(p.sample_id, p.polarity_id)])
    axs[1].invert_yaxis(); axs[1].axvline(0, color='#888', lw=.8)
    axs[1].set_xlim(-3.2, 1.5); axs[1].set_xlabel('预测情感强度  无测试标签')
    fig.tight_layout()
    save(fig, '图6_附件4全量预测与解释')

    # ---- 图7: 错误分组 ----
    v['group'] = pd.cut(np.abs(v.true_intensity), [-.001, 0, .5, 1.5, 3],
                        labels=['中性 0', '弱情感 ≤0.5', '中等 ≤1.5', '强情感 >1.5'])
    v['correct'] = (v.true_polarity == v.predicted_polarity)
    groups = v.groupby('group', observed=False).agg(n=('correct', 'size'),
                                                     accuracy=('correct', 'mean'),
                                                     mae=('absolute_error', 'mean'))
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.5))
    axs[0].bar(groups.index.astype(str), groups.accuracy, color=COL[0])
    axs[0].set_ylim(0, 1); axs[0].set_ylabel('Accuracy')
    axs[1].bar(groups.index.astype(str), groups.mae, color=COL[1])
    axs[1].set_ylabel('MAE')
    for ax in axs:
        ax.tick_params(axis='x', rotation=12)
    for j, row in enumerate(groups.itertuples()):
        axs[0].text(j, row.accuracy + .025, f'n={row.n}', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, '图7_验证集错误分组')

    # ---- 样本解释卡 (选几个典型) ----
    # Find samples with different principal modalities
    principal = p.principal_modality.values
    text_samples = p[p.principal_modality == 'text'].sample_id.tolist()[:3]
    vision_samples = p[p.principal_modality == 'vision'].sample_id.tolist()[:1]
    audio_samples = p[p.principal_modality == 'audio'].sample_id.tolist()[:1]
    selected = text_samples + vision_samples + audio_samples
    selected = selected[:5]

    ig = ta['ig']  # (20, 50, 3, 2)
    for sid in selected:
        idx = list(p.sample_id).index(sid)
        row = p[p.sample_id == sid].iloc[0]
        m = KEYS.index(row.principal_modality)
        raw = a4_raw.get(sid, '')
        words = raw.split()
        n_words = min(len(words), 50)
        # IG for principal modality, class-prob (index 1)
        word_ig = ig[idx, :n_words, m, 1]
        fig = plt.figure(figsize=(10.5, 5.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4])
        a0 = fig.add_subplot(gs[0])
        a1 = fig.add_subplot(gs[1])
        ph = ta['shap'][idx, :, 1]
        a0.barh(NAMES, ph, color=COL)
        a0.axvline(0, color='#888', lw=.8)
        a0.set_xlabel('对当前预测类别概率的 Shapley 贡献')
        a0.set_title(f"样本 {sid}   {CLASS[row.polarity_id]}   强度 {row.intensity:.3f}\n"
                     f"主要参考 {NAMES[m]}   类别概率 {row.predicted_class_probability:.3f}",
                     loc='left', fontsize=11)
        a1.bar(range(n_words), word_ig, color=[COL[0] if s >= 0 else '#b75952' for s in word_ig])
        a1.axhline(0, color='#888', lw=.7)
        a1.set_xticks(range(n_words))
        a1.set_xticklabels(words[:n_words], rotation=55, ha='right', fontsize=8)
        a1.set_ylabel(f'{NAMES[m]} 词级积分梯度')
        a1.set_xlabel('原文词序  正值支持当前类别  负值反对当前类别')
        fig.tight_layout(rect=[0, .04, 1, 1])
        fig.text(.01, .008, '特征级归因；BERT含上下文。', fontsize=9, color='#555')
        save(fig, f'样本{sid}_解释卡')

    # ---- 诊断信息 ----
    faith_20 = faith[faith.rate == .2].groupby('method').probability_drop_mean.mean().to_dict()
    diag = {
        'q2_model': 'matrix_no_aug (epoch 4, classification_projected)',
        'valid_metrics': {k: met[k] for k in ('accuracy', 'macro_f1', 'mae', 'pearson')},
        'q3_valid_metrics': q3_met,
        'principal_modality_counts': dict(zip(NAMES, counts)),
        'faithfulness_20pct': faith_20,
        'selected_cases': selected,
        'top_errors': v.sort_values('absolute_error', ascending=False).head(8)[
            ['sample_id', 'true_intensity', 'predicted_intensity', 'absolute_error',
             'true_polarity', 'predicted_polarity']].to_dict('records')
    }
    (R / 'report_diagnostics.json').write_text(
        json.dumps(diag, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print("Visualizations complete. Figures saved to", F)


if __name__ == '__main__':
    main()
