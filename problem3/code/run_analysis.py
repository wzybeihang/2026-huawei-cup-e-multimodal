"""Apply Problem 2 matrix-completion model to Problem 3 data.
Runs prediction, exact 3-player Shapley, Gauss-Legendre integrated gradients,
and faithfulness deletion experiment on 附件2 valid and 附件4.
"""
import argparse
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import KEYS, ROOT, read, fields, standardize, tensors, metrics, dump, sha
from models import LocalMissingModel


def model_output(model, features, observed, domain):
    """Return (logits, raw_intensity, class_probabilities, attention_weights)."""
    out = model(features, observed, domain)
    prob = torch.softmax(out['logits'], -1)
    return out['logits'], out['raw'], prob, out['attention']


@torch.no_grad()
def predict_all(model, data, device, batch_size=64):
    """Run clean prediction on a dataset dict. Returns logits, raw, attention."""
    model.eval()
    all_logits, all_raw, all_att = [], [], []
    xs, obs, dom = tensors(data, device)
    for at in range(0, len(obs), batch_size):
        sl = slice(at, at + batch_size)
        feat = {k: xs[m][sl] for m, k in enumerate(KEYS)}
        logits, raw, prob, att = model_output(model, feat, obs[sl], dom[sl])
        all_logits.append(logits.cpu().numpy())
        all_raw.append(raw.cpu().numpy())
        all_att.append(att.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_raw), np.concatenate(all_att)


def decode(logits, raw):
    """Q2 selected rule: classification_projected."""
    c = logits.argmax(-1)
    mag = np.maximum(np.abs(raw), 1e-6)
    r = np.where(c == 0, -mag, np.where(c == 2, mag, 0))
    return c.astype(np.int64), r.astype(np.float32)


@torch.no_grad()
def shapley(model, data, device, batch_size=16):
    """Exact 3-player Shapley over 8 coalitions.
    Attribution targets: intensity (raw) and predicted-class probability.
    Baseline = zero features (standardized zero = training mean).
    """
    model.eval()
    xs, obs, dom = tensors(data, device)
    chunks = {k: [] for k in ('reg', 'prob', 'att', 'target', 'coalitions', 'shap')}
    for at in range(0, len(obs), batch_size):
        sl = slice(at, at + batch_size)
        feat = {k: xs[m][sl] for m, k in enumerate(KEYS)}
        _, r, p, att = model_output(model, feat, obs[sl], dom[sl])
        target = p.argmax(-1)
        values = []
        for bits in range(8):
            f_ = {k: (xs[m][sl] if bits & (1 << m) else torch.zeros_like(xs[m][sl]))
                  for m, k in enumerate(KEYS)}
            _, rr, pp, _ = model_output(model, f_, obs[sl], dom[sl])
            values.append(torch.stack([rr, pp.gather(1, target[:, None]).squeeze(1)], -1))
        values = torch.stack(values, 1)  # (B, 8, 2)
        phi = torch.zeros((len(obs[sl]), 3, 2), device=device)
        for m in range(3):
            for s in range(8):
                if s & (1 << m):
                    continue
                count = s.bit_count()
                w = math.factorial(count) * math.factorial(2 - count) / 6
                phi[:, m] += w * (values[:, s | (1 << m)] - values[:, s])
        chunks['reg'].append(r.cpu().numpy())
        chunks['prob'].append(p.cpu().numpy())
        chunks['att'].append(att.cpu().numpy())
        chunks['target'].append(target.cpu().numpy())
        chunks['coalitions'].append(values.cpu().numpy())
        chunks['shap'].append(phi.cpu().numpy())
    return {k: np.concatenate(v) for k, v in chunks.items()}


def integrated_gradients(model, data, target, device, steps=64, batch_size=8):
    """Gauss-Legendre IG for intensity and target-class probability.
    Returns (B, 50, 3, 2): last dim = [intensity, class_prob].
    """
    model.eval()
    xs, obs, dom = tensors(data, device)
    n = len(obs)
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    nodes = (nodes + 1) / 2
    weights = weights / 2
    acc = [[torch.zeros_like(x) for x in xs] for _ in range(2)]
    for at in range(0, n, batch_size):
        sl = slice(at, min(at + batch_size, n))
        xb = [x[sl] for x in xs]
        ob = obs[sl]
        db = dom[sl]
        tb = target[sl]
        bs = len(ob)
        for node_start in range(0, steps, 8):
            alpha = torch.tensor(nodes[node_start:node_start + 8], device=device, dtype=torch.float32)
            weight = torch.tensor(weights[node_start:node_start + 8], device=device, dtype=torch.float32)
            na = len(alpha)
            xx = [(alpha[:, None, None, None] * x[None]).flatten(0, 1).requires_grad_(True)
                  for x in xb]
            oo = ob.repeat(na, 1, 1)
            dd = db.repeat(na, 1)
            tt = tb.repeat(na)
            feat = {k: xx[m] for m, k in enumerate(KEYS)}
            _, r, p, _ = model_output(model, feat, oo, dd)
            q = p.gather(1, tt[:, None]).squeeze(1)
            for j, z in enumerate([r, q]):
                grad = torch.autograd.grad(z.sum(), xx, retain_graph=j == 0)
                for m, g in enumerate(grad):
                    g_reshaped = g.reshape(na, bs, 50, -1)
                    acc[j][m][sl] += torch.einsum('a,abtd->btd', weight, g_reshaped)
    ig = torch.stack([torch.stack([(g * x).sum(-1) for g, x in zip(gg, xs)], -1)
                      for gg in acc], -1)
    return ig.detach().cpu().numpy()


@torch.no_grad()
def faithfulness(model, data, attr, device, batch_size=64):
    """Deletion experiment: replace top-r% positions with zero (training mean),
    measure predicted-class probability drop. Compare IG vs attention vs random."""
    model.eval()
    xs, obs, dom = tensors(data, device)
    target = torch.tensor(attr['target'], device=device)
    base = torch.tensor(attr['prob'].max(-1), device=device)
    rows = []
    sample_rows = []
    ig_scores = attr['ig'][:, :, :, 1]  # (B, 50, 3) class-prob IG
    att_scores = attr['att']
    if att_scores.ndim == 2:  # Q2 model: position-only attention (B, 50)
        att_scores = np.repeat(att_scores[:, :, None], 3, axis=2)  # -> (B, 50, 3)
    for rate in [.1, .2, .3, .5]:
        for mode in ['integrated_gradients', 'attention', 'hashed_uniform']:
            repeat = 5 if mode == 'hashed_uniform' else 1
            for rep in range(repeat):
                keep = data['obs'].copy()
                for i, sid in enumerate(data['ids']):
                    valid = np.flatnonzero(keep[i].ravel())
                    n = max(1, int(np.ceil(len(valid) * rate)))
                    if mode == 'integrated_gradients':
                        score = ig_scores[i].ravel()[valid]
                    elif mode == 'attention':
                        score = att_scores[i].ravel()[valid]
                    else:
                        score = np.array([int.from_bytes(
                            hashlib.sha256(f'{sid}|{rep}|{int(k)}'.encode()).digest()[:8],
                            'little') / 2 ** 64 for k in valid])
                    chosen = valid[np.argsort(-score, kind='stable')[:n]]
                    keep[i].ravel()[chosen] = False
                km = torch.tensor(keep, device=device)
                drops = []
                for at in range(0, len(obs), batch_size):
                    sl = slice(at, at + batch_size)
                    feat = {k: xs[m][sl] * km[sl, :, m, None] for m, k in enumerate(KEYS)}
                    _, _, pp, _ = model_output(model, feat, obs[sl], dom[sl])
                    drops.append((base[sl] - pp.gather(1, target[sl][:, None]).squeeze(1)).cpu().numpy())
                drop = np.concatenate(drops)
                rows.append(dict(rate=rate, method=mode, repeat=rep,
                                 probability_drop_mean=float(drop.mean()),
                                 probability_drop_std=float(drop.std()),
                                 positive_drop_fraction=float((drop > 0).mean())))
                if rate == .2:
                    sample_rows.extend(dict(id=str(sid), method=mode, repeat=rep, drop=float(z))
                                       for sid, z in zip(data['ids'], drop))
    return rows, sample_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'<project_root>\附件2-数据集特征文件\aligned_50.pkl')
    ap.add_argument('--attachment4',
                    default=r'<project_root>\附件4-可解释专项视频样本与特征文件\附件4-可解释专项视频样本与特征文件\对齐版本')
    ap.add_argument('--q2_result', required=True, help='path to the authorized Question 2 experiment output directory')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.set_num_threads(4)
    R = ROOT / '实验结果'
    R.mkdir(exist_ok=True)

    # ---- Load Q2 model ----
    q2r = Path(args.q2_result)
    ckpt = torch.load(q2r / 'matrix_no_aug' / 'best.pt', map_location='cpu', weights_only=False)
    model = LocalMissingModel(ckpt['spec']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    scaler = dict(np.load(q2r / 'scaler.npz'))
    print(f"Loaded Q2 matrix_no_aug model (epoch {ckpt['epoch']})")

    # ---- Load 附件2 valid ----
    all_data = read(args.data)
    valid_raw = fields(all_data['valid'])
    valid = standardize(valid_raw, scaler)
    valid['y'] = np.asarray(all_data['valid']['regression_labels']).reshape(-1)
    valid['c'] = np.asarray(all_data['valid']['classification_labels'], int).reshape(-1)
    print(f"Valid samples: {len(valid['ids'])}")

    # ---- Load 附件4 ----
    paths = sorted(Path(args.attachment4).glob('*.pkl'))
    items = [read(p) for p in paths]
    test_raw = fields({k: np.stack([x[k] for x in items]) for k in items[0]})
    test = standardize(test_raw, scaler)
    print(f"附件4 samples: {len(test['ids'])}")
    assert len(test['ids']) == len(set(test['ids'])) == 20

    # ---- Prediction on valid ----
    logits_v, raw_v, att_v = predict_all(model, valid, args.device)
    pred_c_v, pred_r_v = decode(logits_v, raw_v)
    m = metrics(valid['y'], valid['c'].astype(np.int64), pred_r_v, pred_c_v)
    dump(R / 'valid_metrics.json', m)
    pd.DataFrame(dict(sample_id=valid['ids'], raw_text=valid['raw'],
                      true_intensity=valid['y'], true_polarity=valid['c'],
                      predicted_intensity=pred_r_v, predicted_polarity=pred_c_v,
                      absolute_error=np.abs(valid['y'] - pred_r_v),
                      logits_negative=logits_v[:, 0], logits_neutral=logits_v[:, 1],
                      logits_positive=logits_v[:, 2])).to_csv(
        R / '验证集逐样本预测.csv', index=False, encoding='utf-8-sig')
    print(f"Valid: Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} MAE={m['mae']:.4f} Pearson={m['pearson']:.4f}")

    # ---- Shapley on valid ----
    va = shapley(model, valid, args.device)
    # IG with residual check
    va['ig'] = integrated_gradients(model, valid, torch.tensor(va['target'], device=args.device), args.device)
    delta = va['coalitions'][:, 7] - va['coalitions'][:, 0]
    ig_residual = np.abs(va['ig'].sum((1, 2)) - delta).max()
    if ig_residual > .002:
        print(f"IG residual {ig_residual:.6f} > 0.002, recomputing with 128 points")
        va['ig'] = integrated_gradients(model, valid, torch.tensor(va['target'], device=args.device),
                                        args.device, steps=128)
    np.savez_compressed(R / 'valid_attributions.npz', **va)
    print(f"Valid Shapley residual: {np.abs(va['shap'].sum(1) - delta).max():.2e}")

    # ---- Faithfulness on valid ----
    faith_rows, faith_sample = faithfulness(model, valid, va, args.device)
    pd.DataFrame(faith_rows).to_csv(R / '解释忠实性删除实验.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(faith_sample).to_csv(R / '解释忠实性逐样本.csv', index=False, encoding='utf-8-sig')
    faith_20 = {r['method']: r['probability_drop_mean'] for r in faith_rows
                if r['rate'] == .2 and r['repeat'] == 0}
    print(f"Faithfulness@20%: IG={faith_20.get('integrated_gradients', 0):.4f} "
          f"Attn={faith_20.get('attention', 0):.4f} Rand={faith_20.get('hashed_uniform', 0):.4f}")

    # ---- Prediction + Shapley + IG on 附件4 ----
    logits_t, raw_t, att_t = predict_all(model, test, args.device)
    pred_c_t, pred_r_t = decode(logits_t, raw_t)
    ta = shapley(model, test, args.device)
    ta['ig'] = integrated_gradients(model, test, torch.tensor(ta['target'], device=args.device), args.device)
    delta_t = ta['coalitions'][:, 7] - ta['coalitions'][:, 0]
    ig_res_t = np.abs(ta['ig'].sum((1, 2)) - delta_t).max()
    if ig_res_t > .002:
        ta['ig'] = integrated_gradients(model, test, torch.tensor(ta['target'], device=args.device),
                                        args.device, steps=128)
    np.savez_compressed(R / 'attachment4_attributions.npz', **ta)

    # ---- Build submission CSV for 附件4 ----
    records = []
    for i, sid in enumerate(test['ids']):
        phi = ta['shap'][i]
        w = np.abs(phi[:, 1])
        w = w / w.sum() if w.sum() > 1e-12 else np.zeros(3)
        row = dict(sample_id=str(sid), polarity_id=int(ta['target'][i]),
                   polarity=['Negative', 'Neutral', 'Positive'][ta['target'][i]],
                   intensity=float(ta['reg'][i]),
                   principal_modality=KEYS[int(w.argmax())] if w.sum() else 'none',
                   predicted_class_probability=float(ta['prob'][i].max()),
                   probability_baseline=float(ta['coalitions'][i, 0, 1]),
                   intensity_baseline=float(ta['coalitions'][i, 0, 0]))
        for mm, k in enumerate(KEYS):
            row.update({k + '_effect': float(w[mm]),
                        k + '_shap_probability': float(phi[mm, 1]),
                        k + '_shap_intensity': float(phi[mm, 0])})
        records.append(row)
    submit = ROOT / '提交结果'
    submit.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(submit / '问题3_附件4预测与解释结果_Q2模型.csv', index=False, encoding='utf-8-sig')

    # ---- Audit ----
    audits = {}
    for key, attr in [('valid', va), ('attachment4', ta)]:
        delta = attr['coalitions'][:, 7] - attr['coalitions'][:, 0]
        audits[key] = {
            'n': len(delta),
            'shapley_max_absolute_residual': float(np.abs(attr['shap'].sum(1) - delta).max()),
            'ig_max_absolute_residual': float(np.abs(attr['ig'].sum((1, 2)) - delta).max()),
            'principal_modality_counts': np.bincount(np.abs(attr['shap'][:, :, 1]).argmax(1), minlength=3).tolist()
        }
    audits['test_files'] = [dict(filename=p.name, sha256=sha(p)) for p in paths]
    audits['model_source'] = 'Q2 matrix_no_aug (epoch 4), classification_projected rule'
    audits['definition'] = ('fixed valid masks; standardized zero training-mean baseline; '
                            'class attribution targets selected class probability; '
                            'regression attribution targets intensity; '
                            'IG feature interventions keep BERT context in precomputed features')
    dump(R / 'attribution_audit.json', audits)
    print("Audit:", json.dumps(audits, ensure_ascii=False)[:300])

    print("Analysis complete. Results saved to", R)


import json

if __name__ == '__main__':
    main()
