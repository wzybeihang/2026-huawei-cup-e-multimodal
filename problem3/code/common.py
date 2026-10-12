"""Data loading and standardization for applying Q2 model to Q3 data.
Uses official aligned_50 text field (verified equal to BERT output of text_bert).
"""
from pathlib import Path
import json, hashlib, pickle
import numpy as np
import torch

KEYS = ('text', 'audio', 'vision')
DIMS = (768, 74, 35)
ROOT = Path(__file__).resolve().parents[1]


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def content_mask(tb):
    """Compute domain and text-observed mask from text_bert (same as Q2 protocol)."""
    ids = np.asarray(tb)[:, 0].astype(np.int64)
    att = np.asarray(tb)[:, 1] > 0
    domain = np.zeros_like(att)
    for i in range(len(ids)):
        sep = np.where((ids[i] == 102) & att[i])[0]
        if not len(sep):
            raise ValueError(f'No visible SEP for sample {i}')
        domain[i, 1:int(sep[-1])] = True
    text_obs = domain & att & (ids != 0) & (ids != 103)
    return domain, text_obs


def fields(d):
    """Extract standardized-ready fields from a data dict (single split or stacked)."""
    tb = np.asarray(d['text_bert'], np.int64).reshape(-1, 3, 50)
    ids = np.asarray(d['id'], str).reshape(-1)
    n = len(ids)
    domain, text_obs = content_mask(tb)
    xs = [np.asarray(d[k], np.float32).reshape(n, 50, dim) for k, dim in zip(KEYS, DIMS)]
    obs_list = [text_obs]
    for m in range(1, 3):
        obs_list.append(domain & np.isfinite(xs[m]).all(-1) & np.any(xs[m] != 0, -1))
    obs = np.stack(obs_list, -1)
    return dict(xs=xs, mask=domain, obs=obs, ids=ids, tb=tb,
                raw=np.asarray(d['raw_text'], str).reshape(-1))


def standardize(d, s):
    """Apply Q2 scaler (text_mean, text_scale, audio_mean, ...)."""
    d['xs'] = [np.where(d['obs'][:, :, m, None],
                        np.clip((x - s[k + '_mean']) / s[k + '_scale'], -8, 8), 0).astype(np.float32)
               for m, (k, x) in enumerate(zip(KEYS, d['xs']))]
    return d


def tensors(d, device):
    return [torch.as_tensor(x, device=device) for x in d['xs']], \
           torch.as_tensor(d['obs'], device=device), \
           torch.as_tensor(d['mask'], device=device)


def metrics(y, c, p, pc):
    """y=true intensity, c=true class, p=pred intensity, pc=pred class."""
    pred = pc
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (c, pred), 1)
    support = cm.sum(1)
    f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + support, 1)
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else 0.
    return dict(n=len(y), accuracy=float((pred == c).mean()), macro_f1=float(f1.mean()),
                weighted_f1=float((f1 * support).sum() / len(y)), mae=float(abs(y - p).mean()),
                pearson=corr, per_class_f1=f1.tolist(), support=support.tolist(),
                confusion_matrix=cm.tolist())
