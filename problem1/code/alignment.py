"""Auditable temporal alignment and uncertainty-aware feature filtering.

All time intervals are half-open [start, end). Times may be seconds, but one
consistent unit must be used within a call. This module depends only on NumPy.
No fallback invents token times when forced alignment is infeasible.
"""

from __future__ import annotations

import numpy as np


def ctc_forced_align(log_probs, targets, blank_id=0):
    """Viterbi-align a prescribed token sequence to CTC frame probabilities.

    Args:
        log_probs: Log probabilities, shape (frames, vocabulary). Negative
            infinity is allowed for impossible emissions; NaN/+inf are not.
        targets: One-dimensional integer token IDs, excluding ``blank_id``.
            These must use the acoustic model's tokenizer, not BERT tokens.
        blank_id: CTC blank ID in the acoustic model vocabulary.

    Returns:
        A list of dictionaries in target order. Each contains ``token_index``,
        ``token_id``, ``start_frame`` (inclusive), ``end_frame`` (exclusive),
        and ``posterior`` (mean exp(log probability) over assigned frames).
        Posterior is an emission-quality proxy, not calibrated alignment
        correctness. Blank frames are excluded from token spans. Map frame
        boundaries to seconds using the acoustic model's documented stride
        and receptive-field convention, and record that convention.

    Raises:
        ValueError: Invalid arrays or no finite legal CTC path. Repeated
            adjacent target IDs require at least one intervening blank frame.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[1] == 0:
        raise ValueError("log_probs must have shape (T, V) with V > 0")
    if np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError("log_probs may contain -inf, but not NaN or +inf")
    if (lp > 1e-6).any():
        raise ValueError("log_probs must be log probabilities (values <= 0)")
    if not isinstance(blank_id, (int, np.integer)) or not 0 <= blank_id < lp.shape[1]:
        raise ValueError("blank_id is outside the vocabulary")
    raw_targets = np.asarray(targets)
    if raw_targets.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if raw_targets.size and not np.issubdtype(raw_targets.dtype, np.integer):
        raise ValueError("targets must contain integer IDs")
    tokens = raw_targets.astype(np.int64)
    if np.any((tokens < 0) | (tokens >= lp.shape[1]) | (tokens == blank_id)):
        raise ValueError("targets must be nonblank IDs inside the vocabulary")
    t_count, _ = lp.shape
    l_count = len(tokens)
    if not l_count:
        if t_count and not np.isfinite(lp[:, blank_id]).all():
            raise ValueError("No finite CTC path for the empty target sequence")
        return []
    min_frames = l_count + int(np.sum(tokens[1:] == tokens[:-1]))
    if t_count < min_frames:
        raise ValueError(f"Infeasible CTC alignment: {t_count} frames < {min_frames} required")

    states = np.full(2 * l_count + 1, blank_id, dtype=np.int64)
    states[1::2] = tokens
    s_count = len(states)
    previous = np.full(s_count, -np.inf)
    previous[0] = lp[0, blank_id]
    previous[1] = lp[0, tokens[0]]
    back = np.full((t_count, s_count), -1, dtype=np.int32)
    can_skip = np.zeros(s_count, dtype=bool)
    can_skip[2:] = (states[2:] != blank_id) & (states[2:] != states[:-2])
    indices = np.arange(s_count)
    for t in range(1, t_count):
        # Ties prefer staying, then a one-state move, then a two-state move.
        candidates = np.full((3, s_count), -np.inf)
        candidates[0] = previous
        candidates[1, 1:] = previous[:-1]
        candidates[2, 2:] = np.where(can_skip[2:], previous[:-2], -np.inf)
        move = np.argmax(candidates, axis=0)
        best = candidates[move, indices]
        current = best + lp[t, states]
        back[t] = indices - move
        back[t, ~np.isfinite(current)] = -1
        previous = current
    state = s_count - 2 + int(previous[-1] > previous[-2])
    if not np.isfinite(previous[state]):
        raise ValueError("No finite legal CTC path for the supplied transcript")
    path = np.empty(t_count, dtype=np.int32)
    for t in range(t_count - 1, -1, -1):
        path[t] = state
        if t:
            state = int(back[t, state])
            if state < 0:
                raise ValueError("CTC traceback reached an unreachable state")
    aligned = []
    for j, token in enumerate(tokens):
        frames = np.flatnonzero(path == 2 * j + 1)
        if not frames.size:
            raise ValueError(f"CTC traceback omitted target position {j}")
        aligned.append({
            "token_index": j,
            "token_id": int(token),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1] + 1),
            "posterior": float(np.exp(lp[frames, token]).mean()),
        })
    return aligned


def _intervals(values, name, strictly_positive=False):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite array of shape (N, 2)")
    duration = arr[:, 1] - arr[:, 0]
    if np.any(duration <= 0 if strictly_positive else duration < 0):
        raise ValueError(f"{name} contains an invalid interval duration")
    return arr


def _union_duration(intervals):
    if len(intervals) == 0:
        return 0.0
    ordered = intervals[np.argsort(intervals[:, 0], kind="stable")]
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return float(total + end - start)


def overlap_pool(native_features, native_intervals, target_intervals, quality=None, valid=None):
    """Pool native observations onto word or fixed-grid target intervals.

    Args:
        native_features: Shape (N, D); each row has a traceable source record.
        native_intervals: Shape (N, 2); zero-duration native rows contribute
            nothing. Overlapping analysis windows are permitted.
        target_intervals: Shape (K, 2), with strictly positive duration.
        quality: Optional (N,) finite confidence proxy in [0, 1]. Defaults to
            one. A zero-quality row contributes neither weight nor coverage.
        valid: Optional (N,) Boolean observation mask. Invalid feature values
            may be NaN; valid, positive-quality rows must have finite values.

    Returns:
        Dictionary with ``features`` (K,D), ``mask`` (K,), ``coverage`` (K,),
        and CSR arrays ``source_indptr`` (K+1,), ``source_indices``,
        ``source_weights``, ``source_overlap``. CSR indices refer to original
        native row positions. Normalized weights are overlap_duration times
        quality; summing mapped source rows times these weights reconstructs
        every nonmissing pooled row. ``source_overlap`` stores raw intersection
        durations. Coverage uses the clipped interval UNION, divided by target
        duration, so overlapping windows never produce coverage above one.
        Missing output rows are zero-filled with mask False and coverage zero.

    Boundary rule: [a,b) and [b,c) have zero overlap. No interpolation or
    nearest-neighbor filling silently creates observations in temporal gaps.
    """
    features = np.asarray(native_features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("native_features must have shape (N, D)")
    intervals = _intervals(native_intervals, "native_intervals")
    targets = _intervals(target_intervals, "target_intervals", strictly_positive=True)
    n, d = features.shape
    if len(intervals) != n:
        raise ValueError("native_features and native_intervals row counts differ")
    q = np.ones(n) if quality is None else np.asarray(quality, dtype=np.float64)
    if q.shape != (n,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("quality must have shape (N,) and values in [0, 1]")
    v = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if v.shape != (n,):
        raise ValueError("valid must have shape (N,)")
    usable = v & (q > 0)
    if not np.isfinite(features[usable]).all():
        raise ValueError("Usable native feature rows must contain finite values")
    output = np.zeros((len(targets), d), dtype=np.float64)
    mask = np.zeros(len(targets), dtype=bool)
    coverage = np.zeros(len(targets), dtype=np.float64)
    indptr, source_ids, source_weights, source_overlap = [0], [], [], []
    for k, (start, end) in enumerate(targets):
        left = np.maximum(intervals[:, 0], start)
        right = np.minimum(intervals[:, 1], end)
        overlap = np.maximum(0.0, right - left)
        selected = np.flatnonzero(usable & (overlap > 0))
        if selected.size:
            weights = overlap[selected] * q[selected]
            weights /= weights.sum()
            output[k] = weights @ features[selected]
            mask[k] = True
            clipped = np.column_stack((left[selected], right[selected]))
            coverage[k] = min(1.0, _union_duration(clipped) / (end - start))
            source_ids.extend(selected.tolist())
            source_weights.extend(weights.tolist())
            source_overlap.extend(overlap[selected].tolist())
        indptr.append(len(source_ids))
    return {
        "features": output,
        "mask": mask,
        "coverage": coverage,
        "source_indptr": np.asarray(indptr, dtype=np.int64),
        "source_indices": np.asarray(source_ids, dtype=np.int64),
        "source_weights": np.asarray(source_weights, dtype=np.float64),
        "source_overlap": np.asarray(source_overlap, dtype=np.float64),
    }


def _dimension_parameter(value, dimensions, name, strictly_positive=False):
    arr = np.asarray(value, dtype=np.float64)
    try:
        arr = np.broadcast_to(arr, (dimensions,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or shape (D,)") from exc
    if not np.isfinite(arr).all() or np.any(arr <= 0 if strictly_positive else arr < 0):
        raise ValueError(f"{name} has invalid variance values")
    return arr


def diagonal_kalman_filter(observations, observed_mask=None, quality=None,
                           process_variance=1e-3, observation_variance=1.0,
                           initial_mean=0.0, initial_variance=1.0,
                           quality_floor=0.05, huber_delta=2.5, dt=1.0):
    """Quality-adaptive, Huber-robust diagonal random-walk Kalman filter.

    State model per feature: x[t] = x[t-1] + w[t], Var(w)=Q*dt[t].
    Observation model: y[t] = x[t] + v[t]. Start with
    R0[t,d] = observation_variance[d] / max(quality[t,d],quality_floor)^2.
    For standardized innovation z=(y-x_pred)/sqrt(P_pred+R0), set Huber
    weight h=min(1,huber_delta/abs(z)), then R=R0/h. This is one robust
    reweighting step, not an exact Bayesian posterior under a Gaussian model.

    Args:
        observations: Shape (T,D), preferably standardized using training-only
            statistics. Missing positions may contain NaN.
        observed_mask: Optional Boolean (T,) or (T,D); default all True.
            A row mask broadcasts across dimensions. It is never promoted by
            predictions. Observations where the mask is True must be finite.
        quality: Optional finite (T,) or (T,D) values in [0,1], default one.
            Zero quality inflates R using quality_floor; it does not itself
            alter observed_mask. Mark unavailable observations explicitly.
        process_variance, observation_variance, initial_variance: Scalar or
            (D,); Q and initial P nonnegative, base R strictly positive.
        initial_mean: Scalar or (D,) prior mean before the first prediction.
        quality_floor: Positive lower bound <=1 for numerical stability.
        huber_delta: Positive standardized-innovation threshold.
        dt: Nonnegative scalar or (T,) elapsed times for every prediction,
            including t=0. Set dt[0]=0 to avoid an initial process increment.

    Returns:
        Dictionary of (T,D) arrays ``mean``, ``variance``, ``predicted_mean``,
        ``predicted_variance``, ``gain``, ``innovation``, ``huber_weight``,
        ``effective_observation_variance``, and ``observed_mask``. The returned
        mask is the input mask broadcast to (T,D), unchanged. Missing points
        perform prediction only: mean=predicted_mean, variance=predicted_variance,
        gain=0, innovation=NaN, Huber weight=0, effective R=NaN. Thus predicted
        values cannot be mistaken for actual modality observations.
    """
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("observations must have shape (T, D)")
    t_count, dimensions = y.shape
    mask = np.ones_like(y, dtype=bool) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
    if mask.shape == (t_count,):
        mask = np.repeat(mask[:, None], dimensions, axis=1)
    if mask.shape != y.shape:
        raise ValueError("observed_mask must have shape (T,) or (T,D)")
    mask = mask.copy()
    if not np.isfinite(y[mask]).all():
        raise ValueError("Observed values must be finite; mark missing values False")
    q = np.ones_like(y) if quality is None else np.asarray(quality, dtype=np.float64)
    if q.shape == (t_count,):
        q = np.repeat(q[:, None], dimensions, axis=1)
    if q.shape != y.shape or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("quality must have shape (T,) or (T,D) and values in [0,1]")
    if not np.isfinite(quality_floor) or not 0 < quality_floor <= 1:
        raise ValueError("quality_floor must be in (0,1]")
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    q_process = _dimension_parameter(process_variance, dimensions, "process_variance")
    r_base = _dimension_parameter(observation_variance, dimensions, "observation_variance", True)
    p = _dimension_parameter(initial_variance, dimensions, "initial_variance")
    try:
        x = np.broadcast_to(np.asarray(initial_mean, dtype=np.float64), (dimensions,)).copy()
        steps = np.broadcast_to(np.asarray(dt, dtype=np.float64), (t_count,)).copy()
    except ValueError as exc:
        raise ValueError("initial_mean must be scalar/(D,), dt scalar/(T,)") from exc
    if not np.isfinite(x).all() or not np.isfinite(steps).all() or np.any(steps < 0):
        raise ValueError("initial_mean must be finite and dt finite/nonnegative")
    result = {key: np.zeros_like(y) for key in (
        "mean", "variance", "predicted_mean", "predicted_variance", "gain", "huber_weight")}
    result["innovation"] = np.full_like(y, np.nan)
    result["effective_observation_variance"] = np.full_like(y, np.nan)
    result["observed_mask"] = mask
    for t in range(t_count):
        p = p + q_process * steps[t]
        result["predicted_mean"][t] = x
        result["predicted_variance"][t] = p
        selected = mask[t]
        if selected.any():
            innovation = y[t, selected] - x[selected]
            r0 = r_base[selected] / np.maximum(q[t, selected], quality_floor) ** 2
            z = np.abs(innovation) / np.sqrt(p[selected] + r0)
            h = np.minimum(1.0, huber_delta / np.maximum(z, np.finfo(float).tiny))
            r = r0 / h
            gain = p[selected] / (p[selected] + r)
            x[selected] += gain * innovation
            # Joseph form retains nonnegative covariance at round-off limits.
            p[selected] = (1 - gain) ** 2 * p[selected] + gain ** 2 * r
            result["gain"][t, selected] = gain
            result["innovation"][t, selected] = innovation
            result["huber_weight"][t, selected] = h
            result["effective_observation_variance"][t, selected] = r
        result["mean"][t] = x
        result["variance"][t] = p
    return result
