"""plotlibs.fast — vectorized speed helpers (decimation, limits, ticks)."""
from __future__ import annotations

import numpy as np

DECIMATE_THRESHOLD = 2000  # max points drawn per line; rest min-max decimated


def as_xy(x, y=None):
    """Normalize plot(x) / plot(x, y) inputs to (x, y) float arrays."""
    if y is None:
        y = np.asarray(x, dtype=float).ravel()
        x = np.arange(y.size, dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if not bool(np.all(mask)):
        x, y = x[mask], y[mask]
    return x, y


def decimate(x: np.ndarray, y: np.ndarray, max_points: int = DECIMATE_THRESHOLD):
    """Min-max bucket decimation preserving visual envelope. O(n), numpy only.

    Splits data into `max_points//2` buckets and keeps min+max of each bucket,
    so spikes don't disappear like naive every-Nth sampling.
    """
    n = x.size
    if n <= max_points or max_points < 4:
        return x, y
    nb = max_points // 2
    # bucket index per point
    idx = np.linspace(0, n, nb + 1).astype(np.int64)
    xs = np.empty(nb * 2, dtype=float)
    ys = np.empty(nb * 2, dtype=float)
    k = 0
    for b in range(nb):
        s, e = int(idx[b]), int(idx[b + 1])
        if e <= s:
            continue
        segx, segy = x[s:e], y[s:e]
        jmin = int(np.argmin(segy))
        jmax = int(np.argmax(segy))
        if jmin <= jmax:
            xs[k], ys[k] = segx[jmin], segy[jmin]
            xs[k + 1], ys[k + 1] = segx[jmax], segy[jmax]
        else:
            xs[k], ys[k] = segx[jmax], segy[jmax]
            xs[k + 1], ys[k + 1] = segx[jmin], segy[jmin]
        k += 2
    # sort by x to keep polyline order (cheap: buckets already ordered)
    return xs[:k], ys[:k]


def nice_limits(vmin: float, vmax: float, pad: float = 0.05):
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmin == vmax:
        d = abs(vmin) * 0.1 if vmin != 0 else 1.0
        return vmin - d, vmax + d
    span = vmax - vmin
    return vmin - span * pad, vmax + span * pad


def nice_ticks(vmin: float, vmax: float, nbins: int = 5):
    """Matplotlib-like MaxNLocator simplified, fully vectorized."""
    if vmin == vmax or not (np.isfinite(vmin) and np.isfinite(vmax)):
        return np.array([vmin])
    span = vmax - vmin
    raw = span / max(1, nbins)
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            step = m * mag
            break
    else:
        step = 10 * mag
    t0 = np.ceil(vmin / step) * step
    t1 = np.floor(vmax / step) * step
    if t1 < t0:
        return np.array([(vmin + vmax) / 2])
    n = int(round((t1 - t0) / step)) + 1
    n = min(n, 20)
    return np.linspace(t0, t0 + step * (n - 1), n)


def fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-3:
        return f"{v:.1e}"
    if a >= 100:
        return f"{v:.0f}" if v == int(v) else f"{v:.1f}"
    if a >= 1:
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        return s
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s


def gaussian_kde_1d(v: np.ndarray, points: int = 200, bw: float | None = None):
    """تقدير الكثافة Gaussian KDE بـ numpy فقط (Silverman bandwidth)."""
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        xs = np.array([0.0, 1.0])
        return xs, np.zeros(2)
    if v.min() == v.max():
        xs = np.linspace(v.min() - 1, v.max() + 1, points)
        ys = np.exp(-0.5 * ((xs - v.min())) ** 2)
        return xs, ys / ys.max()
    std = v.std() or 1.0
    n = v.size
    bw = bw or (1.06 * std * n ** (-1 / 5)) or std * 0.3
    xs = np.linspace(v.min(), v.max(), points)
    # vectorized: (points, n) قد تكون كبيرة -> chunked
    ys = np.zeros(points)
    chunk = 64
    for i in range(0, points, chunk):
        seg = xs[i:i + chunk, None] - v[None, :]
        ys[i:i + chunk] = np.exp(-0.5 * (seg / bw) ** 2).sum(axis=1)
    ys /= (n * bw * np.sqrt(2 * np.pi))
    return xs, ys


def box_stats(v: np.ndarray) -> dict:
    """إحصاءات boxplot: min/q1/med/q3/max + outliers (1.5*IQR)."""
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(q1=0, med=0, q3=0, lo=0, hi=0, out=np.array([]), mean=0)
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    iqr = q3 - q1 or 1.0
    lo = max(v.min(), q1 - 1.5 * iqr)
    hi = min(v.max(), q3 + 1.5 * iqr)
    out = v[(v < lo) | (v > hi)]
    return dict(q1=float(q1), med=float(med), q3=float(q3),
                lo=float(lo), hi=float(hi), out=out, mean=float(v.mean()))


def roc_from_scores(y_true, y_score, n_pts: int = 200):
    """منحنى ROC بـ numpy فقط. يعيد (fpr, tpr, auc)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n = min(y_true.size, y_score.size)
    y_true, y_score = y_true[:n] == 1, y_score[:n]
    order = np.argsort(-y_score, kind="stable")
    yt = y_true[order].astype(float)
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    P, N = tp[-1] if len(tp) else 0, fp[-1] if len(fp) else 0
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    tpr = np.concatenate([[0], tp / P, [1]])
    fpr = np.concatenate([[0], fp / N, [1]])
    # تخفيف النقاط
    if len(fpr) > n_pts:
        idx = np.linspace(0, len(fpr) - 1, n_pts).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, abs(auc)


def pr_from_scores(y_true, y_score, n_pts: int = 200):
    """منحنى Precision-Recall بـ numpy فقط. يعيد (recall, precision, ap)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n = min(y_true.size, y_score.size)
    y_true, y_score = (y_true[:n] == 1).astype(float), y_score[:n]
    order = np.argsort(-y_score, kind="stable")
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    P = yt.sum() or 1.0
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    prec = np.concatenate([[1], prec])
    rec = np.concatenate([[0], rec])
    if len(rec) > n_pts:
        idx = np.linspace(0, len(rec) - 1, n_pts).astype(int)
        rec, prec = rec[idx], prec[idx]
    ap = float(np.trapezoid(prec, rec))
    return rec, prec, abs(ap)
