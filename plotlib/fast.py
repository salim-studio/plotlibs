"""plotlib.fast — vectorized speed helpers (decimation, limits, ticks)."""
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
