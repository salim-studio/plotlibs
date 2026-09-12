"""plotlibs.data — جسر البيانات: pandas / numpy / csv بدون اعتماديات إجبارية.

الهدف: محلل البيانات يكتب سطرين فقط:
    import plotlibs as pl
    df = pl.load_csv("sales.csv")
    pl.quick_eda(df)

كل الدوال تعمل مع:
- pandas.DataFrame (إن وُجدت pandas)
- dict of columns
- numpy 2D array
- list of dicts (rows)
"""
from __future__ import annotations

import csv
import math

import numpy as np


def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except Exception:
        return False


def to_frame(data):
    """حوّل أي مصدر شائع إلى pandas.DataFrame إن أمكن، وإلا dict."""
    if data is None:
        raise ValueError("data is None")
    if _has_pandas():
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.to_frame()
        if isinstance(data, dict):
            return pd.DataFrame(data)
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], dict):
            return pd.DataFrame(data)
        arr = np.asarray(data)
        if arr.ndim == 2:
            return pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])
        return pd.DataFrame({"value": np.asarray(data).ravel()})
    # بدون pandas: أعد dict من الأعمدة
    if isinstance(data, dict):
        return {k: np.asarray(v) for k, v in data.items()}
    arr = np.asarray(data)
    if arr.ndim == 2:
        return {f"c{i}": arr[:, i] for i in range(arr.shape[1])}
    return {"value": np.asarray(arr).ravel()}


def columns_of(data) -> list[str]:
    if _has_pandas():
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return list(data.columns.astype(str))
    if isinstance(data, dict):
        return list(data.keys())
    return []


def column_values(data, col):
    """أعد عموداً كـ numpy array نظيف."""
    if _has_pandas():
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
    if isinstance(data, dict):
        return np.asarray(data[col], dtype=float).ravel()
    raise KeyError(f"unknown column {col!r}")


def numeric_columns(data) -> list[str]:
    if isinstance(data, dict):
        out = []
        for k, v in data.items():
            try:
                arr = np.asarray(v)
                if arr.dtype.kind in "iufb":
                    out.append(k)
                    continue
                # حاول تحويل عينة
                np.asarray(v, dtype=float)
                out.append(k)
            except Exception:
                pass
        return out
    if _has_pandas():
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return [str(c) for c in data.select_dtypes(include=[np.number]).columns]
        return []
    return []


def load_csv(path, **kw):
    """قراءة CSV سريعة — pandas إن وُجدت وإلا csv stdlib."""
    if _has_pandas():
        import pandas as pd
        return pd.read_csv(path, **kw)
    with open(path, newline="", encoding=kw.get("encoding", "utf-8")) as f:
        reader = csv.DictReader(f)
        cols: dict[str, list] = {}
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
        # حاول التحويل لرقمي
        out: dict[str, np.ndarray] = {}
        for k, vals in cols.items():
            try:
                out[k] = np.array(vals, dtype=float)
            except Exception:
                out[k] = np.array(vals)
        return out


def save_csv(data, path, **kw):
    if _has_pandas():
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=kw.get("index", False))
            return path
    if isinstance(data, dict):
        keys = list(data.keys())
        n = max(len(np.asarray(data[k]).ravel()) for k in keys)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(n):
                row = []
                for k in keys:
                    col = np.asarray(data[k]).ravel()
                    row.append(col[i] if i < len(col) else "")
                w.writerow(row)
        return path
    raise ValueError("unsupported data type for save_csv")


def describe(data) -> dict:
    """إحصاءات وصفية سريعة (count/mean/std/min/25/50/75/max) لكل عمود رقمي."""
    result: dict[str, dict[str, float]] = {}
    for col in numeric_columns(data):
        try:
            v = column_values(data, col)
        except Exception:
            continue
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
        result[col] = {
            "count": float(v.size),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(q[0]),
            "25%": float(q[1]),
            "50%": float(q[2]),
            "75%": float(q[3]),
            "max": float(q[4]),
        }
    return result


def corr_matrix(data, cols=None) -> tuple[np.ndarray, list[str]]:
    """مصفوفة ارتباط Pearson بـ numpy فقط."""
    cols = cols or numeric_columns(data)
    if len(cols) < 2:
        raise ValueError("need >= 2 numeric columns for correlation")
    mat = np.column_stack([column_values(data, c) for c in cols])
    # احذف الصفوف التي فيها NaN
    mask = np.all(np.isfinite(mat), axis=1)
    mat = mat[mask]
    if mat.shape[0] < 2:
        return np.eye(len(cols)), cols
    c = np.corrcoef(mat, rowvar=False)
    c = np.nan_to_num(c, nan=0.0)
    return c, cols


def sample_rows(data, n: int = 5):
    if _has_pandas():
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.head(n)
    if isinstance(data, dict):
        keys = list(data.keys())
        return {k: np.asarray(data[k]).ravel()[:n] for k in keys}
    return data


def train_test_split_simple(*arrays, test_size=0.2, seed=0):
    """تقسيم سريع بدون sklearn (للمبتدئين)."""
    rng = np.random.default_rng(seed)
    n = len(np.asarray(arrays[0]))
    idx = np.arange(n)
    rng.shuffle(idx)
    k = int(math.ceil(n * test_size))
    te, tr = idx[:k], idx[k:]
    out = []
    for a in arrays:
        a = np.asarray(a)
        out += [a[tr], a[te]]
    return out
