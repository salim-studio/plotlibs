"""plotlibs.eda — تحليل استكشافي بسطر واحد للمحللين و Data Science.

    import plotlibs as pl
    pl.quick_eda(df)            # شكل 2x2: توزيع + box + ارتباط + قيم مفقودة
    pl.scatter_matrix(df)       # مصفوفة انتشار
    pl.plot_missing(df)         # خريطة القيم المفقودة
    pl.plot_trend(df, "date", "sales")  # خط + متوسط متحرك
"""
from __future__ import annotations

import numpy as np


def _get_pyplot():
    from . import pyplot as plt
    return plt


def quick_eda(data, max_cols: int = 6, savefig=None, title: str = "Quick EDA"):
    """تقرير بصري سريع 2x2. يعيد Figure."""
    from .data import numeric_columns, column_values, describe
    plt = _get_pyplot()
    nums = numeric_columns(data)[:max_cols]
    if not nums:
        raise ValueError("quick_eda needs at least 1 numeric column")
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title)
    # 1) توزيع أول عمود
    v0 = column_values(data, nums[0])
    v0 = v0[np.isfinite(v0)]
    axs[0, 0].hist(v0, bins=30)
    axs[0, 0].kde(v0, label="KDE")
    axs[0, 0].set_title(f"dist: {nums[0]}")
    # 2) boxplot كل الأعمدة
    cols = [column_values(data, c) for c in nums]
    axs[0, 1].boxplot(cols, labels=nums)
    axs[0, 1].set_title("boxplot")
    # 3) ارتباط
    try:
        axs[1, 0].corr(data, cols=nums, annot=len(nums) <= 8)
        axs[1, 0].set_title("correlation")
    except Exception as e:
        axs[1, 0].set_title(f"corr N/A: {e}")
    # 4) نص إحصاءات وصفية
    desc = describe(data)
    lines = []
    for c in nums[:4]:
        d = desc.get(c, {})
        if d:
            lines.append(f"{c}: μ={d['mean']:.2f} σ={d['std']:.2f} med={d['50%']:.2f}")
    axs[1, 1].plot([0, 1], [0, 0], label="baseline")
    axs[1, 1].set_title("summary")
    axs[1, 1].text(0.02, 0.9, "\n".join(lines) or "no stats", fontsize=9, color="black")
    # حفظ اختياري
    if savefig:
        fig.savefig(savefig)
    _ = desc
    return fig


def scatter_matrix(data, cols=None, figsize=(9, 9), **kw):
    """مصفوفة انتشار n×n (بديل pandas.plotting.scatter_matrix) — سريعة."""
    from .data import numeric_columns, column_values
    plt = _get_pyplot()
    cols = cols or numeric_columns(data)[:5]
    n = len(cols)
    if n < 2:
        raise ValueError("need >= 2 numeric columns")
    fig, axs = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axs = np.array([[axs]])
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axs[i, j]
            x = column_values(data, cj)
            y = column_values(data, ci)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if i == j:
                ax.hist(x, bins=20)
            else:
                ax.scatter(x[:5000], y[:5000])
            if j == 0:
                ax.set_ylabel(ci)
            if i == n - 1:
                ax.set_xlabel(cj)
    fig.suptitle("scatter matrix")
    return fig


def plot_missing(data, **kw):
    """شريط القيم المفقودة لكل عمود (بديل missingno)."""
    plt = _get_pyplot()
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            miss = data.isna().mean().sort_values(ascending=False)
            cols = list(miss.index.astype(str))
            vals = miss.to_numpy()
            plt.bar(cols, vals)
            plt.ylabel("missing ratio")
            plt.title("missing values")
            return vals
    except ImportError:
        pass
    # fallback: dict
    from .data import columns_of
    ratios = []
    labels = []
    for c in columns_of(data):
        v = np.asarray(data[c]).ravel()
        try:
            vf = v.astype(float)
            r = float(np.isnan(vf).mean())
        except Exception:
            r = 0.0
        labels.append(str(c))
        ratios.append(r)
    plt.bar(labels, np.array(ratios))
    plt.ylabel("missing ratio")
    plt.title("missing values")
    return np.array(ratios)


def plot_trend(data, x_col, y_col, window: int = 7, **kw):
    """خط زمني + متوسط متحرك (للمحللين الماليين ومحللي المبيعات)."""
    from .data import column_values
    plt = _get_pyplot()
    x = column_values(data, x_col) if x_col else None
    y = column_values(data, y_col)
    n = y.size
    if x is None or x.size != n:
        x = np.arange(n, dtype=float)
    plt.plot(x, y, label=y_col)
    if n >= window:
        ma = np.convolve(y, np.ones(window) / window, mode="same")
        plt.plot(x, ma, label=f"MA({window})")
    plt.xlabel(x_col or "t")
    plt.ylabel(y_col)
    plt.title(f"trend: {y_col}")
    plt.legend()
    return ma if n >= window else y


def value_counts_plot(values, top: int = 15, horizontal: bool = False, **kw):
    """رسم تكرار الفئات الأعلى (لمحللي البيانات الفئوية)."""
    plt = _get_pyplot()
    vals = np.asarray(values).ravel()
    uniq, counts = np.unique(vals, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:top]
    uniq, counts = uniq[order], counts[order]
    if horizontal:
        plt.barh(uniq, counts, **kw)
    else:
        plt.bar(uniq, counts, **kw)
    plt.title("value counts")
    return uniq, counts
