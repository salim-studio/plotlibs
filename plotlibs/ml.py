"""plotlibs.ml — رسوم تعلم الآلة والتعلم العميق (تُشهر المكتبة عند ML/DL).

تعمل بـ numpy فقط — لا تحتاج sklearn/keras للرسم، لكن تقبل مخرجاتها.

    import plotlibs as pl
    pl.plot_history({"loss": [...], "val_loss": [...]})
    pl.plot_confusion_matrix(y_true, y_pred)
    pl.plot_roc(y_true, y_score)
    pl.plot_feature_importance(names, values)
    pl.plot_images(images)  # للتعلم العميق / رؤية حاسوبية
"""
from __future__ import annotations

import numpy as np


def _plt():
    from . import pyplot as plt
    return plt


def plot_history(history, metrics=("loss",), savefig=None, title="training history", smooth: int = 1):
    """ارسم history من Keras (dict) أو DataFrame أو dict of lists.

    history = {"loss": [...], "val_loss": [...], "accuracy": [...]}
    """
    plt = _plt()
    if hasattr(history, "history"):  # keras History object
        history = history.history
    if hasattr(history, "to_dict"):
        try:
            history = history.to_dict(orient="list")
        except Exception:
            pass
    keys = [k for k in history.keys() if not k.startswith("val_")] if isinstance(history, dict) else []
    if not keys and isinstance(history, dict):
        keys = list(history.keys())
    if metrics == ("loss",) and keys:
        # تلقائياً: كل المقاييس الموجودة
        metrics = tuple(keys)
    fig_axes = []
    for m in metrics:
        train = np.asarray(history[m], dtype=float).ravel() if m in history else None
        val = np.asarray(history.get(f"val_{m}", []), dtype=float).ravel() if isinstance(history, dict) else None
        if train is None:
            continue
        if smooth > 1 and train.size >= smooth:
            train = np.convolve(train, np.ones(smooth) / smooth, mode="same")
        plt.figure()
        plt.plot(train, label=f"train {m}")
        if val is not None and val.size:
            plt.plot(val, label=f"val {m}")
        plt.xlabel("epoch")
        plt.ylabel(m)
        plt.title(title if len(metrics) == 1 else f"{title} — {m}")
        plt.legend()
        plt.grid(True)
        fig_axes.append(plt.gcf())
        if savefig and len(metrics) == 1:
            plt.savefig(savefig)
    return fig_axes[0] if len(fig_axes) == 1 else fig_axes


def confusion_matrix_data(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = min(y_true.size, y_pred.size)
    y_true, y_pred = y_true[:n], y_pred[:n]
    if labels is None:
        labels = sorted(map(str, np.unique(np.concatenate([y_true, y_pred]))))
    lab_to_i = {str(l): i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(y_true, y_pred):
        i = lab_to_i.get(str(t))
        j = lab_to_i.get(str(p))
        if i is not None and j is not None:
            cm[i, j] += 1
    return cm, labels


def plot_confusion_matrix(y_true=None, y_pred=None, cm=None, labels=None,
                          normalize: bool = False, title="confusion matrix", **kw):
    """مصفوفة الالتباس — تقبل (y_true, y_pred) أو مصفوفة جاهزة cm."""
    plt = _plt()
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("pass (y_true, y_pred) or cm=")
        cm, labels = confusion_matrix_data(y_true, y_pred, labels)
    else:
        cm = np.asarray(cm, dtype=float)
        labels = labels or [str(i) for i in range(cm.shape[0])]
    disp = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    plt.gca().heatmap(disp, xticks=labels, yticks=labels, annot=True, cmap="viridis")
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.xlabel("predicted")
    plt.ylabel("true")
    return disp


def plot_roc(y_true, y_score, label="", **kw):
    """منحنى ROC + AUC (numpy فقط)."""
    from .fast import roc_from_scores
    plt = _plt()
    fpr, tpr, auc = roc_from_scores(y_true, y_score)
    plt.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}" if label else f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(True)
    return fpr, tpr, auc


def plot_pr(y_true, y_score, label="", **kw):
    """منحنى Precision-Recall + AP."""
    from .fast import pr_from_scores
    plt = _plt()
    rec, prec, ap = pr_from_scores(y_true, y_score)
    plt.plot(rec, prec, label=f"{label} AP={ap:.3f}" if label else f"AP={ap:.3f}")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("precision-recall curve")
    plt.legend()
    plt.grid(True)
    return rec, prec, ap


def plot_feature_importance(names, values, top: int = 20, title="feature importance", **kw):
    """أهمية الميزات (XGBoost/sklearn/RF) كأشرطة أفقية مرتبة."""
    plt = _plt()
    names = np.asarray(names).ravel()
    values = np.asarray(values, dtype=float).ravel()
    n = min(names.size, values.size)
    names, values = names[:n], values[:n]
    order = np.argsort(values, kind="stable")[-top:]
    plt.barh([str(names[i]) for i in order], values[order])
    plt.title(title)
    plt.grid(True)
    return order


def plot_residuals(y_true, y_pred, title="residuals", **kw):
    """بقايا الانحدار: predicted مقابل residual + هستوغرام."""
    plt = _plt()
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = min(y_true.size, y_pred.size)
    res = y_true[:n] - y_pred[:n]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].scatter(y_pred[:n], res)
    axs[0].axhline(0)
    axs[0].set_xlabel("predicted")
    axs[0].set_ylabel("residual")
    axs[0].set_title(title)
    axs[1].hist(res, bins=30)
    axs[1].set_title("residual dist")
    fig.suptitle(title)
    return res


def plot_elbow(k_list, inertia, title="elbow method", **kw):
    plt = _plt()
    plt.plot(np.asarray(k_list), np.asarray(inertia, dtype=float), marker="o")
    plt.xlabel("k")
    plt.ylabel("inertia")
    plt.title(title)
    plt.grid(True)


def plot_images(images, ncols: int = 8, max_images: int = 64, title="", cmap="gray", **kw):
    """شبكة صور للتعلم العميق (MNIST/CIFAR) — numpy فقط."""
    plt = _plt()
    arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr[:max_images]
    elif arr.ndim == 4:
        arr = arr[:max_images]
    else:
        raise ValueError("images must be (N,H,W) or (N,H,W,C)")
    n = arr.shape[0]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.4))
    axs = np.asarray(axs, dtype=object).ravel()
    for i in range(len(axs)):
        if i < n:
            axs[i].imshow(arr[i], cmap=cmap)
        axs[i].set_title("")
    if title:
        fig.suptitle(title)
    return fig


def plot_clusters_2d(X, labels, title="clusters", **kw):
    """عناقيد 2D ملونة (KMeans/DBSCAN) — سريعة حتى 50k نقطة."""
    plt = _plt()
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).ravel()
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        plt.scatter(X[m, 0][:10000], X[m, 1][:10000], label=f"c{lab}")
    plt.title(title)
    plt.legend()
    return plt.gca()
