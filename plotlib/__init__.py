"""Deprecated alias: ``import plotlib`` now maps to ``plotlibs``.

The library was renamed from ``plotlib`` to ``plotlibs`` (see
https://github.com/salim-studio/plotlibs). This shim keeps old code working.
Please migrate::

    # old
    import plotlib as pl
    # new
    import plotlibs as pl
"""
from __future__ import annotations

import warnings

warnings.warn(
    "The package was renamed 'plotlib' -> 'plotlibs'. "
    "Please use 'import plotlibs as pl'. The 'plotlib' alias will be removed in 1.0.",
    DeprecationWarning,
    stacklevel=2,
)

from plotlibs import *  # noqa: F401,F403
from plotlibs import (  # noqa: F401
    Figure, figure, subplots, plt, pyplot, style, rcParams,
    data, db, eda, ml,
    plot, scatter, bar, barh, hist, imshow, pie, fill_between,
    step, errorbar, axhline, axvline, boxplot, violinplot, kde, density,
    heatmap, corr, countplot, area, stackplot, hist2d, stem,
    xlabel, ylabel, title, legend, grid, xlim, ylim,
    savefig, show, close, gca, gcf,
    load_csv, save_csv, describe, db_connect, read_sql, to_sql,
    quick_eda, scatter_matrix, plot_missing,
    plot_history, plot_confusion_matrix, plot_roc, plot_pr,
    plot_feature_importance, figure_to_image,
)

__version__ = "0.3.0"
