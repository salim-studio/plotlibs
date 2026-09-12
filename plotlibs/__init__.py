"""plotlibs — fast, beautiful Python plotting for everyone.

Drop-in matplotlib alternative (4x faster), plus one-liners for
DataFrames, SQL databases, EDA, machine learning and deep learning.

    import plotlibs as pl
    pl.plot([1, 2, 3], [1, 4, 9], label="x^2")
    pl.xlabel("x"); pl.ylabel("y"); pl.legend(); pl.savefig("out.png")

    # Data analysts:
    df = pl.load_csv("sales.csv")
    pl.quick_eda(df)

    # Databases:
    conn = pl.db_connect("sales.db")
    df2 = pl.read_sql("SELECT * FROM sales", conn)

    # ML / DL:
    pl.plot_history({"loss": [...], "val_loss": [...]})
    pl.plot_confusion_matrix(y_true, y_pred)
"""
from __future__ import annotations

from . import pyplot as plt
from .figure import Figure, figure, subplots
from . import style
from .style import rcParams

# Data + databases + EDA + ML
from . import data as data
from . import db as db
from . import eda as eda
from . import ml as ml

__version__ = "0.3.1"
__brand__ = "plotlibs"
__all__ = ["Figure", "figure", "subplots", "plt", "pyplot", "style", "rcParams",
           "data", "db", "eda", "ml",
           "plot", "scatter", "bar", "barh", "hist", "imshow", "pie",
           "fill_between", "boxplot", "violinplot", "kde", "heatmap",
           "corr", "countplot", "area", "hist2d", "stem",
           "savefig", "show", "figure_to_image",
           "load_csv", "read_sql", "db_connect", "quick_eda",
           "plot_history", "plot_confusion_matrix", "plot_roc"]

from . import pyplot as pyplot  # noqa: E402

plot = pyplot.plot
scatter = pyplot.scatter
bar = pyplot.bar
barh = pyplot.barh
hist = pyplot.hist
imshow = pyplot.imshow
pie = pyplot.pie
fill_between = pyplot.fill_between
step = pyplot.step
errorbar = pyplot.errorbar
axhline = pyplot.axhline
axvline = pyplot.axvline
boxplot = pyplot.boxplot
violinplot = pyplot.violinplot
kde = pyplot.kde
density = pyplot.kde
heatmap = pyplot.heatmap
corr = pyplot.corr
countplot = pyplot.countplot
area = pyplot.area
stackplot = pyplot.area
hist2d = pyplot.hist2d
stem = pyplot.stem
xlabel = pyplot.xlabel
ylabel = pyplot.ylabel
title = pyplot.title
legend = pyplot.legend
grid = pyplot.grid
xlim = pyplot.xlim
ylim = pyplot.ylim
savefig = pyplot.savefig
show = pyplot.show
close = pyplot.close
gca = pyplot.gca
gcf = pyplot.gcf
figure_fn = figure
subplots = subplots

# --- data / DB shortcuts ---
load_csv = data.load_csv
save_csv = data.save_csv
describe = data.describe

db_connect = db.connect
read_sql = db.read_sql
to_sql = db.to_sql

# --- EDA shortcuts ---
quick_eda = eda.quick_eda
scatter_matrix = eda.scatter_matrix
plot_missing = eda.plot_missing

# --- ML/DL shortcuts ---
plot_history = ml.plot_history
plot_confusion_matrix = ml.plot_confusion_matrix
plot_roc = ml.plot_roc
plot_pr = ml.plot_pr
plot_feature_importance = ml.plot_feature_importance


def figure_to_image(fig=None):
    from .backends.renderer import render_to_image
    return render_to_image(fig or pyplot.gcf())
