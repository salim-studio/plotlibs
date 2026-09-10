"""plotlib — fast matplotlib-compatible plotting library.

Same API as matplotlib.pyplot, but faster:
- min-max decimation for big data (no visual loss)
- Pillow C-level rasterizer instead of heavy layout engine
- vectorized numpy pipeline, lazy draw, font caching

Example:
    import plotlib as pl
    pl.plot([1, 2, 3], [1, 4, 9], label="x^2")
    pl.xlabel("x"); pl.ylabel("y"); pl.legend(); pl.savefig("out.png")
"""
from __future__ import annotations

from . import pyplot as plt
from .figure import Figure, figure, subplots
from . import style
from .style import rcParams

__version__ = "0.1.0"
__all__ = ["Figure", "figure", "subplots", "plt", "pyplot", "style", "rcParams",
           "plot", "scatter", "bar", "barh", "hist", "imshow", "pie",
           "fill_between", "savefig", "show", "figure_to_image"]

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


def figure_to_image(fig=None):
    from .backends.renderer import render_to_image
    return render_to_image(fig or pyplot.gcf())
