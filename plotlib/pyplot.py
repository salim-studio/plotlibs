"""plotlib.pyplot — matplotlib.pyplot-compatible state machine, faster."""
from __future__ import annotations

from . import style
from .figure import Figure, subplots as _subplots

_figures: dict[int, Figure] = {}
_current: Figure | None = None
_counter = [0]


def _cur() -> Figure:
    global _current
    if _current is None:
        _current = figure()
    return _current


def figure(figsize=None, dpi=None, facecolor=None, num=None):
    global _current
    _counter[0] += 1
    n = num if num is not None else _counter[0]
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=facecolor, num=n)
    _figures[n] = fig
    _current = fig
    return fig


def gcf():
    return _cur()


def gca():
    fig = _cur()
    if not fig.axes:
        return fig.add_subplot(1, 1, 1)
    return fig.axes[-1]


def subplots(nrows=1, ncols=1, figsize=None, dpi=None, **kw):
    global _current
    fig = Figure(figsize=figsize, dpi=dpi)
    axs = fig.subplots(nrows, ncols)
    _counter[0] += 1
    _figures[_counter[0]] = fig
    _current = fig
    return fig, axs


def subplot(nrows, ncols, index, **kw):
    return _cur().add_subplot(nrows, ncols, index)


def plot(*args, **kw):
    return gca().plot(*args, **kw)


def scatter(*args, **kw):
    return gca().scatter(*args, **kw)


def bar(*args, **kw):
    return gca().bar(*args, **kw)


def barh(*args, **kw):
    return gca().barh(*args, **kw)


def hist(*args, **kw):
    return gca().hist(*args, **kw)


def imshow(*args, **kw):
    return gca().imshow(*args, **kw)


def pie(*args, **kw):
    return gca().pie(*args, **kw)


def fill_between(*args, **kw):
    return gca().fill_between(*args, **kw)


def step(*args, **kw):
    return gca().step(*args, **kw)


def errorbar(*args, **kw):
    return gca().errorbar(*args, **kw)


def axhline(*args, **kw):
    return gca().axhline(*args, **kw)


def axvline(*args, **kw):
    return gca().axvline(*args, **kw)


def text(x, y, s, **kw):
    return gca().text(x, y, s, **kw)


def xlim(*args, **kw):
    ax = gca()
    if not args and not kw:
        return ax.get_xlim()
    if len(args) == 2:
        return ax.set_xlim(args[0], args[1])
    if len(args) == 1:
        return ax.set_xlim(args[0][0], args[0][1])
    left = kw.pop("left", None)
    right = kw.pop("right", None)
    return ax.set_xlim(left, right)


def ylim(*args, **kw):
    ax = gca()
    if not args and not kw:
        return ax.get_ylim()
    if len(args) == 2:
        return ax.set_ylim(args[0], args[1])
    if len(args) == 1:
        return ax.set_ylim(args[0][0], args[0][1])
    bottom = kw.pop("bottom", None)
    top = kw.pop("top", None)
    return ax.set_ylim(bottom, top)


def xlabel(s, **kw):
    gca().set_xlabel(s)


def ylabel(s, **kw):
    gca().set_ylabel(s)


def title(s, **kw):
    gca().set_title(s)


def suptitle(s, **kw):
    _cur().suptitle(s)


def legend(*args, **kw):
    return gca().legend(*args, **kw)


def grid(visible=True, **kw):
    gca().grid(visible, **kw)


def xticks(ticks=None, labels=None, **kw):
    ax = gca()
    if ticks is None:
        from .fast import nice_ticks
        xl = ax.get_xlim()
        return nice_ticks(xl[0], xl[1]), None
    ax.set_xticks(ticks)
    return ticks, labels


def yticks(ticks=None, labels=None, **kw):
    ax = gca()
    if ticks is None:
        from .fast import nice_ticks
        yl = ax.get_ylim()
        return nice_ticks(yl[0], yl[1]), None
    ax.set_yticks(ticks)
    return ticks, labels


def xscale(s, **kw):
    gca().set_xscale(s)


def yscale(s, **kw):
    gca().set_yscale(s)


def semilogx(*a, **k):
    r = plot(*a, **k)
    gca().set_xscale("log")
    return r


def semilogy(*a, **k):
    r = plot(*a, **k)
    gca().set_yscale("log")
    return r


def loglog(*a, **k):
    r = plot(*a, **k)
    gca().set_xscale("log")
    gca().set_yscale("log")
    return r


def tight_layout(**kw):
    _cur().tight_layout(**kw)


def savefig(fname, dpi=None, **kw):
    _cur().savefig(fname, dpi=dpi, **kw)


def show(**kw):
    _cur().show()


def close(*args, **kw):
    global _current
    _figures.clear()
    _current = None


def clf():
    _cur().clf()


def cla():
    gca().cla()


def rcParams_update(d):
    style.rcParams.update(d)


rcParams = style.rcParams
style_use = style.use
