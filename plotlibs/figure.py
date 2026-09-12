"""plotlibs.figure — Figure + Axes with matplotlib-compatible API.

Speed design:
- artists are plain dicts (no heavy objects)
- limits computed vectorized with numpy
- actual rasterization deferred to backends/renderer.py (Pillow, C-level draws)
- auto min-max decimation for big lines
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from . import style
from .colors import CYCLE, to_rgb
from .fast import as_xy, decimate, nice_limits


class Axes:
    _ids = itertools.count()

    def __init__(self, fig: "Figure", rect=(0.125, 0.11, 0.775, 0.77)):
        self.figure = fig
        self.id = next(Axes._ids)
        self.rect = tuple(rect)  # (left, bottom, width, height) in 0-1
        self.lines: list[dict] = []
        self.scatters: list[dict] = []
        self.bars: list[dict] = []
        self.hists: list[dict] = []
        self.images: list[dict] = []
        self.fills: list[dict] = []
        self.pies: list[dict] = []
        self.texts: list[dict] = []
        self._hlines: list[dict] = []
        self._vlines: list[dict] = []
        # --- stats / EDA / ML artists (جديد 0.2.0) ---
        self.boxes: list[dict] = []
        self.violins: list[dict] = []
        self.kdes: list[dict] = []
        self.heatmaps: list[dict] = []
        self.areas: list[dict] = []
        self.stems: list[dict] = []
        self._color_idx = 0
        self._xlim = None
        self._ylim = None
        self._xlabel = ""
        self._ylabel = ""
        self._title = ""
        self._legend_labels: list[str] = []
        self._grid = style.rcParams.get("axes.grid", False)
        self._xscale = "linear"
        self._yscale = "linear"
        self.facecolor = style.rcParams.get("axes.facecolor", "white")

    # ---------- internal ----------
    def _next_color(self, given=None):
        if given is not None:
            return given
        c = CYCLE[self._color_idx % len(CYCLE)]
        self._color_idx += 1
        return c

    # ---------- plotting API (matplotlib-compatible) ----------
    def plot(self, x, y=None, fmt: str = "", **kw):
        """plot(x, y) / plot(y) / plot(x, y, 'ro--'). Returns list[Line2D-dict]."""
        color = kw.pop("color", kw.pop("c", None))
        label = kw.pop("label", "")
        lw = kw.pop("linewidth", kw.pop("lw", style.rcParams["lines.linewidth"]))
        ls = kw.pop("linestyle", kw.pop("ls", "-"))
        marker = kw.pop("marker", None)
        ms = kw.pop("markersize", kw.pop("ms", style.rcParams["lines.markersize"]))
        alpha = kw.pop("alpha", 1.0)
        if isinstance(y, str) and fmt == "":
            fmt, y = y, None
        if fmt:
            # parse 'ro--' style fmt
            f = fmt.strip()
            for ch, col in (("b", "b"), ("g", "g"), ("r", "r"), ("c", "c"),
                            ("m", "m"), ("y", "y"), ("k", "k"), ("w", "w")):
                if ch in f and color is None:
                    color = col
                    break
            for m in ("o", "s", "^", "v", "D", "x", "+", "*", ".", "d"):
                if m in f:
                    marker = marker or m
            if "--" in f:
                ls = "--"
            elif "-." in f:
                ls = "-."
            elif ":" in f:
                ls = ":"
            elif "-" in f:
                ls = "-"
            elif f.strip("bgrcmykwos^vDx+*.d") == "":
                ls = "none"
        x, y = as_xy(x, y)
        line = dict(x=x, y=y, color=self._next_color(color), label=label,
                    linewidth=float(lw), linestyle=ls, marker=marker,
                    markersize=float(ms), alpha=float(alpha))
        self.lines.append(line)
        if label:
            self._legend_labels.append(label)
        return [line]

    def scatter(self, x, y, s=None, c=None, color=None, label="", alpha=1.0,
                marker="o", **kw):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        col = c if c is not None else color
        if isinstance(col, (list, np.ndarray)) and np.asarray(col).ndim == 2:
            colors = [tuple(v) for v in np.asarray(col)]
        elif isinstance(col, (list, np.ndarray)) and np.asarray(col).ndim == 1 and len(np.asarray(col)) == n and n > 4:
            # scalar-mapped -> simple colormap (viridis approx: blue->yellow)
            v = np.asarray(col, dtype=float)
            v = (v - v.min()) / (v.ptp() or 1.0)
            colors = [(int(68 + v_i * 187), int(1 + v_i * 230), int(84 + v_i * 60)) for v_i in v]
        else:
            colors = self._next_color(col)
        sc = dict(x=x, y=y, s=s, colors=colors, label=label, alpha=float(alpha), marker=marker)
        self.scatters.append(sc)
        if label:
            self._legend_labels.append(label)
        return sc

    def bar(self, x, height, width=0.8, color=None, label="", alpha=1.0, **kw):
        xa = np.asarray(x).ravel()
        h = np.asarray(height, dtype=float).ravel()
        if xa.dtype.kind in "USO" or (xa.size != h.size):
            # categorical or mismatched -> positions 0..n-1
            xpos = np.arange(h.size, dtype=float)
            try:
                self._xticklabels = [str(v) for v in xa.ravel()[: h.size]]
                self._xtickpos = xpos
            except Exception:
                pass
        else:
            xpos = xa.astype(float)
        bottom = kw.pop("bottom", 0.0)
        b = np.asarray(bottom, dtype=float).ravel()
        if b.size == 1:
            b = np.full(h.size, float(b[0]))
        bars = dict(x=np.asarray(xpos).ravel(), h=h, w=float(width) if np.ndim(width) == 0 else np.asarray(width),
                    bottom=b, color=self._next_color(color), label=label, alpha=float(alpha), horizontal=False)
        self.bars.append(bars)
        if label:
            self._legend_labels.append(label)
        return bars

    def barh(self, y, width, height=0.8, color=None, label="", alpha=1.0, **kw):
        y = np.asarray(y).ravel()
        w = np.asarray(width, dtype=float).ravel()
        if y.size != w.size and np.asarray(y).dtype.kind in "iuf":
            y = np.arange(w.size)
        bars = dict(x=y, h=w, w=float(height) if np.ndim(height) == 0 else np.asarray(height),
                    bottom=np.zeros(w.size), color=self._next_color(color),
                    label=label, alpha=float(alpha), horizontal=True)
        self.bars.append(bars)
        if label:
            self._legend_labels.append(label)
        return bars

    def hist(self, x, bins=10, color=None, label="", alpha=1.0, density=False, **kw):
        x = np.asarray(x, dtype=float).ravel()
        x = x[np.isfinite(x)]
        counts, edges = np.histogram(x, bins=bins, density=density)
        h = dict(counts=counts, edges=edges, color=self._next_color(color),
                 label=label, alpha=float(alpha))
        self.hists.append(h)
        if label:
            self._legend_labels.append(label)
        return counts, edges, h

    def imshow(self, X, cmap=None, vmin=None, vmax=None, origin="upper",
               extent=None, aspect="auto", **kw):
        img = dict(data=np.asarray(X), cmap=cmap or style.rcParams["image.cmap"],
                   vmin=vmin, vmax=vmax, origin=origin, extent=extent)
        self.images.append(img)
        return img

    def pie(self, x, labels=None, autopct=None, colors=None, startangle=0, **kw):
        fracs = np.asarray(x, dtype=float).ravel()
        s = fracs.sum() or 1.0
        fracs = fracs / s
        p = dict(fracs=fracs, labels=list(labels) if labels is not None else None,
                 autopct=autopct, colors=colors, startangle=float(startangle))
        self.pies.append(p)
        return fracs

    def fill_between(self, x, y1, y2=0, color=None, alpha=0.35, label="", **kw):
        x = np.asarray(x, dtype=float).ravel()
        y1 = np.asarray(y1, dtype=float).ravel()
        y2 = np.asarray(y2, dtype=float).ravel() if np.ndim(y2) else np.full_like(x, float(y2))
        n = min(x.size, y1.size, y2.size)
        f = dict(x=x[:n], y1=y1[:n], y2=y2[:n], color=self._next_color(color),
                 alpha=float(alpha), label=label)
        self.fills.append(f)
        if label:
            self._legend_labels.append(label)
        return f

    def step(self, x, y, color=None, label="", linewidth=None, **kw):
        x, y = as_xy(x, y)
        # convert to step path (pre)
        xs = np.repeat(x, 2)[1:]
        xs = np.append(xs, x[-1]) if x.size else xs
        ys = np.repeat(y, 2)[:-1] if y.size else y
        n = min(xs.size, ys.size)
        return self.plot(xs[:n], ys[:n], color=color, label=label,
                         linewidth=linewidth or style.rcParams["lines.linewidth"])[0]

    def errorbar(self, x, y, yerr=None, xerr=None, fmt="o", color=None,
                 label="", capsize=3, **kw):
        line = self.plot(x, y, fmt, color=color, label=label, **kw)[0]
        if yerr is not None:
            x_, y_ = as_xy(x, y)
            e = np.asarray(yerr, dtype=float).ravel()
            if e.size == 1:
                e = np.full_like(y_, float(e[0]))
            self._vlines.append(dict(kind="errorbar-y", x=x_, y=y_, e=e[: len(x_)],
                                     color=line["color"]))
        if xerr is not None:
            x_, y_ = as_xy(x, y)
            e = np.asarray(xerr, dtype=float).ravel()
            if e.size == 1:
                e = np.full_like(x_, float(e[0]))
            self._hlines.append(dict(kind="errorbar-x", x=x_, y=y_, e=e[: len(x_)],
                                     color=line["color"]))
        return line

    def axhline(self, y=0, color="k", linestyle="--", linewidth=1.0, **kw):
        self._hlines.append(dict(kind="axhline", y=float(y), color=color,
                                 linestyle=linestyle, linewidth=float(linewidth)))
        return self._hlines[-1]

    def axvline(self, x=0, color="k", linestyle="--", linewidth=1.0, **kw):
        self._vlines.append(dict(kind="axvline", x=float(x), color=color,
                                 linestyle=linestyle, linewidth=float(linewidth)))
        return self._vlines[-1]

    def text(self, x, y, s, fontsize=None, color="black", **kw):
        t = dict(x=x, y=y, s=str(s), fontsize=fontsize or style.rcParams["font.size"],
                 color=color, data_coords=True)
        self.texts.append(t)
        return t

    # ---------- stats / EDA (جديد — لجعل المكتبة مشهورة عند المحللين) ----------
    def boxplot(self, data, labels=None, color=None, **kw):
        """boxplot(list of arrays). متوافق مع matplotlib."""
        if isinstance(data, np.ndarray) and data.ndim == 2:
            cols = [data[:, i] for i in range(data.shape[1])]
        elif isinstance(data, (list, tuple)) and data and np.ndim(data[0]) == 0:
            cols = [np.asarray(data, dtype=float)]
        else:
            try:
                cols = [np.asarray(c, dtype=float).ravel() for c in data]
            except Exception:
                cols = [np.asarray(data, dtype=float).ravel()]
        b = dict(cols=cols, labels=list(labels) if labels is not None else None,
                 color=self._next_color(color))
        self.boxes.append(b)
        return b

    def violinplot(self, data, labels=None, color=None, points=120, **kw):
        if isinstance(data, np.ndarray) and data.ndim == 2:
            cols = [data[:, i] for i in range(data.shape[1])]
        else:
            try:
                cols = [np.asarray(c, dtype=float).ravel() for c in data]
            except Exception:
                cols = [np.asarray(data, dtype=float).ravel()]
        v = dict(cols=cols, labels=list(labels) if labels is not None else None,
                 color=self._next_color(color), points=int(points))
        self.violins.append(v)
        return v

    def kde(self, data, color=None, label="", fill=True, bw=None, **kw):
        """منحنى الكثافة KDE (بديل seaborn.kdeplot)."""
        from .fast import gaussian_kde_1d
        v = np.asarray(data, dtype=float).ravel()
        xs, ys = gaussian_kde_1d(v, bw=bw) if v.size else (np.array([0, 1]), np.zeros(2))
        k = dict(x=xs, y=ys, color=self._next_color(color), label=label,
                 fill=bool(fill))
        self.kdes.append(k)
        if label:
            self._legend_labels.append(label)
        return k

    density = kde

    def heatmap(self, matrix, xticks=None, yticks=None, annot=False, cmap="viridis",
                vmin=None, vmax=None, **kw):
        m = np.asarray(matrix, dtype=float)
        h = dict(data=m, xticks=list(xticks) if xticks is not None else None,
                 yticks=list(yticks) if yticks is not None else None,
                 annot=bool(annot), cmap=cmap, vmin=vmin, vmax=vmax)
        self.heatmaps.append(h)
        return h

    def corr(self, data, cols=None, annot=True, cmap="viridis", **kw):
        """مصفوفة ارتباط جاهزة من DataFrame/dict مباشرة."""
        from .data import corr_matrix
        if isinstance(data, np.ndarray):
            m = np.asarray(data, dtype=float)
            c = np.corrcoef(m, rowvar=False) if m.ndim == 2 else np.eye(1)
            labels = cols or [f"c{i}" for i in range(c.shape[0])]
            return self.heatmap(np.nan_to_num(c), xticks=labels, yticks=labels,
                                annot=annot, cmap=cmap)
        c, labels = corr_matrix(data, cols)
        return self.heatmap(c, xticks=labels, yticks=labels, annot=annot, cmap=cmap,
                            vmin=-1, vmax=1)

    def corrcoef(self, *a, **k):
        return self.corr(*a, **k)

    def countplot(self, values, color=None, label="", **kw):
        """رسم تكرار الفئات (بديل seaborn.countplot)."""
        vals = np.asarray(values).ravel()
        uniq, counts = np.unique(vals, return_counts=True)
        order = np.argsort(-counts, kind="stable")
        uniq, counts = uniq[order], counts[order]
        xpos = np.arange(len(uniq), dtype=float)
        b = dict(x=xpos, h=counts.astype(float), w=np.full(len(uniq), 0.6),
                 bottom=np.zeros(len(uniq)), color=self._next_color(color),
                 label=label, alpha=1.0, horizontal=False)
        self.bars.append(b)
        try:
            self._xticklabels = [str(v) for v in uniq]
            self._xtickpos = xpos
        except Exception:
            pass
        if label:
            self._legend_labels.append(label)
        return b

    count = countplot

    def area(self, x, *ys, labels=None, colors=None, alpha=0.5, **kw):
        """منحنى مساحي مكدّس (stackplot)."""
        x = np.asarray(x, dtype=float).ravel()
        arrs = [np.asarray(y, dtype=float).ravel()[: x.size] for y in ys]
        a = dict(x=x, ys=arrs,
                 labels=list(labels) if labels is not None else [""] * len(arrs),
                 colors=list(colors) if colors is not None else [self._next_color(None) for _ in arrs],
                 alpha=float(alpha))
        self.areas.append(a)
        for lb in a["labels"]:
            if lb:
                self._legend_labels.append(lb)
        return a

    stackplot = area

    def hist2d(self, x, y, bins=30, cmap="viridis", **kw):
        """كثافة ثنائية الأبعاد (بديل plt.hist2d)."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        H, xe, ye = np.histogram2d(x[:n], y[:n], bins=bins)
        h = dict(data=H.T, xticks=None, yticks=None, annot=False, cmap=cmap,
                 vmin=None, vmax=None, extent=(xe[0], xe[-1], ye[0], ye[-1]))
        self.heatmaps.append(h)
        return H, xe, ye, h

    def stem(self, x, y=None, color=None, label="", **kw):
        if y is None:
            y = np.asarray(x, dtype=float).ravel()
            x = np.arange(y.size, dtype=float)
        s = dict(x=np.asarray(x, dtype=float).ravel(),
                 y=np.asarray(y, dtype=float).ravel(),
                 color=self._next_color(color), label=label)
        self.stems.append(s)
        if label:
            self._legend_labels.append(label)
        return s

    def table(self, data, col_labels=None, row_labels=None, **kw):
        t = dict(kind="table")
        self.texts.append(dict(x=0.5, y=-0.05, s=f"[table {np.asarray(data).shape}]",
                               fontsize=9, color="black", data_coords=False))
        return t

    # ---------- cosmetics, matplotlib-compatible ----------
    def set_xlim(self, left=None, right=None):
        l, r = self.get_xlim()
        if left is not None:
            l = left
        if right is not None:
            r = right
        self._xlim = (l, r)
        return self._xlim

    def set_ylim(self, bottom=None, top=None):
        b, t = self.get_ylim()
        if bottom is not None:
            b = bottom
        if top is not None:
            t = top
        self._ylim = (b, t)
        return self._ylim

    def get_xlim(self):
        if self._xlim is not None:
            return self._xlim
        return self._auto_limits("x")

    def get_ylim(self):
        if self._ylim is not None:
            return self._ylim
        return self._auto_limits("y")

    def _auto_limits(self, axis):
        lo, hi = np.inf, -np.inf
        def upd(v):
            nonlocal lo, hi
            try:
                v = np.asarray(v, dtype=float).ravel()
            except Exception:
                return
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = min(lo, v.min()), max(hi, v.max())
        for ln in self.lines:
            upd(ln["x"] if axis == "x" else ln["y"])
        for sc in self.scatters:
            upd(sc["x"] if axis == "x" else sc["y"])
        for b in self.bars:
            if not b["horizontal"]:
                if axis == "x":
                    upd(b["x"])
                else:
                    upd(np.concatenate([b["bottom"], b["bottom"] + b["h"]]))
            else:
                if axis == "x":
                    upd(np.concatenate([b["bottom"], b["bottom"] + b["h"]]))
                else:
                    upd(b["x"])
        for h in self.hists:
            if axis == "x":
                upd(h["edges"])
            else:
                upd(np.append(h["counts"], 0))
        for f in self.fills:
            if axis == "x":
                upd(f["x"])
            else:
                upd(np.concatenate([f["y1"], f["y2"]]))
        for im in self.images:
            if im["extent"] is not None:
                e = im["extent"]
                upd([e[0], e[1]] if axis == "x" else [e[2], e[3]])
            else:
                d = im["data"]
                upd([0, d.shape[1]] if axis == "x" else [0, d.shape[0]])
        for k in self.kdes:
            upd(k["x"] if axis == "x" else k["y"])
        for b in self.boxes + self.violins:
            if axis == "x":
                upd([1, len(b["cols"])])
            else:
                for c in b["cols"]:
                    upd(c)
        for hm in self.heatmaps:
            d = np.asarray(hm["data"])
            if axis == "x":
                upd([0, d.shape[1]])
            else:
                upd([0, d.shape[0]])
        for a in self.areas:
            if axis == "x":
                upd(a["x"])
            else:
                if a["ys"]:
                    try:
                        upd(np.sum(np.stack([np.asarray(v, dtype=float) for v in a["ys"]]), axis=0))
                        for v in a["ys"]:
                            upd(v)
                    except Exception:
                        pass
        for s in self.stems:
            upd(s["x"] if axis == "x" else s["y"])
        if not np.isfinite(lo):
            return (0.0, 1.0)
        a, bb = nice_limits(lo, hi)
        return (a, bb)

    def set_xlabel(self, s):
        self._xlabel = str(s)

    def set_ylabel(self, s):
        self._ylabel = str(s)

    def set_title(self, s):
        self._title = str(s)

    xlabel = set_xlabel
    ylabel = set_ylabel
    title = set_title

    def set_xscale(self, s):
        self._xscale = s

    def set_yscale(self, s):
        self._yscale = s

    def grid(self, visible=True, **kw):
        self._grid = bool(visible)

    def legend(self, *a, **kw):
        # labels already collected; renderer draws them
        return self._legend_labels

    def tick_params(self, **kw):
        pass

    def set_xticks(self, t):
        self._xticks = list(t)

    def set_yticks(self, t):
        self._yticks = list(t)

    def cla(self):
        self.__init__(self.figure, self.rect)

    # ---------- draw / save ----------
    def _rendered_lines(self):
        """Lines after fast decimation — the core speedup vs matplotlib."""
        out = []
        for ln in self.lines:
            x, y = ln["x"], ln["y"]
            if x.size > 2000:
                x, y = decimate(x, y)
            out.append({**ln, "x": x, "y": y})
        return out


class Figure:
    def __init__(self, figsize=None, dpi=None, facecolor=None, num=None):
        fs = figsize or style.rcParams["figure.figsize"]
        self.figsize = tuple(fs)
        self.dpi = int(dpi or style.rcParams["figure.dpi"])
        self.facecolor = facecolor or style.rcParams["figure.facecolor"]
        self.num = num
        self.axes: list[Axes] = []
        self._suptitle = ""

    @property
    def pixel_size(self):
        return (int(self.figsize[0] * self.dpi), int(self.figsize[1] * self.dpi))

    def add_subplot(self, *args, **kw):
        # support (nrows, ncols, index) or (111)
        if len(args) == 1 and isinstance(args[0], int) and args[0] >= 100:
            code = args[0]
            nr, nc, ix = code // 100, (code // 10) % 10, code % 10
        elif len(args) == 3:
            nr, nc, ix = args
        elif len(args) == 1 and isinstance(args[0], tuple):
            nr, nc, ix = args[0]
        else:
            nr, nc, ix = 1, 1, 1
        left = 0.125 + (0.775 / nc) * ((ix - 1) % nc)
        # row from top
        row = (ix - 1) // nc
        h = 0.77 / nr
        bottom = 0.11 + 0.77 - (row + 1) * h
        w = 0.775 / nc
        ax = Axes(self, rect=(left, bottom, max(w - 0.03, 0.05), max(h - 0.05, 0.05)))
        self.axes.append(ax)
        return ax

    def add_axes(self, rect):
        ax = Axes(self, rect=tuple(rect))
        self.axes.append(ax)
        return ax

    def subplots(self, nrows=1, ncols=1, **kw):
        for i in range(1, nrows * ncols + 1):
            self.add_subplot(nrows, ncols, i)
        if nrows * ncols == 1:
            return self.axes[0]
        import numpy as _np
        return _np.array(self.axes, dtype=object).reshape(nrows, ncols)

    def suptitle(self, s):
        self._suptitle = str(s)

    def tight_layout(self, **kw):
        pass

    def savefig(self, fname, dpi=None, **kw):
        from .backends.renderer import render_figure
        render_figure(self, fname, dpi=dpi or self.dpi)

    def show(self):
        from .backends.renderer import render_figure
        render_figure(self, None)

    def clf(self):
        self.axes.clear()

    def canvas_draw(self):
        from .backends.renderer import render_to_image
        return render_to_image(self)


def figure(figsize=None, dpi=None, facecolor=None, num=None):
    return Figure(figsize=figsize, dpi=dpi, facecolor=facecolor, num=num)


def subplots(nrows=1, ncols=1, figsize=None, dpi=None, **kw):
    fig = Figure(figsize=figsize, dpi=dpi)
    axs = fig.subplots(nrows, ncols)
    return fig, axs
