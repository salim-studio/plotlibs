"""plotlibs.style — rcParams + styles, matplotlib-compatible subset."""
from __future__ import annotations

rcParams: dict = {
    "figure.figsize": (6.4, 4.8),
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.grid": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "savefig.dpi": 100,
    "image.cmap": "viridis",
}

BRAND = {
    "primary": "#4F46E5",
    "accent": "#06B6D4",
    "ink": "#0F172A",
    "paper": "#F8FAFC",
    "lime": "#A3E635",
    "amber": "#FACC15",
}

_STYLES = {
    "default": {},
    "plotlibs": {"axes.facecolor": "white", "axes.grid": True,
                 "figure.facecolor": "white", "axes.edgecolor": "#0F172A",
                 "lines.linewidth": 2.2, "font.size": 11,
                 "axes.labelsize": 11, "axes.titlesize": 13,
                 "image.cmap": "viridis"},
    "plotlibs-dark": {"figure.facecolor": "#0F172A", "axes.facecolor": "#1E293B",
                      "axes.edgecolor": "white", "axes.grid": True,
                      "lines.linewidth": 2.2, "font.size": 11},
    "fast": {"axes.grid": True, "figure.dpi": 80},
    "dark": {"figure.facecolor": "#1e1e1e", "axes.facecolor": "#1e1e1e",
             "axes.edgecolor": "white", "axes.grid": True},
    "ggplot": {"axes.facecolor": "#E5E5E5", "axes.grid": True,
               "figure.facecolor": "#F5F5F5"},
    "grayscale": {},
    # extra gallery styles
    "seaborn": {"axes.facecolor": "white", "axes.grid": True,
                "figure.facecolor": "white", "lines.linewidth": 2.0,
                "font.size": 11, "axes.labelsize": 11},
    "seaborn-darkgrid": {"axes.facecolor": "#EAEAF2", "axes.grid": True,
                         "figure.facecolor": "white", "axes.edgecolor": "white",
                         "lines.linewidth": 2.0},
    "plotly": {"axes.facecolor": "white", "axes.grid": True,
               "figure.facecolor": "white", "lines.linewidth": 2.2,
               "font.size": 12},
    "publication": {"axes.facecolor": "white", "axes.grid": False,
                    "figure.dpi": 150, "lines.linewidth": 1.5,
                    "font.size": 10},
    "darkgrid": {"figure.facecolor": "#212121", "axes.facecolor": "#2E2E2E",
                 "axes.edgecolor": "white", "axes.grid": True},
}

_available_ = sorted(_STYLES)


def use(name: str):
    if name not in _STYLES:
        raise ValueError(f"Unknown style {name!r}. Available: {_available_}")
    base_face = rcParams.get("figure.facecolor", "white")
    rcParams.update(_STYLES[name])
    return base_face


def available():
    return list(_available_)


class _StyleCtx:
    def __init__(self, name):
        self.name = name
        self._saved = None

    def __enter__(self):
        self._saved = dict(rcParams)
        use(self.name)
        return self

    def __exit__(self, *a):
        rcParams.clear()
        rcParams.update(self._saved)
        return False


def context(name: str):
    return _StyleCtx(name)
