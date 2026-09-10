"""plotlib.backends.renderer — Pillow-based fast raster renderer."""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..colors import to_rgb
from ..fast import fmt_tick, nice_ticks
from ..style import rcParams

_FONT_CACHE: dict[int, object] = {}


def _font(size: int):
    size = max(8, int(size))
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.load_default(size=size)
        except TypeError:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _parse_color(c):
    if isinstance(c, tuple) and len(c) == 3 and all(isinstance(v, (int, np.integer)) for v in c):
        return tuple(int(v) for v in c)
    return to_rgb(c)


def _cmap_color(name, t: float):
    """Tiny fast viridis/gray/jet approximation, vectorized externally."""
    t = max(0.0, min(1.0, float(t)))
    if name in ("gray", "grey", "binary", "gist_yarg"):
        v = int(t * 255)
        return (v, v, v)
    if name in ("hot",):
        return (int(t * 255), int(t * t * 255), int(t * t * t * 128))
    if name in ("cool", "jet", "hsv", "rainbow"):
        r = int(255 * max(0, min(1, 1.5 - abs(4 * t - 3))))
        g = int(255 * max(0, min(1, 1.5 - abs(4 * t - 2))))
        b = int(255 * max(0, min(1, 1.5 - abs(4 * t - 1))))
        return (r, g, b)
    # viridis approx
    r = int(68 + t * (253 - 68))
    g = int(1 + t * (231 - 1))
    b = int(84 + t * (37 - 84))
    return (r, g, b)


def _data_to_px(x, y, xlim, ylim, box):
    """Vectorized data -> pixel mapping. box=(x0,y0,x1,y1) PIL coords."""
    x0, y0, x1, y1 = box
    span_x = (xlim[1] - xlim[0]) or 1.0
    span_y = (ylim[1] - ylim[0]) or 1.0
    px = x0 + (np.asarray(x, dtype=float) - xlim[0]) / span_x * (x1 - x0)
    py = y1 - (np.asarray(y, dtype=float) - ylim[0]) / span_y * (y1 - y0)
    return px, py


def _draw_line(draw: ImageDraw.ImageDraw, px, py, color, width, linestyle, marker, ms):
    width = max(1, int(round(width)))
    pts = list(zip(px.astype(int), py.astype(int)))
    if len(pts) >= 2 and linestyle not in ("none", "None", ""):
        if linestyle in ("--", ":"):
            # dashed: draw segments with gaps (fast, chunked)
            dash = 6 if linestyle == "--" else 2
            gap = 4 if linestyle == "--" else 3
            for i in range(len(pts) - 1):
                if (i % (dash + gap)) < dash:
                    draw.line([pts[i], pts[i + 1]], fill=color, width=width)
        else:
            draw.line(pts, fill=color, width=width, joint="curve")
    if marker and len(pts):
        r = max(2, int(ms / 2))
        step = max(1, len(pts) // 200)  # cap markers for speed
        for (mx, my) in pts[::step]:
            if marker == "s":
                draw.rectangle([mx - r, my - r, mx + r, my + r], fill=color, outline=color)
            elif marker in ("^", "v"):
                d = r + 1
                tri = [(mx, my - d), (mx - d, my + d), (mx + d, my + d)] if marker == "^" else \
                      [(mx, my + d), (mx - d, my - d), (mx + d, my - d)]
                draw.polygon(tri, fill=color)
            elif marker in ("x", "+"):
                draw.line([mx - r, my, mx + r, my], fill=color, width=width)
                draw.line([mx, my - r, mx, my + r], fill=color, width=width)
            else:  # o, ., D, *, d -> circle
                draw.ellipse([mx - r, my - r, mx + r, my + r], fill=color, outline=color)


def _draw_axes(img: Image.Image, draw: ImageDraw.ImageDraw, ax, W, H):
    l, b, w, h = ax.rect
    x0, y1 = int(l * W), int((1 - b) * H)
    x1, y0 = int((l + w) * W), int((1 - (b + h)) * H)
    box = (x0, y0, x1, y1)
    # face
    try:
        face = _parse_color(ax.facecolor)
    except Exception:
        face = (255, 255, 255)
    draw.rectangle(box, fill=face)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # grid (under data)
    if ax._grid:
        for t in nice_ticks(xlim[0], xlim[1]):
            if not (xlim[0] <= t <= xlim[1]):
                continue
            gx = int(x0 + (t - xlim[0]) / ((xlim[1] - xlim[0]) or 1) * (x1 - x0))
            draw.line([(gx, y0), (gx, y1)], fill=(210, 210, 210), width=1)
        for t in nice_ticks(ylim[0], ylim[1]):
            if not (ylim[0] <= t <= ylim[1]):
                continue
            gy = int(y1 - (t - ylim[0]) / ((ylim[1] - ylim[0]) or 1) * (y1 - y0))
            draw.line([(x0, gy), (x1, gy)], fill=(210, 210, 210), width=1)

    # images (imshow) first
    for im in ax.images:
        data = im["data"]
        try:
            if data.ndim == 2:
                vmin = im["vmin"] if im["vmin"] is not None else float(np.nanmin(data))
                vmax = im["vmax"] if im["vmax"] is not None else float(np.nanmax(data))
                norm = (data - vmin) / ((vmax - vmin) or 1.0)
                norm = np.clip(norm, 0, 1)
                hh, ww = norm.shape
                rgb = np.empty((hh, ww, 3), dtype=np.uint8)
                # fast vectorized viridis-approx
                rgb[..., 0] = (68 + norm * 185).astype(np.uint8)
                rgb[..., 1] = (1 + norm * 230).astype(np.uint8)
                rgb[..., 2] = (84 - norm * 47).astype(np.uint8)
                pil = Image.fromarray(rgb)
            else:
                a = np.asarray(data)
                if a.dtype != np.uint8:
                    a = np.clip(a, 0, 255).astype(np.uint8) if a.max() > 1 else (a * 255).astype(np.uint8)
                pil = Image.fromarray(a)
            if im["origin"] == "upper":
                pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
            pil = pil.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.BILINEAR)
            img.paste(pil, (x0, y0))
        except Exception:
            pass

    # fill_between
    for f in ax.fills:
        c = _parse_color(f["color"])
        px1, py1 = _data_to_px(f["x"], f["y1"], xlim, ylim, box)
        _, py2 = _data_to_px(f["x"], f["y2"], xlim, ylim, box)
        poly = list(zip(px1.astype(int), py1.astype(int))) + \
               list(zip(px1.astype(int)[::-1], py2.astype(int)[::-1]))
        if len(poly) >= 3:
            draw.polygon(poly, fill=c + (90,) if len(c) == 3 else c)

    # bars + hists
    def _rect_from_data(dx0, dy0, dx1, dy1, color):
        pxa, pya = _data_to_px([dx0, dx1], [dy0, dy1], xlim, ylim, box)
        xa, xb = int(min(pxa[0], pxa[1])), int(max(pxa[0], pxa[1]))
        ya, yb = int(min(pya[0], pya[1])), int(max(pya[0], pya[1]))
        if xb - xa < 1:
            xb = xa + 1
        draw.rectangle([xa, ya, xb, yb], fill=color, outline=tuple(max(0, v - 40) for v in color))

    for bb in ax.bars:
        c = _parse_color(bb["color"])
        n = len(bb["h"])
        widths = bb["w"] if isinstance(bb["w"], np.ndarray) else np.full(n, float(bb["w"]))
        xs = np.asarray(bb["x"]).ravel()
        if xs.size != n:
            xs = np.arange(n)
        for i in range(n):
            if not bb["horizontal"]:
                cx = float(xs[i]) if i < xs.size else i
                _rect_from_data(cx - widths[i] / 2, float(bb["bottom"][i]),
                                cx + widths[i] / 2, float(bb["bottom"][i] + bb["h"][i]), c)
            else:
                cy = float(xs[i]) if i < xs.size else i
                _rect_from_data(float(bb["bottom"][i]), cy - widths[i] / 2,
                                float(bb["bottom"][i] + bb["h"][i]), cy + widths[i] / 2, c)

    for hh in ax.hists:
        c = _parse_color(hh["color"])
        for i, cnt in enumerate(hh["counts"]):
            _rect_from_data(float(hh["edges"][i]), 0.0, float(hh["edges"][i + 1]), float(cnt), c)

    # lines (decimated — speedup core)
    for ln in ax._rendered_lines():
        c = _parse_color(ln["color"])
        if ln["x"].size == 0:
            continue
        px, py = _data_to_px(ln["x"], ln["y"], xlim, ylim, box)
        # clip to box to avoid Pillow huge-coord slowdown
        px = np.clip(px, x0 - 50, x1 + 50)
        py = np.clip(py, y0 - 50, y1 + 50)
        _draw_line(draw, px, py, c, ln["linewidth"], ln["linestyle"], ln["marker"], ln["markersize"])

    # scatters
    for sc in ax.scatters:
        n = len(sc["x"])
        px, py = _data_to_px(sc["x"], sc["y"], xlim, ylim, box)
        cols = sc["colors"]
        single = isinstance(cols, tuple)
        base = _parse_color(cols) if single else None
        s = sc["s"]
        if s is None:
            r = 3
        elif np.ndim(s) == 0:
            r = max(1, int(math.sqrt(float(s)) / 2))
        else:
            r = None
        # fast path: tiny squares for huge clouds
        if n > 20000:
            for i in range(0, n, 2):
                xi, yi = int(px[i]), int(py[i])
                if x0 <= xi <= x1 and y0 <= yi <= y1:
                    cc = base if single else _parse_color(cols[i]) if i < len(cols) else base
                    img.putpixel((xi, yi), cc) if False else draw.point((xi, yi), fill=cc or (0, 0, 0))
            continue
        for i in range(n):
            xi, yi = int(px[i]), int(py[i])
            if not (x0 - 20 <= xi <= x1 + 20 and y0 - 20 <= yi <= y1 + 20):
                continue
            cc = base if single else (_parse_color(cols[i]) if i < len(cols) else (31, 119, 180))
            rr = r if r is not None else max(1, int(math.sqrt(float(np.asarray(s).ravel()[i])) / 2))
            draw.ellipse([xi - rr, yi - rr, xi + rr, yi + rr], fill=cc, outline=cc)

    # hlines / vlines / errorbars (simplified)
    for hl in ax._hlines:
        if hl["kind"] == "axhline":
            _, pyy = _data_to_px([xlim[0]], [hl["y"]], xlim, ylim, box)
            draw.line([(x0, int(pyy[0])), (x1, int(pyy[0]))], fill=_parse_color(hl["color"]), width=int(hl["linewidth"]))
    for vl in ax._vlines:
        if vl["kind"] == "axvline":
            pxx, _ = _data_to_px([vl["x"]], [ylim[0]], xlim, ylim, box)
            draw.line([(int(pxx[0]), y0), (int(pxx[0]), y1)], fill=_parse_color(vl["color"]), width=int(vl["linewidth"]))

    # pie
    for p in ax.pies:
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        rr = min(x1 - x0, y1 - y0) // 2 - 10
        ang = float(p["startangle"])
        cols = p["colors"]
        for i, fr in enumerate(p["fracs"]):
            sweep = float(fr) * 360
            cc = _parse_color(cols[i]) if cols and i < len(cols) else [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)][i % 4]
            draw.pieslice([cx - rr, cy - rr, cx + rr, cy + rr], start=ang, end=ang + sweep, fill=cc,
                          outline=(255, 255, 255))
            ang += sweep

    # spines
    edge = _parse_color(rcParams.get("axes.edgecolor", "black"))
    draw.rectangle(box, outline=edge, width=1)

    # ticks + labels
    fs = int(rcParams.get("xtick.labelsize", 9))
    font = _font(fs)
    for t in nice_ticks(xlim[0], xlim[1]):
        if not (xlim[0] <= t <= xlim[1]):
            continue
        gx = int(x0 + (t - xlim[0]) / ((xlim[1] - xlim[0]) or 1) * (x1 - x0))
        draw.line([(gx, y1), (gx, y1 + 4)], fill=(0, 0, 0), width=1)
        draw.text((gx - 10, y1 + 5), fmt_tick(t), fill=(0, 0, 0), font=font)
    for t in nice_ticks(ylim[0], ylim[1]):
        if not (ylim[0] <= t <= ylim[1]):
            continue
        gy = int(y1 - (t - ylim[0]) / ((ylim[1] - ylim[0]) or 1) * (y1 - y0))
        draw.line([(x0 - 4, gy), (x0, gy)], fill=(0, 0, 0), width=1)
        draw.text((max(2, x0 - 38), gy - 7), fmt_tick(t), fill=(0, 0, 0), font=font)

    # xlabel / ylabel / title
    if ax._xlabel:
        draw.text(((x0 + x1) // 2 - 20, y1 + 22), ax._xlabel, fill=(0, 0, 0), font=_font(int(rcParams["axes.labelsize"])))
    if ax._ylabel:
        draw.text((max(2, x0 - 55), (y0 + y1) // 2 - 10), ax._ylabel, fill=(0, 0, 0), font=_font(int(rcParams["axes.labelsize"])))
    if ax._title:
        draw.text(((x0 + x1) // 2 - len(ax._title) * 3, max(2, y0 - 22)), ax._title, fill=(0, 0, 0),
                  font=_font(int(rcParams["axes.titlesize"])))

    # legend (fast, top-right)
    labels = [(ln.get("label"), _parse_color(ln["color"])) for ln in ax.lines if ln.get("label")]
    labels += [(s.get("label"), _parse_color(s["colors"]) if isinstance(s["colors"], tuple) else (31, 119, 180))
               for s in ax.scatters if s.get("label")]
    if labels:
        lx1, ly1 = x1 - 8, y0 + 8
        lw_box, lh = 110, 18 * len(labels) + 10
        draw.rectangle([lx1 - lw_box, ly1, lx1, ly1 + lh], fill=(255, 255, 255), outline=(150, 150, 150))
        for i, (lb, cc) in enumerate(labels):
            yy = ly1 + 8 + i * 18
            draw.line([(lx1 - lw_box + 8, yy), (lx1 - lw_box + 28, yy)], fill=cc, width=2)
            draw.text((lx1 - lw_box + 32, yy - 8), str(lb)[:16], fill=(0, 0, 0), font=_font(9))


def render_to_image(fig) -> Image.Image:
    W, H = fig.pixel_size
    try:
        bg = to_rgb(fig.facecolor)
    except Exception:
        bg = (255, 255, 255)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img, "RGBA")
    if not fig.axes:
        from ..figure import Axes as _A
        fig.axes.append(_A(fig))
    for ax in fig.axes:
        _draw_axes(img, draw, ax, W, H)
    if getattr(fig, "_suptitle", ""):
        draw.text((W // 2 - len(fig._suptitle) * 4, 6), fig._suptitle, fill=(0, 0, 0),
                  font=_font(int(rcParams["axes.titlesize"]) + 1))
    return img


def render_figure(fig, fname=None, dpi=None, show=False):
    img = render_to_image(fig)
    if fname is None or show:
        try:
            img.show()
        except Exception:
            # headless: save to memory / show via matplotlib if present
            try:
                import matplotlib.pyplot as _plt
                import numpy as _np
                _plt.figure(figsize=fig.figsize, dpi=fig.dpi)
                _plt.imshow(_np.asarray(img))
                _plt.axis("off")
                _plt.show()
            except Exception:
                pass
        if fname is None:
            return img
    fn = str(fname)
    lo = fn.lower()
    if lo.endswith((".pdf", ".svg", ".eps")):
        img.save(fn)
    else:
        img.save(fn, dpi=(dpi or fig.dpi, dpi or fig.dpi))
    return img
