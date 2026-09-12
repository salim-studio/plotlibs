"""plotlibs.backends.renderer — Pillow-based fast raster renderer."""
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

    # images (imshow) first + heatmaps (جديد: corr/heatmap/hist2d)
    for im in list(ax.images) + [dict(data=h["data"], cmap=h.get("cmap", "viridis"),
                                      vmin=h.get("vmin"), vmax=h.get("vmax"),
                                      origin="upper", extent=h.get("extent"))
                                 for h in getattr(ax, "heatmaps", [])]:
        data = im["data"]
        try:
            if np.asarray(data).ndim == 2:
                data = np.asarray(data, dtype=float)
                vmin = im["vmin"] if im["vmin"] is not None else float(np.nanmin(data))
                vmax = im["vmax"] if im["vmax"] is not None else float(np.nanmax(data))
                norm = (data - vmin) / ((vmax - vmin) or 1.0)
                norm = np.clip(norm, 0, 1)
                cmap = im.get("cmap", "viridis")
                hh, ww = norm.shape
                rgb = np.empty((hh, ww, 3), dtype=np.uint8)
                if cmap in ("gray", "grey", "binary", "gist_yarg", "Greys"):
                    v = (norm * 255).astype(np.uint8)
                    rgb[..., 0] = v; rgb[..., 1] = v; rgb[..., 2] = v
                elif cmap in ("hot",):
                    rgb[..., 0] = (norm * 255).astype(np.uint8)
                    rgb[..., 1] = ((norm ** 2) * 255).astype(np.uint8)
                    rgb[..., 2] = ((norm ** 3) * 128).astype(np.uint8)
                elif cmap in ("cool", "jet", "hsv", "rainbow", "plasma", "inferno", "magma"):
                    t = norm
                    rgb[..., 0] = (255 * np.clip(1.5 - np.abs(4 * t - 3), 0, 1)).astype(np.uint8)
                    rgb[..., 1] = (255 * np.clip(1.5 - np.abs(4 * t - 2), 0, 1)).astype(np.uint8)
                    rgb[..., 2] = (255 * np.clip(1.5 - np.abs(4 * t - 1), 0, 1)).astype(np.uint8)
                elif cmap in ("RdBu", "RdBu_r", "coolwarm", "bwr", "seismic"):
                    # أحمر-أزرق للارتباط: -1 أحمر ... +1 أزرق
                    t = np.clip(norm, 0, 1)
                    rgb[..., 0] = (255 * (1 - t)).astype(np.uint8)
                    rgb[..., 1] = (255 * (1 - np.abs(2 * t - 1))).astype(np.uint8)
                    rgb[..., 2] = (255 * t).astype(np.uint8)
                else:  # viridis approx
                    rgb[..., 0] = (68 + norm * 185).astype(np.uint8)
                    rgb[..., 1] = (1 + norm * 230).astype(np.uint8)
                    rgb[..., 2] = (84 - norm * 47).astype(np.uint8)
                pil = Image.fromarray(rgb, "RGB")
            else:
                a = np.asarray(data)
                if a.dtype != np.uint8:
                    a = np.clip(a, 0, 255).astype(np.uint8) if np.nanmax(a) > 1 else (a * 255).astype(np.uint8)
                pil = Image.fromarray(a)
            if im.get("origin", "upper") == "upper":
                pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
            pil = pil.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.BILINEAR)
            img.paste(pil, (x0, y0))
        except Exception:
            pass
    # heatmap annotations + labels
    for h in getattr(ax, "heatmaps", []):
        try:
            m = np.asarray(h["data"], dtype=float)
            nr, nc = m.shape
            if h.get("annot") and nr <= 14 and nc <= 14:
                for i in range(nr):
                    for j in range(nc):
                        # خلية (j,i) -> بكسل
                        cx = int(x0 + (j + 0.5) / nc * (x1 - x0))
                        cy = int(y0 + (i + 0.5) / nr * (y1 - y0))
                        draw.text((cx - 12, cy - 7), f"{m[nr - 1 - i, j]:.2f}",
                                  fill=(255, 255, 255), font=_font(8))
            if h.get("xticks") and nc == len(h["xticks"]):
                for j, lab in enumerate(h["xticks"]):
                    cx = int(x0 + (j + 0.5) / nc * (x1 - x0))
                    draw.text((cx - 12, y1 + 5), str(lab)[:10], fill=(0, 0, 0), font=_font(8))
            if h.get("yticks") and nr == len(h["yticks"]):
                for i, lab in enumerate(h["yticks"]):
                    cy = int(y0 + (i + 0.5) / nr * (y1 - y0))
                    draw.text((max(2, x0 - 52), cy - 7), str(lab)[:10], fill=(0, 0, 0), font=_font(8))
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

    # stacked area (جديد)
    for a in getattr(ax, "areas", []):
        try:
            xs = np.asarray(a["x"], dtype=float)
            px_base, _ = _data_to_px(xs, np.zeros_like(xs), xlim, ylim, box)
            cumul = np.zeros_like(xs, dtype=float)
            for j, yj in enumerate(a["ys"]):
                yj = np.asarray(yj, dtype=float)[: len(xs)]
                prev = cumul.copy()
                cumul = cumul + yj
                c = _parse_color(a["colors"][j] if j < len(a["colors"]) else (79, 70, 229))
                _, py_top = _data_to_px(xs, cumul, xlim, ylim, box)
                _, py_bot = _data_to_px(xs, prev, xlim, ylim, box)
                poly = list(zip(px_base.astype(int), py_top.astype(int))) + \
                    list(zip(px_base.astype(int)[::-1], py_bot.astype(int)[::-1]))
                if len(poly) >= 3:
                    draw.polygon(poly, fill=c)
                if len(px_base):
                    draw.line(list(zip(px_base.astype(int), py_top.astype(int))), fill=tuple(max(0, v - 50) for v in c), width=2)
        except Exception:
            pass

    # KDE curves (جديد)
    for k in getattr(ax, "kdes", []):
        try:
            c = _parse_color(k["color"])
            px, py = _data_to_px(k["x"], k["y"], xlim, ylim, box)
            px = np.clip(px, x0 - 50, x1 + 50)
            py = np.clip(py, y0 - 50, y1 + 50)
            if k.get("fill"):
                _, py0 = _data_to_px(k["x"], np.zeros_like(k["y"]), xlim, ylim, box)
                poly = list(zip(px.astype(int), py.astype(int))) + \
                    list(zip(px.astype(int)[::-1], py0.astype(int)[::-1]))
                if len(poly) >= 3:
                    draw.polygon(poly, fill=c + (90,) if len(c) == 3 else c)
            _draw_line(draw, px, py, c, 2.0, "-", None, 6)
        except Exception:
            pass

    # stems (جديد)
    for s in getattr(ax, "stems", []):
        try:
            c = _parse_color(s["color"])
            px, py = _data_to_px(s["x"], s["y"], xlim, ylim, box)
            _, py0 = _data_to_px(s["x"], np.zeros_like(s["y"]), xlim, ylim, box)
            for i in range(len(px)):
                draw.line([(int(px[i]), int(py0[i])), (int(px[i]), int(py[i]))], fill=c, width=1)
                draw.ellipse([int(px[i]) - 3, int(py[i]) - 3, int(px[i]) + 3, int(py[i]) + 3], fill=c)
        except Exception:
            pass

    # boxplots (جديد)
    for bg in getattr(ax, "boxes", []):
        try:
            from ..fast import box_stats
            n = len(bg["cols"])
            base_c = _parse_color(bg["color"])
            for i, col in enumerate(bg["cols"]):
                st = box_stats(np.asarray(col, dtype=float))
                pos = float(i + 1)
                pxx, _ = _data_to_px([pos], [0], xlim, ylim, box)
                cx = int(pxx[0])
                _, pyy = _data_to_px([0, 0, 0, 0, 0],
                                     [st["lo"], st["q1"], st["med"], st["q3"], st["hi"]],
                                     xlim, ylim, box)
                y_lo, y_q1, y_med, y_q3, y_hi = [int(v) for v in pyy]
                wpx = max(14, int((x1 - x0) / max(n * 3, 1)))
                # whiskers
                draw.line([(cx, y_lo), (cx, y_hi)], fill=(60, 60, 60), width=1)
                draw.line([(cx - wpx // 3, y_lo), (cx + wpx // 3, y_lo)], fill=(60, 60, 60), width=1)
                draw.line([(cx - wpx // 3, y_hi), (cx + wpx // 3, y_hi)], fill=(60, 60, 60), width=1)
                # box q1-q3
                draw.rectangle([cx - wpx // 2, min(y_q1, y_q3), cx + wpx // 2, max(y_q1, y_q3)],
                               fill=base_c, outline=(20, 20, 20))
                draw.line([(cx - wpx // 2, y_med), (cx + wpx // 2, y_med)], fill=(255, 255, 255), width=2)
                # mean
                _, pymean = _data_to_px([0], [st["mean"]], xlim, ylim, box)
                draw.ellipse([cx - 3, int(pymean[0]) - 3, cx + 3, int(pymean[0]) + 3], fill=(0, 0, 0))
                # outliers
                if st["out"].size:
                    _, pyo = _data_to_px(np.zeros_like(st["out"]), st["out"], xlim, ylim, box)
                    for yo in pyo:
                        draw.ellipse([cx - 2, int(yo) - 2, cx + 2, int(yo) + 2], outline=(120, 120, 120))
                if bg.get("labels") and i < len(bg["labels"]):
                    draw.text((cx - 14, y1 + 5), str(bg["labels"][i])[:12], fill=(0, 0, 0), font=_font(8))
        except Exception:
            pass

    # violins (جديد: KDE عمودي مرسوم كمرآة)
    for vg in getattr(ax, "violins", []):
        try:
            from ..fast import gaussian_kde_1d
            n = len(vg["cols"])
            base_c = _parse_color(vg["color"])
            for i, col in enumerate(vg["cols"]):
                v = np.asarray(col, dtype=float)
                v = v[np.isfinite(v)]
                if v.size < 3:
                    continue
                pos = float(i + 1)
                pxx, _ = _data_to_px([pos], [0], xlim, ylim, box)
                cx = int(pxx[0])
                xs, ys = gaussian_kde_1d(v, points=int(vg.get("points", 120)))
                _, pyy = _data_to_px(np.zeros_like(ys), ys if False else xs, xlim, ylim, box)
                # ys هنا كثافة -> عرض أفقي
                mx = ys.max() or 1.0
                wpx = max(18, int((x1 - x0) / max(n * 2.5, 1)))
                half = (ys / mx * (wpx / 2)).astype(int)
                right = [(cx + int(h), int(p)) for h, p in zip(half, pyy)]
                left = [(cx - int(h), int(p)) for h, p in zip(half[::-1], pyy[::-1])]
                if len(right) + len(left) >= 6:
                    draw.polygon(right + left, fill=base_c, outline=(20, 20, 20))
                if vg.get("labels") and i < len(vg["labels"]):
                    draw.text((cx - 14, y1 + 5), str(vg["labels"][i])[:12], fill=(0, 0, 0), font=_font(8))
        except Exception:
            pass

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
            cc = base if single else (_parse_color(cols[i]) if i < len(cols) else (79, 70, 229))
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
            cc = _parse_color(cols[i]) if cols and i < len(cols) else [(79, 70, 229), (255, 127, 14), (44, 160, 44), (214, 39, 40)][i % 4]
            draw.pieslice([cx - rr, cy - rr, cx + rr, cy + rr], start=ang, end=ang + sweep, fill=cc,
                          outline=(255, 255, 255))
            ang += sweep

    # spines
    edge = _parse_color(rcParams.get("axes.edgecolor", "black"))
    draw.rectangle(box, outline=edge, width=1)

    # ticks + labels (مع دعم التسميات الفئوية من bar/count)
    fs = int(rcParams.get("xtick.labelsize", 9))
    font = _font(fs)
    cat_labels = getattr(ax, "_xticklabels", None)
    cat_pos = getattr(ax, "_xtickpos", None)
    if cat_labels is not None and cat_pos is not None and len(cat_labels):
        try:
            for p, lab in zip(np.asarray(cat_pos, dtype=float), cat_labels):
                if not (xlim[0] - 1 <= p <= xlim[1] + 1):
                    continue
                gx = int(x0 + (p - xlim[0]) / ((xlim[1] - xlim[0]) or 1) * (x1 - x0))
                draw.line([(gx, y1), (gx, y1 + 4)], fill=(0, 0, 0), width=1)
                draw.text((gx - 12, y1 + 5), str(lab)[:12], fill=(0, 0, 0), font=font)
        except Exception:
            pass
    else:
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
    labels += [(s.get("label"), _parse_color(s["colors"]) if isinstance(s["colors"], tuple) else (79, 70, 229))
               for s in ax.scatters if s.get("label")]
    labels += [(k.get("label"), _parse_color(k["color"])) for k in getattr(ax, "kdes", []) if k.get("label")]
    for a in getattr(ax, "areas", []):
        for lb, cc in zip(a.get("labels", []), a.get("colors", [])):
            if lb:
                labels.append((lb, _parse_color(cc)))
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
