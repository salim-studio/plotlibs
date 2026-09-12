"""plotlibs.colors — fast matplotlib-compatible color parsing."""
from __future__ import annotations

TABLEAU = {
    "C0": (31, 119, 180),
    "C1": (255, 127, 14),
    "C2": (44, 160, 44),
    "C3": (214, 39, 40),
    "C4": (148, 103, 189),
    "C5": (140, 86, 75),
    "C6": (227, 119, 194),
    "C7": (127, 127, 127),
    "C8": (188, 189, 34),
    "C9": (23, 190, 207),
}

NAMED = {
    "b": (0, 0, 255), "g": (0, 128, 0), "r": (255, 0, 0),
    "c": (0, 191, 191), "m": (191, 0, 191), "y": (191, 191, 0),
    "k": (0, 0, 0), "w": (255, 255, 255),
    "blue": (0, 0, 255), "green": (0, 128, 0), "red": (255, 0, 0),
    "cyan": (0, 191, 191), "magenta": (191, 0, 191), "yellow": (255, 255, 0),
    "black": (0, 0, 0), "white": (255, 255, 255),
    "gray": (128, 128, 128), "grey": (128, 128, 128),
    "orange": (255, 165, 0), "purple": (128, 0, 128),
    "brown": (165, 42, 42), "pink": (255, 192, 203),
    "lime": (0, 255, 0), "navy": (0, 0, 128), "teal": (0, 128, 128),
    "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211),
}

BRAND = {
    # plotlibs visual identity v1 (see assets/logo.svg)
    "primary": (79, 70, 229),      # #4F46E5 indigo
    "accent": (6, 182, 212),       # #06B6D4 cyan
    "ink": (15, 23, 42),           # #0F172A
    "paper": (248, 250, 252),      # #F8FAFC
    "lime": (163, 230, 53),        # #A3E635
    "amber": (250, 204, 21),       # #FACC15
    "coral": (251, 113, 133),      # #FB7185
    "violet": (167, 139, 250),     # #A78BFA
    "sky": (56, 189, 248),         # #38BDF8
    "slate": (100, 116, 139),      # #64748B
}

CYCLE = [
    BRAND["primary"],
    BRAND["accent"],
    BRAND["coral"],
    (34, 197, 94),
    BRAND["amber"],
    BRAND["violet"],
    BRAND["sky"],
    BRAND["slate"],
    (244, 114, 182),
    (45, 212, 191),
]


def to_rgb(color) -> tuple[int, int, int]:
    """Parse any matplotlib-like color spec to (r, g, b) 0-255."""
    if color is None:
        return BRAND["primary"]
    if isinstance(color, (tuple, list)):
        vals = list(color)
        if len(vals) in (3, 4):
            # float 0-1 or int 0-255 ?
            if all(isinstance(v, float) or (isinstance(v, int) and False) for v in vals):
                pass
            mx = max(vals[:3])
            if mx <= 1.0 and any(isinstance(v, float) for v in vals):
                return tuple(int(max(0, min(1, v)) * 255) for v in vals[:3])
            return tuple(int(max(0, min(255, v))) for v in vals[:3])
        raise ValueError(f"Bad color tuple {color!r}")
    if not isinstance(color, str):
        raise ValueError(f"Bad color {color!r}")
    c = color.strip()
    if c in TABLEAU:
        return TABLEAU[c]
    if c in NAMED:
        return NAMED[c]
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) == 6:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        if len(h) == 8:  # #rrggbbaa -> ignore alpha
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    # gray string like "0.5"
    try:
        f = float(c)
        v = int(max(0.0, min(1.0, f)) * 255)
        return (v, v, v)
    except ValueError:
        pass
    raise ValueError(f"Unknown color {color!r}")


def to_rgba(color, alpha: float | None = None) -> tuple[int, int, int, int]:
    r, g, b = to_rgb(color)
    a = 255
    if isinstance(color, (tuple, list)) and len(color) == 4:
        v = color[3]
        a = int(v * 255) if isinstance(v, float) and v <= 1.0 else int(v)
    if alpha is not None:
        a = int(max(0.0, min(1.0, float(alpha))) * 255)
    return (r, g, b, a)
