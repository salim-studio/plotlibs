# plotlibs — visual identity v1

## Brand
- **Name:** plotlibs (always lowercase in text/logo)
- **Tagline:** "Fast, beautiful Python plotting for everyone."
- **Voice:** fast, friendly, practical. English-first docs.

## Logo
- `assets/logo.svg` — 256×256 rounded square (`rx=56`), gradient indigo `#4F46E5` → cyan `#06B6D4`, white axes + chart line, lime/amber/white dots.
- `assets/banner.svg` — 1280×420 GitHub header: midnight background, logo + wordmark + pills ("4x faster", "matplotlib-compatible", "SQL • EDA • ML/DL") + mini chart.
- Clear space = height of the "p" dot. Minimum size: 24px.

## Palette (code + design must match)
| Token | Hex | RGB | Use |
|---|---|---|---|
| primary | `#4F46E5` | 79, 70, 229 | default line C0, links, logo start |
| accent | `#06B6D4` | 6, 182, 212 | second series C1, banner end |
| ink | `#0F172A` | 15, 23, 42 | text, dark theme bg |
| paper | `#F8FAFC` | 248, 250, 252 | light bg |
| lime | `#A3E635` | 163, 230, 53 | highlight |
| amber | `#FACC15` | 250, 204, 21 | highlight |
| coral | `#FB7185` | 251, 113, 133 | third series / errors |
| violet | `#A78BFA` | 167, 139, 250 | extra series |
| sky | `#38BDF8` | 56, 189, 248 | extra series |
| slate | `#64748B` | 100, 116, 139 | grid/secondary text |

`plotlibs/colors.py:BRAND` and `CYCLE` implement this order:
primary → accent → coral → green → amber → violet → sky → slate → pink → teal.

## Typography
- Display/wordmark: Inter 800, letter-spacing -2 (fallback: Segoe UI, Arial).
- Docs/code: Inter for prose, ui-monospace for code.

## Themes
- `pl.style.use("plotlibs")` — white bg, ink spines, grid on, linewidth 2.2.
- `pl.style.use("plotlibs-dark")` — ink bg `#0F172A`, axes `#1E293B`, white spines.
- Gallery styles kept for compatibility: `seaborn`, `plotly`, `ggplot`, `publication`.

## GitHub presentation (English)
- README starts with banner + logo + centered title/tagline + badges.
- Badges: version 0.3.0 (indigo), speed (cyan), deps (lime), license (amber), python (official).
- Sections: Why → Install → 30-second tour → Visual identity → Compatibility → Why faster → Layout → Migration → Roadmap → Contributing → License.
- Social preview: use `assets/banner.svg` (1280×420).

## Don'ts
- Don't stretch the logo; don't change the gradient angle; don't use the old name `plotlib` in new docs (only in the migration section).
