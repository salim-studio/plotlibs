# Changelog

## 0.3.1 — Fix PyPI banner/logo images
- README banner/logo now use absolute `raw.githubusercontent.com` URLs (relative `assets/...` paths don't render on PyPI).
- Doc links (`assets/`, `BRANDING.md`, `LICENSE`) now absolute GitHub URLs for PyPI compatibility.
- No code changes.

## 0.3.0 — Rebrand to plotlibs
- **Renamed** package `plotlib` → `plotlibs`; new home https://github.com/salim-studio/plotlibs.
- Backward-compatible shim `import plotlib` (DeprecationWarning, removed in 1.0).
- **English-first** README + new visual identity: logo/banner in `assets/`, brand palette (`#4F46E5`/`#06B6D4`/`#0F172A`), signature themes `plotlibs` / `plotlibs-dark`, brand default color cycle.
- Docs: `BRANDING.md`, `LICENSE` (MIT), CI workflow.
- No breaking API changes besides the import name.

## 0.2.0
- Data/DB/EDA/ML expansion: `data`, `db`, `eda`, `ml` modules; `boxplot/violin/kde/heatmap/corr/count/area/hist2d/stem`; `quick_eda`; `read_sql/to_sql/plot_from_sql`; `plot_history/confusion/ROC/PR/importance`.

## 0.1.0
- Initial fast matplotlib-compatible core (Pillow + numpy, min-max decimation).
