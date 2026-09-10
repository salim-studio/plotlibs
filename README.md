# plotlib — أسرع بديل متوافق مع matplotlib

`plotlib` نفس كود `matplotlib` تقريباً، لكن أسرع بكثير (≈4x في الرسم الكبير).

```python
import plotlib as pl
pl.plot([1,2,3],[1,4,9], label="x^2")
pl.xlabel("x"); pl.ylabel("y"); pl.legend()
pl.savefig("out.png")   # أو pl.show()
```

## لماذا أسرع؟
1. **Min-max decimation**: أي خط > 2000 نقطة يُختزل تلقائياً مع الحفاظ على القمم (O(n) بـ numpy).
2. **Pillow rasterizer**: رسم C-level مباشر بدون محرك layout الثقيل في matplotlib.
3. **Vectorized pipeline**: تحويل data->pixel دفعة واحدة، ticks/limits بـ numpy.
4. **Caching**: خطوط + layout تُحسب مرة واحدة.

## التوافق مع matplotlib
| matplotlib | plotlib |
|---|---|
| `plt.plot / scatter / bar / hist / imshow / pie / fill_between / step / errorbar` | ✅ نفس التوقيع |
| `xlabel / ylabel / title / legend / grid / xlim / ylim / savefig / show / subplots` | ✅ |
| `fig.add_subplot / savefig` | ✅ |
| `style.use / rcParams` | ✅ subset |

## بنية المشروع
```
plotlib/            # repo root (standalone)
  plotlib/          # package
    __init__.py  pyplot.py  figure.py  colors.py  fast.py  style.py
    backends/renderer.py
  tests/
  benchmarks/
  examples/
```

## تشغيل
```bash
pip install numpy pillow
pip install -e .
python -m pytest tests -q
python benchmarks/bench.py
python examples/demo.py
```

## Benchmark (n=200k)
```
plotlib   : ~370 ms/plot
matplotlib: ~1560 ms/plot
speedup   : ~4.2x faster
```

License: MIT
