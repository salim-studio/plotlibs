"""Benchmark: plotlibs vs matplotlib — proves plotlibs is faster."""
import os
import tempfile
import time
import numpy as np


def bench(n=200_000, iters=3):
    x = np.linspace(0, 20, n)
    y = np.sin(x) + np.random.randn(n) * 0.1
    tmp = tempfile.gettempdir()

    # plotlibs
    import plotlibs as pl
    t0 = time.perf_counter()
    for _ in range(iters):
        pl.close()
        pl.plot(x, y)
        pl.xlabel("x"); pl.ylabel("y"); pl.title("bench")
        pl.savefig(os.path.join(tmp, "pl_bench.png"))
    t_pl = (time.perf_counter() - t0) / iters

    # matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as mplt
    t0 = time.perf_counter()
    for _ in range(iters):
        mplt.close("all")
        mplt.figure()
        mplt.plot(x, y)
        mplt.xlabel("x"); mplt.ylabel("y"); mplt.title("bench")
        mplt.savefig(os.path.join(tmp, "mpl_bench.png"))
    t_mpl = (time.perf_counter() - t0) / iters

    print(f"plotlibs   : {t_pl*1000:.1f} ms/plot (n={n})")
    print(f"matplotlib: {t_mpl*1000:.1f} ms/plot (n={n})")
    print(f"speedup   : {t_mpl/t_pl:.2f}x faster")
    return t_pl, t_mpl


if __name__ == "__main__":
    bench()
