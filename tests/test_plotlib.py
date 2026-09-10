import numpy as np


def test_plot_save(tmp_path):
    import plotlib as pl
    pl.close()
    pl.plot([1, 2, 3], [1, 4, 9], label="x^2")
    pl.xlabel("x"); pl.ylabel("y"); pl.legend(); pl.grid(True)
    f = tmp_path / "t.png"
    pl.savefig(str(f))
    assert f.exists() and f.stat().st_size > 1000


def test_subplots_scatter_bar_hist():
    import plotlib as pl
    pl.close()
    fig, axs = pl.subplots(2, 2)
    axs[0, 0].plot(np.arange(10), np.arange(10) ** 2)
    axs[0, 1].scatter(np.random.rand(50), np.random.rand(50))
    axs[1, 0].bar(["a", "b"], [3, 7])
    axs[1, 1].hist(np.random.randn(500), bins=20)
    img = fig.canvas_draw()
    assert img.size[0] > 100


def test_big_data_decimation():
    from plotlib.fast import decimate
    x = np.linspace(0, 10, 200_000)
    y = np.sin(x)
    xd, yd = decimate(x, y)
    assert len(xd) <= 2000
    # peaks preserved
    assert yd.max() > 0.99


def test_matplotlib_compat_signatures():
    import plotlib as pl
    pl.close()
    pl.plot([1, 2], [3, 4], "ro--", label="fmt")
    pl.scatter([1, 2], [3, 4], c="red")
    pl.bar([1, 2], [3, 4])
    pl.hist([1, 2, 2, 3], bins=3)
    pl.imshow([[1, 2], [3, 4]])
    pl.fill_between([1, 2], [1, 1], [2, 3])
    pl.axhline(1); pl.axvline(1)
    pl.xlim(0, 5); pl.ylim(0, 5)
    assert pl.gca().get_xlim() == (0, 5)
