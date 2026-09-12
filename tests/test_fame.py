"""Fame tests: DataFrames + SQL + EDA + ML visuals."""
import numpy as np


def test_data_describe_corr():
    from plotlibs import data
    rng = np.random.default_rng(0)
    df = {"a": rng.normal(size=100), "b": rng.normal(size=100)}
    d = data.describe(df)
    assert "a" in d and abs(d["a"]["mean"]) < 0.5
    c, cols = data.corr_matrix(df)
    assert c.shape == (2, 2) and cols == ["a", "b"]


def test_db_sqlite_memory():
    import plotlibs as pl
    conn = pl.db_connect(":memory:")
    import pandas as pd
    df = pd.DataFrame({"region": ["A", "B", "A"], "amount": [10.0, 20.0, 30.0]})
    pl.to_sql(df, "sales", conn)
    assert "sales" in pl.db.list_tables(conn)
    out = pl.read_sql("SELECT region, amount FROM sales ORDER BY amount", conn)
    assert len(out) == 3


def test_stats_plots_save(tmp_path):
    import plotlibs as pl
    rng = np.random.default_rng(1)
    pl.close()
    pl.boxplot([rng.normal(size=80), rng.normal(1, size=80)], labels=["x", "y"])
    pl.savefig(str(tmp_path / "box.png"))
    pl.close()
    pl.violinplot([rng.normal(size=80), rng.normal(size=80)])
    pl.savefig(str(tmp_path / "violin.png"))
    pl.close()
    pl.kde(rng.normal(size=300), label="kde")
    pl.savefig(str(tmp_path / "kde.png"))
    pl.close()
    pl.heatmap(np.eye(4), annot=True)
    pl.savefig(str(tmp_path / "hm.png"))
    pl.close()
    pl.countplot(["a", "b", "a", "c", "a"])
    pl.savefig(str(tmp_path / "count.png"))
    pl.close()
    pl.area(np.arange(10), rng.random(10), rng.random(10), labels=["u", "v"])
    pl.savefig(str(tmp_path / "area.png"))
    pl.close()
    pl.hist2d(rng.normal(size=500), rng.normal(size=500))
    pl.savefig(str(tmp_path / "h2.png"))
    pl.close()
    pl.stem([1, 2, 3], [2, 1, 3])
    pl.savefig(str(tmp_path / "stem.png"))
    for f in ["box.png", "violin.png", "kde.png", "hm.png", "count.png", "area.png", "h2.png", "stem.png"]:
        assert (tmp_path / f).stat().st_size > 500


def test_corr_and_eda(tmp_path):
    import plotlibs as pl
    import pandas as pd
    df = pd.DataFrame({"a": np.random.randn(60), "b": np.random.randn(60), "c": np.random.randn(60)})
    pl.close()
    pl.corr(df, annot=True)
    pl.savefig(str(tmp_path / "corr.png"))
    pl.close()
    fig = pl.quick_eda(df)
    fig.savefig(str(tmp_path / "eda.png"))
    assert (tmp_path / "eda.png").stat().st_size > 1000


def test_ml_plots():
    import plotlibs as pl
    pl.close()
    fpr, tpr, auc = pl.plot_roc([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert 0.5 <= auc <= 1.0
    pl.close()
    cm = pl.plot_confusion_matrix([0, 1, 1, 0], [0, 0, 1, 0])
    assert cm.shape == (2, 2)
    pl.close()
    pl.plot_history({"loss": [0.9, 0.5, 0.3], "val_loss": [1.0, 0.6, 0.4]})
    pl.close()
    pl.plot_feature_importance(["a", "b", "c"], [0.2, 0.5, 0.3])
    pl.close()
