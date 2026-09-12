"""plotlibs showcase: DataFrames + SQL + EDA + ML in one script (English)."""
import numpy as np
import plotlibs as pl

print(f"plotlibs {pl.__version__}")

# 1) بيانات + EDA بسطر واحد
rng = np.random.default_rng(42)
import pandas as pd
df = pd.DataFrame({
    "sales": rng.gamma(2.0, 150, size=500),
    "qty": rng.integers(1, 20, size=500),
    "profit": rng.normal(50, 15, size=500),
    "region": rng.choice(["Cairo", "Oran", "Riyadh", "Dubai"], size=500),
})
pl.close()
fig = pl.quick_eda(df, title="Sales EDA")
fig.savefig("demo_eda.png")
print("saved demo_eda.png — describe:", list(pl.describe(df).keys()))

# 2) قاعدة بيانات: SQL -> رسم مباشر
conn = pl.db_connect(":memory:")
pl.to_sql(df, "sales", conn)
df2 = pl.read_sql("SELECT qty, sales FROM sales LIMIT 300", conn)
pl.close()
pl.scatter(df2["qty"] if hasattr(df2, "__getitem__") else [1, 2], df2["sales"] if hasattr(df2, "__getitem__") else [1, 2])
pl.title("SQL -> scatter")
pl.savefig("demo_sql.png")
print("saved demo_sql.png — tables:", pl.db.list_tables(conn))

# 3) رسوم إحصائية للمحللين
pl.close()
pl.boxplot([df["sales"], df["profit"]], labels=["sales", "profit"])
pl.title("boxplot")
pl.savefig("demo_box.png")

pl.close()
pl.corr(df[["sales", "qty", "profit"]], annot=True)
pl.title("correlation")
pl.savefig("demo_corr.png")

pl.close()
pl.countplot(df["region"])
pl.title("sales by region")
pl.savefig("demo_count.png")

# 4) تعلم آلة / عميق
pl.close()
y_true = (rng.random(200) > 0.5).astype(int)
y_score = np.clip(y_true * 0.6 + rng.random(200) * 0.4, 0, 1)
pl.plot_roc(y_true, y_score, label="model")
pl.savefig("demo_roc.png")

pl.close()
pl.plot_confusion_matrix(y_true, (y_score > 0.5).astype(int))
pl.savefig("demo_cm.png")

pl.close()
pl.plot_history({"loss": [0.9, 0.6, 0.4, 0.3], "val_loss": [1.0, 0.7, 0.5, 0.45],
                 "accuracy": [0.6, 0.75, 0.82, 0.86]})
pl.savefig("demo_history.png")

print("all fame demos saved.")
