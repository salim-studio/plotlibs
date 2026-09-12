"""plotlibs.db — مهام قواعد البيانات في سطرين.

    import plotlibs as pl
    conn = pl.db_connect("sales.db")
    df = pl.read_sql("SELECT * FROM sales LIMIT 1000", conn)
    pl.quick_eda(df)

يدعم:
- sqlite3 من المكتبة القياسية (بدون أي تنصيب)
- sqlalchemy / duckdb إن وُجدت (URLs مثل sqlite:///x.db أو duckdb:///x.ddb)
- كتابة DataFrame إلى جدول + إنشاء جداول تجريبية + رسم مباشر من SQL
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def _is_duckdb_url(url: str) -> bool:
    return url.startswith("duckdb:")


def _is_sqla_url(url: str) -> bool:
    return "://" in url and not _is_duckdb_url(url) and url.split("://")[0] not in ("sqlite",)


def connect(url_or_path: str = ":memory:"):
    """اتصال ذكي: مسار sqlite أو :memory: أو URL.

    يعيد كائن اتصال sqlite3، أو sqlalchemy Connection، أو duckdb connection.
    """
    s = str(url_or_path)
    if _is_duckdb_url(s):
        try:
            import duckdb  # type: ignore
        except ImportError as e:
            raise ImportError("duckdb not installed: pip install duckdb") from e
        path = s.replace("duckdb://", "").replace("duckdb:", "") or ":memory:"
        return duckdb.connect(path if path else ":memory:")
    if s.startswith("sqlite:///"):
        return sqlite3.connect(s.replace("sqlite:///", ""))
    if "://" in s:
        # sqlalchemy generic (postgres/mysql/...)
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as e:
            raise ImportError("sqlalchemy not installed: pip install sqlalchemy") from e
        eng = create_engine(s)
        return eng.connect()
    # مسار ملف أو :memory:
    return sqlite3.connect(s)


db_connect = connect


def read_sql(query: str, conn):
    """نفّذ SELECT وأعد DataFrame (أو dict إن لم توجد pandas)."""
    # duckdb
    try:
        import duckdb  # type: ignore
        if isinstance(conn, duckdb.DuckDBPyConnection):
            return conn.execute(query).fetchdf()
    except Exception:
        pass
    # sqlalchemy
    try:
        from sqlalchemy.engine import Connection as _SAConn  # type: ignore
        if isinstance(conn, _SAConn):
            import pandas as pd
            return pd.read_sql(query, conn)
    except Exception:
        pass
    # sqlite3
    try:
        import pandas as pd
        return pd.read_sql_query(query, conn)
    except ImportError:
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        import numpy as np
        out: dict = {}
        for i, c in enumerate(cols):
            col = [r[i] for r in rows]
            try:
                out[c] = np.array(col, dtype=float)
            except Exception:
                out[c] = np.array(col)
        return out


def to_sql(data, table: str, conn, if_exists: str = "replace"):
    """اكتب DataFrame/dict إلى جدول SQL."""
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            # sqlalchemy conn أم sqlite3؟
            try:
                from sqlalchemy.engine import Connection as _SAConn  # type: ignore
                if isinstance(conn, _SAConn):
                    data.to_sql(table, conn, if_exists=if_exists, index=False)
                    return table
            except Exception:
                pass
            data.to_sql(table, conn, if_exists=if_exists, index=False)
            return table
    except ImportError:
        pass
    # dict fallback عبر sqlite3
    import numpy as np
    if isinstance(data, dict) and hasattr(conn, "execute"):
        keys = list(data.keys())
        n = max(len(np.asarray(data[k]).ravel()) for k in keys)
        if if_exists == "replace":
            try:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            except Exception:
                pass
        coldefs = ", ".join(f'"{k}" TEXT' for k in keys)
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({coldefs})')
        for i in range(n):
            vals = []
            for k in keys:
                col = np.asarray(data[k]).ravel()
                vals.append(str(col[i]) if i < len(col) else None)
            conn.execute(
                f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in vals)})', vals
            )
        conn.commit()
        return table
    raise ValueError("unsupported to_sql input")


def list_tables(conn) -> list[str]:
    try:
        import duckdb  # type: ignore
        if isinstance(conn, duckdb.DuckDBPyConnection):
            rows = conn.execute("SHOW TABLES").fetchall()
            return [r[0] for r in rows]
    except Exception:
        pass
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [r[0] for r in cur.fetchall()]
    except Exception:
        pass
    try:
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def demo_db(path: str = ":memory:"):
    """أنشئ قاعدة تجريبية sales(region, amount, qty) للتعليم والعروض."""
    import numpy as np
    conn = connect(path)
    rng = np.random.default_rng(7)
    regions = rng.choice(["Cairo", "Oran", "Riyadh", "Dubai"], size=300)
    amounts = rng.gamma(2.0, 150.0, size=300).round(2)
    qty = rng.integers(1, 20, size=300)
    try:
        import pandas as pd
        df = pd.DataFrame({"region": regions, "amount": amounts, "qty": qty})
    except ImportError:
        df = {"region": regions, "amount": amounts, "qty": qty}
    to_sql(df, "sales", conn, if_exists="replace")
    return conn


def plot_from_sql(query: str, conn, kind: str = "line", **kw):
    """ارسم مباشرة من استعلام SQL: SELECT x, y FROM t."""
    from . import pyplot as plt
    from .data import numeric_columns, column_values, columns_of

    df = read_sql(query, conn)
    cols = columns_of(df)
    nums = numeric_columns(df)
    if len(nums) >= 2:
        x = column_values(df, nums[0])
        y = column_values(df, nums[1])
        if kind == "bar":
            plt.bar(x, y, **kw)
        elif kind == "scatter":
            plt.scatter(x, y, **kw)
        else:
            plt.plot(x, y, **kw)
        plt.xlabel(nums[0])
        plt.ylabel(nums[1])
        plt.title(query[:60])
    elif cols:
        # تجميع فئوي: SELECT region, SUM(amount)
        try:
            import pandas as _pd  # noqa
            vals = df
        except Exception:
            vals = df
        _ = vals
    return df
