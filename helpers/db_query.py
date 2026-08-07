import os
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(
    os.environ.get(
        "FF_DB_PATH",
        r"C:\Users\terre\OneDrive\Desktop\Projects\Fantasy_Football_Database\data\fantasy_football.db",
    )
)

def _native(value):
    """Convert a numpy scalar to the Python type sqlite3 can actually bind.

    sqlite3 does not raise on a numpy int64 - it binds it as a value that
    matches nothing, so `WHERE season BETWEEN ? AND ?` with numpy bounds
    returns zero rows and every downstream join quietly fills with NaN. Since
    numpy ints are what `df["season"].min()` hands back, this is easy to hit
    and invisible when it happens.
    """
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def query_db(sql, params=None):
    """Run a query, return a DataFrame. Opens/closes a connection per call."""
    if params is not None:
        params = (
            {k: _native(v) for k, v in params.items()}
            if isinstance(params, dict)
            else tuple(_native(v) for v in params)
        )
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(sql, con, params=params)