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

def query_db(sql, params=None):
    """Run a query, return a DataFrame. Opens/closes a connection per call."""
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(sql, con, params=params)