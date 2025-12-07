"""
src/utils.py

Utility functions: feature engineering, common checks, SQL loader skeleton.
"""

import pandas as pd
from typing import List

def ensure_datetime(df: pd.DataFrame, col_candidates: List[str]) -> str:
    """
    Given df and a list of possible column names, find a datetime column and convert it.
    Returns the selected column name or raises.
    """
    for c in col_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return c
    raise ValueError("No datetime column found among candidates: " + ", ".join(col_candidates))

def top_n_players_by_wager(transactions: pd.DataFrame, player_col: str, amount_col: str, n: int = 20):
    agg = transactions.groupby(player_col)[amount_col].sum().reset_index().sort_values(amount_col, ascending=False).head(n)
    return agg
