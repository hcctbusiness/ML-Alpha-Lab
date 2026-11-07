"""Price loading on the US trading calendar (Yahoo Finance via ``quantcore.data``)."""
from __future__ import annotations

import pandas as pd

import quantcore as qc

from .universe import MARKET, VIX


def load(tickers: list[str], start: str = "2009-01-01") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (stock closes, SPY close, VIX level) on SPY's trading days, gaps filled ≤ 5 bars."""
    px = qc.data.prices(list(dict.fromkeys(tickers + [MARKET, VIX])), start=start)
    cal = px[MARKET].dropna().index
    px = px.reindex(cal).ffill(limit=5)
    stocks = px[[t for t in tickers if t in px.columns and px[t].notna().sum() > 500]]
    return stocks, px[MARKET], px[VIX]
