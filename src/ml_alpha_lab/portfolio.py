"""Turn out-of-sample scores into weekly-rebalanced portfolios and backtest them with QuantCore."""
from __future__ import annotations

import numpy as np
import pandas as pd

import quantcore as qc

COSTS = qc.CostModel(commission_bps=1.0, slippage_bps=2.0, borrow_bps_annual=50.0)


def _wide(s: pd.Series) -> pd.DataFrame:
    return s.unstack("ticker")


def long_short_targets(score: pd.Series, q: float = 0.2) -> pd.DataFrame:
    """Dollar-neutral: +100% equally across the top quantile, −100% across the bottom one."""
    S = _wide(score)
    r = S.rank(axis=1, pct=True)
    top, bot = (r > 1 - q).astype(float), (r <= q).astype(float)
    return (top.div(top.sum(axis=1).replace(0, np.nan), axis=0) - bot.div(bot.sum(axis=1).replace(0, np.nan), axis=0)).fillna(0.0)


def long_only_targets(score: pd.Series, q: float = 0.2) -> pd.DataFrame:
    S = _wide(score)
    top = (S.rank(axis=1, pct=True) > 1 - q).astype(float)
    return top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def equal_weight_targets(score: pd.Series) -> pd.DataFrame:
    a = _wide(score).notna().astype(float)
    return a.div(a.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def signed_targets(size: pd.Series) -> pd.DataFrame:
    """Weights = signed size / number of names available that week (meta-labelling books)."""
    S = _wide(size)
    n = S.notna().sum(axis=1).replace(0, np.nan)
    return S.fillna(0.0).div(n, axis=0).fillna(0.0)


def weekly_spread(score: pd.Series, fwd_ret: pd.Series, q: float = 0.2) -> pd.Series:
    """Fast gross scoring: weekly top-minus-bottom-quantile next-week return (used inside CV)."""
    df = pd.DataFrame({"s": score, "r": fwd_ret}).dropna()
    r = df.groupby(level="t0")["s"].rank(pct=True)
    top = df["r"][r > 1 - q].groupby(level="t0").mean()
    bot = df["r"][r <= q].groupby(level="t0").mean()
    return (top - bot).dropna()


def drift(targets: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Daily weights for QuantCore (lag=1) that trade only on rebalance dates and drift in between.

    ``targets`` is indexed by rebalance dates (decided at that close). The engine holds
    weights[t-1] on day t, so weights[t] must equal the book held on t+1.
    """
    idx = prices.index
    cols = targets.columns
    rets = prices[cols].pct_change(fill_method=None).fillna(0.0).to_numpy()
    tgt = targets.reindex(idx).to_numpy()
    is_reb = targets.reindex(idx).notna().any(axis=1).to_numpy()
    held = np.zeros((len(idx), len(cols)))
    for t in range(1, len(idx)):
        if is_reb[t - 1]:
            held[t] = np.nan_to_num(tgt[t - 1])
        else:
            h, r = held[t - 1], rets[t - 1]
            g = 1 + h @ r
            held[t] = h * (1 + r) / g if g != 0 else h
    w = pd.DataFrame(held, index=idx, columns=cols).shift(-1)
    w.iloc[-1] = w.iloc[-2]
    return w.fillna(0.0)


def run(targets: pd.DataFrame, prices: pd.DataFrame, name: str, start: str | None = None,
        costs: qc.CostModel = COSTS) -> qc.BacktestResult:
    px = prices.loc[start:] if start else prices
    tg = targets.loc[px.index[0]:]
    w = drift(tg.reindex(columns=px.columns).fillna(0.0), px.ffill())
    # a fresh rebalance can only begin once the first target exists
    w.loc[: tg.index[0] - pd.Timedelta(days=1)] = 0.0
    return qc.backtest(w, px.ffill(), costs=costs, lag=1, name=name)
