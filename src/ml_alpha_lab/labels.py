"""Event sampling, triple-barrier labels and sample-uniqueness weights (AFML ch. 3–4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarrierParams:
    horizon: int = 10        # vertical barrier, trading days
    pt: float = 1.0          # profit-take width in units of sigma_daily * sqrt(horizon)
    sl: float = 1.0          # stop-loss width, same units
    vol_window: int = 63     # look-back for sigma_daily (rolling std of log returns)


def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar week."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W-FRI")).max().to_numpy())


def triple_barrier_path(logp: np.ndarray, i0: int, upper: float, lower: float, horizon: int) -> tuple[int, int]:
    """First barrier touched on one path. Returns (i1, barrier) with barrier +1 / -1 / 0 (vertical)."""
    end = min(i0 + horizon, len(logp) - 1)
    path = logp[i0 + 1:end + 1] - logp[i0]
    hit_u = np.flatnonzero(path >= upper)
    hit_l = np.flatnonzero(path <= -lower)
    ju = hit_u[0] if hit_u.size else np.inf
    jl = hit_l[0] if hit_l.size else np.inf
    if ju == jl == np.inf:
        return end, 0
    return (i0 + 1 + int(ju), 1) if ju < jl else (i0 + 1 + int(jl), -1)


def triple_barrier(px: pd.DataFrame, event_dates: pd.DatetimeIndex, p: BarrierParams = BarrierParams()) -> pd.DataFrame:
    """Triple-barrier labels for every (event date, ticker) with a complete label window.

    Barriers are ±width × sigma_daily(t0) × sqrt(horizon), where sigma_daily uses data up to t0.
    ``ret`` is the simple return from t0 to the first touch, ``bin`` = 1 if ret > 0.
    ``fwd_ret`` is the next-week return from t0+1 to t0+6 (used to score weekly portfolios).
    """
    lp_all = np.log(px)
    sig_all = lp_all.diff().rolling(p.vol_window, min_periods=p.vol_window // 2).std()
    pos = px.index.get_indexer(event_dates)
    pos = pos[(pos >= 0) & (pos + p.horizon < len(px))]
    H = p.horizon
    out = []
    for t in px.columns:
        lp = lp_all[t].to_numpy()
        sig = sig_all[t].to_numpy()
        ok = pos[np.isfinite(sig[pos]) & np.isfinite(lp[pos]) & np.isfinite(lp[np.minimum(pos + H, len(lp) - 1)])]
        if not ok.size:
            continue
        W = np.lib.stride_tricks.sliding_window_view(lp, H + 1)[ok]
        path = W[:, 1:] - W[:, [0]]
        width = sig[ok] * np.sqrt(H)
        up = path >= (p.pt * width)[:, None]
        dn = path <= -(p.sl * width)[:, None]
        ju = np.where(up.any(axis=1), up.argmax(1), H)
        jl = np.where(dn.any(axis=1), dn.argmax(1), H)
        j = np.minimum(np.minimum(ju, jl), H - 1)
        barrier = np.where((ju < jl) & (ju < H), 1, np.where((jl < ju) & (jl < H), -1, 0))
        i1 = ok + 1 + j
        r = np.exp(lp[i1] - lp[ok]) - 1
        f_end = np.minimum(ok + 6, len(lp) - 1)
        fwd = np.where(ok + 6 < len(lp), np.exp(lp[f_end] - lp[np.minimum(ok + 1, len(lp) - 1)]) - 1, np.nan)
        out.append(pd.DataFrame({"t0": px.index[ok], "ticker": t, "t1": px.index[i1], "i0": ok, "i1": i1,
                                 "trgt": width, "ret": r, "barrier": barrier, "bin": (r > 0).astype(int), "fwd_ret": fwd}))
    ev = pd.concat(out, ignore_index=True).sort_values(["t0", "ticker"])
    return ev.set_index(["t0", "ticker"])


# ----------------------------------------------------------------------------- uniqueness
def concurrency(i0: np.ndarray, i1: np.ndarray, n: int | None = None) -> np.ndarray:
    """Number of labels alive on each bar; a label spans bars i0..i1 inclusive."""
    i0, i1 = np.asarray(i0, int), np.asarray(i1, int)
    n = int(i1.max()) + 1 if n is None else n
    c = np.zeros(n + 1)
    np.add.at(c, i0, 1)
    np.add.at(c, i1 + 1, -1)
    return np.cumsum(c)[:n]


def average_uniqueness(i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    """Average over each label's life of 1 / concurrency (AFML snippet 4.2)."""
    i0, i1 = np.asarray(i0, int), np.asarray(i1, int)
    if not len(i0):
        return np.array([])
    c = concurrency(i0, i1)
    inv = np.concatenate([[0.0], np.cumsum(1.0 / np.where(c > 0, c, np.inf))])
    return (inv[i1 + 1] - inv[i0]) / (i1 - i0 + 1)


def uniqueness_weights(events: pd.DataFrame) -> pd.Series:
    """Per-ticker average uniqueness (labels of the same stock overlap in time)."""
    u = pd.Series(np.nan, index=events.index)
    for _, g in events.groupby(level="ticker"):
        u.loc[g.index] = average_uniqueness(g["i0"].to_numpy(), g["i1"].to_numpy())
    return u.rename("uniqueness")
