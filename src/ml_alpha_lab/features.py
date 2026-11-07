"""Causal features for a daily stock panel, plus fixed-width-window fractional differencing.

Every feature at date t uses closes up to and including t only (rolling windows, no
centring, no future fill). ``tests/test_ml_alpha_lab.py`` checks this by perturbing future
prices. Features are stored as a long DataFrame indexed by (date, ticker).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

FAMILY = {
    "ret_5": "Returns", "ret_21": "Returns", "ret_63": "Returns", "ret_126": "Returns", "mom_12_1": "Returns",
    "vol_21": "Volatility", "vol_63": "Volatility", "vol_ratio": "Volatility",
    "z_ma20": "Mean reversion", "z_ma100": "Mean reversion", "rsi_14": "Mean reversion",
    "ffd_logp": "Frac. diff.",
    "xs_ret_21": "Cross-sectional rank", "xs_mom": "Cross-sectional rank", "xs_vol": "Cross-sectional rank",
    "spy_ret_21": "Market regime", "spy_trend": "Market regime", "vix": "Market regime", "vix_chg_21": "Market regime",
}
FEATURES = list(FAMILY)


# ----------------------------------------------------------------------------- FFD
def ffd_weights(d: float, thresh: float = 1e-4, max_len: int = 10_000) -> np.ndarray:
    """Fixed-width-window weights w_k = -w_{k-1} (d - k + 1) / k, truncated when |w_k| < thresh.

    Returned oldest-first so that ``np.dot(w, x[t-L+1 : t+1])`` is the FFD value at t.
    """
    w = [1.0]
    k = 1
    while k < max_len:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(x: pd.Series | pd.DataFrame, d: float, thresh: float = 1e-4) -> pd.Series | pd.DataFrame:
    """Fractionally differentiate each column with a fixed-width window (causal)."""
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda c: frac_diff_ffd(c, d, thresh))
    w = ffd_weights(d, thresh)
    L = len(w)
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= L:
        win = np.lib.stride_tricks.sliding_window_view(v, L)
        out[L - 1:] = win @ w  # NaN anywhere in the window gives NaN
    return pd.Series(out, index=x.index, name=x.name)


def adf_pvalue(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if len(x) < 100 or x.std() == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(adfuller(x.to_numpy(), maxlag=1, regression="c", autolag=None)[1])


def choose_d(logp: pd.DataFrame, grid=np.round(np.arange(0.0, 1.01, 0.1), 2), thresh: float = 1e-4,
             alpha: float = 0.05, sample: int = 30) -> tuple[float, pd.DataFrame]:
    """Smallest d whose FFD series passes ADF (median p < alpha) on the given (training) data."""
    cols = list(logp.columns[:: max(1, len(logp.columns) // sample)])
    rows = []
    for d in grid:
        ps = [adf_pvalue(frac_diff_ffd(logp[c], d, thresh)) for c in cols]
        ffd = frac_diff_ffd(logp[cols], d, thresh)
        corr = np.nanmedian([logp[c].corr(ffd[c]) for c in cols])
        rows.append({"d": d, "median ADF p": float(np.nanmedian(ps)), "share p<0.05": float(np.nanmean(np.array(ps) < alpha)),
                     "corr with log price": float(corr), "window": len(ffd_weights(d, thresh))})
    tab = pd.DataFrame(rows).set_index("d")
    ok = tab.index[tab["median ADF p"] < alpha]
    return (float(ok[0]) if len(ok) else 1.0), tab


# ----------------------------------------------------------------------------- panel features
def _rsi(px: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    dlt = px.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def daily_vol(px: pd.DataFrame, n: int = 63) -> pd.DataFrame:
    return np.log(px).diff().rolling(n, min_periods=n // 2).std()


def build(px: pd.DataFrame, spy: pd.Series, vix: pd.Series, d: float, thresh: float = 1e-4) -> pd.DataFrame:
    """Long (date, ticker) panel of all FEATURES. Purely backward-looking."""
    lp = np.log(px)
    r1 = lp.diff()
    f: dict[str, pd.DataFrame] = {}
    for n in (5, 21, 63, 126):
        f[f"ret_{n}"] = lp - lp.shift(n)
    f["mom_12_1"] = lp.shift(21) - lp.shift(252)
    f["vol_21"] = r1.rolling(21, min_periods=15).std() * np.sqrt(252)
    f["vol_63"] = r1.rolling(63, min_periods=40).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21"] / f["vol_63"]
    for n in (20, 100):
        f[f"z_ma{n}"] = (lp - lp.rolling(n).mean()) / lp.rolling(n).std()
    f["rsi_14"] = _rsi(px)
    f["ffd_logp"] = frac_diff_ffd(lp, d, thresh)
    f["xs_ret_21"] = f["ret_21"].rank(axis=1, pct=True)
    f["xs_mom"] = f["mom_12_1"].rank(axis=1, pct=True)
    f["xs_vol"] = f["vol_63"].rank(axis=1, pct=True)
    ls = np.log(spy)
    mkt = {
        "spy_ret_21": ls - ls.shift(21),
        "spy_trend": ls - ls.rolling(200).mean(),
        "vix": vix.ffill(limit=5),
        "vix_chg_21": np.log(vix.ffill(limit=5)).diff(21),
    }
    for k, s in mkt.items():
        f[k] = pd.DataFrame(np.repeat(s.to_numpy()[:, None], px.shape[1], 1), index=px.index, columns=px.columns)
    panel = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1)
    panel.index.names = ["date", "ticker"]
    return panel[FEATURES]


def stationarity_table(panel: pd.DataFrame, logp: pd.DataFrame, end: str, sample: int = 20) -> pd.DataFrame:
    """Median ADF p-value per feature across a sample of stocks, training period only."""
    tick = list(logp.columns[:: max(1, logp.shape[1] // sample)])
    rows = {"log price (raw)": [adf_pvalue(logp.loc[:end, t]) for t in tick]}
    sub = panel.loc[(slice(None, end), tick), :]
    for c in panel.columns:
        s = sub[c].unstack()
        if FAMILY[c] == "Market regime":
            rows[c] = [adf_pvalue(s.iloc[:, 0])]
        else:
            rows[c] = [adf_pvalue(s[t]) for t in tick if t in s]
    out = pd.DataFrame({k: {"median ADF p": float(np.nanmedian(v)), "share stationary (p<0.05)": float(np.nanmean(np.array(v) < 0.05))}
                        for k, v in rows.items()}).T
    out["family"] = [FAMILY.get(k, "—") for k in out.index]
    return out
