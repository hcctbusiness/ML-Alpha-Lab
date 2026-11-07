"""Backtest-overfitting diagnostics: PSR, Deflated Sharpe, MinBTL and PBO via CSCV.

References: Bailey & López de Prado (2012, 2014), Bailey, Borwein, López de Prado & Zhu (2017).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649015329


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else np.nan


def psr(sr: float, T: int, sk: float, ku: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio; ``sr`` and ``sr_star`` are per-period, ``ku`` is raw kurtosis."""
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """E[max SR] of n_trials unskilled strategies whose Sharpe estimates have variance var_sr."""
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns: pd.Series, n_trials: int, var_sr: float) -> dict:
    """DSR = PSR against the expected maximum Sharpe of ``n_trials`` skill-less trials.

    ``var_sr`` is the cross-trial variance of per-period Sharpe ratios (same frequency as returns).
    """
    r = pd.Series(returns).dropna().to_numpy()
    sr, T = sharpe(r), len(r)
    sk, ku = float(skew(r)), float(kurtosis(r, fisher=False))
    sr0 = expected_max_sr(n_trials, var_sr)
    return {"SR (per period)": sr, "T": T, "skew": sk, "kurtosis": ku, "SR0 (per period)": sr0,
            "PSR (vs 0)": psr(sr, T, sk, ku, 0.0), "DSR": psr(sr, T, sk, ku, sr0), "N trials": n_trials}


def min_backtest_length(n_trials: int, target_sr_annual: float) -> float:
    """Years of data needed so that the best of ``n_trials`` skill-less configurations is not
    expected to show an annualised Sharpe of ``target_sr_annual`` (Bailey et al. 2014)."""
    if n_trials <= 1:
        return 0.0
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return float(((1 - EULER) * z1 + EULER * z2) ** 2 / target_sr_annual ** 2)


def pbo_cscv(M: pd.DataFrame | np.ndarray, n_blocks: int = 16) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    ``M`` is a (T × N) matrix of per-period returns for N configurations. Rows are cut into
    ``n_blocks`` contiguous blocks; for every half/half split the best in-sample configuration's
    out-of-sample relative rank ω gives the logit λ = ln(ω / (1 − ω)). PBO = P(λ ≤ 0).
    """
    X = np.asarray(M, float)
    T, N = X.shape
    X = X[: T - T % n_blocks]
    blocks = np.array_split(np.arange(len(X)), n_blocks)
    # per-block sufficient statistics make every split O(N)
    s1 = np.array([X[b].sum(0) for b in blocks])
    s2 = np.array([(X[b] ** 2).sum(0) for b in blocks])
    n = np.array([len(b) for b in blocks])
    logits, is_best, oos_best = [], [], []
    allb = set(range(n_blocks))
    for c in combinations(range(n_blocks), n_blocks // 2):
        o = sorted(allb - set(c))
        def sr(ix):
            m = s1[list(ix)].sum(0) / n[list(ix)].sum()
            v = s2[list(ix)].sum(0) / n[list(ix)].sum() - m ** 2
            return m / np.sqrt(np.maximum(v, 1e-18))
        sr_is, sr_oos = sr(c), sr(o)
        b = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[b]).sum() + 1  # 1 = worst … N = best
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_best.append(sr_is[b]); oos_best.append(sr_oos[b])
    logits = np.array(logits)
    return {"PBO": float((logits <= 0).mean()), "logits": logits, "IS best SR": np.array(is_best),
            "OOS SR of IS best": np.array(oos_best), "n_splits": len(logits)}
