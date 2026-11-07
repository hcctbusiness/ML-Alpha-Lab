"""Meta-labelling and bet sizing (AFML ch. 3.6 and 10)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score, precision_score, recall_score


def primary_side(X: pd.DataFrame) -> np.ndarray:
    """Primary model: time-series 12-1 momentum. Long if the stock's 12-1 return is positive, else short."""
    return np.where(X["mom_12_1"].to_numpy() >= 0, 1, -1)


def meta_labels(ret: np.ndarray, side: np.ndarray) -> np.ndarray:
    """1 if the primary bet made money by the first barrier touch, else 0."""
    return (np.asarray(ret) * np.asarray(side) > 0).astype(int)


def bet_size(p: np.ndarray, step: float = 0.0) -> np.ndarray:
    """Size from the meta-model probability that the primary bet is right.

    z = (p − ½) / √(p(1 − p)), size = 2Φ(z) − 1, floored at 0 (no bet when p ≤ ½) and
    optionally discretised to multiples of ``step``. Always in [0, 1].
    """
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    m = np.clip(2 * norm.cdf(z) - 1, 0.0, 1.0)
    if step > 0:
        m = np.round(m / step) * step
    return np.where(np.isfinite(m), m, 0.0)


def precision_recall(y_true: np.ndarray, act: np.ndarray) -> dict:
    """Positive class = 'the primary trade is profitable'. ``act`` = 1 where a trade is taken."""
    return {"Precision": precision_score(y_true, act, zero_division=0), "Recall": recall_score(y_true, act, zero_division=0),
            "F1": f1_score(y_true, act, zero_division=0), "Trades taken": float(np.mean(act))}
