"""Model zoo, out-of-sample prediction helpers and feature importance (MDI / MDA / SHAP)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "lgbm"            # "lgbm", "logit" or "momentum"
    num_leaves: int = 15
    n_estimators: int = 300
    learning_rate: float = 0.03
    min_child_samples: int = 200
    C: float = 1.0
    seed: int = 0

    @property
    def name(self) -> str:
        if self.kind == "lgbm":
            return f"LGBM leaves={self.num_leaves} trees={self.n_estimators} min_leaf={self.min_child_samples}"
        if self.kind == "logit":
            return f"Logit C={self.C:g}"
        return "Momentum 12-1 rule"

    def as_dict(self) -> dict:
        return asdict(self)


MAIN = ModelConfig()                    # fixed upfront, never tuned on the sample
LOGIT = ModelConfig(kind="logit", C=1.0)
MOMENTUM = ModelConfig(kind="momentum")


class Model:
    """Thin wrapper giving all three model kinds the same fit / predict_proba interface."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.est = None
        self.cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None = None) -> "Model":
        self.cols = list(X.columns)
        c = self.cfg
        if c.kind == "lgbm":
            import lightgbm as lgb

            self.est = lgb.LGBMClassifier(
                n_estimators=c.n_estimators, learning_rate=c.learning_rate, num_leaves=c.num_leaves,
                min_child_samples=c.min_child_samples, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                reg_lambda=1.0, random_state=c.seed, n_jobs=4, verbose=-1,
            )
            self.est.fit(X, y, sample_weight=w)
        elif c.kind == "logit":
            self.scaler = StandardScaler().fit(X)
            self.est = LogisticRegression(C=c.C, max_iter=500)
            self.est.fit(self.scaler.transform(X), y, sample_weight=w)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        c = self.cfg
        if c.kind == "momentum":  # P(up) proxy = cross-sectional momentum rank, no fitting
            return X["xs_mom"].to_numpy(dtype=float)
        if c.kind == "logit":
            return self.est.predict_proba(self.scaler.transform(X[self.cols]))[:, 1]
        return self.est.predict_proba(X[self.cols])[:, 1]


def oos_predict(cfg: ModelConfig, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, splits) -> np.ndarray:
    """Concatenate out-of-sample probabilities over (train, test) splits; untested events are NaN."""
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        p[te] = Model(cfg).fit(X.iloc[tr], y[tr], w[tr]).predict_proba(X.iloc[te])
    return p


def walk_forward_predict(cfg: ModelConfig, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, events: pd.DataFrame,
                         Xp: pd.DataFrame, bars: pd.DatetimeIndex, first_test: str, retrain: str = "YS") -> pd.Series:
    """Expanding-window walk-forward with purging.

    At the start of every period (``retrain`` frequency, from ``first_test``) the model is refit on
    all labelled events whose label ended strictly before the period's first bar, then predicts
    every row of ``Xp`` (features on rebalance dates, labelled or not) dated inside the period.
    """
    tp = Xp.index.get_level_values(0)
    i1 = events["i1"].to_numpy()
    out = pd.Series(np.nan, index=Xp.index)
    starts = pd.date_range(first_test, tp.max(), freq=retrain)
    if len(starts) == 0 or starts[0] > pd.Timestamp(first_test):
        starts = starts.insert(0, pd.Timestamp(first_test))
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else tp.max() + pd.Timedelta(days=1)
        te = (tp >= s) & (tp < e)
        if not te.any():
            continue
        first_bar = int(bars.searchsorted(s))
        tr = np.flatnonzero(i1 < first_bar)
        out[te] = Model(cfg).fit(X.iloc[tr], y[tr], w[tr]).predict_proba(Xp[te])
    return out


def classification_scores(y: np.ndarray, p: np.ndarray) -> dict:
    m = np.isfinite(p)
    y, p = y[m], np.clip(p[m], 1e-6, 1 - 1e-6)
    return {"Accuracy": accuracy_score(y, p > 0.5), "AUC": roc_auc_score(y, p), "Log loss": log_loss(y, p), "N": int(m.sum())}


# ----------------------------------------------------------------------------- importance
def mdi(model: Model) -> pd.Series:
    """Mean decrease in impurity: LightGBM total split gain, normalised to sum to 1 (in-sample)."""
    g = model.est.booster_.feature_importance(importance_type="gain")
    s = pd.Series(g, index=model.cols)
    return s / s.sum()


def mda(cfg: ModelConfig, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, splits, groups: dict[str, list[str]] | None = None,
        seed: int = 0) -> pd.DataFrame:
    """Purged mean decrease in accuracy: increase in OOS log loss when a feature (or a feature
    group, which handles substitution effects) is permuted in the test fold. Mean and std over folds."""
    rng = np.random.default_rng(seed)
    units = {c: [c] for c in X.columns} if groups is None else groups
    res = {u: [] for u in units}
    for tr, te in splits:
        m = Model(cfg).fit(X.iloc[tr], y[tr], w[tr])
        Xt = X.iloc[te].copy()
        base = log_loss(y[te], np.clip(m.predict_proba(Xt), 1e-6, 1 - 1e-6), sample_weight=w[te])
        for u, cols in units.items():
            Xp = Xt.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]
            ll = log_loss(y[te], np.clip(m.predict_proba(Xp), 1e-6, 1 - 1e-6), sample_weight=w[te])
            res[u].append(ll - base)
    return pd.DataFrame({"mean": {u: np.mean(v) for u, v in res.items()}, "std": {u: np.std(v) for u, v in res.items()}})


def shap_importance(model: Model, X: pd.DataFrame, n: int = 5000, seed: int = 0) -> pd.Series:
    """Mean |TreeSHAP| contribution (LightGBM ``pred_contrib``), on a random sample of rows."""
    Xs = X.sample(min(n, len(X)), random_state=seed)[model.cols]
    contrib = model.est.booster_.predict(Xs, pred_contrib=True)[:, :-1]
    s = pd.Series(np.abs(contrib).mean(0), index=model.cols)
    return s / s.sum()
