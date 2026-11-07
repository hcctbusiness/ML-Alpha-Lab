"""Charts for the ML Alpha Lab study (matplotlib, HCCT Group palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#0B3C5D", "#D9822B", "#328CC1", "#7A7A7A", "#2E7D32", "#B23A48"]
FOOT = "HCCT Group · ML Alpha Lab"


def _style(ax):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path):
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="#7A7A7A")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def cv_comparison(tab: pd.DataFrame, base_rate: float, path):
    """tab: rows MultiIndex (model, scheme) with columns AUC, Accuracy, L/S Sharpe."""
    models = list(dict.fromkeys(tab.index.get_level_values(0)))
    schemes = list(dict.fromkeys(tab.index.get_level_values(1)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = np.arange(len(models))
    wd = 0.8 / len(schemes)
    for ax, col, ref, lab in zip(axes, ["Accuracy", "AUC", "L/S Sharpe"], [base_rate, 0.5, 0.0],
                                 ["Accuracy (base rate dashed)", "ROC AUC (0.5 = coin flip)", "Weekly L/S Sharpe from OOS predictions (gross)"]):
        for j, s in enumerate(schemes):
            v = [tab.loc[(m, s), col] for m in models]
            bars = ax.bar(x + (j - (len(schemes) - 1) / 2) * wd, v, wd, color=PALETTE[j], label=s)
            for b, val in zip(bars, v):
                up = val >= 0
                ax.annotate(f"{val:.2f}", (b.get_x() + b.get_width() / 2, val), xytext=(0, 2 if up else -2),
                            textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=7)
        ax.axhline(ref, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, models)
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=10.5)
        _style(ax)
        if col != "L/S Sharpe":
            lo = min(tab[col].min(), ref) - 0.03
            ax.set_ylim(lo, tab[col].max() + 0.05)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Same models, four validation schemes: shuffled K-fold leaks, purged CV does not",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, path)


def cpcv(paths: dict[str, np.ndarray], refs: dict[str, float], path):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    names = list(paths)
    for i, n in enumerate(names):
        v = np.asarray(paths[n])
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), s=55,
                   color=PALETTE[i], alpha=0.85, zorder=3)
        ax.boxplot(v, positions=[i], vert=False, widths=0.45, showfliers=False,
                   medianprops=dict(color=PALETTE[i], lw=2), boxprops=dict(color="#7A7A7A"), whiskerprops=dict(color="#7A7A7A"))
        ax.text(v.max() + 0.05, i + 0.22, f"mean {v.mean():.2f} · min {v.min():.2f} · max {v.max():.2f}", fontsize=8.5, va="center")
    for j, (lab, val) in enumerate(refs.items()):
        ax.axvline(val, color=PALETTE[3 + j % 3], ls="--", lw=1.2, label=lab)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Annualised Sharpe of weekly L/S quintile spread (gross) on each CPCV backtest path")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    lo = min(min(np.min(v) for v in paths.values()), min(refs.values()), 0) - 0.2
    hi = max(max(np.max(v) for v in paths.values()), max(refs.values())) + 1.0
    ax.set_xlim(lo, hi)
    ax.set_title("Combinatorial purged CV: a distribution of out-of-sample Sharpe ratios, not one number",
                 loc="left", fontweight="bold")
    _style(ax)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, path)


def feature_importance(mdi: pd.Series, mda: pd.DataFrame, shap: pd.Series, mda_group: pd.DataFrame, path):
    order = mda["mean"].sort_values().index
    fig, axes = plt.subplots(1, 4, figsize=(18, 6.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
    axes[0].barh(order, mdi.reindex(order), color=PALETTE[0])
    axes[0].set_title("MDI (split gain, in-sample)", loc="left", fontweight="bold", fontsize=10.5)
    axes[1].barh(order, mda["mean"].reindex(order) * 1e3, xerr=mda["std"].reindex(order) * 1e3, color=PALETTE[1], ecolor="#7A7A7A")
    axes[1].axvline(0, color="k", lw=0.7)
    axes[1].set_title("MDA, purged CV (Δ log loss × 1e3)", loc="left", fontweight="bold", fontsize=10.5)
    axes[2].barh(order, shap.reindex(order), color=PALETTE[2])
    axes[2].set_title("Mean |SHAP| (TreeSHAP)", loc="left", fontweight="bold", fontsize=10.5)
    go = mda_group["mean"].sort_values().index
    axes[3].barh(go, mda_group["mean"].reindex(go) * 1e3, xerr=mda_group["std"].reindex(go) * 1e3, color=PALETTE[4], ecolor="#7A7A7A")
    axes[3].axvline(0, color="k", lw=0.7)
    axes[3].set_title("Clustered MDA by family (× 1e3)", loc="left", fontweight="bold", fontsize=10.5)
    for a in axes[1:3]:
        a.set_yticklabels([])
    for a in axes:
        _style(a)
    fig.suptitle("LightGBM feature importance: in-sample MDI vs out-of-sample purged MDA vs SHAP",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, path)


def meta_labelling(pr: pd.DataFrame, equity: dict[str, pd.Series], path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw=dict(width_ratios=[1, 1.5]))
    ax = axes[0]
    cols = ["Precision", "Recall", "F1"]
    x = np.arange(len(cols))
    wd = 0.8 / len(pr)
    for j, (n, row) in enumerate(pr.iterrows()):
        b = ax.bar(x + (j - (len(pr) - 1) / 2) * wd, row[cols].astype(float), wd, color=PALETTE[j], label=n)
        for bb, v in zip(b, row[cols]):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, cols)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("OOS classification of 'primary bet is profitable'", loc="left", fontweight="bold", fontsize=10.5)
    _style(ax)
    ax = axes[1]
    for i, (n, r) in enumerate(equity.items()):
        ax.plot((1 + r).cumprod(), color=PALETTE[i], lw=1.8 if i == 1 else 1.3, label=n)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Growth of $1, net of costs (log scale)", loc="left", fontweight="bold", fontsize=10.5)
    _style(ax)
    fig.suptitle("Meta-labelling: momentum decides the side, LightGBM decides whether and how much to bet",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, path)


def pbo(res: dict, path, n_configs: int):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.3))
    ax = axes[0]
    lg = res["logits"]
    ax.hist(lg, bins=40, color=PALETTE[0], alpha=0.85)
    ax.axvline(0, color=PALETTE[5], lw=1.5, ls="--")
    ax.text(0.02, 0.95, f"PBO = P(λ ≤ 0) = {res['PBO']:.2f}\n{res['n_splits']:,} CSCV splits, {n_configs} configs",
            transform=ax.transAxes, va="top", fontsize=10, bbox=dict(fc="white", ec="#7A7A7A"))
    ax.set_xlabel("Logit λ of the in-sample winner's out-of-sample relative rank")
    ax.set_title("Distribution of logits (λ ≤ 0: the IS winner is below median OOS)", loc="left", fontweight="bold", fontsize=10.5)
    _style(ax)
    ax = axes[1]
    a, b = res["IS best SR"] * np.sqrt(252), res["OOS SR of IS best"] * np.sqrt(252)
    ax.scatter(a, b, s=6, alpha=0.25, color=PALETTE[2])
    k = np.polyfit(a, b, 1)
    xs = np.linspace(a.min(), a.max(), 10)
    ax.plot(xs, np.polyval(k, xs), color=PALETTE[1], lw=2, label=f"OLS slope {k[0]:.2f}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("In-sample Sharpe of the IS-best config (annualised)")
    ax.set_ylabel("Same config, out-of-sample Sharpe")
    ax.legend(frameon=False)
    ax.set_title("Performance degradation: IS winner vs its OOS Sharpe", loc="left", fontweight="bold", fontsize=10.5)
    _style(ax)
    fig.suptitle("Probability of Backtest Overfitting (CSCV, Bailey et al. 2017)", fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, path)


def triple_barrier(px: pd.Series, ev: pd.DataFrame, path, ticker: str):
    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.plot(px.index, px.values, color=PALETTE[0], lw=1.4, label=f"{ticker} adjusted close")
    col = {1: PALETTE[4], -1: PALETTE[5], 0: PALETTE[3]}
    lab = {1: "profit-take hit first", -1: "stop-loss hit first", 0: "vertical barrier (10 days)"}
    used = set()
    for (t0, _), r in ev.iterrows():
        p0 = px.loc[t0]
        up, dn = p0 * np.exp(r["trgt"]), p0 * np.exp(-r["trgt"])
        ax.fill_between([t0, r["tv"]], dn, up, color=col[r["barrier"]], alpha=0.10, lw=0)
        ax.hlines([up, dn], t0, r["tv"], color=col[r["barrier"]], lw=0.8, alpha=0.8)
        b = int(r["barrier"])
        ax.scatter([r["t1"]], [px.loc[r["t1"]]], color=col[b], s=36, zorder=4, edgecolor="white",
                   label=lab[b] if b not in used else None)
        used.add(b)
    ax.set_title(f"Triple-barrier labels on {ticker}: weekly events, ±1.0 × σ_daily × √10 barriers, 10-day vertical barrier",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, path)


def ffd(logp: pd.Series, ffd_series: pd.Series, dtab: pd.DataFrame, d: float, ticker: str, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    ax.plot(logp.index, logp.values, color=PALETTE[0], lw=1.2, label=f"log price {ticker}")
    ax2 = ax.twinx()
    ax2.plot(ffd_series.index, ffd_series.values, color=PALETTE[1], lw=0.9, label=f"FFD d={d:.1f}")
    ax.set_title(f"{ticker}: log price vs fractionally differentiated series (d = {d:.1f})", loc="left", fontweight="bold", fontsize=10.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    _style(ax); ax2.spines[["top"]].set_visible(False)
    ax = axes[1]
    ax.plot(dtab.index, dtab["median ADF p"], "o-", color=PALETTE[0], label="median ADF p-value")
    ax.axhline(0.05, color=PALETTE[5], ls="--", lw=1, label="5% critical level")
    axr = ax.twinx()
    axr.plot(dtab.index, dtab["corr with log price"], "s-", color=PALETTE[2], label="corr. with log price")
    axr.set_ylim(0, 1.05)
    ax.axvline(d, color="#7A7A7A", lw=0.8)
    ax.set_xlabel("d (chosen on 2009–2013 training data only)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center right", fontsize=8.5)
    ax.set_title("Stationarity vs memory", loc="left", fontweight="bold", fontsize=10.5)
    _style(ax); axr.spines[["top"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, path)
