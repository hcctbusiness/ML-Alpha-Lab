"""Offline unit tests on synthetic data (no network)."""
from math import comb

import numpy as np
import pandas as pd
import pytest
from scipy.special import binom

from ml_alpha_lab import cv
from ml_alpha_lab import features as F
from ml_alpha_lab import labels as L
from ml_alpha_lab import meta as ML
from ml_alpha_lab import models as M
from ml_alpha_lab import overfitting as OF
from ml_alpha_lab import portfolio as P


def synthetic_prices(n=800, k=8, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    r = rng.normal(0.0003, 0.015, (n, k))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), idx, [f"S{i}" for i in range(k)])


def synthetic_events(n=800, k=8, seed=0):
    px = synthetic_prices(n, k, seed)
    ev = L.triple_barrier(px, L.weekly_dates(px.index), L.BarrierParams(horizon=10))
    return px, ev


# ----------------------------------------------------------------------------- purged CV
@pytest.mark.parametrize("embargo", [0, 5, 21])
def test_purged_kfold_never_trains_inside_test_span_plus_embargo(embargo):
    _, ev = synthetic_events()
    i0, i1 = ev["i0"].to_numpy(), ev["i1"].to_numpy()
    n_tested = 0
    for tr, te in cv.PurgedKFold(5, embargo).split(i0, i1):
        a, b = i0[te].min(), i1[te].max()
        assert not np.intersect1d(tr, te).size
        overlap = (i1[tr] >= a) & (i0[tr] <= b + embargo)  # label touches [a, b + embargo]
        assert not overlap.any()
        n_tested += len(te)
    assert n_tested == len(ev)


def test_cpcv_purges_around_every_test_block():
    _, ev = synthetic_events()
    i0, i1 = ev["i0"].to_numpy(), ev["i1"].to_numpy()
    sp = cv.CombinatorialPurgedKFold(6, 2, embargo=5)
    groups = sp.groups(i0)
    for tr, te, c in sp.split(i0, i1):
        for g in c:
            a, b = i0[groups[g]].min(), i1[groups[g]].max()
            assert not ((i1[tr] >= a) & (i0[tr] <= b + 5)).any()


def test_walk_forward_trains_only_on_labels_ending_before_test():
    _, ev = synthetic_events()
    i0, i1 = ev["i0"].to_numpy(), ev["i1"].to_numpy()
    for tr, te in cv.WalkForward(5).split(i0, i1):
        assert i1[tr].max() < i0[te].min()


@pytest.mark.parametrize("N,k", [(6, 2), (8, 2), (10, 3), (5, 1)])
def test_cpcv_path_count(N, k):
    sp = cv.CombinatorialPurgedKFold(N, k)
    assert sp.n_splits == comb(N, k)
    assert sp.n_paths == comb(N, k) * k // N
    m = sp.path_map()
    assert m.shape == (sp.n_paths, N) and (m >= 0).all()
    combos = sp.combos()
    for p in range(sp.n_paths):  # each path uses, for group g, a split that actually tested g
        for g in range(N):
            assert g in combos[m[p, g]]
    assert len(list(sp.split(*synthetic_events(n=400, k=3)[1][["i0", "i1"]].to_numpy().T))) == comb(N, k)


# ----------------------------------------------------------------------------- triple barrier
def test_triple_barrier_hits_correct_barrier_on_constructed_paths():
    base = np.log(100.0)
    up = np.r_[base, base + np.array([0.01, 0.02, 0.06, -0.10, 0, 0, 0, 0, 0, 0])]
    dn = np.r_[base, base + np.array([-0.01, -0.03, -0.07, 0.10, 0, 0, 0, 0, 0, 0])]
    flat = np.r_[base, base + np.full(10, 0.01)]
    assert L.triple_barrier_path(up, 0, 0.05, 0.05, 10) == (3, 1)
    assert L.triple_barrier_path(dn, 0, 0.05, 0.05, 10) == (3, -1)
    assert L.triple_barrier_path(flat, 0, 0.05, 0.05, 10) == (10, 0)
    # stop-loss touched before profit-take even though the path ends higher
    whip = np.r_[base, base + np.array([-0.06, 0.08, 0.1, 0, 0, 0, 0, 0, 0, 0])]
    assert L.triple_barrier_path(whip, 0, 0.05, 0.05, 10) == (1, -1)


def test_vectorised_triple_barrier_matches_reference_and_jumps():
    px, ev = synthetic_events(n=600, k=4, seed=3)
    lp = np.log(px)
    for (t0, tk), r in ev.sample(200, random_state=0).iterrows():
        i1, b = L.triple_barrier_path(lp[tk].to_numpy(), int(r["i0"]), r["trgt"], r["trgt"], 10)
        assert (i1, b) == (r["i1"], r["barrier"])
        assert r["bin"] == int(r["ret"] > 0)
    # a +15% jump the day after an event must be a profit-take hit on day 1
    px2 = synthetic_prices(300, 1, seed=1)
    d0 = px2.index[200]
    px2.iloc[201:] *= 1.15
    e = L.triple_barrier(px2, pd.DatetimeIndex([d0]))
    assert e["barrier"].iloc[0] == 1 and e["i1"].iloc[0] == 201


# ----------------------------------------------------------------------------- FFD
@pytest.mark.parametrize("d", [0.2, 0.35, 0.5, 0.8])
def test_ffd_weights_match_binomial_formula(d):
    w = F.ffd_weights(d, thresh=1e-5)[::-1]  # newest first: w_0, w_1, ...
    k = np.arange(len(w))
    np.testing.assert_allclose(w, (-1.0) ** k * binom(d, k), rtol=1e-10, atol=1e-14)
    assert abs(w[-1]) >= 1e-5
    np.testing.assert_allclose(F.ffd_weights(1.0)[::-1], [1.0, -1.0])  # d = 1 is the first difference
    assert len(F.ffd_weights(0.0)) == 1


def test_ffd_of_random_walk_is_stationary():
    rng = np.random.default_rng(0)
    rw = pd.Series(np.cumsum(rng.normal(size=4000)))
    assert F.adf_pvalue(rw) > 0.05                           # the level is a unit root
    assert F.adf_pvalue(F.frac_diff_ffd(rw, 1.0)) < 0.01     # d = 1 removes it
    assert F.adf_pvalue(F.frac_diff_ffd(rw, 0.6, 1e-3)) < 0.05  # fractional d already passes ADF
    np.testing.assert_allclose(F.frac_diff_ffd(rw, 1.0).dropna(), rw.diff().dropna())


# ----------------------------------------------------------------------------- uniqueness
def test_uniqueness_on_known_overlaps():
    # A spans bars 0-2, B spans 1-3, C spans 5-5 (alone)
    i0, i1 = np.array([0, 1, 5]), np.array([2, 3, 5])
    np.testing.assert_array_equal(L.concurrency(i0, i1), [1, 2, 2, 1, 0, 1])
    np.testing.assert_allclose(L.average_uniqueness(i0, i1), [2 / 3, 2 / 3, 1.0])
    # three identical labels each get 1/3
    np.testing.assert_allclose(L.average_uniqueness(np.zeros(3, int), np.full(3, 4)), [1 / 3] * 3)
    # per-ticker: overlap across *different* stocks does not reduce uniqueness
    idx = pd.MultiIndex.from_tuples([("2020-01-01", "A"), ("2020-01-01", "B")], names=["t0", "ticker"])
    ev = pd.DataFrame({"i0": [0, 0], "i1": [3, 3]}, index=idx)
    np.testing.assert_allclose(L.uniqueness_weights(ev), [1.0, 1.0])


# ----------------------------------------------------------------------------- overfitting
def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0006, 0.01, 2500))
    dsr = [OF.deflated_sharpe(r, n, var_sr=0.03 ** 2)["DSR"] for n in (1, 5, 20, 100, 1000)]
    assert all(a > b for a, b in zip(dsr, dsr[1:]))
    assert OF.deflated_sharpe(r, 1, 0.03 ** 2)["DSR"] == pytest.approx(OF.deflated_sharpe(r, 1, 0.03 ** 2)["PSR (vs 0)"])
    assert OF.expected_max_sr(100, 1.0) > OF.expected_max_sr(10, 1.0) > 0
    assert OF.min_backtest_length(100, 1.0) > OF.min_backtest_length(10, 1.0) > 0


def test_pbo_is_about_half_on_noise():
    # PBO of a single noise panel is very dispersed, so average over many independent panels
    pbos = [OF.pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (600, 20)), n_blocks=10)["PBO"] for s in range(60)]
    assert 0.42 < np.mean(pbos) < 0.58
    rng = np.random.default_rng(42)
    # a genuinely better strategy is not "overfit"
    X = rng.normal(0, 0.01, (2000, 20))
    X[:, 3] += 0.003
    assert OF.pbo_cscv(X, n_blocks=12)["PBO"] < 0.1


# ----------------------------------------------------------------------------- look-ahead
def test_features_have_no_look_ahead():
    px = synthetic_prices(700, 6, seed=5)
    rng = np.random.default_rng(1)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 700))), px.index)
    vix = pd.Series(20 + np.abs(rng.normal(0, 3, 700)), px.index)
    cut = px.index[500]
    base = F.build(px, spy, vix, d=0.4)
    px2, spy2, vix2 = px.copy(), spy.copy(), vix.copy()
    px2.loc[px.index > cut] *= rng.uniform(0.5, 1.5, px2.loc[px.index > cut].shape)
    spy2.loc[spy.index > cut] *= 1.3
    vix2.loc[vix.index > cut] *= 2.0
    alt = F.build(px2, spy2, vix2, d=0.4)
    a, b = base.loc[:cut], alt.loc[:cut]
    pd.testing.assert_frame_equal(a, b)
    assert not base.loc[px.index[-1]].equals(alt.loc[px.index[-1]])
    assert base.loc[:cut].notna().any().all()


def test_labels_never_use_data_before_event_and_barrier_is_causal():
    px, ev = synthetic_events(n=500, k=3)
    assert (ev["i1"] > ev["i0"]).all() and (ev["i1"] - ev["i0"] <= 10).all()
    # perturbing prices after t0 + horizon leaves an event's label unchanged
    t = ev.index[50]
    i_end = int(ev.loc[t, "i0"]) + 10
    px2 = px.copy()
    px2.iloc[i_end + 1:] *= 3.0
    ev2 = L.triple_barrier(px2, pd.DatetimeIndex([t[0]]))
    assert ev2.loc[t, ["i1", "barrier", "bin"]].tolist() == ev.loc[t, ["i1", "barrier", "bin"]].tolist()


# ----------------------------------------------------------------------------- meta-labelling
def test_bet_sizes_within_unit_interval():
    p = np.r_[np.linspace(0, 1, 1001), 0.0, 1.0, 0.5]
    s = ML.bet_size(p)
    assert np.all((s >= 0) & (s <= 1))
    assert np.all(s[p <= 0.5] == 0)
    assert np.all(np.diff(ML.bet_size(np.linspace(0.5, 0.999, 100))) >= 0)  # monotone
    assert np.all((ML.bet_size(p, step=0.1) >= 0) & (ML.bet_size(p, step=0.1) <= 1))
    assert ML.meta_labels(np.array([0.02, -0.01, 0.03]), np.array([1, 1, -1])).tolist() == [1, 0, 0]


def test_end_to_end_walk_forward_on_synthetic_data():
    px, ev = synthetic_events(n=900, k=6, seed=2)
    rng = np.random.default_rng(0)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(px)))), px.index)
    vix = pd.Series(20 + np.abs(rng.normal(0, 3, len(px))), px.index)
    panel = F.build(px, spy, vix, d=0.4)
    X = panel.reindex(pd.MultiIndex.from_arrays([ev.index.get_level_values(0), ev.index.get_level_values(1)]))
    X.index = ev.index
    ok = X.notna().all(axis=1).to_numpy()
    X, ev = X[ok], ev[ok]
    w = L.uniqueness_weights(ev).to_numpy()
    cfg = M.ModelConfig(n_estimators=30, min_child_samples=20)
    Xp = X.copy()
    s = M.walk_forward_predict(cfg, X, ev["bin"].to_numpy(), w, ev, Xp, px.index, "2017-06-01").dropna()
    assert s.between(0, 1).all() and s.index.get_level_values(0).min() >= pd.Timestamp("2017-06-01")
    res = P.run(P.long_short_targets(s), px, "test", start="2017-06-01")
    assert np.isfinite(res.returns).all()
    tg = P.long_short_targets(s)
    assert np.allclose(tg.sum(axis=1), 0, atol=1e-9)              # dollar-neutral targets
    assert np.allclose(tg.abs().sum(axis=1), 2.0, atol=1e-9)      # 100% long + 100% short
    held = res.weights.loc[res.weights.abs().sum(axis=1) > 0]
    assert (held.abs().sum(axis=1) < 2.5).all()                   # drifts, but stays near gross 2
