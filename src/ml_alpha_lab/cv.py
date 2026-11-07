"""Cross-validation schemes for overlapping financial labels (AFML ch. 7 and 12).

All splitters take the bar index where each label starts (``i0``) and ends (``i1``) and
yield positional (train, test) index arrays. Folds are formed from contiguous blocks of
event *dates*, so every stock observed on a date lands in the same fold.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from sklearn.model_selection import KFold


def _date_groups(i0: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Split the unique start bars into contiguous groups; return event positions per group."""
    u = np.unique(i0)
    blocks = np.array_split(u, n_groups)
    return [np.flatnonzero(np.isin(i0, b)) for b in blocks]


def purge_train(i0: np.ndarray, i1: np.ndarray, test: np.ndarray, embargo: int = 0,
                candidates: np.ndarray | None = None) -> np.ndarray:
    """Training positions whose label span [i0, i1] does not overlap the test span
    [min i0_test, max i1_test] and does not start within ``embargo`` bars after it."""
    a, b = i0[test].min(), i1[test].max()
    cand = np.setdiff1d(np.arange(len(i0)) if candidates is None else candidates, test)
    keep = (i1[cand] < a) | (i0[cand] > b + embargo)
    return cand[keep]


class PurgedKFold:
    """K contiguous folds; training labels overlapping the test span are purged and the
    ``embargo`` bars after each test fold are dropped from training."""

    def __init__(self, n_splits: int = 5, embargo: int = 10):
        self.n_splits, self.embargo = n_splits, embargo

    def split(self, i0: np.ndarray, i1: np.ndarray):
        i0, i1 = np.asarray(i0), np.asarray(i1)
        for test in _date_groups(i0, self.n_splits):
            yield purge_train(i0, i1, test, self.embargo), test


class ContiguousKFold:
    """K contiguous date folds with **no** purging or embargo (leaks through overlapping labels)."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, i0: np.ndarray, i1: np.ndarray):
        i0 = np.asarray(i0)
        for test in _date_groups(i0, self.n_splits):
            yield np.setdiff1d(np.arange(len(i0)), test), test


class ShuffledKFold:
    """Textbook shuffled K-fold at the event level — the naive (wrong) choice for time series."""

    def __init__(self, n_splits: int = 5, seed: int = 0):
        self.kf = KFold(n_splits, shuffle=True, random_state=seed)

    def split(self, i0: np.ndarray, i1: np.ndarray):
        for tr, te in self.kf.split(np.asarray(i0)):
            yield tr, te


class WalkForward:
    """Expanding-window walk-forward over K contiguous folds: train on all earlier labels that
    end before the test fold starts (purged), test on the next fold. Fold 0 is never tested."""

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, i0: np.ndarray, i1: np.ndarray):
        i0, i1 = np.asarray(i0), np.asarray(i1)
        groups = _date_groups(i0, self.n_splits)
        for test in groups[1:]:
            a = i0[test].min()
            yield np.flatnonzero(i1 < a), test


class CombinatorialPurgedKFold:
    """Combinatorial purged CV: N contiguous groups, every choice of k test groups.

    Produces C(N, k) splits and C(N, k) · k / N complete backtest paths.
    """

    def __init__(self, n_groups: int = 8, k_test: int = 2, embargo: int = 10):
        self.N, self.k, self.embargo = n_groups, k_test, embargo

    @property
    def n_splits(self) -> int:
        return comb(self.N, self.k)

    @property
    def n_paths(self) -> int:
        return comb(self.N, self.k) * self.k // self.N

    def combos(self) -> list[tuple[int, ...]]:
        return list(combinations(range(self.N), self.k))

    def path_map(self) -> np.ndarray:
        """(n_paths × N) array: entry [p, g] is the split whose test prediction for group g
        forms part of path p (each group's j-th appearance goes to path j)."""
        seen = np.zeros(self.N, int)
        m = np.full((self.n_paths, self.N), -1)
        for s, c in enumerate(self.combos()):
            for g in c:
                m[seen[g], g] = s
                seen[g] += 1
        return m

    def split(self, i0: np.ndarray, i1: np.ndarray):
        i0, i1 = np.asarray(i0), np.asarray(i1)
        groups = _date_groups(i0, self.N)
        for c in self.combos():
            train = np.arange(len(i0))
            for g in c:  # purge + embargo around every test block separately
                train = purge_train(i0, i1, groups[g], self.embargo, candidates=train)
            test = np.concatenate([groups[g] for g in c])
            yield train, test, c

    def groups(self, i0: np.ndarray) -> list[np.ndarray]:
        return _date_groups(np.asarray(i0), self.N)
