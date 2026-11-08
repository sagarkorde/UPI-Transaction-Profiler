"""Budget-constrained review policies.

All policies receive a period sorted by review window and a per-window budget k_w and
return a boolean selection mask with exactly k_w selected rows in window w.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def window_slices(window: np.ndarray):
    """(start, end) index pairs for each contiguous window of a sorted window array."""
    window = np.asarray(window)
    assert np.all(np.diff(window) >= 0), "rows must be sorted by window"
    starts = np.r_[0, np.flatnonzero(np.diff(window)) + 1]
    ends = np.r_[starts[1:], len(window)]
    return list(zip(starts.tolist(), ends.tolist()))


def allocate_budget(n_w, rate: float):
    """Integer budgets k_w ~= rate * n_w with fractional remainders carried forward, so that
    sum_w k_w = floor(rate * sum_w n_w) up to carry, and k_w <= n_w."""
    k = np.zeros(len(n_w), dtype=int)
    acc = 0.0
    for i, n in enumerate(n_w):
        acc += rate * n
        ki = min(int(math.floor(acc + 1e-9)), int(n))
        k[i] = ki
        acc -= ki
    return k


def _order(primary, secondary, idx):
    """Indices idx sorted by primary desc, then secondary desc, then position asc."""
    return idx[np.lexsort((idx, -secondary[idx], -primary[idx]))]


def topk(score, tiebreak, slices, k_w):
    sel = np.zeros(len(score), dtype=bool)
    for (a, b), k in zip(slices, k_w):
        if k <= 0:
            continue
        idx = np.arange(a, b)
        sel[_order(score, tiebreak, idx)[:k]] = True
    return sel


@dataclass
class DRTState:
    rho: float
    S_A: float = 0.0
    N_A: float = 0.0
    S_B: float = 0.0
    N_B: float = 0.0
    acc_B: float = 0.0
    trace: list = field(default_factory=list)
    pending: list = field(default_factory=list)


def _observe(y_true, noise, rng):
    """Analyst-reported fraud count for reviewed rows; exact when noise is None."""
    if noise is None or len(y_true) == 0:
        return float(np.sum(y_true))
    u = rng.random(len(y_true))
    rep = np.where(y_true == 1, u < noise["fraud_detected_prob"], u < noise["false_alarm_prob"])
    return float(rep.sum())


def drt(s, b, y, slices, k_w, p, state: DRTState | None = None, adaptive=True, rho_fixed=None,
        delay=0, noise=None, rng=None):
    """Disagreement-Reserved Triage.

    Channel A (exploit) reviews the top (1-rho) share of the budget by supervised score s.
    Channel B (disagreement) reviews the top rho share among the *remaining* transactions by
    an independent signal b (anomaly percentile in the proposed method). After each window
    the analyst outcomes of the marginal (lowest-ranked) picks of each channel update
    discounted Beta estimates of marginal precision, and rho moves by eta towards the channel
    with the higher marginal precision (projected onto [rho_min, rho_max]). Only labels of
    reviewed transactions are used, as in deployment.
    """
    if state is None:
        state = DRTState(rho=p["rho0"] if rho_fixed is None else rho_fixed)
    q, lam, eta = p["marginal_fraction"], p["discount"], p["eta"]
    a0, b0 = p["prior"]
    sel = np.zeros(len(s), dtype=bool)
    for (a, e), k in zip(slices, k_w):
        rho = state.rho if rho_fixed is None else rho_fixed
        state.acc_B += rho * k
        kB = min(int(math.floor(state.acc_B + 1e-9)), int(k))
        state.acc_B -= kB
        kA = int(k) - kB
        idx = np.arange(a, e)
        pickA = _order(s, b, idx)[:kA] if kA > 0 else idx[:0]
        rest = np.setdiff1d(idx, pickA, assume_unique=True)
        pickB = _order(b, s, rest)[:kB] if kB > 0 else idx[:0]
        sel[pickA] = True
        sel[pickB] = True
        # feedback from reviewed marginal picks (optionally noisy and delayed; amendment v1.1).
        # Retrieval is always scored with true labels; only the feedback driving rho is degraded.
        mA = pickA[len(pickA) - math.ceil(q * len(pickA)):] if len(pickA) else pickA
        mB = pickB[len(pickB) - math.ceil(q * len(pickB)):] if len(pickB) else pickB
        obs = lambda m: _observe(y[m], noise, rng)
        state.pending.append((obs(mA), len(mA), obs(mB), len(mB)))
        fA = fB = 0.0; nA = nB = 0
        if len(state.pending) > delay:
            fA, nA, fB, nB = state.pending.pop(0)
        state.S_A = lam * state.S_A + fA
        state.N_A = lam * state.N_A + nA
        state.S_B = lam * state.S_B + fB
        state.N_B = lam * state.N_B + nB
        if adaptive and rho_fixed is None:
            pA = (state.S_A + a0) / (state.N_A + a0 + b0)
            pB = (state.S_B + a0) / (state.N_B + a0 + b0)
            if pB > pA:
                state.rho = min(p["rho_max"], state.rho + eta)
            elif pB < pA:
                state.rho = max(p["rho_min"], state.rho - eta)
        state.trace.append(state.rho if rho_fixed is None else rho_fixed)
    return sel, state


def per_window(sel, y, amount, novel, slices):
    """Per-window counts of caught frauds, caught amount and caught novel-group frauds."""
    cs = np.r_[0, np.cumsum(sel & (y == 1))]
    ca = np.r_[0, np.cumsum(np.where(sel & (y == 1), amount, 0.0))]
    cn = np.r_[0, np.cumsum(sel & novel)]
    st = np.array([a for a, _ in slices]); en = np.array([b for _, b in slices])
    return cs[en] - cs[st], ca[en] - ca[st], cn[en] - cn[st]
