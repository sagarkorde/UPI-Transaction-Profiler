"""Robustness of the inference to temporal dependence between review windows (reviewer request, v11).

The protocol's tests treat review windows as exchangeable units: sign flips and bootstrap resamples act
on single windows, after held-out scenarios are summed within each test window. This script checks that
assumption on the same per-window counts used in the paper (c = 1):

  * lag-1..3 autocorrelation of the per-window paired differences d_w and a Ljung-Box test;
  * block sign-flip permutation p-values (one sign per block of L consecutive windows; exact
    enumeration when there are <= 20 blocks, else 10,000 draws);
  * moving-block bootstrap 95% CIs of the recall difference (block length L, circular).

L = 1 reproduces the protocol's tests. Writes results/analysis/dependence.csv (read by make_supplement.py).
This is a post-hoc robustness check of the inference, not a new comparison.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dstriage import experiment as E  # noqa: E402
from dstriage import stats as S  # noqa: E402

BLOCKS = [1, 2, 5, 10]
R, B = 10000, 2000
C1 = 1.0
pol, win, meta = S.load_runs(os.path.join(ROOT, "results", "runs"))
POLICIES = [p for p in E.MAIN_POLICIES if p != "SUP"]


def acf(d, k):
    d = d - d.mean()
    den = (d ** 2).sum()
    return float((d[:-k] * d[k:]).sum() / den) if den > 0 and len(d) > k else np.nan


def ljung_box(d, h=3):
    n = len(d)
    q = n * (n + 2) * sum(acf(d, k) ** 2 / (n - k) for k in range(1, h + 1))
    return float(q), float(sps.chi2.sf(q, h))


def block_ids(n, L):
    return np.arange(n) // L


def block_signflip_p(d, L, rng):
    if not np.any(d != 0):
        return 1.0, 1.0
    g = block_ids(len(d), L)
    sums = np.bincount(g, weights=d)
    nb = len(sums)
    T = abs(sums.sum())
    if nb <= 20:
        pats = np.array(list(itertools.product([-1.0, 1.0], repeat=nb)))
        Tp = np.abs(pats @ sums)
        p = float(np.mean(Tp >= T - 1e-12))
        pmin = 2.0 / 2 ** nb
    else:
        s = rng.choice([-1.0, 1.0], size=(R, nb))
        Tp = np.abs(s @ sums)
        p = float((1 + np.sum(Tp >= T - 1e-12)) / (R + 1))
        pmin = 1.0 / (R + 1)
    return p, pmin


def block_boot_ci(cp, cr, den, L, rng):
    n = len(den)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(B, -1)[:, :n]
    dd = den[idx].sum(1)
    ok = dd > 0
    diff = (cp[idx].sum(1)[ok] - cr[idx].sum(1)[ok]) / dd[ok]
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


def series(ds, scenario_list, novel):
    acc = None
    for sc in scenario_list:
        piv, Wm, _ = S.seed_avg(pol, win, ds, sc, C1)
        part = {"caught": piv["caught_novel" if novel else "caught"],
                "den": (Wm.novel_frauds if novel else Wm.frauds).to_numpy(float)}
        if acc is None:
            acc = part
        else:
            acc = {"caught": acc["caught"] + part["caught"], "den": acc["den"] + part["den"]}
    return acc


rows = []
for di, ds in enumerate(["ulb", "banksim", "tabformer"]):
    scen = sorted(set(pol[(pol.dataset == ds) & (pol.scenario != "base")].scenario))
    for setting, sl, novel in [("stationary", ["base"], False), ("held-out (novel)", scen, True)]:
        acc = series(ds, sl, novel)
        ref = acc["caught"]["SUP"].to_numpy(float)
        den = acc["den"]
        for pi_, P in enumerate(POLICIES):
            cp = acc["caught"][P].to_numpy(float)
            d = cp - ref
            if den.sum() == 0:
                continue
            q, qp = ljung_box(d) if np.any(d != 0) else (np.nan, np.nan)
            row = dict(dataset=ds, setting=setting, policy=P, windows=len(d), diff=float(d.sum() / den.sum()),
                       acf1=acf(d, 1), acf2=acf(d, 2), acf3=acf(d, 3), ljung_box_q=q, ljung_box_p=qp)
            for L in BLOCKS:
                rg = np.random.default_rng([20261108, di, int(novel), pi_, L])
                p, pmin = block_signflip_p(d, L, rg)
                lo, hi = block_boot_ci(cp, ref, den, L, rg)
                row.update({f"p_L{L}": p, f"pmin_L{L}": pmin, f"lo_L{L}": lo, f"hi_L{L}": hi})
            rows.append(row)
out = pd.DataFrame(rows)
path = os.path.join(ROOT, "results", "analysis", "dependence.csv")
out.to_csv(path, index=False)
show = out[out.policy.isin(["DRT", "FUSE60"])][["dataset", "setting", "policy", "diff", "acf1", "ljung_box_p",
                                               "p_L1", "p_L2", "p_L5", "pmin_L5", "p_L10", "lo_L1", "hi_L1",
                                               "lo_L5", "hi_L5", "lo_L10", "hi_L10"]]
pd.set_option("display.width", 250)
print(show.round(4).to_string(index=False))
print("wrote", path)
