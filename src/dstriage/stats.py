"""Statistical analysis of per-window results.

Unit of resampling = review window (moving-block structure is respected because windows are
the operational batches; all policies are evaluated on the same windows, so comparisons are
paired). Seeds are averaged per window before resampling (seed variance is reported separately).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd


def load_runs(run_dir):
    pol, win, meta = [], [], []
    for f in sorted(glob.glob(os.path.join(run_dir, "*__policy.parquet"))):
        ds, scen, s = os.path.basename(f).replace("__policy.parquet", "").split("__")
        seed = int(s[1:])
        p = pd.read_parquet(f); p["dataset"], p["scenario"], p["seed"] = ds, scen, seed
        w = pd.read_parquet(f.replace("__policy", "__windows")); w["dataset"], w["scenario"], w["seed"] = ds, scen, seed
        m = json.load(open(f.replace("__policy.parquet", "__meta.json")))
        pol.append(p); win.append(w); meta.append(m)
    return pd.concat(pol, ignore_index=True), pd.concat(win, ignore_index=True), meta


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def paired_stats(caught_p, caught_ref, denom, rng, B=2000, R=10000):
    """Recall of policy and reference over windows (seed-averaged counts), difference,
    window-bootstrap percentile CIs and a two-sided sign-flip permutation p-value."""
    caught_p, caught_ref, denom = map(np.asarray, (caught_p, caught_ref, denom))
    D = denom.sum()
    if D == 0:
        return dict(recall=np.nan, ref=np.nan, diff=np.nan, lo=np.nan, hi=np.nan,
                    diff_lo=np.nan, diff_hi=np.nan, p=np.nan)
    r, r0 = caught_p.sum() / D, caught_ref.sum() / D
    n = len(denom)
    idx = rng.integers(0, n, size=(B, n))
    dd = denom[idx].sum(1)
    ok = dd > 0
    rb = caught_p[idx].sum(1)[ok] / dd[ok]
    r0b = caught_ref[idx].sum(1)[ok] / dd[ok]
    d = caught_p - caught_ref
    T = d.sum()
    signs = rng.choice([-1.0, 1.0], size=(R, n))
    Tp = (signs * d).sum(1)
    p = (1 + np.sum(np.abs(Tp) >= abs(T) - 1e-12)) / (R + 1) if np.any(d != 0) else 1.0
    return dict(recall=r, ref=r0, diff=r - r0,
                lo=np.percentile(rb, 2.5), hi=np.percentile(rb, 97.5),
                diff_lo=np.percentile(rb - r0b, 2.5), diff_hi=np.percentile(rb - r0b, 97.5), p=p)


def seed_avg(pol, win, dataset, scenario, c):
    """Per-window seed-averaged caught counts per policy, and window totals."""
    P = pol[(pol.dataset == dataset) & (pol.scenario == scenario) & (pol.c == c)]
    W = win[(win.dataset == dataset) & (win.scenario == scenario) & (win.c == c)]
    Wm = W.groupby("wi")[["frauds", "novel_frauds", "fraud_amt", "k", "n"]].mean().sort_index()
    piv = {col: P.pivot_table(index="wi", columns="policy", values=col, aggfunc="mean").sort_index()
           for col in ["caught", "caught_novel", "caught_amt"]}
    per_seed = P.groupby(["policy", "seed"])[["caught", "caught_novel"]].sum().reset_index()
    tot = W.groupby("seed")[["frauds", "novel_frauds"]].sum()
    per_seed["recall"] = per_seed.apply(lambda x: x.caught / tot.loc[x.seed, "frauds"], axis=1)
    per_seed["novel_recall"] = per_seed.apply(
        lambda x: x.caught_novel / tot.loc[x.seed, "novel_frauds"] if tot.loc[x.seed, "novel_frauds"] else np.nan, axis=1)
    sd = per_seed.groupby("policy")[["recall", "novel_recall"]].std(ddof=1)
    return piv, Wm, sd
