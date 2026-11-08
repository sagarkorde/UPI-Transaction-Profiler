"""Operational metrics for held-out scenarios (internal review, v1.4; post-hoc, descriptive).

1. Capacity in held-out scenarios. The protocol sets capacity from the pre-test fraud rate q (A3). In a held-out
   scenario the held-out group's frauds are absent from the pre-test data, so q and the budget shrink. We report
   the test-period budget relative to the stationary (base) budget and relative to the held-out frauds.
2. Time to first detection. For emerging fraud the first retrieved cases matter most: they give analysts labels to
   act on. Per scenario and seed (c = 1), we record the first test window in which the held-out mechanism occurs,
   the first window in which a policy retrieves one of its frauds, the delay between the two (in windows; a window
   is one day on BankSim and TabFormer and ten minutes on ULB), and the novel frauds retrieved in the first seven
   windows after the mechanism first occurs. Never-detected cases are reported as censored.

Writes results/analysis/capacity_heldout.csv and results/analysis/time_to_detection.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.environ.get("RUN_DIR", os.path.join(ROOT, "results", "runs"))
OUT = os.path.join(ROOT, "results", "analysis")
C1 = 1.0
POL = ["SUP", "DRT", "FUSE60", "STACK", "DRT-LD", "DRT-AMT", "SUP-NOID", "DRT-NOID", "DRT-ADAPT", "SUP-IDF", "DRT-IDF"]

runs = []
for f in sorted(glob.glob(os.path.join(RUN, "*__policy.parquet"))):
    ds, sc, sd = os.path.basename(f).split("__")[:3]
    runs.append((ds, sc, int(sd[1:]), f))

# ---- 1. capacity
cap = []
for ds in sorted({r[0] for r in runs}):
    base = [r for r in runs if r[0] == ds and r[1] == "base"]
    kb = np.mean([pd.read_parquet(r[3].replace("__policy", "__windows")).query("c == @C1").k.sum() for r in base])
    for sc in sorted({r[1] for r in runs if r[0] == ds and r[1] != "base"}):
        ws = [pd.read_parquet(r[3].replace("__policy", "__windows")).query("c == @C1")
              for r in runs if r[0] == ds and r[1] == sc]
        k = np.mean([w.k.sum() for w in ws]); fr = ws[0].frauds.sum(); nov = ws[0].novel_frauds.sum()
        cap.append(dict(dataset=ds, scenario=sc, budget=k, base_budget=kb, budget_vs_base=k / kb,
                        test_frauds=fr, novel_frauds=nov, budget_vs_frauds=k / fr, novel_share=nov / fr))
cap = pd.DataFrame(cap)
cap.to_csv(os.path.join(OUT, "capacity_heldout.csv"), index=False)

# ---- 2. time to first detection
rows = []
for ds, sc, sd, f in runs:
    if sc == "base":
        continue
    p = pd.read_parquet(f).query("c == @C1")
    w = pd.read_parquet(f.replace("__policy", "__windows")).query("c == @C1").sort_values("wi")
    nw = len(w)
    occ = w.wi[w.novel_frauds > 0]
    if occ.empty:
        continue
    first_occ = int(occ.iloc[0])
    for pol in POL:
        q = p[p.policy == pol].sort_values("wi")
        if q.empty:
            continue
        hit = q.wi[q.caught_novel > 0]
        first = int(hit.iloc[0]) if not hit.empty else None
        early = q[(q.wi >= first_occ) & (q.wi < first_occ + 7)].caught_novel.sum()
        early_tot = w[(w.wi >= first_occ) & (w.wi < first_occ + 7)].novel_frauds.sum()
        rows.append(dict(dataset=ds, scenario=sc, seed=sd, policy=pol, windows=nw, first_occurrence=first_occ,
                         first_detection=first, detected=first is not None,
                         delay=(first - first_occ) if first is not None else np.nan,
                         novel_first7=float(early), novel_total_first7=float(early_tot)))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(OUT, "time_to_detection.csv"), index=False)

pd.set_option("display.width", 220)
print(cap.round(3).to_string(index=False))
summ = (t.groupby(["dataset", "scenario", "policy"])
        .agg(detected=("detected", "mean"), median_delay=("delay", "median"),
             first7=("novel_first7", "mean"), of=("novel_total_first7", "mean")).reset_index())
print(summ.round(2).to_string(index=False))
print("wrote capacity_heldout.csv, time_to_detection.csv")
