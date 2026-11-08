"""Compare two run directories policy-by-policy and window-by-window.

Usage: python scripts/compare_runs.py results/runs_v1_1 results/runs [--ignore-policy NAME ...]

Policies listed with --ignore-policy (e.g. variants added by a later amendment) are dropped from the
newer directory before comparison; every other policy row, window row and metadata field (except
run time and later-amendment hashes) must be identical.
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("old"); ap.add_argument("new")
ap.add_argument("--ignore-policy", nargs="*", default=[])
a = ap.parse_args()
same = diff = rows = 0
bad = []
for f in sorted(glob.glob(os.path.join(a.old, "*__policy.parquet"))):
    stem = os.path.basename(f)[: -len("__policy.parquet")]
    g = os.path.join(a.new, os.path.basename(f))
    if not os.path.exists(g):
        bad.append(f"missing {stem}"); diff += 1; continue
    o = pd.read_parquet(f).reset_index(drop=True)
    n = pd.read_parquet(g)
    n = n[~n.policy.isin(a.ignore_policy)].reset_index(drop=True)
    wo = pd.read_parquet(f.replace("__policy", "__windows"))
    wn = pd.read_parquet(g.replace("__policy", "__windows"))
    mo = json.load(open(f.replace("__policy.parquet", "__meta.json")))
    mn = json.load(open(g.replace("__policy.parquet", "__meta.json")))
    for m in (mo, mn):
        m.pop("seconds", None); m.pop("amendment_v1_2_sha256", None)
        for p in a.ignore_policy:
            m.get("rho_trace", {}).pop(p, None)
    ok = o.equals(n) and wo.equals(wn) and mo == mn
    rows += len(o)
    if ok:
        same += 1
    else:
        diff += 1; bad.append(stem)
print(f"runs identical: {same}  different: {diff}  policy rows compared: {rows}")
for b in bad:
    print("  DIFF", b)
sys.exit(1 if diff else 0)
