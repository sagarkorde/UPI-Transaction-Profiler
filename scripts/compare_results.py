"""Compare a regenerated results/analysis directory with a reference copy.

Usage: python scripts/compare_results.py results/analysis released_results/analysis [--tol 1e-9]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("new"); ap.add_argument("ref"); ap.add_argument("--tol", type=float, default=1e-9)
a = ap.parse_args()
worst, bad = 0.0, []
for f in sorted(glob.glob(os.path.join(a.ref, "*.csv"))):
    g = os.path.join(a.new, os.path.basename(f))
    if not os.path.exists(g):
        bad.append(f"missing {os.path.basename(f)}"); continue
    r, n = pd.read_csv(f), pd.read_csv(g)
    if r.shape != n.shape or list(r.columns) != list(n.columns):
        bad.append(f"shape/columns differ in {os.path.basename(f)}"); continue
    num = r.select_dtypes("number").columns
    d = np.nanmax(np.abs(r[num].to_numpy(float) - n[num].to_numpy(float))) if len(num) else 0.0
    same_nan = np.array_equal(np.isnan(r[num].to_numpy(float)), np.isnan(n[num].to_numpy(float)))
    worst = max(worst, d)
    status = "OK" if (d <= a.tol and same_nan and r.drop(columns=num).equals(n.drop(columns=num))) else "DIFF"
    if status == "DIFF":
        bad.append(os.path.basename(f))
    print(f"{status:4s} {os.path.basename(f):24s} max|Δ| = {d:.3g}")
print(f"\nworst absolute difference: {worst:.3g}")
sys.exit(1 if bad else 0)
