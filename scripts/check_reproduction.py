"""Compare a re-run (results/runs_v11) with the original v1.0 runs (results/runs) for every policy
and window present in both. ULB held-out scenarios are excluded (redefined by amendment v1.1)."""
import glob, os, sys, json
import pandas as pd
new_dir, old_dir = sys.argv[1], sys.argv[2]
ok = bad = skipped = 0
for f in sorted(glob.glob(os.path.join(old_dir, "*__policy.parquet"))):
    name = os.path.basename(f)
    if name.startswith("ulb__cluster"):
        skipped += 1; continue
    g = os.path.join(new_dir, name)
    if not os.path.exists(g):
        continue
    a = pd.read_parquet(f); b = pd.read_parquet(g)
    b = b[b.policy.isin(set(a.policy))].reset_index(drop=True)
    w1 = pd.read_parquet(f.replace("__policy", "__windows")); w2 = pd.read_parquet(g.replace("__policy", "__windows"))
    m1 = json.load(open(f.replace("__policy.parquet", "__meta.json"))); m2 = json.load(open(g.replace("__policy.parquet", "__meta.json")))
    same = a.equals(b) and w1.equals(w2) and m1["signal"] == m2["signal"]
    ok += same; bad += (not same)
    if not same: print("DIFFERENT:", name)
print(f"identical: {ok}  different: {bad}  skipped (redefined ULB scenarios): {skipped}")
sys.exit(1 if bad else 0)
