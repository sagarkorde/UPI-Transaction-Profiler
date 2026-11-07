"""Dev-only analysis (held-in periods): which ordering of the reserved channel works best?"""
import glob, sys, os, numpy as np, pandas as pd
sys.path.insert(0, "src")
from dstriage import policies as P
import yaml
dp = yaml.safe_load(open("configs/protocol.yaml"))["policies"]["drt_adaptive"]
rhos = [0.05, 0.1, 0.2, 0.3, 0.5]; ws = [0.5, 0.6, 0.7, 0.8, 0.9]
rows = []
for f in sorted(glob.glob("results/dev_runs/*__scores.parquet")):
    d = pd.read_parquet(f); t = d[d.period == "test"].reset_index(drop=True)
    sl = P.window_slices(t.w.to_numpy()); n_w = np.array([b - a for a, b in sl])
    k = P.allocate_budget(n_w, 1.0 * d.pi_hat.iloc[0]); y = t.y.to_numpy(); nov = t.novel.to_numpy()
    s, sp, ap = t.s_raw.to_numpy(), t.s_pct.to_numpy(), t.a_pct.to_numpy()
    name = os.path.basename(f).replace("__scores.parquet", "")
    def ev(sel): return (sel & (y == 1)).sum() / y.sum(), (sel & nov).sum() / max(nov.sum(), 1)
    sup = ev(P.topk(s, s, sl, k))
    chans = {"a": ap + 1e-9 * sp, "avg": (sp + ap) / 2, "min": np.minimum(sp, ap) + 1e-6 * (sp + ap),
             "geo": np.sqrt(sp * ap)}
    for ch, b in chans.items():
        for r in rhos:
            rec, nr = ev(P.drt(s, b, y, sl, k, dp, None, False, r)[0])
            rows.append((name, "DRT-" + ch, r, rec, nr, sup[0], sup[1]))
    for w in ws:
        rec, nr = ev(P.topk(w * sp + (1 - w) * ap, s, sl, k))
        rows.append((name, "FUSE", w, rec, nr, sup[0], sup[1]))
R = pd.DataFrame(rows, columns=["run", "pol", "param", "rec", "nov", "sup_rec", "sup_nov"])
R["rel_cost"] = 1 - R.rec / R.sup_rec
pd.set_option("display.width", 250)
for run, g in R.groupby("run"):
    print(run, "SUP rec=%.3f nov=%.3f" % (g.sup_rec.iloc[0], g.sup_nov.iloc[0]))
    print(g.pivot_table(index="pol", columns="param", values=["rec", "nov"]).round(3).to_string())
# frontier summary: novel recall at matched relative cost (linear interpolation), cost budgets 2%,5%,10%
def nov_at(g, cmax):
    g = g.sort_values("rel_cost"); xs = np.r_[0, g.rel_cost.clip(lower=0)]; ys = np.r_[g.sup_nov.iloc[0], g.nov]
    best = ys[xs <= cmax].max() if (xs <= cmax).any() else np.nan
    return best
out = []
for (run, pol), g in R.groupby(["run", "pol"]):
    if g.sup_nov.iloc[0] == 0 and "base" not in run and g.nov.max() == 0: pass
    out.append((run, pol, *[nov_at(g, c) for c in (0.02, 0.05, 0.10, 0.20)]))
O = pd.DataFrame(out, columns=["run", "pol", "c2", "c5", "c10", "c20"])
print("\nBest novel recall achievable within relative recall cost <= 2/5/10/20%")
print(O[~O.run.str.contains("base")].pivot_table(index="run", columns="pol", values=["c5", "c10"]).round(3).to_string())
print(O[~O.run.str.contains("base")].groupby("pol")[["c2", "c5", "c10", "c20"]].mean().round(3))
