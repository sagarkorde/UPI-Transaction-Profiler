"""Pre-registered confirmatory analysis of amendment v1.6 (configs/amendment_v1_6.yaml).

Written and committed before any replication run existed. It reads only the replication runs and
applies the hypotheses, tests, multiplicity correction and decision rules fixed in the amendment.

Usage: python scripts/confirm_v1_6.py [--runs results/replication_v1_6/runs]
Outputs (next to the runs directory):
  confirmatory.json        H1-H5 outcomes, with every number they rest on
  descriptive_heldout.csv  held-out recall of every policy at every budget (pooled over held-out groups)
  descriptive_base.csv     base-scenario recall of every policy at every budget
  descriptive_groups.csv   per-group held-out recall at c = 1
  time_to_detection.csv    first detection per group, seed and policy at c = 1 (censored if never)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dstriage import experiment as E  # noqa: E402
from dstriage import stats as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.join(E.ROOT, "results", "replication_v1_6", "runs"))
ap.add_argument("--published", default=os.path.join(E.ROOT, "results", "runs"),
                help="published v1.5 runs, for the bit-identity check of the banksim_u base scenario")
a = ap.parse_args()
OUT = os.path.dirname(os.path.abspath(a.runs))

cfg, cfg_hash = E.load_cfg()
am_path = os.path.join(E.ROOT, "configs", "amendment_v1_6.yaml")
am, am_hash = yaml.safe_load(open(am_path)), E.sha256_file(am_path)
cf = am["confirmatory"]
st = cfg["statistics"]
B, R = st["bootstrap_resamples"], st["permutation_resamples"]
C1 = cf["budget_multiplier"]
RHO = cfg["policies"]["drt_rho"]
ALPHA = cf["alpha"]
REPS = am["replications"]


def rng(i):
    return np.random.default_rng([*cf["rng_seed"], i])


pol, win, meta = S.load_runs(a.runs)
assert sorted(set(pol.dataset)) == sorted(REPS), sorted(set(pol.dataset))
for m in meta:  # every run must come from the frozen protocol and this amendment
    assert m["config_sha256"] == cfg_hash
    assert m["amendment_v1_6_sha256"] == am_hash
    assert m["replication_of"] == REPS[m["dataset"]]["inherits"]
seeds = sorted(set(pol.seed))
assert seeds == cfg["seeds"], seeds
scen = {ds: sorted(set(pol[(pol.dataset == ds) & (pol.scenario != "base")].scenario)) for ds in REPS}
for ds in REPS:
    for sc in ["base"] + scen[ds]:
        got = sorted(set(pol[(pol.dataset == ds) & (pol.scenario == sc)].seed))
        assert got == seeds, (ds, sc, got)
man = json.load(open(os.path.join(a.runs, "_manifest.json"))) if os.path.exists(
    os.path.join(a.runs, "_manifest.json")) else {}


def pooled(ds, c):
    """Seed-averaged per-window counts summed across held-out scenarios (same test windows)."""
    acc = None
    for sc in scen[ds]:
        piv, Wm, _ = S.seed_avg(pol, win, ds, sc, c)
        part = dict(piv); part["frauds"], part["novel"] = Wm.frauds, Wm.novel_frauds
        acc = part if acc is None else {k: acc[k] + part[k] for k in acc}
    return acc


res = dict(amendment_sha256=am_hash, config_sha256=cfg_hash, seeds=seeds, n_runs=len(meta),
           heldout_groups=scen, decision_rule=cf["success"],
           test_frauds_by_group={ds: man.get("scenarios", {}).get(ds, {}).get("test_fraud_counts") for ds in REPS},
           git_commits=man.get("git_commits", [man.get("git_commit")]))

# ---------------------------------------------------------------- H1-H4 (one Holm family)
H = cf["hypotheses"]
testable = []
for i, h in enumerate(sorted(H), start=1):
    spec = H[h]; ds = spec["replication"]
    if not scen[ds]:
        res[h] = dict(spec, status="untestable: no held-out group with >= 10 test frauds")
        continue
    acc = pooled(ds, C1)
    N = acc["novel"].to_numpy()
    r = S.paired_stats(acc["caught_novel"][spec["policy"]].to_numpy(),
                       acc["caught_novel"][spec["reference"]].to_numpy(), N, rng(i), B, R)
    res[h] = dict(spec, heldout_recall=r["recall"], ref_recall=r["ref"], ci=[r["lo"], r["hi"]],
                  diff=r["diff"], diff_ci=[r["diff_lo"], r["diff_hi"]], p=r["p"],
                  heldout_frauds_pooled=float(N.sum()), windows=int(len(N)))
    testable.append(h)
adj = S.holm([res[h]["p"] for h in testable]) if testable else []
for h, p_adj in zip(testable, adj):
    res[h]["p_holm"] = float(p_adj)
    res[h]["supported"] = bool(res[h]["diff"] > 0 and p_adj < ALPHA)

# ---------------------------------------------------------------- H5 (stationary cost, one-sided bound)
h5 = cf["H5_stationary_cost"]; ds = h5["replication"]
piv, Wm, _ = S.seed_avg(pol, win, ds, "base", C1)
s_sup, s_idf = piv["caught"]["SUP"].to_numpy(), piv["caught"]["DRT-IDF"].to_numpy()
F = Wm.frauds.to_numpy()
rel = 1 - s_idf.sum() / s_sup.sum()
g = rng(5); n = len(F); bs = []
for _ in range(h5["resamples"]):
    ii = g.integers(0, n, n)
    if s_sup[ii].sum() > 0:
        bs.append(1 - s_idf[ii].sum() / s_sup[ii].sum())
ub = float(np.percentile(bs, 95))
res["H5"] = dict(replication=ds, statement=h5["statement"].strip(), rel_cost=float(rel), upper95=ub, bound=RHO,
                 supported=bool(ub <= RHO), recall_sup=float(s_sup.sum() / F.sum()),
                 recall_drt_idf=float(s_idf.sum() / F.sum()), abs_cost=float((s_sup.sum() - s_idf.sum()) / F.sum()),
                 test_frauds=float(F.sum()), windows=int(n))

# ---------------------------------------------------------------- secondary: v1.0 primary hypothesis
res["secondary_DRT_vs_SUP"] = {}
for j, ds in enumerate(sorted(REPS)):
    if not scen[ds]:
        continue
    acc = pooled(ds, C1)
    r = S.paired_stats(acc["caught_novel"]["DRT"].to_numpy(), acc["caught_novel"]["SUP"].to_numpy(),
                       acc["novel"].to_numpy(), rng(10 + j), B, R)
    res["secondary_DRT_vs_SUP"][ds] = dict(heldout_recall=r["recall"], ref_recall=r["ref"], diff=r["diff"],
                                           diff_ci=[r["diff_lo"], r["diff_hi"]], p=r["p"])

# ---------------------------------------------------------------- data description
res["split"] = {}
for ds in REPS:
    m0 = [m for m in meta if m["dataset"] == ds and m["scenario"] == "base"][0]
    res["split"][ds] = dict(rows=m0.get("n"), frauds=m0.get("frauds"))

# ---------------------------------------------------------------- bit-identity check (banksim_u base)
chk = []
for sd_ in seeds:
    f_new = os.path.join(a.runs, f"banksim_u__base__s{sd_}__policy.parquet")
    f_pub = os.path.join(a.published, f"banksim__base__s{sd_}__policy.parquet")
    if os.path.exists(f_new) and os.path.exists(f_pub):
        x, y = pd.read_parquet(f_new), pd.read_parquet(f_pub)
        chk.append(bool(x.shape == y.shape and x.equals(y)))
res["banksim_u_base_identical_to_published"] = dict(seeds_checked=len(chk), identical=int(sum(chk)))

# ---------------------------------------------------------------- descriptive tables
d_held, d_base, grp, ttd = [], [], [], []
for ds in sorted(REPS):
    for c in cfg["budget_multipliers"]:
        if scen[ds]:
            acc = pooled(ds, c)
            for p in acc["caught_novel"].columns:
                d_held.append(dict(replication=ds, c=c, policy=p,
                                   heldout_recall=acc["caught_novel"][p].sum() / acc["novel"].sum(),
                                   overall_recall=acc["caught"][p].sum() / acc["frauds"].sum()))
        piv, Wm, sd = S.seed_avg(pol, win, ds, "base", c)
        for p in piv["caught"].columns:
            d_base.append(dict(replication=ds, c=c, policy=p, base_recall=piv["caught"][p].sum() / Wm.frauds.sum(),
                               sd_seed=sd.loc[p, "recall"]))
    for sc in scen[ds]:
        piv, Wm, sd = S.seed_avg(pol, win, ds, sc, C1)
        for p in piv["caught"].columns:
            grp.append(dict(replication=ds, group=sc, policy=p,
                            heldout_recall=piv["caught_novel"][p].sum() / Wm.novel_frauds.sum(),
                            sd_seed=sd.loc[p, "novel_recall"], heldout_frauds=float(Wm.novel_frauds.sum())))
        for sd_ in seeds:
            w = win[(win.dataset == ds) & (win.scenario == sc) & (win.c == C1) & (win.seed == sd_)].sort_values("wi")
            occ = w[w.novel_frauds > 0].wi
            if occ.empty:
                continue
            first_occ = int(occ.iloc[0])
            for p in ["SUP", "DRT", "DRT-AMT", "SUP-IDF", "DRT-IDF"]:
                q = pol[(pol.dataset == ds) & (pol.scenario == sc) & (pol.c == C1) & (pol.seed == sd_)
                        & (pol.policy == p)].sort_values("wi")
                hit = q[q.caught_novel > 0].wi
                first = int(hit.iloc[0]) if not hit.empty else None
                ttd.append(dict(replication=ds, group=sc, seed=sd_, policy=p, first_occurrence=first_occ,
                                first_detection=first, detected=first is not None,
                                delay=(first - first_occ) if first is not None else np.nan))
pd.DataFrame(d_held).to_csv(os.path.join(OUT, "descriptive_heldout.csv"), index=False)
pd.DataFrame(d_base).to_csv(os.path.join(OUT, "descriptive_base.csv"), index=False)
pd.DataFrame(grp).to_csv(os.path.join(OUT, "descriptive_groups.csv"), index=False)
pd.DataFrame(ttd).to_csv(os.path.join(OUT, "time_to_detection.csv"), index=False)

json.dump(res, open(os.path.join(OUT, "confirmatory.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k in ("H1", "H2", "H3", "H4", "H5")}, indent=1, default=float))
