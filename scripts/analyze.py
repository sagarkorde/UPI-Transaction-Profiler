"""Produce all statistics used in the paper from results/runs (no manual numbers).

Outputs (results/analysis/):
  main_base.csv          recall@budget per policy, dataset, budget (base scenario) + CIs + Holm p
  emerging_pooled.csv    overall & novel recall per policy (held-out scenarios pooled) + tests
  emerging_groups.csv    per held-out group
  frontier_points.csv    DRT@rho / FUSE@w points (overall vs novel recall; base cost)
  frontier_summary.csv   best novel recall within relative cost tau, DRT vs FUSE, bootstrap CI
  ablation.csv           channel-signal and adaptivity ablations
  signals.csv            AP / ROC-AUC (overall and novel-only) mean +- sd over seeds
  premium.csv            calibration-estimated vs realised stationary premium of rho
  summary.json           headline numbers
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dstriage import experiment as E  # noqa: E402
from dstriage import stats as S  # noqa: E402

cfg, cfg_hash = E.load_cfg()
st = cfg["statistics"]
B, R = st["bootstrap_resamples"], st["permutation_resamples"]
RUN = os.environ.get("RUN_DIR", os.path.join(E.ROOT, "results", "runs"))
OUT = os.environ.get("OUT_DIR", os.path.join(E.ROOT, "results", "analysis"))
os.makedirs(OUT, exist_ok=True)
pol, win, meta = S.load_runs(RUN)
assert all(m["config_sha256"] == cfg_hash for m in meta), "results from a different protocol"
DATASETS = [d for d in ["ulb", "banksim", "tabformer"] if d in set(pol.dataset)]
MAIN = E.MAIN_POLICIES
BUDGETS = cfg["budget_multipliers"]
C1 = cfg["primary_budget_multiplier"]


def rng(*key):
    return np.random.default_rng([st["rng_seed"], *key])


# --------------------------------------------------------------- 1. base scenario
rows = []
for di, ds in enumerate(DATASETS):
    for c in BUDGETS:
        piv, Wm, sd = S.seed_avg(pol, win, ds, "base", c)
        ref = piv["caught"]["SUP"].to_numpy()
        block = []
        for pi_, P in enumerate(MAIN):
            r = S.paired_stats(piv["caught"][P].to_numpy(), ref, Wm.frauds.to_numpy(),
                               rng(di, int(c * 100), pi_), B, R)
            amt = piv["caught_amt"][P].sum() / Wm.fraud_amt.sum()
            block.append(dict(dataset=ds, c=c, policy=P, **r, sd_seed=sd.loc[P, "recall"],
                              amount_recall=amt, test_frauds=Wm.frauds.sum(), windows=len(Wm)))
        ps = [b["p"] for b in block if b["policy"] != "SUP"]
        adj = S.holm(ps); j = 0
        for b in block:
            if b["policy"] != "SUP":
                b["p_holm"] = adj[j]; j += 1
            else:
                b["p_holm"] = np.nan
        rows += block
main = pd.DataFrame(rows)
main.to_csv(os.path.join(OUT, "main_base.csv"), index=False)


# --------------------------------------------------------------- 2. emerging fraud
def pooled(ds, c, scen_list):
    """Sum seed-averaged per-window counts across held-out scenarios (same test windows)."""
    acc = None
    for sc in scen_list:
        piv, Wm, _ = S.seed_avg(pol, win, ds, sc, c)
        part = {k: v for k, v in piv.items()}
        part["frauds"], part["novel"] = Wm.frauds, Wm.novel_frauds
        if acc is None:
            acc = part
        else:
            for k in acc:
                acc[k] = acc[k] + part[k]
    return acc


FB = ["DRT-ADAPT-D1", "DRT-ADAPT-D3", "DRT-ADAPT-D7", "DRT-ADAPT-NOISY", "DRT-ADAPT-AUDIT"]
EMERG_POL = MAIN + ["DRT-ADAPT", "DRT-UNC", "DRT-RAND"] + [f for f in FB if f in set(pol.policy)]
erows, grows = [], []
scen_of = {ds: sorted(set(pol[(pol.dataset == ds) & (pol.scenario != "base")].scenario)) for ds in DATASETS}
for di, ds in enumerate(DATASETS):
    if not scen_of[ds]:
        continue
    for c in BUDGETS:
        acc = pooled(ds, c, scen_of[ds])
        ref, refn = acc["caught"]["SUP"].to_numpy(), acc["caught_novel"]["SUP"].to_numpy()
        blk = []
        for pi_, P in enumerate(EMERG_POL):
            o = S.paired_stats(acc["caught"][P].to_numpy(), ref, acc["frauds"].to_numpy(), rng(10 + di, int(c * 100), pi_), B, R)
            n = S.paired_stats(acc["caught_novel"][P].to_numpy(), refn, acc["novel"].to_numpy(), rng(20 + di, int(c * 100), pi_), B, R)
            blk.append(dict(dataset=ds, c=c, policy=P, groups=len(scen_of[ds]),
                            overall=o["recall"], overall_lo=o["lo"], overall_hi=o["hi"],
                            overall_diff=o["diff"], overall_diff_lo=o["diff_lo"], overall_diff_hi=o["diff_hi"], overall_p=o["p"],
                            novel=n["recall"], novel_lo=n["lo"], novel_hi=n["hi"],
                            novel_diff=n["diff"], novel_diff_lo=n["diff_lo"], novel_diff_hi=n["diff_hi"], novel_p=n["p"],
                            novel_frauds=float(acc["novel"].sum())))
        for key in ["overall_p", "novel_p"]:
            idx = [i for i, b in enumerate(blk) if b["policy"] in MAIN and b["policy"] != "SUP"]
            adj = S.holm([blk[i][key] for i in idx])
            for i, a in zip(idx, adj):
                blk[i][key + "_holm"] = a
        erows += blk
        if c == C1:
            for sc in scen_of[ds]:
                piv, Wm, sd = S.seed_avg(pol, win, ds, sc, c)
                Wg = win[(win.dataset == ds) & (win.scenario == sc) & (win.c == c)]
                cap_novel = Wg.groupby("seed").k.sum().mean() / max(Wg.groupby("seed").novel_frauds.sum().mean(), 1)
                share_novel = Wg.groupby("seed").novel_frauds.sum().mean() / Wg.groupby("seed").frauds.sum().mean()
                for P in EMERG_POL:
                    grows.append(dict(dataset=ds, group=sc, policy=P, capacity_over_novel=cap_novel, novel_share=share_novel,
                                      overall=piv["caught"][P].sum() / Wm.frauds.sum(),
                                      novel=piv["caught_novel"][P].sum() / Wm.novel_frauds.sum(),
                                      novel_frauds=Wm.novel_frauds.sum(), sd_novel=sd.loc[P, "novel_recall"]))
emerg = pd.DataFrame(erows); emerg.to_csv(os.path.join(OUT, "emerging_pooled.csv"), index=False)
pd.DataFrame(grows).to_csv(os.path.join(OUT, "emerging_groups.csv"), index=False)

# --------------------------------------------------------------- 3. frontier
pc = cfg["policies"]
fam = {"DRT": [("SUP", 0.0)] + [(f"DRT@{r:.2f}", r) for r in pc["drt_frontier_rhos"]],
       "FUSE": [("SUP", 1.0)] + [(f"FUSE@{w:.2f}", w) for w in sorted(pc["fuse_frontier_weights"], reverse=True)]}
fpts, fsum = [], []
taus = st["frontier_cost_levels"]
for di, ds in enumerate(DATASETS):
    if not scen_of[ds]:
        continue
    acc = pooled(ds, C1, scen_of[ds])
    pivb, Wb, _ = S.seed_avg(pol, win, ds, "base", C1)
    F, N = acc["frauds"].to_numpy(), acc["novel"].to_numpy()
    for f, items in fam.items():
        for name, par in items:
            fpts.append(dict(dataset=ds, family=f, param=par, policy=name,
                             overall=acc["caught"][name].sum() / F.sum(),
                             novel=acc["caught_novel"][name].sum() / N.sum(),
                             base_overall=pivb["caught"][name].sum() / Wb.frauds.sum()))

    def best(idx, f, tau):
        Fi, Ni = F[idx].sum(), N[idx].sum()
        sup = acc["caught"]["SUP"].to_numpy()[idx].sum() / Fi
        b = 0.0
        for name, _ in fam[f]:
            ov = acc["caught"][name].to_numpy()[idx].sum() / Fi
            if 1 - ov / sup <= tau + 1e-12:
                b = max(b, acc["caught_novel"][name].to_numpy()[idx].sum() / Ni)
        return b

    n = len(F); g = rng(30 + di)
    full = np.arange(n)
    for tau in taus:
        d0, f0 = best(full, "DRT", tau), best(full, "FUSE", tau)
        diffs = []
        for _ in range(B):
            ii = g.integers(0, n, n)
            if N[ii].sum() == 0:
                continue
            diffs.append(best(ii, "DRT", tau) - best(ii, "FUSE", tau))
        diffs = np.array(diffs)
        fsum.append(dict(dataset=ds, tau=tau, drt=d0, fuse=f0, diff=d0 - f0,
                         diff_lo=np.percentile(diffs, 2.5), diff_hi=np.percentile(diffs, 97.5),
                         p_boot_le0=float(np.mean(diffs <= 0))))
pd.DataFrame(fpts).to_csv(os.path.join(OUT, "frontier_points.csv"), index=False)
pd.DataFrame(fsum).to_csv(os.path.join(OUT, "frontier_summary.csv"), index=False)

# --------------------------------------------------------------- 4. ablations (c = 1)
ab = []
for di, ds in enumerate(DATASETS):
    piv, Wm, _ = S.seed_avg(pol, win, ds, "base", C1)
    ref = piv["caught"]["SUP"].to_numpy()
    for pi_, P in enumerate(["DRT", "DRT-ECOD", "DRT-UNC", "DRT-RAND", "DRT-ADAPT"] + FB):
        if P not in piv["caught"]:
            continue
        r = S.paired_stats(piv["caught"][P].to_numpy(), ref, Wm.frauds.to_numpy(), rng(40 + di, pi_), B, R)
        row = dict(dataset=ds, policy=P, base_recall=r["recall"], base_cost=r["ref"] - r["recall"],
                   base_cost_lo=-r["diff_hi"], base_cost_hi=-r["diff_lo"])
        e = emerg[(emerg.dataset == ds) & (emerg.c == C1) & (emerg.policy == P)]
        if len(e):
            row.update(novel=e.novel.iloc[0], novel_lo=e.novel_lo.iloc[0], novel_hi=e.novel_hi.iloc[0],
                       emerg_overall=e.overall.iloc[0])
        ab.append(row)
pd.DataFrame(ab).to_csv(os.path.join(OUT, "ablation.csv"), index=False)

# --------------------------------------------------------------- 5. signal quality
sg = []
for m in meta:
    for k, v in m["signal"].items():
        if k == "calibration":
            continue
        sg.append(dict(dataset=m["dataset"], scenario=m["scenario"], seed=m["seed"], signal=k, **v))
sg = pd.DataFrame(sg)
sg["kind"] = np.where(sg.scenario == "base", "base", "holdout")
sig = sg.groupby(["dataset", "kind", "signal"]).agg(
    ap=("ap", "mean"), ap_sd=("ap", "std"), auc=("auc", "mean"), auc_sd=("auc", "std"),
    ap_novel=("ap_novel", "mean"), ap_novel_sd=("ap_novel", "std"),
    auc_novel=("auc_novel", "mean"), auc_novel_sd=("auc_novel", "std")).reset_index()
sig.to_csv(os.path.join(OUT, "signals.csv"), index=False)

# ------------------------------------------- 5b. exploratory: per-group signal quality vs DRT gain
gq = sg[sg.scenario != "base"].groupby(["dataset", "scenario", "signal"])[["ap_novel", "auc_novel"]].mean().unstack("signal")
gq.columns = [f"{a}_{b}" for a, b in gq.columns]
gq = gq.reset_index().rename(columns={"scenario": "group"})
gg = pd.DataFrame(grows)
if len(gg):
    piv_g = gg[gg.policy.isin(["SUP", "DRT"])].pivot_table(index=["dataset", "group"], columns="policy", values="novel").reset_index()
    gq = gq.merge(piv_g, on=["dataset", "group"], how="left")
    gq["drt_gain"] = gq["DRT"] - gq["SUP"]
    gq["ano_adv_ap"] = gq["ap_novel_ANO"] - gq["ap_novel_SUP"]
gq.to_csv(os.path.join(OUT, "group_signal.csv"), index=False)

# --------------------------------------------------------------- 6. premium predictability
pr = []
for ds in DATASETS:
    piv, Wm, _ = S.seed_avg(pol, win, ds, "base", C1)
    sup = piv["caught"]["SUP"].sum() / Wm.frauds.sum()
    ms = [m for m in meta if m["dataset"] == ds and m["scenario"] == "base"]
    for r in pc["drt_frontier_rhos"]:
        est = np.mean([m["premium_calib"][str(C1)][f"{r:.2f}"] for m in ms])
        real = sup - piv["caught"][f"DRT@{r:.2f}"].sum() / Wm.frauds.sum()
        pr.append(dict(dataset=ds, rho=r, premium_calib=est, premium_test=real))
pr = pd.DataFrame(pr); pr.to_csv(os.path.join(OUT, "premium.csv"), index=False)

# --------------------------------------------- 6b. assumption A2: SUP precision by rank band
# SUP's top-k sets are nested across budgets c, so frauds retrieved between consecutive budgets
# divided by the extra reviews give the precision of successive rank bands (stationary test period).
a2 = []
for di, ds in enumerate(DATASETS):
    cs = sorted(BUDGETS)
    prev_c = prev_k = None
    for c in cs:
        piv, Wm, _ = S.seed_avg(pol, win, ds, "base", c)
        caught, k = piv["caught"]["SUP"].to_numpy(), Wm.k.to_numpy()
        if prev_c is None:
            band_f, band_k, lab = caught, k, f"(0, {c}]"
        else:
            band_f, band_k, lab = caught - prev_c, k - prev_k, f"({cs[cs.index(c)-1]}, {c}]"
        g_ = rng(60 + di, int(c * 100)); n = len(k); bs = []
        for _ in range(B):
            ii = g_.integers(0, n, n)
            if band_k[ii].sum() > 0:
                bs.append(band_f[ii].sum() / band_k[ii].sum())
        a2.append(dict(dataset=ds, band=lab, reviews=float(band_k.sum()), frauds=float(band_f.sum()),
                       precision=band_f.sum() / max(band_k.sum(), 1e-9),
                       lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
        prev_c, prev_k = caught, k
a2 = pd.DataFrame(a2)
a2["non_increasing"] = a2.groupby("dataset").precision.transform(lambda x: bool(np.all(np.diff(x.to_numpy()) <= 1e-12)))
a2.to_csv(os.path.join(OUT, "assumption_a2.csv"), index=False)

# --------------------------------------------------------------- 7. summary
rho_final = {}
for ds in DATASETS:
    ms = [m for m in meta if m["dataset"] == ds]
    rho_final[ds] = {sc: float(np.mean([np.mean(m["rho_trace"][str(C1)]) for m in ms if m["scenario"] == sc]))
                     for sc in sorted(set(m["scenario"] for m in ms))}
rel = main[main.policy == "DRT"].merge(main[main.policy == "SUP"][["dataset", "c", "recall"]],
                                       on=["dataset", "c"], suffixes=("", "_sup"))
rel["rel_cost"] = 1 - rel.recall / rel.recall_sup
rho_fb = {}
for ds in DATASETS:
    ms = [m for m in meta if m["dataset"] == ds and m["scenario"] != "base"]
    rho_fb[ds] = {v: {sc: float(np.mean([np.mean(m["rho_trace"][v][str(C1)]) for m in ms if m["scenario"] == sc]))
                      for sc in sorted(set(m["scenario"] for m in ms))}
                  for v in FB if ms and v in ms[0]["rho_trace"]}
summary = dict(config_sha256=cfg_hash, n_runs=len(meta), datasets=DATASETS, scenarios=scen_of,
               amendment_sha256=sorted(set(m.get("amendment_sha256", "none") for m in meta)),
               mean_rho_feedback_variants=rho_fb,
               a2_non_increasing=a2.groupby("dataset").non_increasing.first().to_dict(),
               fuse_w_star_is_1=int(sum(m["fuse_w_star"] == 1.0 for m in meta)),
               drt_rel_cost_max_all_budgets=float(rel.rel_cost.max()),
               drt_rel_cost_by=rel[["dataset", "c", "rel_cost"]].round(4).values.tolist(),
               fuse_w_star={ds: sorted(set(m["fuse_w_star"] for m in meta if m["dataset"] == ds)) for ds in DATASETS},
               mean_rho_adaptive=rho_final,
               premium_corr=float(np.corrcoef(pr.premium_calib, pr.premium_test)[0, 1]),
               group_spearman=(float(spearmanr(gq.ano_adv_ap, gq.drt_gain).statistic) if "drt_gain" in gq else None),
               group_spearman_p=(float(spearmanr(gq.ano_adv_ap, gq.drt_gain).pvalue) if "drt_gain" in gq else None),
               n_groups=int(len(gq)))
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
