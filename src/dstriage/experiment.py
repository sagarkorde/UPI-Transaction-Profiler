"""Run one (dataset, scenario, seed) experiment and write per-window results."""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import data as D
from . import models as M
from . import policies as P

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAIN_POLICIES = ["RAND", "SUP", "ANO", "UNC", "FUSE60", "FUSEW", "AUG", "STACK", "DRT"]
ABLATIONS = ["DRT-UNC", "DRT-RAND", "DRT-ECOD", "DRT-ADAPT"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_cfg():
    import yaml
    path = os.path.join(ROOT, "configs", "protocol.yaml")
    return yaml.safe_load(open(path)), sha256_file(path)


def _auc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def run(dataset: str, scenario: str, seed: int, mode: str = "final", prepared=None, out_dir=None):
    cfg, cfg_hash = load_cfg()
    t0 = time.time()
    Pd = prepared or D.load(dataset)
    df = Pd.df
    fr = dict(cfg["splits"])
    if mode == "dev":
        # Development mode: the real test period is removed and never read.
        real = D.chronological_splits(df, fr)
        df = df[real != "test"].reset_index(drop=True)
        s = fr["train"] + fr["tune"] + fr["calib"]
        fr = {"train": fr["train"] / s, "tune": fr["tune"] / s,
              "calib": fr["calib"] / 2 / s, "test": fr["calib"] / 2 / s}
    split = D.chronological_splits(df, fr)

    # ---- scenario: hold out one fraud mechanism from all pre-test periods
    if scenario != "base":
        drop = (df["y"].to_numpy() == 1) & (df["group"].to_numpy() == scenario) & (split != "test")
        df = df[~drop].reset_index(drop=True)
        split = split[~drop]
    novel_all = (df["y"].to_numpy() == 1) & (df["group"].to_numpy() == scenario)

    sup, cats, ano = Pd.sup_features, Pd.cat_features, Pd.ano_features
    idx = {k: np.flatnonzero(split == k) for k in ["train", "tune", "calib", "test"]}
    y = df["y"].to_numpy()
    rng = np.random.default_rng(seed)

    tr = idx["train"]
    nsr = cfg["datasets"][dataset].get("negative_sampling_rate")
    if nsr:
        keep = (y[tr] == 1) | (rng.random(len(tr)) < nsr)
        tr_sup = tr[keep]
    else:
        tr_sup = tr

    # ---- anomaly detector (unsupervised; fit on the full training period, no labels)
    iso = M.AnomalyModel("iforest", cfg["anomaly"], seed).fit(df.iloc[tr][ano])
    need_a = np.unique(np.r_[tr_sup, idx["tune"], idx["calib"], idx["test"]])
    a_raw = np.full(len(df), np.nan)
    a_raw[need_a] = iso.score(df.iloc[need_a][ano])

    # ---- supervised model (SUP) and anomaly-augmented model (AUG)
    X = df[sup]
    sup_cfg = cfg["supervised"]
    bst = M.fit_lgbm(X.iloc[tr_sup], y[tr_sup], X.iloc[idx["tune"]], y[idx["tune"]], cats, sup_cfg, seed)
    Xa = X.copy(); Xa["if_score"] = a_raw
    bst_aug = M.fit_lgbm(Xa.iloc[tr_sup], y[tr_sup], Xa.iloc[idx["tune"]], y[idx["tune"]], cats, sup_cfg, seed)
    ev = np.r_[idx["tune"], idx["calib"], idx["test"]]
    s_raw = np.full(len(df), np.nan); s_raw[ev] = M.predict_lgbm(bst, X.iloc[ev])
    s_aug = np.full(len(df), np.nan); s_aug[ev] = M.predict_lgbm(bst_aug, Xa.iloc[ev])

    # ---- reference ECDFs (tune), calibration & combiners (calib)
    s_ecdf, a_ecdf = M.ECDF(s_raw[idx["tune"]]), M.ECDF(a_raw[idx["tune"]])
    s_pct, a_pct = s_ecdf(s_raw), a_ecdf(a_raw)
    c_idx = idx["calib"]
    iso_cal = M.isotonic(s_raw[c_idx], y[c_idx])
    s_cal = np.full(len(df), np.nan); s_cal[ev] = iso_cal.predict(s_raw[ev])
    stk = M.fit_stack(s_raw[c_idx], a_pct[c_idx], y[c_idx])
    s_stack = np.full(len(df), np.nan); s_stack[ev] = stk.decision_function(M.stack_features(s_raw[ev], a_pct[ev]))
    pi_hat = float(y[c_idx].mean())
    unc = s_cal * (1 - s_cal)

    ecod_pct = None
    if scenario == "base":
        ec = M.AnomalyModel("ecod", cfg["anomaly"], seed).fit(df.iloc[tr][ano])
        e_raw = np.full(len(df), np.nan)
        e_ix = np.r_[idx["tune"], idx["calib"], idx["test"]]
        e_raw[e_ix] = ec.score(df.iloc[e_ix][ano])
        ecod_pct = M.ECDF(e_raw[idx["tune"]])(e_raw)

    # ---- periods used by policies
    def period(k):
        ii = idx[k]
        return dict(ii=ii, w=df["window"].to_numpy()[ii], y=y[ii],
                    amt=df["amount"].to_numpy()[ii], novel=novel_all[ii])
    cal, tst = period("calib"), period("test")
    cal["sl"], tst["sl"] = P.window_slices(cal["w"]), P.window_slices(tst["w"])
    for per in (cal, tst):
        per["n_w"] = np.array([b - a for a, b in per["sl"]])

    def g(arr, per):
        return arr[per["ii"]]

    # FUSE weight chosen on the calibration period at the primary budget
    k_cal_primary = P.allocate_budget(cal["n_w"], cfg["primary_budget_multiplier"] * pi_hat)
    best_w, best_r = None, -1
    for w in cfg["policies"]["fuse_weight_grid"]:
        sc = w * g(s_pct, cal) + (1 - w) * g(a_pct, cal)
        sel = P.topk(sc, g(s_raw, cal), cal["sl"], k_cal_primary)
        r = (sel & (cal["y"] == 1)).sum()
        if r >= best_r:  # ties -> larger supervised weight
            best_w, best_r = w, r
    pc = cfg["policies"]
    wh, rho = pc["fuse_heuristic_weight"], pc["drt_rho"]
    dp = pc["drt_adaptive"]

    rows, rho_traces, premium_calib = [], {}, {}
    win_tot = []
    for c in cfg["budget_multipliers"]:
        rate = c * pi_hat
        k_cal = P.allocate_budget(cal["n_w"], rate)
        k_tst = P.allocate_budget(tst["n_w"], rate)
        ys, sl = tst["y"], tst["sl"]
        rnd = np.random.default_rng([seed, int(c * 1000)])
        rand_t = rnd.random(len(ys)); rand_c = rnd.random(len(cal["y"]))
        S = lambda arr: g(arr, tst)
        scores = {
            "RAND": (rand_t, rand_t),
            "SUP": (S(s_raw), S(s_raw)),
            "ANO": (S(a_raw), S(s_raw)),
            "UNC": (S(unc), S(s_raw)),
            "FUSE60": (wh * S(s_pct) + (1 - wh) * S(a_pct), S(s_raw)),
            "FUSEW": (best_w * S(s_pct) + (1 - best_w) * S(a_pct), S(s_raw)),
            "AUG": (S(s_aug), S(s_aug)),
            "STACK": (S(s_stack), S(s_raw)),
        }
        for w in pc["fuse_frontier_weights"]:
            scores[f"FUSE@{w:.2f}"] = (w * S(s_pct) + (1 - w) * S(a_pct), S(s_raw))
        sels = {name: P.topk(sc, tb, sl, k_tst) for name, (sc, tb) in scores.items()}

        def drt_fixed(bsig_tst, r):
            return P.drt(S(s_raw), bsig_tst, ys, sl, k_tst, dp, None, False, r)[0]

        sels["DRT"] = drt_fixed(S(a_raw), rho)
        for r in pc["drt_frontier_rhos"]:
            sels[f"DRT@{r:.2f}"] = drt_fixed(S(a_raw), r)
        sels["DRT-UNC"] = drt_fixed(S(unc), rho)
        sels["DRT-RAND"] = drt_fixed(rand_t, rho)
        if ecod_pct is not None:
            sels["DRT-ECOD"] = drt_fixed(S(ecod_pct), rho)
        # adaptive allocation (ablation): burn-in on calibration period, then test
        _, st = P.drt(g(s_raw, cal), g(a_raw, cal), cal["y"], cal["sl"], k_cal, dp, None, True)
        st.trace = []
        sels["DRT-ADAPT"], st = P.drt(S(s_raw), S(a_raw), ys, sl, k_tst, dp, st, True)
        rho_traces[str(c)] = st.trace
        # stationary premium of each reserved share, estimated on the calibration period
        sup_cal = P.topk(g(s_raw, cal), g(s_raw, cal), cal["sl"], k_cal)
        base_c = (sup_cal & (cal["y"] == 1)).sum()
        premium_calib[str(c)] = {}
        for r in pc["drt_frontier_rhos"]:
            sel_c = P.drt(g(s_raw, cal), g(a_raw, cal), cal["y"], cal["sl"], k_cal, dp, None, False, r)[0]
            premium_calib[str(c)][f"{r:.2f}"] = float((base_c - (sel_c & (cal["y"] == 1)).sum())
                                                      / max(cal["y"].sum(), 1))

        for name, sel in sels.items():
            assert sel.sum() == k_tst.sum(), (name, sel.sum(), k_tst.sum())
            cf, ca, cn = P.per_window(sel, ys, tst["amt"], tst["novel"], sl)
            rows.append(pd.DataFrame({"policy": name, "c": c, "wi": np.arange(len(sl)),
                                      "caught": cf, "caught_amt": ca, "caught_novel": cn}))
        fw, fa, fn = P.per_window(np.ones(len(ys), bool), ys, tst["amt"], tst["novel"], sl)
        win_tot.append(pd.DataFrame({"c": c, "wi": np.arange(len(sl)), "window": tst["w"][[a for a, _ in sl]],
                                     "n": tst["n_w"], "k": k_tst, "frauds": fw, "fraud_amt": fa,
                                     "novel_frauds": fn}))

    # ---- threshold-free signal quality on the test period
    yt = tst["y"]; nov = tst["novel"]
    sig = {"SUP": S(s_raw), "ANO": S(a_raw), "AUG": S(s_aug), "STACK": S(s_stack)}
    if ecod_pct is not None:
        sig["ECOD"] = S(ecod_pct)
    signal = {}
    for k_, v in sig.items():
        signal[k_] = {"ap": M.ap(yt, v), "auc": _auc(yt, v)}
        if nov.any():
            m = (yt == 0) | nov
            signal[k_]["ap_novel"] = M.ap(nov[m].astype(int), v[m])
            signal[k_]["auc_novel"] = _auc(nov[m].astype(int), v[m])
    sc_t = S(s_cal)
    signal["calibration"] = {"brier": float(np.mean((sc_t - yt) ** 2)),
                             "base_rate_test": float(yt.mean()), "pi_hat_calib": pi_hat}

    meta = dict(dataset=dataset, scenario=scenario, seed=seed, mode=mode, config_sha256=cfg_hash,
                fuse_w_star=best_w, best_iter_sup=int(bst.best_iteration),
                best_iter_aug=int(bst_aug.best_iteration), stack_coef=stk.coef_.ravel().tolist(),
                n={k: int(len(v)) for k, v in idx.items()},
                frauds={k: int(y[v].sum()) for k, v in idx.items()},
                novel_test_frauds=int(tst["novel"].sum()), signal=signal, rho_trace=rho_traces,
                premium_calib=premium_calib,
                seconds=round(time.time() - t0, 1))
    out_dir = out_dir or os.path.join(ROOT, "results", "runs" if mode == "final" else "dev_runs")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{dataset}__{scenario}__s{seed}")
    if mode == "dev":  # keep held-in scores for offline design analysis (never in final mode)
        pd.DataFrame({"period": np.r_[["calib"] * len(cal["ii"]), ["test"] * len(tst["ii"])],
                      "w": np.r_[cal["w"], tst["w"]], "y": np.r_[cal["y"], tst["y"]],
                      "novel": np.r_[cal["novel"], tst["novel"]],
                      "s_raw": np.r_[g(s_raw, cal), S(s_raw)], "a_raw": np.r_[g(a_raw, cal), S(a_raw)],
                      "s_pct": np.r_[g(s_pct, cal), S(s_pct)], "a_pct": np.r_[g(a_pct, cal), S(a_pct)],
                      "pi_hat": pi_hat}).to_parquet(stem + "__scores.parquet", index=False)
    pd.concat(rows).to_parquet(stem + "__policy.parquet", index=False)
    pd.concat(win_tot).to_parquet(stem + "__windows.parquet", index=False)
    json.dump(meta, open(stem + "__meta.json", "w"), indent=1)
    return meta
