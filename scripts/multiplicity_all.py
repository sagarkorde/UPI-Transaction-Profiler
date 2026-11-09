"""Post-hoc family-wise correction over every held-out comparison ever reported (supplement S16).

Answers "how many paths were tried?": every policy ever evaluated (v1.0 and amendments v1.1-v1.5)
compared with SUP on held-out recall, on every corpus and at every budget, is put in one family and
Holm-adjusted. Uses only results/analysis/emerging_pooled.csv (written by scripts/analyze.py).

Output: results/analysis/multiplicity_all.csv and multiplicity_all.json
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dstriage import experiment as E  # noqa: E402
from dstriage import stats as S  # noqa: E402

A = os.path.join(E.ROOT, "results", "analysis")
AMENDED = ["DRT-ADAPT-D1", "DRT-ADAPT-D3", "DRT-ADAPT-D7", "DRT-ADAPT-NOISY", "DRT-ADAPT-AUDIT", "DRT-LD",
           "DRT-AMT", "SUP-NOID", "DRT-NOID", "SUP-IDF", "DRT-IDF"]

e = pd.read_csv(os.path.join(A, "emerging_pooled.csv"))
e = e[e.policy != "SUP"].dropna(subset=["novel_p"]).copy()
e["p_holm_all"] = S.holm(e.novel_p.to_numpy())
am = e.policy.isin(AMENDED)
e.loc[am, "p_holm_amendments"] = S.holm(e.loc[am, "novel_p"].to_numpy())
e["survives_all"] = (e.novel_diff > 0) & (e.p_holm_all < 0.05)
cols = ["dataset", "c", "policy", "novel", "novel_diff", "novel_diff_lo", "novel_diff_hi", "novel_p",
        "p_holm_all", "p_holm_amendments", "survives_all"]
e[cols].sort_values(["dataset", "c", "novel_p"]).to_csv(os.path.join(A, "multiplicity_all.csv"), index=False)

key = e[(e.c == 1.0) & e.policy.isin(["DRT", "DRT-AMT", "SUP-IDF", "DRT-IDF"])]
out = dict(family_size_all=int(len(e)), family_size_amendments=int(am.sum()),
           policies=int(e.policy.nunique()), datasets=sorted(e.dataset.unique()),
           budgets=sorted(float(c) for c in e.c.unique()),
           min_attainable_p=1 / (E.load_cfg()[0]["statistics"]["permutation_resamples"] + 1),
           survivors_all=int(e.survives_all.sum()),
           key_c1={f"{r.dataset}/{r.policy}": dict(diff=r.novel_diff, p=r.novel_p, p_holm_all=r.p_holm_all,
                                                   survives=bool(r.survives_all)) for r in key.itertuples()})
json.dump(out, open(os.path.join(A, "multiplicity_all.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
