"""How much of each held-out group can raw transaction amount alone rank? (internal review, v1.4; descriptive)

For each corpus: mean amount of fraud and of legitimate transactions over the whole corpus, and, per held-out
group, the average precision of raw amount for that group's frauds against legitimate test transactions.
No model is fitted; the test period is defined by the protocol's chronological split.
Writes results/analysis/amount_signal.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dstriage import data as D  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "protocol.yaml")))
scen = pd.read_csv(os.path.join(ROOT, "results", "analysis", "capacity_heldout.csv"))
rows = []
for ds in ["ulb", "banksim", "tabformer"]:
    df = D.load(ds).df
    split = D.chronological_splits(df, cfg["splits"])
    y = df["y"].to_numpy(); amt = df["amount"].to_numpy(float); grp = df["group"].to_numpy()
    t = split == "test"
    for g in scen[scen.dataset == ds].scenario:
        nov = (grp == g) & (y == 1)
        m = t & ((y == 0) | nov)
        rows.append(dict(dataset=ds, group=g, mean_amount_fraud=float(amt[y == 1].mean()),
                         mean_amount_legit=float(amt[y == 0].mean()), test_novel=int(nov[t].sum()),
                         ap_amount_novel=float(average_precision_score(nov[m], amt[m])),
                         base_rate=float(nov[m].mean())))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results", "analysis", "amount_signal.csv"), index=False)
print(out.round(3).to_string(index=False))
