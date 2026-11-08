"""Audit of held-out scenario selection (reviewer request, v7).

For each corpus, compare the candidate groups chosen by the pre-specified rule (which for BankSim ranks
categories by fraud count over the whole corpus) with the groups the same rule would choose using
pre-test frauds only. Writes results/analysis/scenario_provenance.json, read by make_supplement.py.
"""
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dstriage import data as D  # noqa: E402


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "protocol.yaml")))
    res = {}
    for name in ["ulb", "banksim", "tabformer"]:
        df = D.load(name).df
        split = D.chronological_splits(df, cfg["splits"])
        fr = df.y == 1
        allc = df[fr].group.value_counts()
        pre = df[fr & (split != "test")].group.value_counts()
        rule = cfg["datasets"][name]["holdout_groups"]
        if rule == "top4_category":
            cand_all, cand_pre = allc.index[:4].tolist(), pre.index[:4].tolist()
        else:
            cand_all, cand_pre = sorted(allc.index.tolist()), sorted(pre.index.tolist())
        res[name] = {"rule": rule, "candidates_all_periods": cand_all, "candidates_pre_test_only": cand_pre,
                     "identical": sorted(cand_all) == sorted(cand_pre),
                     "pre_test_fraud_counts": {k: int(v) for k, v in pre.items()}}
        print(name, "identical" if res[name]["identical"] else "DIFFERENT", cand_pre)
    out = os.path.join(ROOT, "results", "analysis", "scenario_provenance.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
