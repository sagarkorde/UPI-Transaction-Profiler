"""Build processed feature tables and print a provenance summary."""
import json, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml
from dstriage import data

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "protocol.yaml")))
summary = {}
for name in ["ulb", "banksim", "tabformer"]:
    t0 = time.time(); P = data.load(name); df = P.df
    split = data.chronological_splits(df, cfg["splits"])
    s = {"rows": int(len(df)), "frauds": int(df.y.sum()), "prevalence": float(df.y.mean()),
         "windows": int(df.window.nunique()),
         "per_split": {k: {"rows": int((split == k).sum()), "frauds": int(df.y[split == k].sum()),
                           "windows": int(df.window[split == k].nunique())}
                       for k in ["train", "tune", "calib", "test"]},
         "groups": df[df.y == 1].group.value_counts().to_dict(),
         "test_groups": df[(df.y == 1) & (split == "test")].group.value_counts().to_dict(),
         "n_sup_features": len(P.sup_features), "n_ano_features": len(P.ano_features),
         "seconds": round(time.time() - t0, 1)}
    summary[name] = s
    print(name, json.dumps(s, indent=1), flush=True)
os.makedirs("results", exist_ok=True)
json.dump(summary, open("results/dataset_summary.json", "w"), indent=1)
