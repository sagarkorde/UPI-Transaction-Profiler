"""Run the complete pre-specified experiment grid (final mode, test period evaluated once).

Usage: python scripts/run_all.py [--datasets ulb banksim tabformer] [--seeds 0 1 2 3 4]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from dstriage import data as D  # noqa: E402
from dstriage import experiment as E  # noqa: E402


def holdout_groups(name, P, cfg):
    """Pre-specified scenario rule (configs/protocol.yaml)."""
    df = P.df
    split = D.chronological_splits(df, cfg["splits"])
    fr = df[df.y == 1]
    test_counts = df[(df.y == 1) & (split == "test")].group.value_counts()
    rep = D.replication_spec(name) if name not in D.LOADERS else None
    rule = (rep or {}).get("holdout_rule") or cfg["datasets"][D.base_dataset(name)]["holdout_groups"]
    if rule == "top4_category":
        cand = fr.group.value_counts().index[:4].tolist()
    elif rule == "category_ranks_5plus":  # amendment v1.6: every category the v1.0 top-4 rule never held out
        cand = fr.group.value_counts().index[4:].tolist()
    else:  # kmeans4 / channel: all groups
        cand = sorted(fr.group.unique().tolist())
    keep = [g for g in cand if test_counts.get(g, 0) >= cfg["min_test_frauds_per_holdout_group"]]
    return keep, {g: int(test_counts.get(g, 0)) for g in cand}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ulb", "banksim", "tabformer"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="skip runs whose results already exist")
    args = ap.parse_args()
    cfg, cfg_hash = E.load_cfg()
    seeds = args.seeds or cfg["seeds"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=E.ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    out = args.out or os.path.join(E.ROOT, "results", "runs")
    os.makedirs(out, exist_ok=True)
    manifest = {"config_sha256": cfg_hash, "git_commit": commit, "python": platform.python_version(),
                "platform": platform.platform(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "scenarios": {}, "runs": []}
    import importlib.metadata as md
    mpath = os.path.join(out, "_manifest.json")
    if args.resume and os.path.exists(mpath):
        old = json.load(open(mpath))
        assert old["config_sha256"] == cfg_hash, "existing results come from a different protocol"
        manifest["runs"], manifest["started"] = old["runs"], old["started"]
        manifest["scenarios"] = old.get("scenarios", {})
        prev = old.get("git_commits", [old.get("git_commit")])
        manifest["git_commits"] = prev + ([commit] if commit not in prev else [])
        manifest["resumed"] = old.get("resumed", []) + [time.strftime("%Y-%m-%dT%H:%M:%S")]
    manifest["packages"] = {p: md.version(p) for p in
                            ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "pyod", "pyarrow"]}
    for name in args.datasets:
        P = D.load(name)
        groups, counts = holdout_groups(name, P, cfg)
        manifest["scenarios"][name] = {"holdout_groups": groups, "test_fraud_counts": counts}
        for scen in ["base"] + groups:
            for seed in seeds:
                if args.resume and os.path.exists(os.path.join(out, f"{name}__{scen}__s{seed}__meta.json")):
                    continue
                t0 = time.time()
                m = E.run(name, scen, seed, "final", P, out)
                manifest["runs"].append({"dataset": name, "scenario": scen, "seed": seed,
                                         "seconds": round(time.time() - t0, 1)})
                print(f"[{time.strftime('%H:%M:%S')}] {name} {scen} seed={seed} "
                      f"SUP_AP={m['signal']['SUP']['ap']:.3f} ANO_AP={m['signal']['ANO']['ap']:.3f} "
                      f"({m['seconds']}s)", flush=True)
                json.dump(manifest, open(os.path.join(out, "_manifest.json"), "w"), indent=1)
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(manifest, open(os.path.join(out, "_manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
