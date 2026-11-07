import sys, glob, pandas as pd, json
d = sys.argv[1]; pat = sys.argv[2]
for f in sorted(glob.glob(f"{d}/{pat}__policy.parquet")):
    p = pd.read_parquet(f); w = pd.read_parquet(f.replace("__policy", "__windows"))
    tot = w.groupby("c")[["frauds","novel_frauds","k"]].sum()
    r = p.groupby(["policy","c"])[["caught","caught_novel"]].sum().reset_index()
    r["recall"] = r.apply(lambda x: x.caught / tot.loc[x.c,"frauds"], axis=1)
    r["nov_recall"] = r.apply(lambda x: x.caught_novel / max(tot.loc[x.c,"novel_frauds"],1), axis=1)
    print(f, tot.to_dict())
    print(r.pivot(index="policy", columns="c", values="recall").round(3).to_string())
    if tot.novel_frauds.sum() > 0:
        print("novel recall"); print(r.pivot(index="policy", columns="c", values="nov_recall").round(3).to_string())
    m = json.load(open(f.replace("__policy.parquet","__meta.json")))
    for c,tr in m["rho_trace"].items(): print("rho", c, [round(x,2) for x in tr[::max(1,len(tr)//12)]])
