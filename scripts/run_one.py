import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dstriage import experiment
ds, scen, seed, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
m = experiment.run(ds, scen, seed, mode)
print(json.dumps({k: m[k] for k in ["dataset","scenario","seed","fuse_w_star","best_iter_sup","best_iter_aug","n","frauds","novel_test_frauds","seconds"]}))
print(json.dumps(m["signal"]))
