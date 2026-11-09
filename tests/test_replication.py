"""Amendment v1.6: replication aliases inherit every setting and use only disjoint data / new segments."""
import importlib.util
import os

import numpy as np
import pandas as pd

from dstriage import data as D

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _run_all():
    spec = importlib.util.spec_from_file_location("run_all", os.path.join(ROOT, "scripts", "run_all.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_aliases_inherit_settings():
    assert D.base_dataset("tabformer") == "tabformer"
    assert D.base_dataset("banksim") == "banksim"
    assert D.base_dataset("tabformer_p5") == "tabformer"
    assert D.base_dataset("banksim_u") == "banksim"


def test_tabformer_window_is_disjoint_from_v1_0():
    rep = D.replication_spec("tabformer_p5")
    assert rep["years"][1] < 2015 and max(rep["warmup_years"]) < rep["years"][0]
    assert rep["last_month"].startswith(str(rep["years"][1]))
    assert rep["time_origin"] == f"{min(rep['warmup_years'])}-01-01"


def test_new_segment_rule_excludes_v1_0_groups():
    cfg = {"splits": {"train": .55, "tune": .15, "calib": .15, "test": .15},
           "datasets": {"banksim": {"holdout_groups": "top4_category"}},
           "min_test_frauds_per_holdout_group": 10}
    counts = {f"g{i}": 300 - 20 * i for i in range(8)}  # g0 most frequent ... g7 least
    rows = []
    for g, n in counts.items():
        rows += [dict(y=1, group=g)] * n
    rows += [dict(y=0, group="")] * 3000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    df["window"] = np.arange(len(df)) // 50
    P = D.Prepared("banksim_u", df, [], [], [])
    ra = _run_all()
    new, _ = ra.holdout_groups("banksim_u", P, cfg)
    old, _ = ra.holdout_groups("banksim", P._replace(name="banksim") if hasattr(P, "_replace") else
                               D.Prepared("banksim", df, [], [], []), cfg)
    assert old == ["g0", "g1", "g2", "g3"]
    assert set(new) <= {"g4", "g5", "g6", "g7"} and len(new) >= 1
    assert not set(new) & set(old)


def test_v1_7_window_is_disjoint_and_registered_once():
    assert os.path.basename(D.replication_source("tabformer_e14")) == "amendment_v1_7.yaml"
    assert os.path.basename(D.replication_source("tabformer_p5")) == "amendment_v1_6.yaml"
    assert D.base_dataset("tabformer_e14") == "tabformer"
    rep = D.replication_spec("tabformer_e14")
    assert rep["years"][1] < 2015 and max(rep["warmup_years"]) < rep["years"][0]
    assert rep["last_month"] == "2014-04"
    assert D.replication_source("nonexistent") is None
