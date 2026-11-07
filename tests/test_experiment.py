"""Integration tests on the real BankSim data in development mode (test period untouched)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dstriage import data as D  # noqa: E402
from dstriage import experiment as E  # noqa: E402

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "banksim.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(RAW), reason="BankSim not downloaded")


@pytest.fixture(scope="module")
def prepared():
    return D.load("banksim")


def test_deterministic_rerun(prepared, tmp_path):
    m1 = E.run("banksim", "base", 0, "dev", prepared, str(tmp_path / "a"))
    m2 = E.run("banksim", "base", 0, "dev", prepared, str(tmp_path / "b"))
    p1 = pd.read_parquet(tmp_path / "a" / "banksim__base__s0__policy.parquet")
    p2 = pd.read_parquet(tmp_path / "b" / "banksim__base__s0__policy.parquet")
    pd.testing.assert_frame_equal(p1, p2)
    assert m1["signal"] == m2["signal"]


def test_holdout_removes_group_before_test(prepared, tmp_path):
    m = E.run("banksim", "es_travel", 0, "dev", prepared, str(tmp_path))
    assert m["novel_test_frauds"] > 0
    w = pd.read_parquet(tmp_path / "banksim__es_travel__s0__windows.parquet")
    assert w[w.c == 1.0].novel_frauds.sum() == m["novel_test_frauds"]


def test_dev_mode_never_reads_test_period(prepared):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(E.ROOT, "configs", "protocol.yaml")))
    real = D.chronological_splits(prepared.df, cfg["splits"])
    last_dev_window = prepared.df.window[real != "test"].max()
    first_test_window = prepared.df.window[real == "test"].min()
    assert last_dev_window < first_test_window
