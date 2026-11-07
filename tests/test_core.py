import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dstriage import data as D  # noqa: E402
from dstriage import policies as P  # noqa: E402

DP = {"rho0": 0.1, "rho_min": 0.05, "rho_max": 0.5, "eta": 0.02, "marginal_fraction": 0.25,
      "discount": 0.95, "prior": [1.0, 1.0]}


# ----------------------------------------------------------------- budget accounting
def test_allocate_budget_carry_and_bounds():
    n = np.array([10, 3, 0, 7, 25, 1])
    k = P.allocate_budget(n, 0.13)
    assert np.all(k <= n) and np.all(k >= 0)
    assert abs(k.sum() - 0.13 * n.sum()) < 1.0


def _random_period(seed=0, n_windows=30):
    rng = np.random.default_rng(seed)
    w = np.sort(rng.integers(0, n_windows, 3000))
    y = (rng.random(len(w)) < 0.03).astype(int)
    s = rng.random(len(w)) + 0.5 * y
    a = rng.random(len(w))
    return w, y, s, a


def test_topk_exact_per_window():
    w, y, s, a = _random_period()
    sl = P.window_slices(w)
    k = P.allocate_budget(np.array([b - a_ for a_, b in sl]), 0.05)
    sel = P.topk(s, s, sl, k)
    for (a_, b), kk in zip(sl, k):
        assert sel[a_:b].sum() == kk


@pytest.mark.parametrize("rho", [0.05, 0.1, 0.3, 0.5])
def test_drt_budget_and_guarantee(rho):
    """DRT reviews exactly k_w per window and always contains SUP's top (k_w - kB_w) picks,
    hence Recall(DRT_rho) >= Recall(SUP with budget (1-rho)k) up to carry rounding."""
    w, y, s, a = _random_period(1)
    sl = P.window_slices(w)
    k = P.allocate_budget(np.array([b - a_ for a_, b in sl]), 0.05)
    sel, _ = P.drt(s, a, y, sl, k, DP, None, False, rho)
    assert sel.sum() == k.sum()
    for (a_, b), kk in zip(sl, k):
        assert sel[a_:b].sum() == kk
    kA = P.allocate_budget(np.array([b - a_ for a_, b in sl]), 0.05 * (1 - rho))
    sup_small = P.topk(s, s, sl, np.minimum(kA, k))
    # every window: SUP top-kA_w is contained in the DRT selection when kA_w <= DRT's own kA_w
    caught_drt = (sel & (y == 1)).sum()
    caught_sup_small = (sup_small & (y == 1)).sum()
    assert caught_drt >= caught_sup_small - len(sl)  # carry rounding slack <= 1 per window


def test_drt_rho_zero_equals_sup():
    w, y, s, a = _random_period(2)
    sl = P.window_slices(w)
    k = P.allocate_budget(np.array([b - a_ for a_, b in sl]), 0.05)
    sel, _ = P.drt(s, a, y, sl, k, DP, None, False, 0.0)
    assert np.array_equal(sel, P.topk(s, a, sl, k))


def test_drt_adaptive_bounds():
    w, y, s, a = _random_period(3)
    sl = P.window_slices(w)
    k = P.allocate_budget(np.array([b - a_ for a_, b in sl]), 0.05)
    _, st = P.drt(s, a, y, sl, k, DP, None, True)
    assert all(DP["rho_min"] - 1e-12 <= r <= DP["rho_max"] + 1e-12 for r in st.trace)


# ------------------------------------------------------------------ splits
def test_chronological_splits_do_not_split_windows():
    df = pd.DataFrame({"window": np.repeat(np.arange(100), 7)})
    lab = D.chronological_splits(df, {"train": 0.55, "tune": 0.15, "calib": 0.15, "test": 0.15})
    order = {"train": 0, "tune": 1, "calib": 2, "test": 3}
    codes = np.array([order[x] for x in lab])
    assert np.all(np.diff(codes) >= 0)
    for wv in range(100):
        assert len(set(lab[df.window == wv])) == 1


# ------------------------------------------------------------------ leakage
def _write_banksim(path, df):
    q = df.copy()
    for c in ["customer", "age", "gender", "zipcodeOri", "merchant", "zipMerchant", "category"]:
        q[c] = "'" + q[c].astype(str) + "'"
    q.to_csv(os.path.join(path, "banksim.csv"), index=False, quotechar='"')


def _toy_banksim(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "step": np.sort(rng.integers(0, 20, n)),
        "customer": rng.choice(["C1", "C2", "C3", "C4"], n),
        "age": rng.choice(["1", "2", "3"], n), "gender": rng.choice(["M", "F"], n),
        "zipcodeOri": "28007", "merchant": rng.choice(["M1", "M2", "M3", "M4", "M5"], n),
        "zipMerchant": "28007", "category": rng.choice(["es_a", "es_b", "es_c"], n),
        "amount": rng.gamma(2.0, 20.0, n).round(2), "fraud": (rng.random(n) < 0.1).astype(int)})


def test_banksim_features_ignore_future_and_labels(tmp_path):
    base = _toy_banksim()
    _write_banksim(tmp_path, base)
    f0 = D.load_banksim(str(tmp_path))
    pert = base.copy()
    fut = pert.step >= 12
    pert.loc[fut, "amount"] *= 50
    pert.loc[fut, "merchant"] = "M9"
    pert["fraud"] = 1 - pert["fraud"]           # labels must never influence features
    _write_banksim(tmp_path, pert)
    f1 = D.load_banksim(str(tmp_path))
    past = f0.df["t"] < 12
    cols = f0.ano_features
    pd.testing.assert_frame_equal(f0.df.loc[past, cols].reset_index(drop=True),
                                  f1.df.loc[past, cols].reset_index(drop=True))


def _toy_tabformer(path, seed=0, n=600, amount_scale=None, flip=False):
    rng = np.random.default_rng(seed)
    yrs = np.r_[np.full(n // 3, 2015), np.full(n - n // 3, 2016)]
    df = pd.DataFrame({
        "User": rng.integers(0, 4, n), "Card": 0, "Year": yrs,
        "Month": rng.integers(1, 13, n), "Day": rng.integers(1, 28, n),
        "Time": [f"{h:02d}:{m:02d}" for h, m in zip(rng.integers(0, 24, n), rng.integers(0, 60, n))],
        "Amount": ["$%.2f" % v for v in rng.gamma(2.0, 30.0, n)],
        "Use Chip": rng.choice(["Chip Transaction", "Swipe Transaction", "Online Transaction"], n),
        "Merchant Name": rng.integers(0, 6, n), "Merchant City": "X",
        "Merchant State": rng.choice(["CA", "NY", "Italy", None], n), "Zip": 1.0,
        "MCC": rng.choice([5411, 5300, 4829], n), "Errors?": rng.choice([None, "Bad PIN"], n, p=[.9, .1]),
        "Is Fraud?": rng.choice(["No", "Yes"], n, p=[.95, .05])})
    return df


def _ts(df):
    hh = df["Time"].str.slice(0, 2).astype(int); mm = df["Time"].str.slice(3, 5).astype(int)
    return pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day, hour=hh, minute=mm))


def test_tabformer_features_ignore_future_and_labels(tmp_path):
    base = _toy_tabformer(tmp_path)
    base.to_csv(tmp_path / "card_transaction.v1.csv", index=False)
    kw = dict(raw_dir=str(tmp_path), years=(2016, 2016), warmup=(2015,), last_month="2016-12", cache=False)
    f0 = D.load_tabformer(**kw)
    cut = pd.Timestamp("2016-07-01")
    pert = base.copy()
    fut = _ts(pert) >= cut
    pert.loc[fut, "Amount"] = "$9999.00"
    pert.loc[fut, "Merchant Name"] = 99
    pert["Is Fraud?"] = np.where(pert["Is Fraud?"] == "Yes", "No", "Yes")
    pert.to_csv(tmp_path / "card_transaction.v1.csv", index=False)
    f1 = D.load_tabformer(**kw)
    tcut = (cut - pd.Timestamp("2015-01-01")) // pd.Timedelta(minutes=1)
    cols = f0.ano_features + ["user_n_prior"]
    a = f0.df[f0.df.t < tcut].sort_values(["t", "user", "log_amount"])[cols].reset_index(drop=True)
    b = f1.df[f1.df.t < tcut].sort_values(["t", "user", "log_amount"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_tabformer_24h_count_is_strict(tmp_path):
    rows = [dict(User=0, Card=0, Year=2016, Month=1, Day=1, Time=t, Amount="$10.00",
                 **{"Use Chip": "Chip Transaction", "Merchant Name": 1, "Merchant City": "X",
                    "Merchant State": "CA", "Zip": 1.0, "MCC": 5411, "Errors?": None, "Is Fraud?": "No"})
            for t in ["10:00", "10:00", "11:00", "12:00"]]
    rows.append(dict(rows[0], Day=2, Time="11:30"))
    pd.DataFrame(rows).to_csv(tmp_path / "card_transaction.v1.csv", index=False)
    f = D.load_tabformer(raw_dir=str(tmp_path), years=(2016, 2016), warmup=(2016,),
                         last_month="2016-12", cache=False).df.sort_values("t")
    # same-minute rows do not see each other; 11:00 sees two; 12:00 sees three;
    # next day 11:30 sees 11:00(prev day)? no: window [11:30-24h, 11:30) contains 12:00 only.
    assert f["n_24h"].tolist() == [0, 0, 2, 3, 1]
