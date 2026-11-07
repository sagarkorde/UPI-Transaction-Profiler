"""Dataset loading and leakage-safe feature engineering.

Every loader returns a :class:`Prepared` object with a time-sorted frame containing
  t       numeric event time (dataset-specific unit), used only for ordering
  window  integer review-batch id (also the bootstrap block)
  y       fraud label (0/1)
  amount  transaction amount (absolute value)
  group   fraud mechanism/group label (only meaningful where y == 1)
plus the feature columns listed in ``sup_features`` / ``ano_features``.

Leakage rules enforced here (and tested in tests/test_leakage.py):
  * behavioural features use only transactions strictly earlier in time
    (BankSim: strictly earlier simulation steps; TabFormer: strictly earlier rows of
    the same user, with 24h counts using the half-open window [t-24h, t));
  * no feature ever uses a label;
  * global frequency features use only past transactions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


@dataclass
class Prepared:
    name: str
    df: pd.DataFrame
    sup_features: list
    cat_features: list
    ano_features: list
    meta: dict = field(default_factory=dict)


def _cache_path(name):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{name}.parquet")


# --------------------------------------------------------------------------- ULB
def load_ulb(raw_dir=RAW) -> Prepared:
    df = pd.read_csv(os.path.join(raw_dir, "creditcard.csv"))
    df = df.reset_index().rename(columns={"index": "row"})
    df = df.sort_values(["Time", "row"], kind="mergesort").reset_index(drop=True)
    vcols = [f"V{i}" for i in range(1, 29)]
    out = pd.DataFrame({
        "t": df["Time"].astype(float),
        "window": (df["Time"] // 600).astype(int),
        "y": df["Class"].astype(int),
        "amount": df["Amount"].astype(float),
    })
    for c in vcols:
        out[c] = df[c].astype(np.float32)
    out["log_amount"] = np.log1p(df["Amount"]).astype(np.float32)
    feats = vcols + ["log_amount"]
    # Held-out mechanism groups: k-means (k=4) on standardized features of frauds.
    fr = out["y"] == 1
    Z = StandardScaler().fit_transform(out.loc[fr, feats].to_numpy(np.float64))
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Z)
    out["group"] = ""
    out.loc[fr, "group"] = [f"cluster{c}" for c in km.labels_]
    return Prepared("ulb", out, feats, [], feats, {"window_unit": "10 minutes"})


# ----------------------------------------------------------------------- BankSim
def load_banksim(raw_dir=RAW) -> Prepared:
    df = pd.read_csv(os.path.join(raw_dir, "banksim.csv"), quotechar="'")
    for c in ["customer", "age", "gender", "merchant", "category"]:
        df[c] = df[c].astype(str).str.strip("'")
    df = df.reset_index().rename(columns={"index": "row"})
    df = df.sort_values(["step", "row"], kind="mergesort").reset_index(drop=True)

    # customer history from strictly earlier steps
    cs = (df.groupby(["customer", "step"], sort=True)["amount"]
            .agg(n="size", s="sum").reset_index())
    g = cs.groupby("customer", sort=False)
    cs["n_prior"] = g["n"].cumsum() - cs["n"]
    cs["s_prior"] = g["s"].cumsum() - cs["s"]
    cs["last_step"] = g["step"].shift(1)
    df = df.merge(cs[["customer", "step", "n_prior", "s_prior", "last_step"]],
                  on=["customer", "step"], how="left", validate="many_to_one")
    df["cust_mean_prior"] = df["s_prior"] / df["n_prior"].replace(0, np.nan)
    df["amt_ratio"] = df["amount"] / df["cust_mean_prior"]
    df["steps_since_last"] = df["step"] - df["last_step"]
    # novelty of merchant / category for the customer (first step the pair appears)
    for key, nm in [("merchant", "merchant_new"), ("category", "category_new")]:
        first = df.groupby(["customer", key])["step"].transform("min")
        df[nm] = (df["step"] == first).astype(np.int8)
    # global frequency of category / merchant in strictly earlier steps
    tot = df.groupby("step").size().sort_index()
    tot_prior = tot.cumsum() - tot
    for key, nm in [("category", "cat_freq_prior"), ("merchant", "merch_freq_prior")]:
        cnt = df.groupby([key, "step"]).size().rename("c").reset_index()
        cnt["c_prior"] = cnt.groupby(key)["c"].cumsum() - cnt["c"]
        df = df.merge(cnt[[key, "step", "c_prior"]], on=[key, "step"], how="left")
        df[nm] = df["c_prior"] / df["step"].map(tot_prior).replace(0, np.nan)
        df = df.drop(columns="c_prior")

    out = pd.DataFrame({
        "t": df["step"].astype(float),
        "window": df["step"].astype(int),
        "y": df["fraud"].astype(int),
        "amount": df["amount"].astype(float),
        "log_amount": np.log1p(df["amount"]).astype(np.float32),
        "cust_n_prior": np.log1p(df["n_prior"]).astype(np.float32),
        "amt_ratio": df["amt_ratio"].astype(np.float32),
        "steps_since_last": df["steps_since_last"].astype(np.float32),
        "merchant_new": df["merchant_new"],
        "category_new": df["category_new"],
        "cat_freq_prior": df["cat_freq_prior"].astype(np.float32),
        "merch_freq_prior": df["merch_freq_prior"].astype(np.float32),
    })
    for c in ["category", "merchant", "age", "gender"]:
        out[c] = df[c].astype("category")
    out["group"] = np.where(out["y"] == 1, df["category"], "")
    num = ["log_amount", "cust_n_prior", "amt_ratio", "steps_since_last", "merchant_new",
           "category_new", "cat_freq_prior", "merch_freq_prior"]
    cats = ["category", "merchant", "age", "gender"]
    return Prepared("banksim", out, num + cats, cats, num, {"window_unit": "1 step (day)"})


# --------------------------------------------------------------------- TabFormer
TAB_COLS = ["User", "Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant Name",
            "Merchant State", "MCC", "Errors?", "Is Fraud?"]


def _read_tabformer(raw_dir, y0, y1, last_month):
    parts = []
    path = os.path.join(raw_dir, "card_transaction.v1.csv")
    for ch in pd.read_csv(path, usecols=TAB_COLS, chunksize=2_000_000,
                          dtype={"Merchant State": "string", "Errors?": "string"}):
        ch = ch[ch["Year"].between(y0, y1)]
        ly, lm = map(int, last_month.split("-"))
        ch = ch[(ch["Year"] * 100 + ch["Month"]) <= ly * 100 + lm]
        parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def load_tabformer(raw_dir=RAW, years=(2016, 2019), warmup=(2015,), last_month="2019-10",
                   cache=True) -> Prepared:
    cp = _cache_path("tabformer")
    if cache and os.path.exists(cp):
        out = pd.read_parquet(cp)
    else:
        df = _read_tabformer(raw_dir, min(warmup + (years[0],)), years[1], last_month)
        hhmm = df["Time"].str.split(":", expand=True).astype(int)
        ts = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"],
                                 hour=hhmm[0], minute=hhmm[1]))
        df["tmin"] = (ts - pd.Timestamp("2015-01-01")) // pd.Timedelta(minutes=1)
        df["hour"] = hhmm[0].astype(np.int8)
        df["weekday"] = ts.dt.weekday.astype(np.int8)
        df = df.reset_index().rename(columns={"index": "row"})
        df = df.sort_values(["User", "tmin", "row"], kind="mergesort").reset_index(drop=True)
        amt = df["Amount"].str.replace("$", "", regex=False).astype(float)
        u = df["User"].to_numpy()
        t = df["tmin"].to_numpy(np.int64)
        # strict 24h count of the user's earlier transactions: [t-1440, t)
        key = u.astype(np.int64) * 10**8 + t
        lo = np.searchsorted(key, key - 1440, side="left")
        hi = np.searchsorted(key, key, side="left")   # excludes same-minute rows
        n24 = hi - lo
        first_of_user = np.r_[True, u[1:] != u[:-1]]
        prev_t = np.r_[0, t[:-1]]
        mins_since_last = np.where(first_of_user, np.nan, t - prev_t).astype(float)
        # expanding mean/std of log|amount| over strictly earlier rows of the user
        la = np.log1p(np.abs(amt.to_numpy()))
        g = pd.Series(la).groupby(u)
        csum = g.cumsum().to_numpy() - la
        csq = pd.Series(la ** 2).groupby(u).cumsum().to_numpy() - la ** 2
        n_prior = g.cumcount().to_numpy()
        mean_p = np.where(n_prior > 0, csum / np.maximum(n_prior, 1), np.nan)
        var_p = np.where(n_prior > 1, csq / np.maximum(n_prior, 1) - mean_p ** 2, np.nan)
        std_p = np.sqrt(np.clip(var_p, 1e-6, None))
        amt_z = (la - mean_p) / std_p
        state = df["Merchant State"].fillna("ONLINE")
        is_foreign = ((state.str.len() > 2) & (state != "ONLINE")).astype(np.int8)
        merch_new = (df.groupby(["User", "Merchant Name"]).cumcount() == 0).astype(np.int8)
        mcc_new = (df.groupby(["User", "MCC"]).cumcount() == 0).astype(np.int8)
        state_new = (pd.Series(state.to_numpy()).groupby([u, state.to_numpy()]).cumcount() == 0
                     ).astype(np.int8).to_numpy()
        out = pd.DataFrame({
            "t": t.astype(float),
            "user": u,
            "y": (df["Is Fraud?"] == "Yes").astype(int),
            "amount": np.abs(amt.to_numpy()),
            "log_amount": la.astype(np.float32),
            "is_refund": (amt.to_numpy() < 0).astype(np.int8),
            "hour": df["hour"].to_numpy(),
            "weekday": df["weekday"].to_numpy(),
            "is_online": (df["Use Chip"] == "Online Transaction").astype(np.int8),
            "is_foreign": is_foreign.to_numpy(),
            "has_error": df["Errors?"].notna().astype(np.int8),
            "mins_since_last": np.log1p(mins_since_last).astype(np.float32),
            "n_24h": n24.astype(np.float32),
            "user_n_prior": np.log1p(n_prior).astype(np.float32),
            "amt_z": amt_z.astype(np.float32),
            "merchant_new": merch_new.to_numpy(),
            "mcc_new": mcc_new.to_numpy(),
            "state_new": state_new,
            "mcc": df["MCC"].astype(int).to_numpy(),
            "use_chip": df["Use Chip"].str.replace(" Transaction", "", regex=False).to_numpy(),
            "row": df["row"].to_numpy(),
        })
        # global MCC frequency among strictly earlier transactions (time order)
        out = out.sort_values(["t", "row"], kind="mergesort").reset_index(drop=True)
        prior_same_t = out.groupby("t").cumcount().to_numpy()
        n_before = np.arange(len(out)) - prior_same_t           # rows with t' < t
        mcc_before = (out.groupby("mcc").cumcount().to_numpy()
                      - out.groupby(["mcc", "t"]).cumcount().to_numpy())
        out["mcc_freq_prior"] = (mcc_before / np.maximum(n_before, 1)).astype(np.float32)
        start = (pd.Timestamp(f"{years[0]}-01-01") - pd.Timestamp("2015-01-01")) // pd.Timedelta(minutes=1)
        out = out[out["t"] >= start].reset_index(drop=True)
        out["window"] = (out["t"] // 1440).astype(int)
        out["group"] = np.where(out["y"] == 1, out["use_chip"], "")
        out = out.drop(columns=["row"])
        if cache:
            out.to_parquet(cp, index=False)
    out["mcc"] = out["mcc"].astype("category")
    out["use_chip"] = out["use_chip"].astype("category")
    num = ["log_amount", "is_refund", "hour", "weekday", "is_online", "is_foreign", "has_error",
           "mins_since_last", "n_24h", "user_n_prior", "amt_z", "merchant_new", "mcc_new",
           "state_new", "mcc_freq_prior"]
    cats = ["mcc", "use_chip"]
    ano = ["log_amount", "hour", "is_online", "is_foreign", "has_error", "mins_since_last",
           "n_24h", "amt_z", "merchant_new", "mcc_new", "state_new", "mcc_freq_prior"]
    return Prepared("tabformer", out, num + cats, cats, ano, {"window_unit": "1 day"})


LOADERS = {"ulb": load_ulb, "banksim": load_banksim, "tabformer": load_tabformer}


def load(name: str) -> Prepared:
    return LOADERS[name]()


def chronological_splits(df: pd.DataFrame, fr: dict) -> np.ndarray:
    """Assign each row to train/tune/calib/test by the cumulative row fraction at the
    start of its window, so a review window is never split across periods."""
    w = df["window"].to_numpy()
    assert np.all(np.diff(w) >= 0), "frame must be sorted by window"
    uniq, start_idx = np.unique(w, return_index=True)
    frac_start = start_idx / len(df)
    edges = np.cumsum([fr["train"], fr["tune"], fr["calib"]])
    lab = np.array(["train", "tune", "calib", "test"])[np.searchsorted(edges, frac_start, side="right")]
    return lab[np.searchsorted(uniq, w)]
