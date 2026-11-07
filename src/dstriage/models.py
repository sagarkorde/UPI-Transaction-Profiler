"""Signal models: supervised fraud score, unsupervised anomaly score, calibration, ECDFs."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def fit_lgbm(Xtr, ytr, Xva, yva, cat_features, cfg, seed):
    params = dict(cfg["params"])
    params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_features or "auto", free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, categorical_feature=cat_features or "auto")
    params["metric"] = "average_precision"
    booster = lgb.train(params, dtr, num_boost_round=cfg["max_rounds"], valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    return booster


def predict_lgbm(booster, X):
    return booster.predict(X, num_iteration=booster.best_iteration)



class AnomalyModel:
    """Isolation Forest (primary) or ECOD (ablation). Higher score = more anomalous.
    Missing values are imputed with training-period medians."""

    def __init__(self, kind, cfg, seed):
        self.kind, self.cfg, self.seed = kind, cfg, seed

    def fit(self, X: pd.DataFrame):
        A = X.to_numpy(np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        self.med = np.nan_to_num(np.nanmedian(A, axis=0))
        A = self._impute(A)
        rng = np.random.default_rng(self.seed)
        n = min(len(A), self.cfg["isolation_forest"]["fit_rows"])
        A = A[rng.choice(len(A), n, replace=False)]
        if self.kind == "iforest":
            p = self.cfg["isolation_forest"]
            self.m = IsolationForest(n_estimators=p["n_estimators"], max_samples=p["max_samples"],
                                     random_state=self.seed, n_jobs=2).fit(A)
        elif self.kind == "ecod":
            from pyod.models.ecod import ECOD
            self.m = ECOD(n_jobs=1).fit(A)
        else:
            raise ValueError(self.kind)
        return self

    def _impute(self, A):
        idx = np.where(np.isnan(A))
        A[idx] = np.take(self.med, idx[1])
        return A

    def score(self, X: pd.DataFrame):
        A = X.to_numpy(np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        A = self._impute(A)
        out = np.empty(len(A))
        for i in range(0, len(A), 500_000):
            blk = A[i:i + 500_000]
            out[i:i + 500_000] = (-self.m.score_samples(blk) if self.kind == "iforest"
                                  else self.m.decision_function(blk))
        return out


class ECDF:
    """Empirical CDF fitted on a reference period; maps scores to percentiles in [0, 1]."""

    def __init__(self, ref):
        self.ref = np.sort(np.asarray(ref, float))

    def __call__(self, x):
        return np.searchsorted(self.ref, np.asarray(x, float), side="right") / len(self.ref)


def isotonic(s_cal, y_cal):
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s_cal, y_cal)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_features(s_raw, a_pct):
    z = _logit(s_raw)
    return np.column_stack([z, a_pct, z * a_pct])


def fit_stack(s_raw, a_pct, y):
    return LogisticRegression(C=1.0, max_iter=1000).fit(stack_features(s_raw, a_pct), y)


def ap(y, s):
    return float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")
