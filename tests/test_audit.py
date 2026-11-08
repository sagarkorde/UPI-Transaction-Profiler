import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dstriage import policies as P  # noqa: E402

DP = dict(rho0=0.1, marginal_fraction=0.25, discount=0.9, eta=0.02, prior=[1, 1], rho_min=0.05, rho_max=0.5)
AU = dict(share=0.05, band_width=1.0)


def _data(seed=0, n=4000, W=20):
    r = np.random.default_rng(seed)
    w = np.sort(r.integers(0, W, n))
    y = (r.random(n) < 0.02).astype(int)
    s = r.random(n) + y * 0.5
    b = r.random(n) + y * 0.3
    return s, b, y, P.window_slices(w)


def test_audit_budget_exact_and_rho_bounded():
    s, b, y, sl = _data()
    k = P.allocate_budget(np.array([e - a for a, e in sl]), 0.05)
    sel, st = P.drt_audit(s, b, y, sl, k, DP, AU, None, np.random.default_rng(1))
    for (a, e), kw in zip(sl, k):
        assert sel[a:e].sum() == kw
    assert all(DP["rho_min"] <= r <= DP["rho_max"] for r in st.trace)


def test_audit_deterministic_given_seed():
    s, b, y, sl = _data(3)
    k = P.allocate_budget(np.array([e - a for a, e in sl]), 0.05)
    s1, _ = P.drt_audit(s, b, y, sl, k, DP, AU, None, np.random.default_rng(7))
    s2, _ = P.drt_audit(s, b, y, sl, k, DP, AU, None, np.random.default_rng(7))
    assert (s1 == s2).all()
