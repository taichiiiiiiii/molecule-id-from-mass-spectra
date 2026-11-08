"""Issue #42 の混合比の点検（合成データ）。"""
import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lb_mixture_check", ROOT / "scripts/cv/lb_mixture_check.py")
LM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LM)


def test_grid_sums_to_one_and_size():
    for n in (2, 3, 5):
        g = LM.simplex_grid(n, 20)
        assert np.allclose(g.sum(1), 1.0)
        assert (g >= 0).all()
        assert len(g) == math.comb(20 + n - 1, n - 1)          # 重複なしで全部
        assert len({tuple(r) for r in np.round(g * 20).astype(int)}) == len(g)


def test_grid_step_values():
    g = LM.simplex_grid(2, 20)
    assert sorted(np.round(g[:, 0], 2).tolist()) == [round(i * 0.05, 2) for i in range(21)]


def test_predict_known_mixture():
    g = np.array([[0.5, 0.3, 0.2]])
    m = np.array([0.8, 0.4, 0.0])
    assert math.isclose(LM.predict(g, m)[0], 0.5 * 0.8 + 0.3 * 0.4)   # 0.52


def test_consistent_requires_both_submissions():
    g = LM.simplex_grid(2, 20)                                 # 比 (p, 1-p)
    mrr = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    ok = LM.consistent(g, mrr, {"a": 0.3, "b": 0.7}, 0.0)
    assert ok.sum() == 1 and np.allclose(g[ok][0], [0.3, 0.7])
    ok2 = LM.consistent(g, mrr, {"a": 0.3, "b": 0.3}, 0.04)   # 両方は満たせない
    assert ok2.sum() == 0
    s = LM.summarize(g, ok, ["x", "y"])
    assert s["n_consistent"] == 1 and s["share_range"]["x"] == [0.3, 0.3]


def test_lp_bounds_matches_simple_case():
    import pytest
    pytest.importorskip("scipy")                               # CI に scipy が無ければ skip
    mrr = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    r = LM.lp_bounds(mrr, {"a": 0.3, "b": 0.7}, 0.05, ["x", "y"])
    assert r["x"] == [0.25, 0.35] and r["y"] == [0.65, 0.75]
