import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))

import n1_density_strata as nd  # noqa: E402


def test_tertile_edges_linear_and_finite_only():
    q1, q2 = nd.tertile_edges([0.0, 0.3, 0.6, 0.9, float("nan")])
    assert q1 == pytest.approx(0.3) and q2 == pytest.approx(0.6)
    q1, q2 = nd.tertile_edges([1.0, 2.0])
    assert q1 == pytest.approx(1 + 1 / 3) and q2 == pytest.approx(1 + 2 / 3)
    with pytest.raises(ValueError):
        nd.tertile_edges([float("nan")])


def test_assign_stratum_edges_absent_nan():
    q1, q2 = 0.3, 0.6
    assert nd.assign_stratum(0.3, True, q1, q2) == "low"
    assert nd.assign_stratum(0.30001, True, q1, q2) == "mid"
    assert nd.assign_stratum(0.6, True, q1, q2) == "mid"
    assert nd.assign_stratum(0.61, True, q1, q2) == "high"
    assert nd.assign_stratum(float("nan"), True, q1, q2) == "low"
    assert nd.assign_stratum(None, False, q1, q2) == "absent"
    assert nd.assign_stratum(0.9, False, q1, q2) == "absent"


def test_molecule_values_max_and_missing():
    g = np.array(["A", "A", "A", "B", "B", "C"])
    y = np.array([1, 1, 0, 0, 0, 1])
    x = np.array([0.2, 0.5, 0.9, 0.7, 0.8, float("nan")])
    vmax, cnt = nd.molecule_values(g, y, x)
    assert vmax["A"] == 0.5 and cnt["A"] == 2
    assert "B" not in vmax and "B" not in cnt
    assert np.isnan(vmax["C"]) and cnt["C"] == 1


def test_summarize_type_counts_and_weighted_delta():
    ids = np.array(["m1", "m2", "m3", "m4", "m5"])
    vmax = {"m1": 0.1, "m2": 0.5, "m3": 0.9, "m4": float("nan")}   # m5 は正解の行なし
    cand = np.array([1.0, 0.5, 1.0, 0.0, 0.0])
    ctrl = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    q1, q2 = nd.tertile_edges([0.1, 0.5, 0.9])

    def boot(a, b):
        d = a - b
        return float(d.min()), float(d.max())

    s = nd.summarize_type(ids, cand - ctrl, cand, ctrl, vmax, q1, q2, boot)
    assert {k: v["n"] for k, v in s.items()} == {"low": 2, "mid": 1, "high": 1, "absent": 1}
    assert s["low"]["delta"] == pytest.approx(0.25)
    assert s["absent"]["delta"] == 0.0
    assert nd.weighted_delta(s) == pytest.approx((cand - ctrl).mean())


def test_summarize_type_empty_stratum():
    s = nd.summarize_type(np.array(["m1"]), np.array([0.0]), np.array([1.0]), np.array([1.0]),
                          {"m1": 0.5}, 0.5, 0.5, lambda a, b: (0.0, 0.0))
    assert s["absent"]["n"] == 0 and s["absent"]["delta"] is None
    assert s["low"]["n"] == 1


def test_reading_rule():
    assert nd.reading(0.01, 0.2) == "not_only_density"
    assert nd.reading(-0.01, 0.2) == "near_zero"
    assert nd.reading(0.0, 0.2) == "near_zero"
    assert nd.reading(-0.2, -0.01) == "low_loses"
    assert nd.reading(None, None) == "no_data"
