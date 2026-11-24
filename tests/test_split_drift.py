"""Issue #222: scripts/research/split_drift.py の純粋な関数を作り物の数で確かめる。実データは読まない。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
import split_drift as S  # noqa: E402


def test_quartiles():
    r = S.quartiles([1, 2, 3, 4, 5])
    assert r == dict(n=5, q1=2.0, median=3.0, q3=4.0)
    r = S.quartiles([1, 2, 3, 4])
    assert r["median"] == 2.5 and r["q1"] == 1.75 and r["q3"] == 3.25
    with pytest.raises(AssertionError):
        S.quartiles([])
    with pytest.raises(AssertionError):
        S.quartiles([1.0, float("nan")])


def test_ks_d_known_values():
    assert S.ks_d([1, 2, 3], [1, 2, 3]) == 0.0
    assert S.ks_d([1, 2, 3], [4, 5, 6]) == 1.0
    # F_x(2)=2/3, F_y(2)=0 -> D = 2/3
    assert S.ks_d([1, 2, 3], [3, 4, 5]) == pytest.approx(2 / 3)
    # 同値を含む: x={1,1,2}, y={1,2,2}: F_x(1)=2/3, F_y(1)=1/3 -> 1/3
    assert S.ks_d([1, 1, 2], [1, 2, 2]) == pytest.approx(1 / 3)


def test_ks_d_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 15, size=rng.integers(1, 40))
        y = rng.integers(0, 15, size=rng.integers(1, 60))
        grid = np.unique(np.concatenate([x, y]))
        brute = max(abs((x <= g).mean() - (y <= g).mean()) for g in grid)
        assert S.ks_d(x, y) == pytest.approx(brute)


def test_p_greater():
    assert S.p_greater([2], [1]) == 1.0
    assert S.p_greater([1], [2]) == 0.0
    assert S.p_greater([1], [1]) == 0.5
    assert S.p_greater([1, 3], [2, 2]) == 0.5
    rng = np.random.default_rng(1)
    x = rng.integers(0, 10, 30)
    y = rng.integers(0, 10, 50)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in x for b in y])
    assert S.p_greater(x, y) == pytest.approx(brute)
    assert S.p_greater(x, y) + S.p_greater(y, x) == pytest.approx(1.0)


def test_band_of_and_mde():
    import split_drift_bands as B
    assert list(B.band_of(np.array([20, 21, 22, 25, 26, 40]), 21.0, 26.0)) == [0, 0, 1, 1, 2, 2]
    assert B.band_mde(0.02, 400, 100) == 0.04
    assert B.band_mde(None, 400, 100) is None
    assert B.band_mde(0.02, 400, 0) is None


def test_heavy_atoms():
    pytest.importorskip("rdkit")
    assert list(S.heavy_atoms(["CCO", "c1ccccc1", "[Na+].[Cl-]"])) == [3, 6, 2]
    with pytest.raises(ValueError):
        S.heavy_atoms(["not_a_smiles(("])
