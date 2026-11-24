"""#218 fragnn3_mechanism の小さな関数のテスト（合成データ）。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
import fragnn3_mechanism as M  # noqa: E402


def test_window_stats_counts_same_formula_and_ignores_missing():
    # CSR 2 行: 行 0 の窓 = [1,2,3]、行 1 の窓 = [0]
    off = np.array([0, 3, 4])
    flat = np.array([1, 2, 3, 0])
    code = np.array([5, 5, -1, 7])          # 2 は式なし
    pos_code = np.array([5, -1])            # 行 1 の正解は式なし → 一致 0
    n, s = M.window_stats(off, flat, pos_code, code)
    assert n.tolist() == [3, 1]
    assert s.tolist() == [1, 0]


def test_change_metrics_small_window_ratio_is_one():
    # 窓が 127 以下なら v2 も全部入る → r = 1、p_change = 0
    m = M.change_metrics(np.array([50, 10]), np.array([5, 0]), np.array([3, 1]))
    assert m["p_change"] == 0.0
    assert m["r"] == pytest.approx(1.0)


def test_change_metrics_big_window():
    # n = 254, s = 10: v2 = 10*127/254 = 5, v3 = 10 → r = 2; 重み 1 の 2 行（1 行は窓 0）
    m = M.change_metrics(np.array([254, 0]), np.array([10, 0]), np.array([1, 1]))
    assert m["p_change"] == pytest.approx(0.5)
    assert m["r"] == pytest.approx(2.0)
    assert m["share_n_zero"] == pytest.approx(0.5)


def test_change_metrics_caps_s_at_kmax():
    m = M.change_metrics(np.array([300]), np.array([200]), np.array([1]))
    assert m["sum_v3"] == pytest.approx(127.0)
    assert m["sum_v2"] == pytest.approx(200 * 127 / 300)


@pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="rdkit")
def test_formula_strips_charge():
    assert M.formula_of("CCO") == "C2H6O"
    assert M.formula_of("C[N+](C)(C)C") == "C4H12N"
    assert M.formula_of("not_a_smiles") is None
    assert M.formula_of(None) is None
