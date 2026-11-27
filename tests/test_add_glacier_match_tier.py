"""Issue #268: add_glacier_match.py の層の行の埋め方（fill_rows・check_jobs）のテスト（モデル・実データは使わない）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/features"))
import add_glacier_match as A  # noqa: E402
import glacier_match as GM  # noqa: E402

TOL = 0.005
C1 = ("[M+H]+", "QTOF", 20.0)
C2 = ("[M+H]+", "QTOF", 40.0)


def _sp(mz, it):
    return GM.prep(np.array(mz, float), np.array(it, float))


def _cols(n):
    return {c: np.full(n, np.nan) for c in GM.SIM_COLS}, np.zeros(n, np.int8)


def test_fill_rows_same_values_for_pool_and_tier_rows():
    # 同じ SMILES・同じ予測なら、プールの行として埋めても層の行として埋めても値は同じ（定義が同じ）
    q = [(_sp([100, 200], [1, 1]), {C1, C2}), (_sp([100, 300], [1, 0.5]), {C1})]
    pred = {("m", "CCO", C1): _sp([100, 200], [1, 0.5]), ("m", "CCO", C2): _sp([200, 300], [1, 1])}
    smis = np.array(["CCO", "CCO"])
    cols, has = _cols(2)
    A.fill_rows(np.array([0]), smis, "m", q, pred, TOL, cols, has)
    A.fill_rows(np.array([1]), smis, "m", q, dict(pred), TOL, cols, has)
    assert has.tolist() == [1, 1]
    for c in GM.SIM_COLS:
        assert cols[c][0] == cols[c][1] and np.isfinite(cols[c][0])
    # 期待値: スペクトルごとに区分の平均 → 分子で平均と最大
    v1 = GM.spectrum_value(q[0][0], [pred[("m", "CCO", C1)], pred[("m", "CCO", C2)]], TOL)
    v2 = GM.spectrum_value(q[1][0], [pred[("m", "CCO", C1)]], TOL)
    agg = GM.aggregate([v1, v2])
    for c in GM.SIM_COLS:
        assert cols[c][0] == pytest.approx(agg[c])


def test_fill_rows_without_prediction_leaves_nan():
    q = [(_sp([100], [1]), {C1})]
    cols, has = _cols(2)
    n_pairs, n_spec = A.fill_rows(np.array([0, 1]), np.array(["C", "N"]), "m", q,
                                  {("m", "C", C1): _sp([100], [1])}, TOL, cols, has)
    assert has.tolist() == [1, 0] and n_pairs == 1 and n_spec == 1
    assert all(np.isnan(cols[c][1]) for c in GM.SIM_COLS)


def test_fill_rows_only_touches_given_rows():
    q = [(_sp([100], [1]), {C1})]
    pred = {("m", s, C1): _sp([100], [1]) for s in ("A", "B", "C")}
    cols, has = _cols(3)
    A.fill_rows(np.array([1]), np.array(["A", "B", "C"]), "m", q, pred, TOL, cols, has)
    assert has.tolist() == [0, 1, 0]


def test_check_jobs():
    assert A.check_jobs("s", "m", {C1, C2}, {"m": {"A", "B"}}, {C1}, {"A", "B"}) is True
    with pytest.raises(AssertionError):           # jobs の候補と表の行が違う
        A.check_jobs("s", "m", {C1}, {"m": {"A"}}, {C1}, {"A", "B"})
    with pytest.raises(AssertionError):           # 先頭 16 本の区分が jobs に無い
        A.check_jobs("s", "m", {C1}, {"m": {"A"}}, {C1, C2}, {"A"})
    # 被覆の無い分子: jobs に行が無い
    assert A.check_jobs("s", "m", set(), {}, set(), {"A"}) is False
    assert A.check_jobs("s", "m", set(), {}, {C1}, set()) is False   # 候補の行が無い分子
