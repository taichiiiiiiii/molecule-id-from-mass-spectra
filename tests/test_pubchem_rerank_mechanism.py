"""#234 pubchem_rerank_mechanism の小さな関数のテスト（合成データ、実データは使わない）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
import pubchem_rerank_mechanism as P  # noqa: E402


def test_truth_rank_first_match_and_none():
    to_key = {"a": "K1", "b": "K2", "c": "K2", "x": None}.get
    assert P.truth_rank(["a", "b", "c"], {"K2"}, to_key) == 2
    assert P.truth_rank(["x", "a"], {"K1"}, to_key) == 2
    assert P.truth_rank(["a", "b"], {"K9"}, to_key) is None
    assert P.truth_rank(["a"], set(), to_key) is None
    assert P.truth_rank([], {"K1"}, to_key) is None


def test_rank_distribution_bins():
    ranks = [1, 1, 2, 5, 6, 25, 26, 60, None, None]
    assert P.rank_distribution(ranks) == {"1-1": 2, "2-5": 2, "6-25": 2, "26-60": 2}


def test_headroom_top60():
    # 1 位 → 0、2 位 → 0.5、4 位 → 0.75、None は足さない、n = 4
    assert P.headroom_top60([1, 2, 4, None], 4) == pytest.approx((0 + 0.5 + 0.75) / 4)


def test_same_formula_share():
    assert P.same_formula_share(["C6H6", "C6H6O", None, "C6H6"], "C6H6") == pytest.approx(0.5)
    assert P.same_formula_share([], "C6H6") is None
    assert P.same_formula_share(["C6H6"], None) is None


def test_frag_win_ties_not_counted_and_none_excluded():
    assert P.frag_win(0.3, 0.2) == 1
    assert P.frag_win(0.2, 0.3) == 0
    assert P.frag_win(0.0, 0.0) == 0      # 同点は「正解 ＞ 1 位」に数えない
    assert P.frag_win(0.25, 0.25) == 0
    assert P.frag_win(None, 0.2) is None
    assert P.frag_win(0.2, None) is None


def test_bootstrap_ci_deterministic_and_brackets_mean():
    x = np.array([1] * 30 + [0] * 20)
    lo, hi = P.bootstrap_ci(x)
    assert (lo, hi) == P.bootstrap_ci(x)          # seed 0 で同じ
    assert lo < 0.6 < hi
    assert P.bootstrap_ci(np.ones(10)) == (1.0, 1.0)


def test_reading_rule_strict():
    assert P.reading(0.51) == "候補にする"
    assert P.reading(0.5) == "作らない"
    assert P.reading(0.2) == "作らない"


def test_formula_of():
    pytest.importorskip("rdkit")
    assert P.formula_of("c1ccccc1O") == "C6H6O"
    assert P.formula_of("not((") is None
    assert P.formula_of("") is None
