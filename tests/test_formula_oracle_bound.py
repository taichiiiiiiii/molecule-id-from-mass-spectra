"""Issue #178: scripts/research/formula_oracle_bound.py の純粋な関数のテスト（rdkit・実データは使わない）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))

import formula_oracle_bound as B  # noqa: E402


def test_oracle_moves_only_other_formula_candidates_above_truth():
    keys = ["a", "b", "c", "T", "d"]
    forms = ["X", "Y", "X", "Y", "X"]          # 正解 T の式は Y。上に Y が 1 つ（b）
    assert B.oracle_rank("T", "Y", keys, forms) == (4, 2)


def test_oracle_hit_absent_and_unknown_formula():
    assert B.oracle_rank("T", "Y", ["T", "a"], ["Y", "X"]) == (1, 1)
    assert B.oracle_rank("T", "Y", ["a", "b"], ["X", "X"]) == (None, None)
    assert B.oracle_rank(None, "Y", ["a"], ["X"]) == (None, None)
    # 式の分からない候補は動かさない（上限を大きく見せない側）
    assert B.oracle_rank("T", "Y", ["a", "b", "T"], [None, "X", "Y"]) == (3, 2)
    # 正解の式が分からなければ動かさない
    assert B.oracle_rank("T", None, ["a", "T"], ["X", "Y"]) == (2, 2)


def test_oracle_ignores_candidates_beyond_25():
    keys = [f"k{i}" for i in range(25)] + ["T"]
    assert B.oracle_rank("T", "Y", keys, ["X"] * 26) == (None, None)


def test_bound_sums_and_counts():
    recs = [(1, 1), (4, 2), (2, 1), (None, None)]
    b = B.bound(recs, 4)
    assert b["mrr_raw"] == pytest.approx((1 + 0.25 + 0.5) / 4)
    assert b["bound_raw"] == pytest.approx(((0.5 - 0.25) + (1 - 0.5)) / 4)
    assert (b["n_hit"], b["n_absent"], b["n_moved"], b["n_top1_other_formula"]) == (1, 1, 2, 1)
