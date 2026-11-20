"""Issue #169: scripts/cv/reporter_reach.py の純粋な関数のテスト（合成のデータ。RDKit は scaffold だけ）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))

import reporter_reach as R  # noqa: E402


def test_scaffold_ring_acyclic_and_unreadable():
    assert R.scaffold("Cc1ccccc1O") == "c1ccccc1"
    assert R.scaffold("CCCCO") == ""
    assert R.scaffold("not_a_smiles") is None
    assert R.scaffold("") is None


def test_best_rank_counts_same_family_and_unreadable_above():
    # 正解は 4 位、上の 3 件は 違う・同じ・読めない → 同じ家族とみなすのは 2 件 → 3 位まで
    assert R.best_rank(["A", "T", None], "T") == 3
    assert R.best_rank(["A", "B"], "T") == 1
    # 正解の骨格が読めなければ動かせない
    assert R.best_rank(["A", "B"], None) == 3


def test_summarize_bound():
    rows = [(1, 1, False), (0, 0, False), (2, 1, False), (4, 2, True)]
    s = R.summarize_bound(rows, 4)
    assert (s["rank1"], s["rank2_25"], s["absent"]) == (1, 2, 1)
    assert s["rank2_25_top1_same_scaffold"] == 1 and s["rank2_25_reach_rank1"] == 1
    assert s["gain_bound"] == pytest.approx(round(((1 - 0.5) + (0.5 - 0.25)) / 4, 4))


def test_coverage_summary():
    c = R.coverage_summary([0, 1, 7, None, 5])
    assert c == dict(n=5, unreadable=1, ge1=3, ge5=2, zero=1, median=3.0)
