"""Issue #160: scripts/cv/net_mde.py の純粋な関数のテスト（合成の rr だけ）。"""
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))

import net_mde as M  # noqa: E402


def test_sd_and_mde():
    assert M.sd([]) == 0.0 and M.sd([1.0]) == 0.0
    assert M.sd([0.0, 1.0]) == pytest.approx(math.sqrt(0.5))
    d = [0.0] * 98 + [1.0, -1.0]
    assert M.mde(d) == pytest.approx(2.8 * M.sd(d) / 10.0)


def test_summarize_counts_and_ignores_unchanged_variants():
    ctrl = {f"m{i}": (1.0 if i % 2 else 0.5) for i in range(10)}
    same = dict(ctrl)
    moved = dict(ctrl, m0=1.0, m2=0.0)
    r = M.summarize(ctrl, {"same": same, "moved": moved})
    assert r["n"] == 10
    assert r["mrr_ctrl"] == 0.75
    assert r["variants"]["same"]["n_changed"] == 0 and r["variants"]["same"]["mde"] == 0.0
    assert r["variants"]["moved"]["n_changed"] == 2
    assert r["variants"]["moved"]["diff"] == 0.0
    # 1 つも変えていない変種は min/median/max に入れない
    assert r["mde_min"] == r["mde_max"] == r["variants"]["moved"]["mde"] > 0


def test_summarize_rejects_different_molecule_sets():
    with pytest.raises(AssertionError):
        M.summarize({"a": 1.0, "b": 0.0}, {"v": {"a": 1.0}})


def test_ctrl_since_uses_oldest_identical_backup(tmp_path):
    cur = tmp_path / "submission_x.csv"
    old_same = tmp_path / "submission_x.csv.bak_1"
    old_diff = tmp_path / "submission_x.csv.bak_0"
    cur.write_text("A")
    old_same.write_text("A")
    old_diff.write_text("B")
    os.utime(old_diff, (100, 100))
    os.utime(old_same, (200, 200))
    os.utime(cur, (300, 300))
    assert M.ctrl_since(cur) == 200


def test_rank_profile():
    r = M.rank_profile({"a": 1.0, "b": 0.5, "c": 0.04, "d": 0.0})
    assert r == dict(n=4, rank1=1, rank2_25=2, absent=1, gain_sum_unlocked=1.46, gain_sum_lock1=0.46)
