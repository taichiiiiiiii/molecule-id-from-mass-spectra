"""Issue #92: spectrum_blind_v3.blind_rank の合成テスト（pop 順・同点で元の順、重複除去、hidden の除外）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

import spectrum_blind_v3 as SB  # noqa: E402


def test_pop_desc_ties_keep_window_order():
    keys = np.array(["A", "B", "C", "D", "E"])
    pop = np.array([1.0, 3.0, 1.0, 3.0, 2.0])
    out = SB.blind_rank(np.arange(5), pop, keys, set())
    assert out == [1, 3, 4, 0, 2]


def test_window_subset_and_order_is_window_order():
    keys = np.array(["A", "B", "C", "D", "E"])
    pop = np.array([5.0, 0.0, 0.0, 9.0, 0.0])
    out = SB.blind_rank(np.array([1, 2, 4]), pop, keys, set())
    assert out == [1, 2, 4]   # 窓の外の 0・3 は出ない、同点は窓の順


def test_dedup_by_key():
    keys = np.array(["A", "A", "B", "B", "C"])
    pop = np.array([1.0, 2.0, 3.0, 0.5, 0.1])
    out = SB.blind_rank(np.arange(5), pop, keys, set())
    assert out == [2, 1, 4]
    assert len({keys[i] for i in out}) == len(out)


def test_hidden_keys_never_appear():
    keys = np.array(["A", "B", "C", "D"])
    pop = np.array([9.0, 8.0, 1.0, 0.0])
    out = SB.blind_rank(np.arange(4), pop, keys, {"A", "D"})
    assert out == [1, 2]


def test_top_k_cap():
    n = 40
    keys = np.array([f"K{i:02d}" for i in range(n)])
    pop = np.arange(n, dtype=float)
    out = SB.blind_rank(np.arange(n), pop, keys, set())
    assert len(out) == SB.K == 25
    assert out[0] == n - 1
