"""Issue #127: 整列チャネルの h の再現の確認（scripts/ship/check_align_h.py）の純粋な関数。

torch・RDKit・PR #110 の align_train.py は使わない（CI に無い・main に無い）。標本の選び方・指紋の詰め方・一致の数え方・1 − cos を確かめる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/ship"))

import check_align_h as C  # noqa: E402


def test_sample_rows_is_seeded_sorted_unique():
    a = C.sample_rows(710_701)
    assert len(a) == 1000 and len(np.unique(a)) == 1000 and (np.diff(a) > 0).all()
    assert np.array_equal(a, np.sort(np.random.default_rng(0).choice(710_701, 1000, replace=False)))
    assert np.array_equal(a, C.sample_rows(710_701, 1000, 0))
    assert not np.array_equal(a, C.sample_rows(710_701, 1000, 1))
    with pytest.raises(ValueError):
        C.sample_rows(10, 11)


def test_pack_bits_is_msb_first_and_1279_bytes():
    bits = np.zeros(C.FP_BITS, np.uint8)
    bits[0] = 1          # 先頭のビット → 1 バイト目の最上位
    bits[9] = 1          # 2 バイト目の上から 2 番目
    bits[-1] = 1         # 10,225 番 → 最後のバイト（1,279 バイト目）の上から 2 番目（10,226 = 1,278·8 + 2）
    p = C.pack_bits(bits)
    assert p.shape == (C.PACKED_BYTES,) == (1279,)
    assert p[0] == 0b1000_0000 and p[1] == 0b0100_0000 and p[-1] == 0b0100_0000
    assert np.array_equal(np.unpackbits(p)[:C.FP_BITS], bits)     # align_train.unpack_bits_np と同じ向き
    with pytest.raises(ValueError):
        C.pack_bits(np.array([0, 2, 1]))
    with pytest.raises(ValueError):
        C.pack_bits(np.zeros((2, 8), np.uint8))


def test_rows_equal_and_first_diff_bits():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (5, C.PACKED_BYTES), dtype=np.uint8)
    b = a.copy()
    b[2, 10] ^= 0b0000_0100          # バイト 10 の下から 3 番目 = ビット 10·8 + 5 = 85
    eq = C.rows_equal(a, b)
    assert eq.tolist() == [True, True, False, True, True]
    assert C.first_diff_bits(a[2], b[2]) == [85]
    assert C.first_diff_bits(a[0], b[0]) == []
    with pytest.raises(ValueError):
        C.rows_equal(a, b[:4])


def test_one_minus_cos_and_summary():
    x = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    y = np.array([[2.0, 0.0], [1.0, 1.00001], [1.0, 0.0], [0.0, 1.0]])
    d = C.one_minus_cos(x, y)
    assert d[0] == pytest.approx(0.0, abs=1e-15)
    assert 0 < d[1] < 1e-9
    assert np.isnan(d[2])                   # 0 ベクトルは nan（一致に数えない）
    assert d[3] == pytest.approx(1.0)
    s = C.summarize_dist(d)
    assert s["n"] == 4 and s["n_finite"] == 3 and s["n_le_tol"] == 2
    assert s["frac_le_tol"] == pytest.approx(0.5)    # 分母は nan も含む全行
    assert s["max"] == pytest.approx(1.0)


def test_fp16_rounding_stays_far_below_tolerance():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((200, 512)).astype(np.float32)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    d = C.one_minus_cos(h, h.astype(np.float16))
    assert d.max() < C.COS_TOL / 100


def test_summarize_all_nan():
    s = C.summarize_dist(np.array([np.nan, np.nan]))
    assert s["n_finite"] == 0 and s["frac_le_tol"] == 0.0


def test_load_align_module_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_align_module(str(tmp_path))
