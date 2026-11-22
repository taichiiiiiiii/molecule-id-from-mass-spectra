"""Issue #196: scripts/train/fragnn_v2_bench.py の純粋な関数（v2 の入力の作り方の候補）のテスト。torch・実データは読まない。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train"))
import fragnn_bench as B  # noqa: E402
import fragnn_v2_bench as V  # noqa: E402


def test_test_adducts_are_the_ten_and_all_have_explicit_carriers():
    assert len(V.TEST_ADDUCTS) == 10 and len(set(V.TEST_ADDUCTS)) == 10
    # どれも fragnn_bench.carriers の表に明示の行がある（末尾 +/- の既定に落ちない）
    na = 22.9897692809
    assert B.carriers("[M+Na]+", 1) == (B.PROTON, na - B.E)
    for a in V.TEST_ADDUCTS:
        c = B.carriers(a, 1 if a.endswith("+") else -1)
        assert c[0] == (B.PROTON if a.endswith("+") else -B.PROTON)
    assert not any(V.keep_adduct(a) for a in ["[2M+Na]+", "[2M+H]+", "[2M-H]-", "[M]+", "[M+2H]2+", None])
    assert all(V.keep_adduct(a) for a in V.TEST_ADDUCTS)


def test_drop_above_precursor():
    mz, it = V.drop_above_precursor([100.0, 151.9, 152.0, 152.1, 300.0], [1, 1, 1, 1, 1], 150.0)
    assert mz.tolist() == [100.0, 151.9, 152.0]
    mz, it = V.drop_above_precursor([100.0, 300.0], [1, 1], float("nan"))
    assert mz.tolist() == [100.0, 300.0]


def test_merge_peak_lists_follows_fpnet_rule():
    a = ([100.0, 200.0], [0.5, 1.0])      # 正規化しても同じ
    b = ([100.003, 300.0], [2.0, 4.0])    # 正規化で 0.5, 1.0。100.003 は 100.0 と 0.005 未満 → 強い方（同点は後ろ）
    mz, it = V.merge_peak_lists([a, b], k=10)
    assert mz.tolist() == [100.003, 200.0, 300.0]
    assert it.tolist() == [0.5, 1.0, 1.0]
    mz, it = V.merge_peak_lists([a, b], k=2)   # 上位 2 本（同点は m/z の昇順、top_peaks と同じ）
    assert mz.tolist() == [200.0, 300.0]
    mz, it = V.merge_peak_lists([([], []), ([1.0], [0.0])])
    assert len(mz) == 0


def test_peers_like_fpnet():
    rng = np.random.default_rng(0)
    assert V.peers(rng, np.array([7]), 7) == [7]
    mem = np.arange(10)
    for _ in range(200):
        p = V.peers(rng, mem, 3)
        assert 3 in p and 2 <= len(p) <= 4 and len(set(p)) == len(p)
    small = np.array([4, 9])
    for _ in range(20):
        assert sorted(V.peers(rng, small, 9)) == [4, 9]


def test_spec_feat_and_pack_v2():
    f = V.spec_feat_v2(1, 64, 40.0, "timsTOF")
    assert f == (1.0, math.log1p(64) / math.log(65), 0.4, 1.0)
    assert V.spec_feat_v2(-1, 0, None, "Orbitrap") == (-1.0, 0.0, 0.0, 0.0)
    rows = [dict(mz=np.arange(1, 101, dtype=float), it=np.linspace(0.01, 1, 100), prec=500.0, adduct="[M+H]+", mode=1,
                 ce=20.0, instrument="timsTOF"),
            dict(mz=[50.0, 60.0], it=[1.0, 0.5], prec=200.0, adduct="[M-H]-", mode=-1, ce=0.0, instrument=None)]
    pk = V.pack_v2(rows)
    assert pk["mz"].shape == (2, 64) and pk["spec_feat"].shape == (2, 4)
    assert pk["n"].tolist() == [64, 2]
    assert pk["mz"][0, :64].tolist() == list(np.arange(37, 101, dtype=float))   # 強度の上位 64 本、m/z の昇順
    assert pk["sh_n"].tolist() == [6, 6]
    assert pk["spec_feat"][1].tolist() == [-1.0, np.float32(math.log1p(2) / math.log(65)), 0.0, 0.0]
    # 同じ行の shift の表は v1 の pack_spectra と同じ
    p1 = B.pack_spectra(rows)
    assert np.array_equal(p1["sh_v"], pk["sh_v"]) and np.array_equal(p1["sh_n"], pk["sh_n"])


def test_group_index():
    g = V.group_index(np.array([5, 5, 6, 5]), np.array([0, 1, 0, 0]))
    assert {k: v.tolist() for k, v in g.items()} == {5 * 64: [0, 3], 5 * 64 + 1: [1], 6 * 64: [2]}


def test_extrapolate_ratio():
    r = V.extrapolate_ratio(0.2, 0.3, v1_real_s=36000.0, v1_steps=100)
    assert abs(r["ratio"] - 1.5) < 1e-12 and abs(r["v2_expected_h"] - 15.0) < 1e-12


def test_parse_footprint():
    txt = """python [4407]: 64-bit    Footprint: 2013742712 B (16384 bytes per page)

      Dirty         Clean   Reclaimable    Regions    Category
1083604992 B           0 B      835584 B         63    IOAccelerator (graphics)
431718400 B           0 B           0 B         14    Malloc Large
 85983232 B           0 B    35651584 B         23    Owned physical footprint (unmapped) (graphics)
"""
    assert V.parse_footprint(txt) == dict(footprint_b=2013742712, graphics_b=1083604992)
