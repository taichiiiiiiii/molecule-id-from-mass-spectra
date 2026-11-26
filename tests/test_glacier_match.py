"""scripts/features/glacier_match.py の単体テスト（Issue #259）。合成データだけ。"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/features"))
import glacier_match as GM  # noqa: E402

TOL = 0.005


def sp(mz, it):
    return GM.prep(np.array(mz, float), np.array(it, float))


def test_same_spectrum_gives_one():
    a = sp([100.0, 150.02, 233.1], [1.0, 0.3, 0.05])
    cos, ent = GM.pair_sims(a[0], a[1], a[0], a[1], TOL)
    assert cos == pytest.approx(1.0, abs=1e-12) and ent == pytest.approx(1.0, abs=1e-12)
    # 強度を定数倍しても 1（cosine も entropy も尺度に依らない）
    b = sp([100.0, 150.02, 233.1], [0.5, 0.15, 0.025])
    cos, ent = GM.pair_sims(a[0], a[1], b[0], b[1], TOL)
    assert cos == pytest.approx(1.0, abs=1e-12) and ent == pytest.approx(1.0, abs=1e-12)


def test_no_overlap_gives_zero():
    a = sp([100.0, 150.0], [1.0, 0.5])
    b = sp([100.0 + 2 * TOL, 300.0], [1.0, 0.2])
    cos, ent = GM.pair_sims(a[0], a[1], b[0], b[1], TOL)
    assert cos == 0.0 and ent == pytest.approx(0.0, abs=1e-12)


def test_tolerance_edge_and_one_to_one():
    a = sp([100.0], [1.0])
    b = sp([100.0 + 0.004], [1.0])
    assert GM.pair_sims(a[0], a[1], b[0], b[1], TOL)[0] == pytest.approx(1.0)
    # 観測 1 本に予測 2 本が許容内: 1 対 1 なので片方だけ対になる
    c = sp([99.999, 100.001], [1.0, 1.0])
    cos, _ = GM.pair_sims(a[0], a[1], c[0], c[1], TOL)
    assert cos == pytest.approx(1.0 / math.sqrt(2.0))


def test_entropy_matches_formula_partial_overlap():
    # 重み付けが効かないように S ≥ 3 の一様なスペクトル（25 本 → S = ln 25 ≈ 3.22）
    mz = np.arange(25) * 10.0 + 100.0
    a = sp(mz, np.ones(25))
    b = sp(np.r_[mz[:20], mz[20:] + 1.0], np.ones(25))   # 20 本が対
    _, ent = GM.pair_sims(a[0], a[1], b[0], b[1], TOL)
    pa = np.ones(25) / 25
    mixed = np.r_[2 * pa[:20], pa[20:], pa[20:]] / 2
    s_ab = -(mixed * np.log(mixed)).sum()
    s = math.log(25)
    assert ent == pytest.approx(1 - (2 * s_ab - 2 * s) / math.log(4), abs=1e-12)


def test_empty_spectrum():
    a = sp([100.0], [1.0])
    e = sp([], [])
    assert GM.pair_sims(a[0], a[1], e[0], e[1], TOL) == (0.0, 0.0)
    assert GM.spectrum_value(e, [a], TOL) is None


def test_prep_drops_nonpositive_and_sorts():
    mz, it = GM.prep(np.array([200.0, 100.0, 150.0]), np.array([1.0, 0.0, 0.5]))
    np.testing.assert_array_equal(mz, [150.0, 200.0])
    np.testing.assert_array_equal(it, [0.5, 1.0])


def test_no_prediction_gives_nan_and_has_gl_zero():
    q = sp([100.0], [1.0])
    assert GM.spectrum_value(q, [], TOL) is None
    a = GM.aggregate([None, None])
    assert a["has_gl"] == 0
    assert all(math.isnan(a[c]) for c in GM.SIM_COLS)
    assert GM.aggregate([])["has_gl"] == 0


def test_spectrum_value_averages_categories():
    q = sp([100.0, 200.0], [1.0, 1.0])
    p1 = sp([100.0, 200.0], [1.0, 1.0])     # cos 1
    p2 = sp([100.0, 300.0], [1.0, 1.0])     # cos 0.5
    cos, _ = GM.spectrum_value(q, [p1, p2], TOL)
    assert cos == pytest.approx(0.75)


def test_aggregate_mean_and_max():
    a = GM.aggregate([(0.2, 0.4), None, (0.6, 0.1)])
    assert a["has_gl"] == 1
    assert a["gl_cos_mean"] == pytest.approx(0.4) and a["gl_cos_max"] == pytest.approx(0.6)
    assert a["gl_ent_mean"] == pytest.approx(0.25) and a["gl_ent_max"] == pytest.approx(0.4)


def test_relativize_masked_only_has_gl_rows():
    df = pd.DataFrame(dict(set="s", molecule_id=["m1"] * 3 + ["m2"] * 2,
                           gl_cos_mean=[0.2, np.nan, 0.6, np.nan, np.nan],
                           gl_ent_mean=[0.1, np.nan, 0.3, np.nan, np.nan], has_gl=[1, 0, 1, 0, 0]))
    out = GM.relativize_masked(df, ["set", "molecule_id"])
    assert out.gl_cos_mean_dmean.tolist()[0] == pytest.approx(-0.2)
    assert out.gl_cos_mean_dmax.tolist()[2] == pytest.approx(0.0)
    assert out.gl_cos_mean_softmax[[0, 2]].sum() == pytest.approx(1.0)
    assert out.loc[[1, 3, 4], [c for c in out.columns if c.startswith("gl_") and "_mean_" in c]].isna().all().all()
    assert list(out.columns[:5]) == list(df.columns)


def test_registry_matches_added_columns():
    sys.path.insert(0, str(ROOT / "scripts/ranker"))
    import lambdarank_oof as LR
    from relativize import SUFFIXES
    added = list(GM.SIM_COLS) + ["has_gl"] + [f"{c}_{s}" for c in GM.REL_SRC for s in SUFFIXES]
    assert list(LR.EXTRA_ALLOWED["glacier"][1]) == added and len(added) == 15
