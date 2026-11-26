"""scripts/features/contrastive_coverage.py の単体テスト（Issue #253）。合成データだけ（frag_rescore があれば一致も見る）。"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/features"))
import contrastive_coverage as cc  # noqa: E402


def test_constants_fixed_from_paper():
    assert cc.TAU_C == 0.1 and cc.W_FLOOR == 1e-8 and cc.SIGMA_FLOOR == 0.02
    assert cc.KAPPA == pytest.approx(0.1 * math.log(1 + math.exp(10.0)), rel=1e-15)
    assert cc.KAPPA == pytest.approx(1.00000454, abs=1e-8)


# ---------------------------------------------------------------- 1. 全候補が説明するピークは寄与 0
def test_peak_explained_by_all_contributes_zero():
    h = np.ones((5, 1))
    np.testing.assert_array_equal(cc.coverage(h, [0.7]), np.zeros(5))
    # 全員が説明するピーク + 誰も説明しないピーク → やはり 0
    h = np.array([[1, 0]] * 4, float)
    np.testing.assert_array_equal(cc.coverage(h, [1.0, 0.25]), np.zeros(4))


# ---------------------------------------------------------------- 2. 1 候補だけが説明するピーク
def test_peak_explained_by_one_candidate():
    I = np.array([1.0, 0.25, 0.04])              # w = 1, 0.5, 0.2 → W = 1.7
    h = np.array([[1, 1, 0],                      # ピーク 0 は全員、ピーク 1 は候補 0 だけ
                  [1, 0, 0],
                  [1, 0, 0]], float)
    got = cc.coverage(h, I)
    assert got[0] == pytest.approx(cc.KAPPA * 0.5 / 1.7, rel=1e-12)
    assert got[1] == 0.0 and got[2] == 0.0


# ---------------------------------------------------------------- 3. 候補数が偶数の median は下側
def test_even_shortlist_uses_lower_median():
    h = np.array([[1], [1], [0], [0]], float)     # 並べると 0, 0, κ, κ → 下側の median = 0（上側なら κ、平均なら κ/2）
    np.testing.assert_array_equal(cc.lower_median(cc.KAPPA * h), [0.0])
    got = cc.coverage(h, [0.36])                  # w = 0.6、W = 0.6
    np.testing.assert_allclose(got, [cc.KAPPA, cc.KAPPA, 0, 0], rtol=1e-12)
    # 3 個が説明すると下側の median = κ → 寄与 0
    np.testing.assert_array_equal(cc.coverage(np.array([[1], [1], [1], [0]], float), [0.36]), np.zeros(4))
    # 奇数は普通の median
    np.testing.assert_array_equal(cc.lower_median(np.array([[3.0], [1.0], [2.0]])), [2.0])


# ---------------------------------------------------------------- 4. W = 0
def test_zero_total_weight_gives_zero():
    np.testing.assert_array_equal(cc.coverage(np.zeros((3, 0)), []), np.zeros(3))       # ピーク無し
    np.testing.assert_array_equal(cc.coverage(np.array([[1, 0], [0, 0]], float), [0.0, 0.0]), np.zeros(2))  # 強度 0


def test_single_candidate_and_zscore():
    assert cc.coverage(np.array([[1, 0, 1]], float), [1, 1, 1]).tolist() == [0.0]       # median = 自分
    assert cc.zscore(np.array([0.3])).tolist() == [0.0]
    v = np.array([0.0, 0.01, 0.02])                                                       # σ < 0.02 → 下限
    np.testing.assert_allclose(cc.zscore(v), (v - 0.01) / 0.02)
    v = np.array([0.0, 1.0])                                                              # σ = 0.5（ddof 0）
    np.testing.assert_allclose(cc.zscore(v), [-1.0, 1.0])


def test_input_checks():
    with pytest.raises(ValueError):
        cc.coverage(np.ones((2, 2)), [1.0])
    with pytest.raises(ValueError):
        cc.coverage(np.ones((2, 1)), [-1.0])


class _FakeFrag:
    """frag_rescore の必要な部分だけの偽物（[M+H]+ の担い手 1 つ、dh = 0 だけ）。"""
    H = 1.00782503207

    @staticmethod
    def resolve_cfg(_):
        return dict(w1=1.0, w2=0.6, w3=0.3, max_breaks=2, tol=0.005, h_shifts=(0,), max_bonds_2=100, max_bonds_3=40)

    @staticmethod
    def _carriers(_adduct, _mode):
        return (1.0,)

    @staticmethod
    def _levels_cached(smi, _c, _cache):
        return {"A": [np.array([100.0]), np.array([50.0]), np.array([]),],
                "B": [np.array([100.0]), np.array([]), np.array([70.0])],
                "bad": None}[smi]


def test_molecule_features_on_fake_fragments():
    sp = [dict(mz=np.array([101.0, 51.0, 71.0, np.nan]), it=np.array([1.0, 0.25, 0.04, 0.5]), adduct="[M+H]+", mode=1)]
    out = cc.molecule_features(sp, ["A", "B", "bad"], _FakeFrag)
    # h: A = [1,1,0]、B = [1,0,1]、bad = 0。下側の median（3 個）: ピーク 0 は κ、1 と 2 は 0
    W = 1.0 + 0.5 + 0.2
    np.testing.assert_allclose(out["cc"], [cc.KAPPA * 0.5 / W, cc.KAPPA * 0.2 / W, 0.0], rtol=1e-12)
    # 作り直しの metfrag_lite: 線形の強度の割合、A = (1 + 0.25)/1.29、B = (1 + 0.6·0.04)/1.29
    assert out["metfrag_lite_rebuilt"][0] == pytest.approx(1.25 / 1.29)
    assert out["metfrag_lite_rebuilt"][1] == pytest.approx(1.024 / 1.29)
    assert out["metfrag_lite_rebuilt"][2] is None
    assert out["n_spectra_with_peaks"] == 1
    none = cc.molecule_features([], ["A", "B"], _FakeFrag)
    assert none["cc"].tolist() == [0.0, 0.0] and none["n_spectra_with_peaks"] == 0


def test_matches_frag_rescore_when_available():
    frag_dir = ROOT / "data/ext/frag"
    if not (frag_dir / "frag_rescore.py").exists():
        pytest.skip("data/ext/frag が無い（CI）")
    pytest.importorskip("rdkit")
    sys.path.insert(0, str(frag_dir))
    import frag_rescore as fr
    sp = [dict(mz=np.array([138.0550, 121.0284, 93.0335, 65.0386]), it=np.array([1.0, 0.6, 0.3, 0.1]),
               adduct="[M+H]+", prec=139.0390, mode=1),
          dict(mz=np.array([92.0257, 136.0404]), it=np.array([1.0, 0.4]), adduct="[M-H]-", prec=137.0244, mode=-1)]
    smis = ["OC(=O)c1ccccc1O", "OC(=O)c1ccc(O)cc1", "CCO", "not_a_smiles"]
    out = cc.molecule_features(sp, smis, fr)
    ref = fr.score_molecule(sp, smis)
    for a, b in zip(out["metfrag_lite_rebuilt"], ref):
        assert (a is None and b is None) or a == pytest.approx(b, abs=1e-12)
    assert np.all(np.isfinite(out["cc"])) and out["cc"][3] == 0.0
