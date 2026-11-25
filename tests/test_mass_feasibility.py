"""scripts/features/mass_feasibility.py の単体テスト（Issue #238）。手で解ける小さな分子と合成の入力だけを使う。"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/features"))
import mass_feasibility as mf  # noqa: E402

DH = mf.parse_adduct("[M+H]+")[2]      # m(H) − m_e


def test_constants_fixed_from_paper():
    assert mf.PPM == 20.0 and mf.MAX_PROPOSALS == 64 and mf.H_SHIFTS == (-2, -1, 0, 1, 2)
    from rdkit import Chem
    assert mf.M_H == Chem.GetPeriodicTable().GetMostCommonIsotopeMass("H")
    assert len(mf.feature_names()) == 26 and len(set(mf.feature_names())) == 26


# ---------------------------------------------------------------- 1. 断片の数
def test_fragment_counts_small_molecules():
    # エタノール: 環でない単結合 2 本 → {CH3}, {CH2OH}, {C2H5}, {OH}
    M, f = mf.proposals("CCO")
    assert len(f) == 4
    assert M == pytest.approx(46.041865, abs=1e-5)
    assert sorted(np.round(f, 4)) == [15.0235, 17.0027, 29.0391, 31.0184]
    # シクロヘキサン: 環の中の 2 本 → 6 員環の連続した弧（1〜5 原子）6×5 = 30 個
    assert len(mf.proposals("C1CCCCC1")[1]) == 30
    # トルエン: 環の弧 30 個 + メチル {CH3} + 環だけ {C6H5} = 32 個
    assert len(mf.proposals("Cc1ccccc1")[1]) == 32
    # 結合が無い・読めない
    assert len(mf.proposals("C")[1]) == 0
    assert mf.proposals("not_a_smiles") is None


def test_fragment_cap_and_determinism():
    # C40 の直鎖: 切り方 39 本 × 両側 = 78 個 → 64 個に間引く
    smi = "C" * 40
    M, f = mf.proposals(smi)
    assert len(f) == 64
    # 同じ構造の別の書き方でも同じ提案（canonical な原子の順）
    a = mf.proposals("OCC(=O)c1ccccc1")
    b = mf.proposals("c1ccc(cc1)C(=O)CO")
    assert a[0] == pytest.approx(b[0]) and np.allclose(np.sort(a[1]), np.sort(b[1]))


# ---------------------------------------------------------------- 2. 20 ppm の境界
def test_direct_ion_20ppm_boundary():
    props = [(500.0, np.array([100.0]))]
    d = 100.0 + DH                       # h = 0 の直接のイオン
    x = np.array([d * (1 + 19.9e-6), d * (1 + 20.1e-6), d * (1 - 19.9e-6), d * (1 - 20.1e-6)])
    k, dp, dfr = mf.parse_adduct("[M+H]+")
    e = mf.edge_counts(props, x, 600.0, k, dp, dfr)   # 前駆体を M と無関係にして中性損失を外す
    assert e.tolist() == [[1, 0, 1, 0]]
    # h = +2 も同じ境界
    d2 = d + 2 * mf.M_H
    e2 = mf.edge_counts(props, np.array([d2 * (1 + 19.9e-6), d2 * (1 + 20.1e-6)]), 600.0, k, dp, dfr)
    assert e2.tolist() == [[1, 0]]
    # h = +3 は範囲の外
    e3 = mf.edge_counts(props, np.array([d + 3 * mf.M_H]), 600.0, k, dp, dfr)
    assert e3.tolist() == [[0]]


# ---------------------------------------------------------------- 3. 中性損失の扱い
def test_neutral_loss_uses_observed_precursor():
    M, f = 500.0, np.array([100.0])
    props = [(M, f)]
    k, dp, dfr = mf.parse_adduct("[M+H]+")
    prec = (M + dp) * (1 + 30e-6)        # 観測の前駆体が理論値から +30 ppm
    x = prec - (M - 100.0)               # m0 − x = ℓ（h = 0）ちょうど
    assert abs(x - (100.0 + dfr)) / (100.0 + dfr) > 100e-6   # 直接のイオンでは説明できない
    assert mf.edge_counts(props, np.array([x]), prec, k, dp, dfr).tolist() == [[1]]
    # 中性損失の 20 ppm 境界（分母は ℓ）
    ell = M - 100.0
    for rel, want in [(19.9e-6, 1), (20.1e-6, 0)]:
        xx = prec - ell * (1 + rel)
        assert mf.edge_counts(props, np.array([xx]), prec, k, dp, dfr).tolist() == [[want]]
    # 多量体（2M）の付加イオンでは中性損失の経路を使わない
    k2, dp2, dfr2 = mf.parse_adduct("[2M+H]+")
    assert k2 == 2
    assert mf.edge_counts(props, np.array([x]), prec, k2, dp2, dfr2).tolist() == [[0]]


def test_neutral_loss_generalised_adduct():
    # [M+NH4]+: (m0 − Δ_前駆体) − (x − Δ_断片) を ℓ と比べる。断片の担い手は H+
    M, f = 500.0, np.array([100.0])
    k, dp, dfr = mf.parse_adduct("[M+NH4]+")
    assert dp == pytest.approx(mf.formula_mass("NH4") - mf.M_E)
    assert dfr == pytest.approx(DH)
    prec = (M + dp) * (1 + 30e-6)
    x = (prec - dp) - (M - 100.0) + dfr
    assert mf.edge_counts([(M, f)], np.array([x]), prec, k, dp, dfr).tolist() == [[1]]
    # [M+H]+・[M+Na]+ では Δ_前駆体 = Δ_断片（論文の m0 − x_p と同じ）
    for a in ("[M+H]+", "[M+Na]+", "[M-H]-"):
        _, p_, q_ = mf.parse_adduct(a)
        assert p_ == pytest.approx(q_, abs=1e-12)


def test_parse_adduct_values():
    assert mf.parse_adduct("[M+Na]+")[1] == pytest.approx(22.98976928 - mf.M_E, abs=1e-6)
    assert mf.parse_adduct("[M-H2O+H]+")[1] == pytest.approx(DH - mf.formula_mass("H2O"))
    assert mf.parse_adduct("[M-2H2O+H]+")[1] == pytest.approx(DH - 2 * mf.formula_mass("H2O"))
    assert mf.parse_adduct("[M+CH2O2-H]-")[1] == pytest.approx(mf.formula_mass("CH2O2") - DH)
    assert mf.parse_adduct("[M+Cl]-")[2] == pytest.approx(mf.formula_mass("Cl") + mf.M_E)
    assert mf.parse_adduct("[M+2H]2+") is None and mf.parse_adduct("junk") is None and mf.parse_adduct(None) is None


# ---------------------------------------------------------------- 4. 他の候補が説明できないピーク（排他）と 13 測定
def test_measurements_by_hand():
    e = np.array([[1, 0, 2], [1, 0, 0]])
    x = np.array([50.0, 100.0, 190.0])
    it = np.array([0.25, 1.0, 0.04])     # w = 0.5, 1.0, 0.2、W = 1.7
    m = mf.measurements(e, x, it, 200.0)
    W = 1.7
    want0 = [2, 2 / 3, math.log(4), 1.5, 0.5, 0.7 / W, 0.2 / W, 1 / 3, 120.0, 0.5 / W, 0.0, 0.2 / W, 0.2 / W]
    want1 = [1, 1 / 3, math.log(2), 1.0, 1.0, 0.5 / W, 0.0, 0.0, 50.0, 0.5 / W, 0.0, 0.0, 0.0]
    assert np.allclose(m[0], want0) and np.allclose(m[1], want1)


def test_exclusive_peak_real_isomers():
    # エタノールとジメチルエーテル（同じ分子式）。C2H5+ (29.0391 + Δ) はエタノールだけが説明できる
    smiles = ["CCO", "COC"]
    props = [mf.proposals(s) for s in smiles]
    M = props[0][0]
    prec = M + DH
    x = np.array([15.0235 + DH, 29.0391 + DH])
    k, dp, dfr = mf.parse_adduct("[M+H]+")
    e = mf.edge_counts(props, np.round(x, 4), prec, k, dp, dfr)
    assert e[0, 1] > 0 and e[1, 1] == 0          # C2H5+ はエタノールだけ
    assert e[0, 0] > 0 and e[1, 0] > 0           # CH3+ は両方
    sp = dict(mz=np.round(x, 4), it=np.array([1.0, 0.25]), prec=prec, adduct="[M+H]+")
    X = mf.group_features(smiles, [sp])
    c = {n: i for i, n in enumerate(mf.feature_names())}
    W = 1.0 + 0.5
    assert X[0, c["mf_excl_int_cov"]] == pytest.approx(0.5 / W)
    assert X[1, c["mf_excl_int_cov"]] == 0.0
    assert X[0, c["mf_excl_peak_frac"]] == pytest.approx(0.5) and X[1, c["mf_excl_peak_frac"]] == 0.0
    # 平均からの差
    assert X[0, c["mf_excl_int_cov_dmean"]] == pytest.approx(0.25 / W)
    assert X[:, 13:].sum(axis=0) == pytest.approx(np.zeros(13), abs=1e-12)


def test_multi_spectrum_mean_skips_unusable():
    smiles = ["CCO", "COC"]
    M = mf.proposals("CCO")[0]
    sp = dict(mz=np.array([30.0464]), it=np.array([1.0]), prec=M + DH, adduct="[M+H]+")
    bad_adduct = dict(sp, adduct="??")
    empty = dict(sp, mz=np.array([]), it=np.array([]))
    a = mf.group_features(smiles, [sp])
    st = {}
    b = mf.group_features(smiles, [sp, bad_adduct, empty], st)
    assert np.array_equal(a, b) and st["n_spectra_used"] == 1 and st["n_spectra_skipped"] == 2
    two = mf.group_features(smiles, [sp, dict(sp, mz=np.array([20.0]))])
    assert np.allclose(two[:, :13], (a[:, :13] + 0.0) / 2)   # 2 本目はどの候補も説明しない → 0 との平均


# ---------------------------------------------------------------- 5. 全列 isfinite
def test_all_columns_finite_on_odd_inputs():
    smiles = ["not_a_smiles", "C", "[Na+].[Cl-]", "CCO", "C" * 80, "C[N+](C)(C)C", "[13CH3]CO", "c1ccc2ccccc2c1"]
    rng = np.random.default_rng(0)
    spectra = [
        dict(mz=np.array([np.nan, 30.0464, 1e6, -5.0]), it=np.array([1.0, 0.5, 1.0, 1.0]), prec=47.0491, adduct="[M+H]+"),
        dict(mz=rng.uniform(10, 500, 300), it=rng.uniform(0, 1, 300), prec=500.0, adduct="[M-H]-"),
        dict(mz=rng.uniform(10, 900, 50), it=rng.uniform(0, 1, 50), prec=900.0, adduct="[2M+Na]+"),
        dict(mz=np.array([30.0]), it=np.array([0.0]), prec=47.0, adduct="[M+Cl]-"),
        dict(mz=np.array([30.0]), it=np.array([1.0]), prec=float("nan"), adduct="[M+H]+"),
    ]
    X = mf.group_features(smiles, spectra)
    assert X.shape == (len(smiles), 26) and np.isfinite(X).all()


def test_prep_peaks_rule():
    mz = np.arange(1, 301, dtype=float)
    it = np.arange(1, 301, dtype=float)
    x, w = mf.prep_peaks(mz, it, prec=250.0)
    assert len(x) == 100 and x.max() == 250.0 and x.min() == 151.0 and np.all(np.diff(x) > 0)


# ---------------------------------------------------------------- 6. ピーク 0 本
def test_zero_peaks_and_empty_inputs():
    smiles = ["CCO", "COC"]
    z = dict(mz=np.array([]), it=np.array([]), prec=47.0491, adduct="[M+H]+")
    above = dict(mz=np.array([60.0]), it=np.array([1.0]), prec=47.0491, adduct="[M+H]+")   # 前駆体より上だけ
    for sp in ([z], [above], []):
        X = mf.group_features(smiles, sp)
        assert X.shape == (2, 26) and np.array_equal(X, np.zeros((2, 26)))
    assert mf.group_features([], [z]).shape == (0, 26)
    assert mf.measurements(np.zeros((2, 0), np.int64), np.array([]), np.array([]), 47.0).shape == (2, 13)
