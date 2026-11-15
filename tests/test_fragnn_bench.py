"""Issue #120: fragnn_bench の純粋な関数（合成データ）。frag_rescore・torch があるときだけ一致の確認もする。"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fragnn_bench", ROOT / "scripts/train/fragnn_bench.py")
fb = importlib.util.module_from_spec(_spec)
sys.modules["fragnn_bench"] = fb
_spec.loader.exec_module(fb)


def test_negative_csr_and_union():
    mass = np.array([100.0, 100.0005, 100.0009, 200.0, 200.001, 300.0])
    allowed = np.array([True, True, False, True, True, True])
    off, flat = fb.negative_csr(np.array([0, 3]), mass, allowed)
    assert off.tolist() == [0, 1, 2] and flat.tolist() == [1, 4]
    assert fb.union_count([0, 3], flat) == 4
    assert fb.union_count([0, 0, 1], [1, 1]) == 2


def test_select_slice_deterministic_and_unique():
    keys = [f"K{i:05d}" for i in range(5000)] + ["K00001"]
    a, b = fb.select_slice(keys, 1000, 0), fb.select_slice(list(reversed(keys)), 1000, 0)
    assert a == b and len(a) == 1000 and len(set(a)) == 1000
    assert fb.select_slice(keys, 1000, 1) != a
    assert fb.select_slice(["A", "B"], 1000, 0) == ["A", "B"]


def test_top_peaks_keeps_most_intense_sorted_by_mz():
    mz = np.arange(40, dtype=float) + 50
    it = np.linspace(0.01, 1.0, 40)
    it[0] = 0.0
    m, i = fb.top_peaks(mz, it, 32)
    assert len(m) == 32 and np.all(np.diff(m) > 0) and m.min() == 58.0 and i.max() == 1.0


def _one_struct():
    # k0: 前駆体 300.0、k1: 150.0, 120.0、k2: 80.0
    return fb.levels_to_csr([[np.array([300.0]), np.array([120.0, 150.0]), np.array([80.0])], None])


def _mkspec(mz, it, prec=301.0, adduct="[M+H]+"):
    return fb.pack_spectra([dict(mz=np.array(mz), it=np.array(it), prec=prec, adduct=adduct, mode=1, ce=20.0)])


@pytest.mark.parametrize("use_numba", [False, True])
def test_pair_features_levels_shift_and_score(use_numba):
    if use_numba and not fb.HAVE_NUMBA:
        pytest.skip("numba なし")
    masses, off = _one_struct()
    P = fb.PROTON
    sp = _mkspec([80.0 + P, 150.0 + P + fb.H, 200.0, 300.0 + P], [0.2, 0.3, 0.4, 0.1])
    feats, fs = fb.batch_features(masses, off, [0, 1], [0, 0], sp, use_numba=use_numba)
    f = feats[0]
    # m/z の昇順: 80+P（k2, dh 0）、150+P+H（k1, dh +1）、200（説明なし）、300+P（k0, dh 0）
    assert f[0, 7] == 1 and f[0, 9 + 2] == 1
    assert f[1, 6] == 1 and f[1, 9 + 3] == 1
    assert f[2, 4] == 1 and f[2, 16] == 0
    assert f[3, 5] == 1 and f[3, 3] == 0
    assert np.allclose(f[:4, 0], [0.2, 0.3, 0.4, 0.1])
    assert math.isclose(float(fs[0]), 0.2 * 0.6 + 0.3 * 1.0 + 0.1 * 1.0, rel_tol=1e-6)
    assert np.all(feats[1][:4, 4] == 1) and float(fs[1]) == 0.0   # 読めない構造は説明なし
    assert np.all(f[4:] == 0)                                       # パディングのピーク


def test_numba_matches_python():
    if not fb.HAVE_NUMBA:
        pytest.skip("numba なし")
    rng = np.random.default_rng(0)
    lv = [[np.sort(rng.uniform(50, 400, n)) for n in (1, 30, 200)] for _ in range(5)]
    masses, off = fb.levels_to_csr(lv)
    rows = [dict(mz=rng.uniform(50, 400, 40), it=rng.uniform(0, 1, 40), prec=401.0, adduct=a, mode=m, ce=10.0)
            for a, m in (("[M+H]+", 1), ("[M+Na]+", 1), ("[M-H]-", -1))]
    sp = fb.pack_spectra(rows)
    cs, ss = np.repeat(np.arange(5), 3), np.tile(np.arange(3), 5)
    a = fb.batch_features(masses, off, cs, ss, sp, use_numba=False)
    b = fb.batch_features(masses, off, cs, ss, sp, use_numba=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_extrapolate_formula_and_verdict():
    r = fb.extrapolate(100_000, 0.01, 360.0, 1.0, 0.25, steps=28_000)
    assert math.isclose(r["precompute_h"], 1000 / 3600)
    assert math.isclose(r["eval_h"], 14 * math.ceil(39_437 / 128) * 0.25 / 3600)
    assert r["verdict"] == "本番へ"
    r2 = fb.extrapolate(100_000, 0.01, 0.0, 64 * 3600 / 28_000 + 0.01, 0.0)
    assert r2["verdict"] == "減らす案へ"


def test_carriers_and_score_match_frag_rescore():
    frag = ROOT / "data/ext/frag"
    if not (frag / "frag_rescore.py").exists():
        pytest.skip("data/ext/frag なし（CI）")
    pytest.importorskip("rdkit")
    sys.path.insert(0, str(frag))
    import frag_rescore as fr
    for a, m in (("[M+H]+", 1), ("[M+Na]+", 1), ("[M+NH4]+", 1), ("[M-H]-", -1), ("[M+Cl]-", -1), ("[M+CH2O2-H]-", -1),
                 ("[M+2H]2+", 1), ("weird", -1)):
        assert fb.carriers(a, m) == fr._carriers(a, m)
    smi = "CCOC(=O)c1ccccc1O"
    lv = fr.fragment_levels(smi)
    masses, off = fb.levels_to_csr([lv])
    rng = np.random.default_rng(1)
    mz = np.concatenate([lv[1][:5] + fb.PROTON, rng.uniform(40, 160, 10)])
    it = rng.uniform(0.05, 1, len(mz))
    sp = fb.pack_spectra([dict(mz=mz, it=it, prec=float(lv[0][0] + fb.PROTON), adduct="[M+H]+", mode=1, ce=0)])
    _, fs = fb.batch_features(masses, off, [0], [0], sp)
    ref = fr.score_molecule([dict(mz=mz, it=it, adduct="[M+H]+", prec=float(lv[0][0] + fb.PROTON), mode=1)], [smi])[0]
    assert math.isclose(float(fs[0]), ref, rel_tol=1e-5)   # ピーク 15 本 ≤ 32 なので全ピークでの値と同じ


def test_model_forward_shape():
    torch = pytest.importorskip("torch")
    m = fb.build_model()
    x = torch.rand(6, fb.TOP_PEAKS, fb.F_PEAK)
    valid = torch.zeros(6, fb.TOP_PEAKS, dtype=torch.bool)
    valid[:, :5] = True
    out = m(x, valid, torch.rand(6, 1), torch.rand(6, 3))
    assert out.shape == (6,)
    assert sum(p.numel() for p in m.parameters()) < 200_000


def test_padded_pairs_and_pack_term():
    assert fb.padded_pairs(1) == 2048 and fb.padded_pairs(2048) == 2048 and fb.padded_pairs(2049) == 4096
    r = fb.extrapolate(0, 0.0, 100.0, 0.0, 0.0, steps=2000, sec_pack_per_spectrum=1e-3, n_spectra=1_000_000)
    assert math.isclose(r["spectra_h"], (100.0 + 1000.0) / 3600)
