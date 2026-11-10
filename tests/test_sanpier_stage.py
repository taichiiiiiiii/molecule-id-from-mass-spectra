"""Issue #60: sanpier 段の混ぜ方、点のキャッシュ、スペクトルの選び方、判定コードの共有（合成データ）。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import run as R  # noqa: E402
import sanpier as SP  # noqa: E402


def test_equal_weight_mix_and_scale_back():
    rows = [(3.0, 10), (2.0, 11), (1.0, 12)]
    n = {10: 0.0, 11: 0.0, 12: 9.0}            # n は 12 を推す
    out = R.sanpier_mix(rows, n)
    s = np.array([3.0, 2.0, 1.0]); zs = (s - s.mean()) / s.std()
    nn = np.array([0.0, 0.0, 9.0]); zn = (nn - nn.mean()) / nn.std()
    want = s.mean() + s.std() * (zs + zn) / 2
    got = {i: v for v, i in out}
    for j, i in enumerate((10, 11, 12)):
        assert abs(got[i] - want[j]) < 1e-9
    assert [i for _, i in out] == [10, 12, 11]   # z の和 0.52 / 0.19 / −0.71


def test_missing_n_is_zero_z():
    rows = [(3.0, 1), (2.0, 2), (1.0, 3)]
    out = R.sanpier_mix(rows, {1: 1.0, 2: 5.0})       # 3 は点無し → z_n = 0
    s = np.array([3.0, 2.0, 1.0]); zs = (s - 2.0) / s.std()
    zn = np.array([-1.0, 1.0, 0.0])                    # 点のある 2 つで z 化
    want = 2.0 + s.std() * (zs + zn) / 2
    got = {i: v for v, i in out}
    assert all(abs(got[i] - want[j]) < 1e-9 for j, i in enumerate((1, 2, 3)))


def test_fallbacks_return_rows_unchanged():
    rows = [(3.0, 1), (2.0, 2), (1.0, 3)]
    assert R.sanpier_mix(rows, {}) == rows                         # 点が無い
    assert R.sanpier_mix(rows, {1: 0.5}) == rows                   # 点が 1 つだけ
    assert R.sanpier_mix(rows, {1: 2.0, 2: 2.0, 3: 2.0}) == rows   # σ_n = 0
    assert R.sanpier_mix([(1.0, 5)], {5: 3.0}) == [(1.0, 5)]       # 候補 1 個
    flat = [(1.0, 1), (1.0, 2)]
    assert R.sanpier_mix(flat, {1: 0.0, 2: 1.0}) == flat           # σ_s = 0


def test_pack_and_load_roundtrip(tmp_path):
    per = {"m1": (np.array([4, 7]), np.array([0.5, np.nan], np.float32)), "m2": (np.array([], np.int64), np.array([], np.float32))}
    np.savez_compressed(tmp_path / "s.npz", **SP.pack(per))
    got = SP.load_scores(tmp_path / "s.npz")
    assert got["m1"][4] == 0.5 and np.isnan(got["m1"][7]) and got["m2"] == {}


class _Feat:
    @staticmethod
    def peaks(mz, it, prec, max_peaks):
        mz = np.asarray(mz)
        if not len(mz):
            raise ValueError("No valid peaks")
        return None, len(mz)


def test_molecule_spectra_drops_short_and_maps_ce():
    te = pd.DataFrame(dict(molecule_id=["a", "a", "a", "b"],
                           ms2_mzs=[[1, 2, 3], [1, 2], [], [5, 6, 7, 8]],
                           ms2_normalized_intensities=[[1, 1, 1], [1, 1], [], [1, 1, 1, 1]],
                           precursor_mz=[100.0] * 4, adduct=["[M+H]+"] * 4, ionization_mode=["positive"] * 4,
                           collision_energy_ev=[20.0, 30.0, 40.0, 50.0]))
    sp = SP.molecule_spectra(te, _Feat, min_peaks=3, max_peaks=96)
    assert len(sp["a"]) == 1 and sp["a"][0]["collision_energy"] == 20.0
    assert len(sp["b"]) == 1


def test_check_uses_frag_judge():
    spec = importlib.util.spec_from_file_location("sanpier_v3_check", ROOT / "scripts/cv/sanpier_v3_check.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import frag_v3_check as fc
    assert m.judge is fc.judge and m.REQUIRED == ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
    assert m.NEW_TAG == "score+sanpier+prior+frag" and m.CTRL_TAG == "score+prior+frag"
