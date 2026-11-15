"""build_holdout_e180（#122）の単体テスト。小さな合成データで、対象の絞り込み・外す理由・決定的な選び方・
1 構造 1 スペクトル・m/z をそのまま使うこと・電子の質量の行の確認・診断の関数を確かめる。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("build_holdout_e180", "scripts/cv/build_holdout_e180.py")


def _spectra():
    # A: [M+H]+ 2 本＋その他 1 本、B: [M-H]- 1 本、C: その他だけ、D: [M+H]+ 1 本（PubChem に無い）、E: プールにある、F: SMILES 不正
    rows = [
        ("AAAA", "CCO", "[M+H]+", "positive", 47.0489, [10.0, 20.0, 30.0], [0.1, 1.0, 0.5]),
        ("AAAA", "CCO", "[M+H]+", "positive", 47.0490, [11.0, 21.0, 31.0, 41.0], [1.0, 0.2, 0.3, 0.4]),
        ("AAAA", "CCO", "[M+Na]+", "positive", 69.03, [12.0, 22.0, 32.0], [1.0, 0.2, 0.3]),
        ("BBBB", "CC(=O)O", "[M-H]-", "negative", 59.0139, [13.0, 23.0, 33.0], [1.0, 0.2, 0.3]),
        ("CCCC", "c1ccccc1", "[M+NH4]+", "positive", 96.08, [14.0, 24.0, 34.0], [1.0, 0.2, 0.3]),
        ("DDDD", "CCN", "[M+H]+", "positive", 46.06, [15.0, 25.0, 35.0], [1.0, 0.2, 0.3]),
        ("EEEE", "CCC", "[M+H]+", "positive", 45.07, [16.0, 26.0, 36.0], [1.0, 0.2, 0.3]),
        ("FFFF", "C1CC", "[M+H]+", "positive", 40.0, [17.0, 27.0, 37.0], [1.0, 0.2, 0.3]),
    ]
    d = pd.DataFrame(rows, columns=["ik14", "smiles", "adduct", "mode", "precursor_mz", "mzs", "intensities"])
    d["mzs"] = d.mzs.map(np.array); d["intensities"] = d.intensities.map(np.array)
    d["row"] = np.arange(len(d))
    return d


def _structs():
    return pd.DataFrame(dict(ik14=["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"],
                             in_pc=[True, True, True, False, True, True], pool_src=[0, 0, 0, 0, 2, 0]))


def _ok(s):
    return isinstance(s, str) and s != "C1CC"


def test_eligible_filters_tier_pool_adduct_and_parse():
    assert B.eligible_structures(_structs(), _spectra(), _ok) == ["AAAA", "BBBB"]


def test_exclusion_reasons_on_every_key_type():
    keys = {"AAAA": {"AAAA", "KA"}, "BBBB": {"BBBB", "KB"}, "GGGG": {"GGGG", "KG"}, "HHHH": {"HHHH"}, "IIII": {"IIII"}}
    ts = [("XXXX", "KB", None), ("ALIAS", "KG", "RAW1"), ("ZZZZ", "ZZZZ", "HHHH")]
    r = B.exclusion_reasons(keys, pool_keys={"KA"}, train_ik14={"BBBB"}, ts_pairs=ts, v3_keys={"IIII"},
                            pubchem_found={"AAAA", "BBBB", "GGGG", "HHHH"})
    assert r["AAAA"] == ["pool"]
    assert r["BBBB"] == ["train_ik14", "train_structs_key"]
    assert r["GGGG"] == ["train_structs_key"]          # 別名: train_structs の別の ik14 が同じ key
    assert r["HHHH"] == ["train_structs_rawkey"]
    assert r["IIII"] == ["v3_truth", "pubchem_not_found"]
    assert B.exclusion_reasons({"OKOK": {"OKOK", None}}, set(), set(), [], set(), {"OKOK"}) == {}


def test_select_is_deterministic_one_spectrum_in_adducts():
    sp = _spectra()
    a = B.select(["BBBB", "AAAA"], sp, n_max=200, seed=0)
    b = B.select(["AAAA", "BBBB"], sp, n_max=200, seed=0)
    assert a == b and len(a) == 2 and len({ik for ik, _ in a}) == 2
    for ik, r in a:
        assert sp.loc[r, "ik14"] == ik and sp.loc[r, "adduct"] in B.ADDUCTS
    # 期待値を同じ手順で手計算
    rng = np.random.default_rng(0)
    order = rng.permutation(2)
    exp = []
    for j in order:
        ik = ["AAAA", "BBBB"][j]
        g = sp[(sp.ik14 == ik) & sp.adduct.isin(B.ADDUCTS)].sort_values("precursor_mz")
        exp.append((ik, int(g.row.iloc[int(rng.integers(len(g)))])))
    assert a == exp


def test_select_caps_at_n_max():
    many = pd.DataFrame(dict(ik14=[f"K{i:03d}" for i in range(300)], smiles="C", adduct="[M+H]+", mode="positive",
                             precursor_mz=1.0, mzs=[np.array([1.0, 2.0, 3.0])] * 300, intensities=[np.ones(3)] * 300))
    many["row"] = np.arange(300)
    s = B.select(sorted(many.ik14), many)
    assert len(s) == 200 and len({k for k, _ in s}) == 200


def test_build_frames_keeps_mz_as_is_and_columns():
    sp = _spectra()
    te, tr, ss = B.build_frames([("BBBB", 3), ("AAAA", 1)], sp, dict(zip(sp.ik14, sp.smiles)))
    assert list(te.columns) == B.TEST_COLS
    assert list(te.molecule_id) == ["m_00000", "m_00001"] and list(te.spectrum_id) == ["s_000000", "s_000001"]
    assert te.precursor_mz.tolist() == [59.0139, 47.0490]           # 電子の質量を二重に引かない
    np.testing.assert_array_equal(te.ms2_mzs.iloc[1], sp.mzs.iloc[1])
    assert te.instrument_type.unique().tolist() == ["timsTOF"]
    assert te.collision_energy_ev.isna().all() and te.base_peak_intensity.isna().all()
    assert tr.inchikey14.tolist() == ["BBBB", "AAAA"] and tr.smiles.tolist() == ["CC(=O)O", "CCO"]
    assert list(ss.columns) == ["molecule_id", "smiles"] and ss.molecule_id.tolist() == tr.molecule_id.tolist()


def test_check_electron_line():
    ok = ("        sh = 0.000548 if pos else 0.0; o = np.argsort(mz)\n"
          "x = dict(precursor_mz=round(float(r['pepmass']) - sh, 4), mzs=np.round(mz[o] - sh, 4))\n")
    r = B.check_electron_line(ok)
    assert r["line_no"] == 1 and r["subtract_again"] is False
    with pytest.raises(ValueError):
        B.check_electron_line("sh = 0.0\n")
    with pytest.raises(ValueError):
        B.check_electron_line("sh = 0.000548 if pos else 0.0\nprecursor_mz=float(r['pepmass'])\n")


def test_mass_diagnostics():
    exact = np.array([100.0, 200.0, 100.0])
    nm = exact * (1 + np.array([2e-6, 4e-6, 1e-6]))
    s = B.ppm_summary(nm, exact, np.array(["[M+H]+", "[M+H]+", "[M-H]-"]))
    assert s["[M+H]+"]["n"] == 2 and s["[M+H]+"]["median"] == pytest.approx(3.0)
    assert s["[M-H]-"]["median"] == pytest.approx(1.0)
    p = B.pos_neg_pair_ppm(np.array(["a", "a", "b", "c"]), np.array([100.0002, 100.0, 50.0, 60.0]),
                           np.array(["[M+H]+", "[M-H]-", "[M+H]+", "[M-H]-"]))
    assert p["n"] == 1 and p["median"] == pytest.approx(2.0)
    assert B.pos_neg_pair_ppm(np.array(["a"]), np.array([1.0]), np.array(["[M+H]+"]))["n"] == 0
