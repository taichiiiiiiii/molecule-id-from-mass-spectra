"""H2: lib_max_sim の装置の系統による置き換え（Issue #287）。合成の参照で定義を確かめる。

casmi（data/ext の外部コード）は CI に無いので、entropy_sim を「問い合わせの mz[0] × 参照の mz[0]」にした偽物を入れる。
"""
import re
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts/local_eval", "scripts/features", "scripts/ranker", "scripts/research", "scripts/validate"):
    sys.path.insert(0, str(ROOT / p))

import library as L  # noqa: E402
import instrument as I  # noqa: E402


@pytest.fixture
def fake_casmi(monkeypatch):
    sp = types.ModuleType("casmi.spectra")
    sp.prep_query = lambda mz, it, *a: (np.asarray(mz, np.float64), np.asarray(it, np.float64))
    sp.entropy_sim = lambda qm, qp, rm, rp, tol, ppm: float(qm[0] * rm[0])
    lib = types.ModuleType("casmi.library")

    class SearchCfg:
        floor, topk, power, ent_weight, tol, ppm_tol = 0.0, 0, 1.0, False, 0.01, 10.0

    lib.SearchCfg = SearchCfg
    pkg = types.ModuleType("casmi")
    pkg.spectra, pkg.library = sp, lib
    monkeypatch.setitem(sys.modules, "casmi", pkg)
    monkeypatch.setitem(sys.modules, "casmi.spectra", sp)
    monkeypatch.setitem(sys.modules, "casmi.library", lib)
    return pkg


def q(inst, adduct="[M+H]+", scale=1.0):
    return dict(mz=np.array([scale]), it=np.array([1.0]), inst=inst, adduct=adduct)


def ref_rows(rows):
    """rows = [(key, sim, instrument, adduct, ingest_lib)] → REF_META_COLS の DataFrame（sim は参照の mz[0]）。"""
    return pd.DataFrame(dict(inchikey14=[r[0] for r in rows], ingest_lib=[r[4] for r in rows],
                             ms2_mzs=[np.array([r[1]]) for r in rows], ms2_normalized_intensities=[np.array([1.0]) for r in rows],
                             instrument_type=pd.Series([r[2] for r in rows], dtype=object),
                             adduct=pd.Series([r[3] for r in rows], dtype=object)))


def test_family_moved_unchanged():
    import cross_instrument_sim as C
    assert C.family is I.family and C.inst_value is I.inst_value
    assert I.family("timsTOF") == "timsTOF" and I.family("Q-Exactive") == "Orbitrap/FT" and I.family("nan") is None


def test_i_same_family_refs_removed(fake_casmi):
    refs = L.refs_with_meta(ref_rows([("K", 0.95, "timsTOF", "[M+H]+", "x"), ("K", 0.60, "Orbitrap", "[M+H]+", "x"),
                                      ("K", 0.70, "timsTOF Pro", "[M+H]+", "x")]))
    full, has, xi, hxi, sa = L.lib_scores_xinst([q("timsTOF")], ["K"], refs)["K"]
    assert full == pytest.approx(0.95) and has == 1
    assert xi == pytest.approx(0.60) and hxi == 1 and sa == pytest.approx(0.60)
    # 問い合わせが Orbitrap なら Orbitrap の参照が外れ、timsTOF の参照が入る
    full, has, xi, hxi, sa = L.lib_scores_xinst([q("Q Exactive HF")], ["K"], refs)["K"]
    assert xi == pytest.approx(0.95) and hxi == 1


def test_ii_missing_instrument_refs_not_used(fake_casmi):
    for miss in (None, float("nan"), "", "nan", "N/A"):
        refs = L.refs_with_meta(ref_rows([("K", 0.9, miss, "[M+H]+", "x"), ("K", 0.3, "timsTOF", "[M+H]+", "x")]))
        full, has, xi, hxi, sa = L.lib_scores_xinst([q("timsTOF")], ["K"], refs)["K"]
        assert full == pytest.approx(0.9) and has == 1
        assert (xi, hxi, sa) == (0.0, 0, 0.0)


def test_iii_missing_instrument_queries_not_used(fake_casmi):
    refs = L.refs_with_meta(ref_rows([("K", 0.5, "Orbitrap", "[M+H]+", "x")]))
    # 装置の欠けた問い合わせ（scale 2 → 1.0）は使わない、他の問い合わせ（timsTOF、scale 1 → 0.5）で計算
    full, has, xi, hxi, sa = L.lib_scores_xinst([q(None, scale=2.0), q("timsTOF", scale=1.0)], ["K"], refs)["K"]
    assert full == pytest.approx(1.0)
    assert xi == pytest.approx(0.5) and hxi == 1
    # 全部欠けた分子は 0
    full, has, xi, hxi, sa = L.lib_scores_xinst([q("nan"), q(float("nan"))], ["K"], refs)["K"]
    assert (xi, hxi, sa) == (0.0, 0, 0.0) and has == 1


def test_iv_no_refs_zero(fake_casmi):
    refs = L.refs_with_meta(ref_rows([("K", 0.5, "Orbitrap", "[M+H]+", "x")]))
    assert L.lib_scores_xinst([q("timsTOF")], ["Z"], refs)["Z"] == (0.0, 0, 0.0, 0, 0.0)
    # 問い合わせが無い分子: full 0（今の dump と同じ）、has_ref は参照の有無、xi は 0
    assert L.lib_scores_xinst([], ["K"], refs)["K"] == (0.0, 1, 0.0, 0, 0.0)


def test_v_hide_rule_applied_before_family(fake_casmi):
    rows = [("K", 0.9, "Orbitrap", "[M+H]+", "enveda-180"), ("K", 0.4, "QTOF", "[M+H]+", "gnps"),
            ("D", 0.8, "Orbitrap", "[M+H]+", "gnps")]
    t = ref_rows(rows)
    # enveda_libs: 隠す構造 K の enveda 系の行だけを除く → K の xi は 0.4
    kept = L.hide_rows(t, dict(rule="enveda_libs"), {"K"})
    r = L.lib_scores_xinst([q("timsTOF")], ["K", "D"], L.refs_with_meta(kept))
    assert r["K"][2] == pytest.approx(0.4) and r["D"][2] == pytest.approx(0.8)
    # all: K の行はすべて除く → K は参照なし
    kept = L.hide_rows(t, dict(rule="all"), {"K"})
    r = L.lib_scores_xinst([q("timsTOF")], ["K"], L.refs_with_meta(kept))
    assert r["K"] == (0.0, 0, 0.0, 0, 0.0)
    with pytest.raises(ValueError):
        L.hide_rows(t, dict(rule="xyz"), {"K"})


def test_same_adduct_column(fake_casmi):
    refs = L.refs_with_meta(ref_rows([("K", 0.9, "Orbitrap", "[M+Na]+", "x"), ("K", 0.5, "Orbitrap", "[M+H]+", "x"),
                                      ("K", 0.99, "Orbitrap", None, "x")]))
    full, has, xi, hxi, sa = L.lib_scores_xinst([q("timsTOF", "[M+H]+")], ["K"], refs)["K"]
    assert xi == pytest.approx(0.99) and sa == pytest.approx(0.5)
    # 同じ付加イオンの組が無ければ sa は 0、xi はある
    full, has, xi, hxi, sa = L.lib_scores_xinst([q("timsTOF", "[M-H]-")], ["K"], refs)["K"]
    assert hxi == 1 and sa == 0.0


def test_full_equals_lib_scores(fake_casmi):
    rows = [("A", 0.3, "timsTOF", "[M+H]+", "x"), ("A", 0.7, None, "[M+H]+", "x"), ("B", 0.2, "QTOF", "[M+H]+", "x")]
    t = ref_rows(rows)
    qs = [q("timsTOF", scale=1.0), q("timsTOF", scale=0.5)]
    old = L.lib_scores(qs, ["A", "B", "C"], {k: [p for p, _f, _a in v] for k, v in L.refs_with_meta(t).items()})
    new = L.lib_scores_xinst(qs, ["A", "B", "C"], L.refs_with_meta(t))
    for k in ("A", "B", "C"):
        assert new[k][0] == old.get(k, 0.0)
        assert new[k][1] == int(k in old)


def test_read_rows_default_cols_unchanged():
    assert L.REF_COLS == ["inchikey14", "ingest_lib", "ms2_mzs", "ms2_normalized_intensities"]
    assert L.REF_META_COLS == L.REF_COLS + ["instrument_type", "adduct"]
    import inspect
    assert inspect.signature(L._read_rows).parameters["cols"].default is None


def test_add_h2_replace_columns():
    import add_h2 as A
    import relativize as RZ
    df = pd.DataFrame(dict(set=["s1"] * 3 + ["s2"] * 2, molecule_id=["m1", "m1", "m2", "m3", "m3"],
                           cand_key=list("abcde"), lib_max_sim=[0.9, 0.1, 0.0, 0.5, 0.4], has_ref=[1, 1, 0, 1, 1],
                           fz=[1.0, 2.0, 3.0, 4.0, 5.0]))
    df = RZ.add_relative(df, ["set", "molecule_id"], ["lib_max_sim"])
    assert A.same_rule_check(df)
    parts = {"s1": pd.DataFrame(dict(set=["s1"] * 3, molecule_id=["m1", "m1", "m2"], cand_key=list("abc"),
                                     lib_max_sim_xi=[0.2, 0.0, 0.0], has_ref_xi=np.array([1, 0, 0], np.int8),
                                     lib_max_sim_xi_sa=[0.1, 0.0, 0.0])),
             "s2": pd.DataFrame(dict(set=["s2"] * 2, molecule_id=["m3", "m3"], cand_key=list("de"),
                                     lib_max_sim_xi=[0.3, 0.4], has_ref_xi=np.array([1, 1], np.int8),
                                     lib_max_sim_xi_sa=[0.0, 0.4]))}
    out, chk = A.replace_columns(df, parts)
    assert not any(c in out.columns for c in A.DROP)
    assert all(c in out.columns for c in A.NEW + A.NEW_REL)
    assert out[["set", "molecule_id", "cand_key", "fz"]].equals(df[["set", "molecule_id", "cand_key", "fz"]])
    exp = RZ.relativize([0.2, 0.0])
    assert np.allclose(out.loc[:1, "lib_max_sim_xi_z"], exp["z"])
    assert out.loc[2, "lib_max_sim_xi_softmax"] == 1.0
    # 順の違う part は止める
    bad = dict(parts, s2=parts["s2"].iloc[::-1].reset_index(drop=True))
    with pytest.raises(ValueError):
        A.replace_columns(df, bad)
    # has_ref_xi = 0 の行に値があれば止める
    bad = dict(parts, s1=parts["s1"].assign(lib_max_sim_xi=[0.2, 0.5, 0.0]))
    with pytest.raises(ValueError):
        A.replace_columns(df, bad)
    with pytest.raises(ValueError):   # NaN
        A.replace_columns(df, dict(parts, s1=parts["s1"].assign(lib_max_sim_xi_sa=[np.nan, 0.0, 0.0])))


def test_harness_h2_registered_and_exclude_absent():
    import lambdarank_oof as LR
    h2 = LR.EXTRA_ALLOWED["h2"][1]
    assert len(h2) == 8 and "lib_max_sim_xi_sa" in h2
    drop = ("lib_max_sim", "has_ref") + tuple(f"lib_max_sim_{r}" for r in LR._REL)
    cols = [c for c in list(LR.ID_COLS) + list(LR.FEATURES) if c not in drop] + list(h2)
    with pytest.raises(LR.HarnessError):            # 既定（exclude なし）では外した列の欠けで止める（今と同じ）
        LR.check_columns(cols, h2)
    rep = LR.check_columns(cols, h2, drop)
    assert rep["missing"] == [] and rep["extra"] == []
    # 外した列が表にあっても止めない
    LR.check_columns(list(LR.ID_COLS) + list(LR.FEATURES) + list(h2), h2, drop)
    with pytest.raises(LR.HarnessError):
        LR.check_columns(cols, h2, ("not_a_feature",))


def test_run_h2_features_are_version_a_minus_7_plus_8():
    import lambdarank_oof as LR
    sh = (ROOT / "scripts/ranker/run_h2.sh").read_text()
    va = (ROOT / "scripts/ranker/run_version_a_combo.sh").read_text()
    var = lambda txt, n: re.search(rf'^{n}="([^"]*)"', txt, re.M).group(1)
    gl, n1, h2, drop = var(sh, "GL"), var(sh, "N1"), var(sh, "H2"), var(sh, "H2_DROP")
    assert gl == var(va, "GL") and n1 == var(va, "N1")
    base = LR.resolve_features("prior_unverified_pop", f"cc,cc_z,{gl},{n1}")
    new = LR.resolve_features(f"prior_unverified_pop,{drop}", f"cc,cc_z,{gl},{n1},{h2}")
    assert len(base) == 81 and len(new) == 82
    assert set(base) - set(new) == set(drop.split(",")) and set(new) - set(base) == set(h2.split(","))
    assert tuple(h2.split(",")) == LR.EXTRA_ALLOWED["h2"][1]


def test_dump_features_h2_constants():
    import dump_features as DF
    assert DF.CHANNELS == ["fz", "tani_pred", "lib_max_sim", "metfrag_lite", "np_score"]
    assert DF.CHANNELS_H2 == ("fz", "tani_pred", "lib_max_sim_xi", "metfrag_lite", "np_score")
    import inspect
    ps = inspect.signature(DF.build).parameters
    assert ps["h2_refs"].default is None and ps["h2_inst"].default is None


def test_h2_aux_set_truth_xi_and_lost_signal():
    import h2_aux as H
    import relativize as RZ
    sub = pd.DataFrame(dict(set=["s"] * 3, molecule_id=["m", "m", "n"], type=["a"] * 3,
                            lib_max_sim_xi=[0.1, 0.5, 0.0], has_ref_xi=[1, 1, 0], lib_max_sim_xi_sa=[0.0, 0.0, 0.0]))
    sub = RZ.add_relative(sub, ["set", "molecule_id"], ["lib_max_sim_xi"])
    ys = np.array([1, 0, 1])
    cf, n = H.set_truth_xi(sub, ys)
    assert n == 2 and list(cf.lib_max_sim_xi) == [0.816, 0.5, 0.816]
    exp = RZ.relativize([0.816, 0.5])
    assert np.allclose(cf.loc[:1, "lib_max_sim_xi_dmax"], exp["dmax"])
    assert list(cf.has_ref_xi) == [1, 1, 0] and list(sub.lib_max_sim_xi) == [0.1, 0.5, 0.0]   # 元は変えない
    ls = H.lost_signal(sub, np.array([0, 0, 1]), np.array([0.3, 0.5, 0.9]), np.array([1, 0, 1]))
    assert ls["a"]["n_decoy"] == 2 and ls["a"]["frac_all"] == 0.5 and ls["a"]["frac_has_ref"] == 1.0
    q = H.quantiles(sub, np.array([1, 0, 0]))
    assert q["a"]["truth"]["n_rows"] == 1 and q["a"]["no_truth_range"] == dict(lo_exclusive=0.1, hi=0.5, n_decoy_rows=1)
