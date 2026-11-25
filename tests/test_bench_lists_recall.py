"""Issue #236: scripts/bench/make_lists.py・candidate_recall.py の純粋な関数のテスト（作り物の分子、実データは使わない）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/bench"))

import candidate_recall as CR  # noqa: E402
import make_lists as ML  # noqa: E402


# ---------------------------------------------------------------- make_lists: 4 鍵の重なり
def _dev():
    return pd.DataFrame(dict(inchikey14=["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"],
                             key=["AAAA", "BBBX", "CCCC", "DDDD", "EEEE"],
                             scorer_key=["AAAA", "BBBX", "CCCY", "DDDD", "EEEE"],
                             identity_group=["g1", "g2", "g3", "g4", "g5"], fold=[1, 2, 3, 4, 1]))


def _hold(**kw):
    base = dict(raw_inchikey14=set(), canonical_key=set(), scorer_key=set(), identity_group=set())
    base.update({k: set(v) for k, v in kw.items()})
    return base


def test_overlap4_no_overlap():
    ov = ML.overlap4(_dev(), _hold(raw_inchikey14=["ZZZZ"], canonical_key=["YYYY"], identity_group=["g9"]))
    assert ov["counts"] == dict(raw_inchikey14=0, canonical_key=0, scorer_key=0, identity_group=0)
    assert not ov["any"].any()


def test_overlap4_each_key_detected():
    d = _dev()
    # 生の鍵だけ
    assert ML.overlap4(d, _hold(raw_inchikey14=["AAAA"]))["counts"]["raw_inchikey14"] == 1
    # 正規化キー（dev の key）だけが holdout の正準キーに当たる
    ov = ML.overlap4(d, _hold(canonical_key=["BBBX"]))
    assert ov["counts"]["canonical_key"] == 1 and ov["counts"]["raw_inchikey14"] == 0 and ov["any"][1]
    # scorer_key だけ（互変異性体の別名）
    ov = ML.overlap4(d, _hold(scorer_key=["CCCY"]))
    assert ov["counts"]["scorer_key"] == 1 and ov["counts"]["canonical_key"] == 0 and ov["any"][2]
    # 群だけ（別の構造でも同じ identity_group）
    ov = ML.overlap4(d, _hold(identity_group=["g4"]))
    assert ov["counts"]["identity_group"] == 1 and ov["any"].sum() == 1 and ov["any"][3]
    # holdout の生の鍵が dev の正規化キーに当たる（鍵の種類をまたぐ）
    ov = ML.overlap4(d, _hold(raw_inchikey14=["BBBX"]))
    assert ov["any"][1]


def test_holdout_key_rows_adds_scorer_and_group():
    st = pd.DataFrame(dict(inchikey14=["AAAA", "QQQQ"], scorer_key=["AAAA", "AAAA"], identity_group=["g1", "g1"],
                           fold_identity=[0, 0]))
    truth = pd.DataFrame(dict(inchikey14=["AAAA"], smiles=["CCO"]))
    rows = ML.holdout_key_rows(truth, {"DROP"}, st, {"AAAA": "AAAA"}, lambda s: "KKKK")
    ks = ML.key_sets(rows)
    assert ks["raw_inchikey14"] == {"AAAA", "DROP"}
    assert "KKKK" in ks["canonical_key"] and ks["scorer_key"] == {"AAAA"} and ks["identity_group"] == {"g1"}


def test_mark_b8_maps_scorer_key_alias():
    dev = _dev().assign(sid=range(5))
    b8 = pd.DataFrame(dict(set=["b8_f1_a_t10", "b8_f3_d_t10"], type=["a", "d"], fold=[1, 3], molecule_id=["m1", "m2"],
                           inchikey14=["AAAA", "CCCY"]))   # 2 つめは scorer_key で当たる
    d = ML.mark_b8(dev, b8)
    assert d.used_b8.tolist() == [1, 0, 1, 0, 0]
    assert d.loc[2, "b8_truth_key"] == "CCCY" and d.loc[2, "b8_type"] == "d"
    with pytest.raises(AssertionError):
        ML.mark_b8(dev, b8.assign(fold=[2, 3]))   # fold の食い違い
    with pytest.raises(AssertionError):
        ML.mark_b8(dev, b8.assign(inchikey14=["AAAA", "NOPE"]))   # dev に無い


# ---------------------------------------------------------------- candidate_recall
def test_heavy_comp_str_basic():
    assert CR.heavy_comp_str("CCO") == (("C", 2), ("O", 1))
    assert CR.heavy_comp_str("c1ccccc1Cl") == (("C", 6), ("Cl", 1))
    assert CR.heavy_comp_str("[nH]1cccc1") == (("C", 4), ("N", 1))
    assert CR.heavy_comp_str("[2H]C([2H])Br") == (("Br", 1), ("C", 1))
    assert CR.heavy_comp_str("[Na+].[O-]C(=O)C") == (("C", 2), ("Na", 1), ("O", 2))
    assert CR.heavy_comp_str("[se]1cccc1") == (("C", 4), ("Se", 1))
    assert CR.heavy_comp_str("[Hg]C") == (("C", 1), ("Hg", 1))
    assert CR.heavy_comp_str("[H][H]") == ()


def test_heavy_comp_str_matches_rdkit():
    Chem = pytest.importorskip("rdkit.Chem")
    from collections import Counter
    for s in ["CC(=O)Nc1ccc(O)cc1", "O=C(O)[C@@H](N)Cc1c[nH]c2ccccc12", "C[N+](C)(C)CC(=O)[O-]", "Brc1ccc(Cl)cc1I",
              "OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1O", "C1CC2CCC1C2", "[Cu+2].[O-]S(=O)(=O)[O-]", "FC(F)(F)c1cccs1"]:
        m = Chem.MolFromSmiles(s, sanitize=False)
        r = tuple(sorted(Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() > 1).items()))
        assert CR.heavy_comp_str(s) == r, s


def test_tautomers_and_stereo_share_comp():
    # to_key が同じ（互変異性体・立体違い）なら文字列の組成も同じ
    assert CR.heavy_comp_str("Oc1ccccn1") == CR.heavy_comp_str("O=c1cccc[nH]1")
    assert CR.heavy_comp_str("C/C=C/C") == CR.heavy_comp_str("C/C=C\\C")
    assert CR.comp_hash(CR.heavy_comp_str("N[C@@H](C)C(=O)O")) == CR.comp_hash(CR.heavy_comp_str("NC(C)C(=O)O"))


def test_window_bounds_edges():
    mass = np.array([99.0, 99.9989, 99.9991, 100.0, 100.0009, 100.0011, 101.0])
    a, b = CR.window_bounds(mass, 100.0, 10.0)   # ±10 ppm = ±0.001 Da
    assert (a, b) == (2, 5)
    # 端ちょうどの値は両端とも含む（searchsorted の left / right）
    t = 100.0
    tol = t * 10.0 * 1e-6
    m2 = np.array([t - tol, t, t + tol])
    assert CR.window_bounds(m2, t, 10.0) == (0, 3)
    assert CR.window_bounds(mass, 50.0) == (0, 0)


def test_center_mass_rules():
    t10 = ("[M+H]+", "[M-H]-")
    assert CR.center_mass([100.0, 102.0, 200.0], ["[M+H]+", "[M-H]-", "[M+X]+"], t10) == 101.0
    assert CR.center_mass([np.nan, 150.0], ["[M+H]+", "[M+X]+"], t10) == 150.0   # 10 種が NaN → 全部の有限値
    assert np.isnan(CR.center_mass([np.nan], ["[M+H]+"], t10))


def test_wl_hash_invariant_to_tautomer_stereo_charge():
    pytest.importorskip("rdkit")
    assert CR.wl_hash("Oc1ccccn1") == CR.wl_hash("O=c1cccc[nH]1")             # 互変異性体
    assert CR.wl_hash("N[C@@H](C)C(=O)O") == CR.wl_hash("NC(C)C(=O)O")        # 立体
    assert CR.wl_hash("C[N+](C)(C)CC(=O)[O-]") == CR.wl_hash("CN(C)(C)CC(=O)O")   # 電荷
    assert CR.wl_hash("Oc1ccccn1") != CR.wl_hash("Oc1cccnc1")                 # 位置異性体は（ふつう）違う
    assert CR.wl_hash("[Na+].[O-]C(=O)C") == 0 and CR.wl_hash("not a smiles((") == 0   # 金属・読めない → 0


def test_compatible_filter_keeps_truth():
    pytest.importorskip("rdkit")
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    import mrr25
    truth = "Oc1ccccn1"                      # 互変異性体 O=c1cccc[nH]1 と同じ to_key
    tk = mrr25.truth_keys(truth)
    th = {CR.comp_hash(CR.heavy_comp_str(truth))}
    tw = {CR.wl_hash(truth)}
    cands = ["c1ccncc1O", "Nc1ccccc1", "O=c1cccc[nH]1", "[Cu]O"]
    ch = np.array([CR.comp_hash(CR.heavy_comp_str(c)) for c in cands])
    cw = np.array([CR.wl_hash(c) for c in cands])
    m_wl = CR.compatible(ch, cw, th, tw)
    m_comp = CR.compatible(ch, None, th, None)
    assert m_comp.tolist() == [True, False, True, False]
    assert m_wl.tolist() == [False, False, True, False]          # 位置異性体は WL で落ち、正解の互変異性体は残る
    found = lambda m: any(mrr25.to_key(c) in tk for c, k in zip(cands, m) if k)
    assert found(m_wl) and found(m_comp)
    # 正解側の WL が 0（金属など）なら WL では絞らない
    assert CR.compatible(ch, cw, th, {0}).tolist() == m_comp.tolist()
    # 候補側の WL が 0 なら両立
    assert CR.compatible(ch, np.zeros(4, np.int64), th, tw).tolist() == m_comp.tolist()


def test_summarize_six_rows_and_total():
    rows = [dict(type=t, found_i=i % 2 == 0, found_ii=i % 3 == 0, n_i=10, n_ii=100, has_center=True)
            for i, t in enumerate(["a", "b", "c", "d", "e", "none", "none"])]
    s = CR.summarize(rows, 7)
    assert set(s) == set(CR.ROW_TYPES) and s["none"]["n"] == 2 and s["a"]["used_b8"] == 1 and s["none"]["used_b8"] == 0
    assert s["a"]["recall_iii"] == 1.0 and s["b"]["recall_iii"] == 0.0
    with pytest.raises(AssertionError):
        CR.summarize(rows, 8)


def test_guard_dev_only_stops_on_fold0():
    dev = _dev().assign(b8_set=[None, "b8_f2_b_t10", None, None, None])
    CR.guard_dev_only(dev, {"ZZZZ"})
    with pytest.raises(AssertionError):
        CR.guard_dev_only(dev, {"CCCY"})                          # fold-0 の構造（scorer_key）が入っている
    with pytest.raises(AssertionError):
        CR.guard_dev_only(dev.assign(fold=[0, 2, 3, 4, 1]), set())  # fold 0 の行
    with pytest.raises(AssertionError):
        CR.guard_dev_only(dev.assign(b8_set=[None, "v3_c1u_t10", None, None, None]), set())   # holdout の集合名


def test_no_holdout_paths_in_candidate_recall():
    src = (ROOT / "scripts/bench/candidate_recall.py").read_text()
    for name in ("v3_c1", "v3_c2", "enveda_v1", "outputs/holdouts", "holdout_eval"):
        assert name not in src, name


def test_recall_miss_center_bins():
    import recall_miss_center as RM
    d = np.array([0.0005, 0.01, -0.3, 1.0, np.nan])
    t = np.array([100.0, 100.0, 300.0, 300.0, 300.0])
    assert RM.bins(d, t) == dict(n_mass=4, le10ppm=1, gt10ppm_le0p5Da=2, gt0p5Da=1)
