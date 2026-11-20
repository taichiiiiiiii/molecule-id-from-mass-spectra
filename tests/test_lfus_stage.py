"""Issue #150: lfus 段（ライブラリの類縁体から点を伝える）の合成データでのテスト。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/cv"))
import lfus as LF  # noqa: E402
import lfus_v3_check as CK  # noqa: E402
import run as R  # noqa: E402

FPS = {"a": [1, 0, 0, 0], "b": [0, 1, 0, 0], "c": [1, 1, 0, 0], "e": [0, 0, 0, 0], "bad": None}


def fpc(smi):
    v = FPS[smi]
    return None if v is None else np.asarray(v, np.float32)


# ---------------------------------------------------------------- Tanimoto と N(c)
def test_tanimoto_matrix_values_and_zero_denominator():
    T = LF.tanimoto_matrix(np.array([[1, 1, 0, 0], [0, 0, 0, 0]]), np.array([[1, 0, 0, 0], [0, 0, 0, 0]]))
    assert T[0, 0] == pytest.approx(0.5)
    assert T[0, 1] == 0 and T[1, 0] == 0 and T[1, 1] == 0   # 分母 0 は 0


def test_neighbour_score_weighted_mean():
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    A = np.array([[1, 0, 0, 0], [1, 1, 0, 0]])
    s = np.array([3.0, 1.0])
    N = LF.neighbour_score(C, A, s)
    # c0: T = [1, 0.5] → (3·1 + 1·0.5)/4 ; c1: T = [0, 0.5] → 0.5/4
    assert N == pytest.approx([3.5 / 4, 0.5 / 4])


def test_neighbour_score_none_without_analogs():
    assert LF.neighbour_score(np.ones((2, 4)), np.zeros((0, 4)), np.array([])) is None
    assert LF.neighbour_score(np.ones((2, 4)), np.ones((1, 4)), np.array([0.0])) is None


# ---------------------------------------------------------------- z の和
def test_lfus_mix_equal_weight_z_sum():
    rows = [(3.0, 10), (2.0, 11), (1.0, 12)]
    N = np.array([0.0, 0.0, 1.0])
    out = LF.lfus_mix(rows, N)
    zs = (np.array([3.0, 2, 1]) - 2) / (np.std([3.0, 2, 1]) + 1e-9)
    zn = (N - N.mean()) / (N.std() + 1e-9)
    f = zs + zn
    exp = np.argsort(-f, kind="stable")
    assert [r[1] for r in out] == [rows[i][1] for i in exp]
    assert [r[0] for r in out] == pytest.approx(list(f[exp]))


def test_lfus_mix_std_zero_keeps_rows():
    rows = [(3.0, 10), (2.0, 11)]
    assert LF.lfus_mix(rows, np.array([0.4, 0.4])) is rows
    assert LF.lfus_mix(rows, None) is rows
    assert LF.lfus_mix([(1.0, 1)], np.array([1.0])) == [(1.0, 1)]


def test_lfus_mix_stable_ties():
    rows = [(1.0, 1), (1.0, 2), (0.0, 3), (0.0, 4)]
    out = LF.lfus_mix(rows, np.array([0.0, 0.0, 1.0, 1.0]))
    assert [r[1] for r in out] == [1, 2, 3, 4]   # z の和が全部同じ → 元の順


def test_lfus_rows_no_analogs_and_fp_fail():
    rows = [(2.0, 0), (1.0, 1)]
    out, info = LF.lfus_rows(rows, ["a", "b"], [], {}, fpc)
    assert out is rows and not info["applied"]
    out, info = LF.lfus_rows(rows, ["a", "b"], None, {}, fpc)
    assert out is rows and not info["applied"]
    smiles = {7: "bad", 8: "b"}
    out, info = LF.lfus_rows(rows, ["a", "b"], [(7, 0.9), (8, 0.5)], smiles, fpc)
    assert info == dict(n_analog=1, n_fp_fail=1, applied=True)
    # 類縁体は b だけ → N = [0, 1]、z の和は 0 と 0 → 同点で元の順
    assert [r[1] for r in out] == [0, 1]
    out, info = LF.lfus_rows([(1.0, 0), (1.0, 1)], ["a", "b"], [(8, 0.5)], smiles, fpc)
    assert info["applied"] and [r[1] for r in out] == [1, 0]   # prior が同点なら類縁体に近い b が先


def test_lfus_rows_all_fp_fail_keeps_rows():
    rows = [(2.0, 0), (1.0, 1)]
    out, info = LF.lfus_rows(rows, ["a", "b"], [(7, 0.9)], {7: "bad"}, fpc)
    assert out is rows and info == dict(n_analog=0, n_fp_fail=1, applied=False)


# ---------------------------------------------------------------- 段の組み合わせ
def test_check_stages_lfus():
    R.check_stages(["score", "prior", "lfus", "frag"])
    R.check_stages(["score", "prior", "frag"])
    for bad in (["score", "nbz", "prior", "lfus", "frag"], ["score", "pmi", "prior", "lfus"],
                ["score", "fragnn", "prior", "lfus"], ["score", "prior", "lfus", "lib"], ["score", "prior", "lfus", "elem"]):
        with pytest.raises(AssertionError):
            R.check_stages(bad)
    with pytest.raises(AssertionError):
        R.check_stages(["score", "lfus", "frag"])   # prior が要る


def test_run_order_lfus_after_prior_before_frag():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    loop = src[src.index("    for mid in truth:\n        if mid not in targets:"):]
    i_prior = loop.index("rows = rerank_with_prior(rows, pool)")
    i_lfus = loop.index("LF.lfus_rows(")
    i_frag = loop.index("rows = frag_rerank(")
    assert i_prior < i_lfus < i_frag


# ---------------------------------------------------------------- 隠す印と混入の検査
def _lib():
    # スペクトル 6 行。sid: 0=A（raw AAA, key KA）, 1=B（raw BBB, key KB, rawkey RB）, 2=C, -1=構造なし
    tr_ik14 = np.array(["AAA", "AAA", "BBX", "CCC", "CCC", "ZZZ"], object)
    tr_lib = np.array(["enveda-180", "gnps", "enveda-np-examples", "gnps", "enveda-180", "gnps"], object)
    sid = np.array([0, 0, 1, 2, 2, -1])
    st_ik14 = np.array(["AAA", "BBX", "CCC"], object)
    st_key = np.array(["KA", "KB", "KC"], object)
    st_rawkey = np.array(["AAA", "RB", "CCC"], object)
    return tr_ik14, tr_lib, sid, st_ik14, st_key, st_rawkey


def test_hide_mask_all_rule_with_key_extension():
    tr_ik14, tr_lib, sid, st_ik14, st_key, st_rawkey = _lib()
    m, st = LF.hide_mask(tr_ik14, tr_lib, sid, st_key, st_rawkey, {"AAA", "KB"}, dict(rule="all"))
    # AAA（raw）の 2 行、KB（正準キー）で sid 1 の行、sid < 0 の行
    assert m.tolist() == [True, True, True, False, False, True]
    assert st == dict(n_hidden_loadrefs_rule=2, n_hidden_extra_key=1, n_sid_negative=1, n_hidden_total=4)


def test_hide_mask_enveda_rule_only_enveda_rows():
    tr_ik14, tr_lib, sid, st_ik14, st_key, st_rawkey = _lib()
    m, _ = LF.hide_mask(tr_ik14, tr_lib, sid, st_key, st_rawkey, {"AAA", "KC"}, dict(rule="enveda_libs"))
    # AAA の enveda 行（0）だけ、gnps（1）は残す。KC は enveda-180 の行 4 だけ。sid < 0 は隠す
    assert m.tolist() == [True, False, False, False, True, True]


def test_hide_mask_matches_load_refs_rule():
    """拡張なし（key/rawkey が drop に当たらない）なら load_refs と同じ判定。"""
    tr_ik14, tr_lib, sid, st_ik14, st_key, st_rawkey = _lib()
    drop = {"CCC"}
    for rule in (dict(rule="all"), dict(rule="enveda_libs")):
        m, st = LF.hide_mask(tr_ik14, tr_lib, sid, st_key, st_rawkey, drop, rule)
        t = pd.DataFrame(dict(inchikey14=tr_ik14, ingest_lib=tr_lib))
        hid = t.inchikey14.isin(drop)
        if rule["rule"] == "enveda_libs":
            hid &= t.ingest_lib.isin(LF.ENVEDA)
        assert (m[sid >= 0] == hid.values[sid >= 0]).all() and st["n_hidden_extra_key"] == 0


def test_leaked_rows_detects_alias_and_respects_rule():
    tr_ik14, tr_lib, sid, st_ik14, st_key, st_rawkey = _lib()
    kw = dict(tr_ik14=tr_ik14, tr_lib=tr_lib, sid=sid, st_ik14=st_ik14, st_key=st_key, st_rawkey=st_rawkey)
    assert LF.leaked_rows(np.arange(6), keys={"RB"}, rule=dict(rule="all"), **kw).tolist() == [2]   # rawkey 経由
    assert LF.leaked_rows(np.arange(6), keys={"KA"}, rule=dict(rule="all"), **kw).tolist() == [0, 1]
    assert LF.leaked_rows(np.arange(6), keys={"KA"}, rule=dict(rule="enveda_libs"), **kw).tolist() == [0]
    # 印を当てた後の代表には混入が無い
    m, _ = LF.hide_mask(tr_ik14, tr_lib, sid, st_key, st_rawkey, {"AAA", "KA", "BBX", "KB", "RB"}, dict(rule="all"))
    assert LF.leaked_rows(np.where(~m)[0], keys={"AAA", "KA", "BBX", "KB", "RB"}, rule=dict(rule="all"), **kw).size == 0


def test_source_keys_rebuilds_aliases():
    tr_ik14, tr_lib, sid, st_ik14, st_key, st_rawkey = _lib()
    keys = LF.source_keys([("BBX", "smi_b"), (None, "smi_c")], {"smi_b": "KB", "smi_c": "KC"}.get, st_ik14, st_key, st_rawkey)
    assert keys == {"BBX", "KB", "RB", "KC", "CCC"}


# ---------------------------------------------------------------- 段 0
def test_random_baseline_window_mode_seed():
    rep_sid = np.array([5, 1, 3, 3, 9, 7])
    rep_mass = np.array([100.0, 150.0, 200.0, 200.0, 250.0, 500.0])
    rep_mode = np.array([1, 1, 1, -1, -1, 1])
    a = LF.random_baseline(rep_sid, rep_mass, rep_mode, 200.0, {1}, 60.0, np.random.default_rng(0))
    assert sorted(a.tolist()) == [1, 3]
    b = LF.random_baseline(rep_sid, rep_mass, rep_mode, 200.0, {1, -1}, 200.0, np.random.default_rng(0), n=2)
    c = LF.random_baseline(rep_sid, rep_mass, rep_mode, 200.0, {1, -1}, 200.0, np.random.default_rng(0), n=2)
    assert b.tolist() == c.tolist() and len(b) == 2 and set(b) <= {1, 3, 5, 9}
    assert LF.random_baseline(rep_sid, rep_mass, rep_mode, 1000.0, {1}, 10.0, np.random.default_rng(0)).size == 0


def test_stage0_molecule_and_exclude_same():
    afp = {1: np.array([1, 0, 0, 0]), 2: np.array([1, 1, 0, 0]), 3: None, 4: np.array([0, 1, 0, 0])}.get
    tfp = np.array([1, 0, 0, 0])
    r = LF.stage0_molecule(tfp, [(1, 0.8), (3, 0.7), (2, 0.2)], afp, np.array([4, 2]))
    assert r["T1"] == 1.0 and r["Tmax10"] == 1.0
    assert r["Nw"] == pytest.approx((0.8 * 1 + 0.2 * 0.5) / 1.0)
    assert r["R1"] == 0.0 and r["Rmax10"] == 0.5 and r["Rmean"] == 0.25 and r["n_random"] == 2
    x = LF.stage0_molecule(tfp, [(1, 0.8), (2, 0.2)], afp, np.array([4]), exclude_same={1})
    assert x["T1"] == 0.5 and "R1" not in x
    e = LF.stage0_molecule(tfp, [], afp, np.array([]))
    assert e["T1"] is None and e["R1"] is None


# ---------------------------------------------------------------- 行の一致の確認
def test_check_alignment():
    lib_ix = {"gnps": 3, "enveda-180": 0}
    tr = dict(inchikey14=np.array(["A", "B", "Z"], object), ingest_lib=np.array(["gnps", "enveda-180", "gnps"], object),
              precursor_mz=np.array([1.0, 2.0, 3.0]), adduct=np.array(["[M+H]+", None, "[M-H]-"], object),
              num_peaks=np.array([5, 6, 7]))
    meta = pd.DataFrame(dict(sid=[0, 1, -1], lib=[3, 0, 3], precursor_mz=[1.0, 2.0, 3.0], adduct=["[M+H]+", None, "[M-H]-"],
                             num_peaks=[5, 6, 7]))
    r = LF.check_alignment(tr, meta, np.array(["A", "B"], object), lib_ix)
    assert r["all_ok"] and r["n_sid_negative"] == 1
    bad = meta.copy(); bad.loc[1, "sid"] = 0
    assert not LF.check_alignment(tr, bad, np.array(["A", "B"], object), lib_ix)["all_ok"]
    bad = meta.copy(); bad.loc[0, "lib"] = 0
    assert not LF.check_alignment(tr, bad, np.array(["A", "B"], object), lib_ix)["all_ok"]
    assert not LF.check_alignment(tr, meta.iloc[:2], np.array(["A", "B"], object), lib_ix)["all_ok"]


def test_lib_ix_from_build_reads_libs(tmp_path):
    p = tmp_path / "build.py"
    p.write_text("import os\nLIBS = ['x', 'y']\n")
    assert LF.lib_ix_from_build(p) == {"x": 0, "y": 1}


# ---------------------------------------------------------------- 判定
def _d(lo, diff=0.03):
    return dict(lo=lo, diff=diff)


def test_judge_rules():
    ok = {"v3_c1": _d(0.001), "v3_c2": _d(0.0, 0.0), "enveda_v1": _d(-0.019), "v3_c2n": _d(-0.01)}
    assert CK.judge(ok)["adopt"]
    assert CK.judge(ok)["adopted_below_planning_mde"] is False
    assert CK.judge(ok | {"v3_c1": _d(0.001, 0.01)})["adopted_below_planning_mde"] is True
    assert not CK.judge(ok | {"v3_c1": _d(0.0)})["adopt"]                    # C1 の下限 > 0 が必須
    assert not CK.judge(ok | {"v3_c2n": _d(-0.021)})["adopt"]
    miss = dict(ok); miss.pop("v3_c2")
    j = CK.judge(miss)
    assert not j["adopt"] and j["missing"] == ["v3_c2"]


def test_check_inputs_asserts():
    aj = dict(align=dict(all_ok=True), mask={}, audit={k: 0 for k in CK.LEAK_KEYS}, drop_keys_sha256="h", stage0={},
              timing={}, secs_total=1.0, limit=None)
    rj = dict(lfus_drop_keys_sha256="h", lfus_analogs_sha256="x")
    man = dict(after=dict(sha256="h"))
    CK.check_inputs("v3_c1", rj, aj, man, 0.85350)
    with pytest.raises(AssertionError):
        CK.check_inputs("v3_c1", rj, aj, man, 0.85)
    with pytest.raises(AssertionError):
        CK.check_inputs("v3_c1", rj, aj, dict(after=dict(sha256="other")), 0.8535)
    with pytest.raises(AssertionError):
        CK.check_inputs("v3_c1", rj, aj | dict(audit={**aj["audit"], "leak_analog_hits": 1}), man, 0.8535)
    with pytest.raises(AssertionError):
        CK.check_inputs("v3_c1", rj, aj | dict(limit=10), man, 0.8535)
    CK.check_inputs("v3_c1", rj, aj, man, 0.8535, analogs_sha256="x")
    with pytest.raises(AssertionError):
        CK.check_inputs("v3_c1", rj, aj, man, 0.8535, analogs_sha256="y")
