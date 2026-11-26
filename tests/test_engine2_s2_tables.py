"""#165（#158 (B)）: 自前 S2 で Engine 2 の rank_train の FP 12 列を作る経路の単体テスト。

(i)  merged が空のとき pv_fp.molecule_logits が single の各スペクトルの平均だけを返す（torch が要る。CI では skip）
(ii) FP 以外の 19 列が #74 の blk_base_fs と同じ入力で一致（rows_for_molecule の s2 経路 = E.blk_base(dict(rec, zlog=z_s2), cfp)）
(iii) クエリの条件（S2 の VAL、別名の漏れ、除外集合）の assert が合成データで漏れを検出する
ほか: build(fp_source='s2') を端から端まで（S2 のロジットは差し替え）、恒等式全式 1.0、sim 表は foldsafe と一致
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_engine2_build_tables as T  # noqa: E402  （合成のライブラリ・プールを使い回す）

BT, E, base_fs, SCH = T.BT, T.E, T.base_fs, T.SCH
NON_FP = [c for c in range(31) if c not in base_fs.FP_COLS]


# ------------------------------------------------------------------------------------------------
# (i) merged が空なら single の各スペクトルの平均だけ
# ------------------------------------------------------------------------------------------------
def test_i_molecule_logits_single_only_is_mean_of_spectra(monkeypatch):
    torch = pytest.importorskip("torch")
    import pv_fp
    torch.manual_seed(0)
    net = pv_fp.FPNet(24, d=32, layers=1, heads=4, drop=0.0).eval()
    rng = np.random.default_rng(0)
    specs, prec = [], []
    for n in (5, 9, 3):
        mz = np.sort(rng.random(n) * 300 + 50).astype(np.float32)
        it = (rng.random(n) + 0.1).astype(np.float32)
        specs.append((mz, it / it.max())); prec.append(np.float32(400.0 + n))
    prec = np.array(prec, np.float32)
    adducts = ["[M+H]+", "[M+Na]+", "[M-H]-"]; instr = ["Orbitrap", "QTOF", "timsTOF"]
    ces = np.array([20.0, 35.0, 25.0], np.float32); modes = np.array([1.0, 1.0, -1.0])

    def boom(*a, **k):
        raise AssertionError("merged branch must not run when merged is empty")
    monkeypatch.setattr(pv_fp, "merge_peaks", boom)
    z = pv_fp.molecule_logits(([net], [], "cpu"), specs, prec, adducts, instr, ces, modes)
    # 参照: スペクトルを 1 本ずつ（パディング無し）通して平均
    per = []
    for j in range(3):
        P = [pv_fp.prep_peaks(specs[j][0], specs[j][1], prec[j])]
        per.append(pv_fp.logits_batch(P, [net], "cpu", prec[j:j + 1], adducts[j:j + 1], instr[j:j + 1], ces[j:j + 1],
                                      modes[j:j + 1]))
    ref = np.mean(per, axis=0)
    assert z.shape == (24,)
    np.testing.assert_allclose(z, ref, rtol=1e-5, atol=1e-6)
    # 同じバッチでの計算（logits_batch に 3 本まとめて）ともバイト一致: molecule_logits はそれ以外を足さない
    P = [pv_fp.prep_peaks(a, b, p) for (a, b), p in zip(specs, prec)]
    assert np.array_equal(z, pv_fp.logits_batch(P, [net], "cpu", prec, adducts, instr, ces, modes))


def test_i_s2_logits_builds_engine_inputs(monkeypatch):
    """build_tables.s2_logits が推論（compute_channels）と同じ入力を pv_fp.molecule_logits に渡す。"""
    pytest.importorskip("torch")
    import pv_fp
    P, L, add, I, rng = T._setup()
    rows = np.flatnonzero(L["row_p"] == 7)[:3]
    meta = dict(mode=np.where(L["pol"] == 1, "positive", "negative").astype(object),
                ce=np.array([[10.0, 30.0], None, "x"] + [25.0] * (len(L["prec"]) - 3), dtype=object),
                instr=np.array(["Orbitrap"] * len(L["prec"]), dtype=object))
    seen = {}

    def fake(models, specs, prec, adduct, instr, ce, mode):
        seen.update(models=models, specs=specs, prec=prec, adduct=adduct, instr=instr, ce=ce, mode=mode)
        return np.ones(6930)
    monkeypatch.setattr(pv_fp, "molecule_logits", fake)
    z = BT.s2_logits(("S", [], "cpu"), L, meta, rows, add[rows])
    assert z.dtype == np.float32 and z.shape == (6930,)
    assert seen["models"] == ("S", [], "cpu")
    ref_specs = BT.query_specs(L, rows)
    assert all(np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(seen["specs"], ref_specs))
    assert seen["prec"].dtype == np.float32 and np.array_equal(seen["prec"], L["prec"][rows].astype(np.float32))
    assert seen["adduct"] == list(add[rows].astype(str)) and seen["instr"] == ["Orbitrap"] * len(rows)
    assert np.array_equal(seen["ce"], np.array([E.parse_ce(v) for v in meta["ce"][rows]], np.float32))
    assert np.array_equal(seen["mode"], np.where(L["pol"][rows] == 1, 1.0, -1.0))


# ------------------------------------------------------------------------------------------------
# (ii) FP 以外の 19 列が #74 の blk_base_fs と一致、s2 経路は E.blk_base(dict(rec, zlog=z_s2), cfp) そのもの
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("zero_lib", [False, True])
def test_ii_s2_rows_non_fp_equal_blk_base_fs(zero_lib):
    P, L, add, I, rng = T._setup()
    rows, modes, target = T._query(L, add, 25)
    cand = E.pool_window(target, E.CFG.PPM_WIN)
    assert len(cand) > 3
    rec = BT.make_rec(E, L, I, rows, add[rows], modes, target, cand, rows, 25, None, None)
    T._frag(rec, np.random.default_rng(1))
    if zero_lib:
        rec["lib"] = np.zeros(len(cand), np.float32)
    cfp = E.pool_fps(cand)
    cfs = (np.random.default_rng(2).random((len(cand), 64)) < 0.3).astype(np.uint8)
    z_fs = np.random.default_rng(3).normal(size=64).astype(np.float32)
    z_s2 = np.random.default_rng(4).normal(size=6930).astype(np.float32)
    Xs_fs, Xr_fs = BT.rows_for_molecule(E, rec, cfp, cfs, z_fs, "foldsafe")
    Xs_s2, Xr_s2 = BT.rows_for_molecule(E, rec, cfp, None, z_s2, "s2")
    assert np.array_equal(Xs_fs, Xs_s2)                                   # sim 表は FP を使わない
    assert len(NON_FP) == 19 and np.array_equal(Xr_fs[:, NON_FP], Xr_s2[:, NON_FP])
    assert np.array_equal(Xr_s2, E.blk_base(dict(rec, zlog=z_s2), cfp).astype(np.float32))
    assert not np.all(Xr_s2[:, base_fs.FP_COLS] == 0)
    # z が None（ロジット無し）なら FP 列は 0
    _, Xr_none = BT.rows_for_molecule(E, rec, cfp, None, None, "s2")
    assert np.all(Xr_none[:, base_fs.FP_COLS] == 0) and np.array_equal(Xr_none[:, NON_FP], Xr_fs[:, NON_FP])
    with pytest.raises(AssertionError):
        BT.rows_for_molecule(E, rec, cfp, cfs, z_fs, "other")


# ------------------------------------------------------------------------------------------------
# (iii) クエリの条件: S2 の VAL・別名の漏れ・除外集合
# ------------------------------------------------------------------------------------------------
def _world_iii():
    keys = [f"Q{i:013d}" for i in range(80)]
    sk = list(keys)
    # 別名: keys[2] と keys[3] が同じ scorer_key（keys[3] は VAL の外 → keys[2] は漏れ）、
    #       keys[5] と keys[6] が同じ scorer_key（両方 VAL → 漏れではない）、
    #       keys[8] と keys[79]（keys[79] は train に出ない → 数えない）
    sk[3] = sk[2]; sk[6] = sk[5]; sk[79] = sk[8]
    fold = pd.DataFrame(dict(scorer_key=sk, fold_identity=[1 + i % 4 for i in range(80)]), index=pd.Index(keys, name="inchikey14"))
    fold.loc[keys[9], "fold_identity"] = 0
    val = set(keys[:40]) - {keys[3]}
    excl = {keys[11], fold.loc[keys[12], "scorer_key"]}
    lib_ik = np.array(keys[:79], dtype=object)                # keys[79] は train に無い
    lib_set = np.array(["gnps"] * 79, dtype=object)
    return keys, fold, val, excl, lib_set, lib_ik


def test_iii_select_and_assert_s2_conditions():
    keys, fold, val, excl, lib_set, lib_ik = _world_iii()
    universe = set(lib_ik)
    assert BT.alias_leak_keys([keys[2], keys[5], keys[6], keys[8]], val, fold, universe) == {keys[2]}
    q, info = BT.select_queries(lib_set, lib_ik, excl, fold, 2000, s2_val=val)
    ch = set(q["gnps"])
    assert ch <= val                                           # クエリ ⊂ VAL
    assert keys[2] not in ch and keys[3] not in ch             # 漏れの別名と VAL の外
    assert {keys[5], keys[6], keys[8]} <= ch                   # 漏れでない別名は残る
    assert keys[9] not in ch and keys[11] not in ch and keys[12] not in ch   # fold 0・ik の除外・scorer_key の除外
    assert info["gnps"]["alias_leak_removed"] == 1 and info["gnps"]["eligible"] == len(ch)
    rep = BT.assert_s2_queries(q, excl, fold, val, universe)
    assert rep["alias_leak"] == 0 and rep["outside_val"] == 0 and rep["unique_queries"] == len(ch)
    # 漏れを入れると assert で止まる（3 種類それぞれ）
    for bad in (keys[2], keys[45], keys[11], keys[12]):          # 別名の漏れ / VAL の外 / ik の除外 / scorer_key の除外
        with pytest.raises(AssertionError):
            BT.assert_s2_queries({"gnps": sorted(ch) + [bad]}, excl, fold, val, universe)
    # 除外集合に入る ik を VAL に入れても（VAL の条件は通る）除外の assert で止まる
    with pytest.raises(AssertionError):
        BT.assert_s2_queries({"gnps": [keys[11]]}, excl, fold, val | {keys[11]}, universe)
    # s2_val=None なら #74 と同じ選び方（VAL の条件なし）
    q0, info0 = BT.select_queries(lib_set, lib_ik, excl, fold, 2000)
    assert "alias_leak_removed" not in info0["gnps"] and set(q0["gnps"]) > ch


def test_iii_check_s2_bits(tmp_path):
    a = np.arange(0, 20, 2, dtype=np.int64); b = a.astype(np.int32)
    np.save(tmp_path / "s2.npy", a); np.save(tmp_path / "coco.npy", b)
    r = BT.check_s2_bits(str(tmp_path / "s2.npy"), str(tmp_path / "coco.npy"), 10)
    assert r["equal"] and r["n_bits"] == 10
    c = a.copy(); c[3] += 1; np.save(tmp_path / "bad.npy", c)
    with pytest.raises(AssertionError):
        BT.check_s2_bits(str(tmp_path / "bad.npy"), str(tmp_path / "coco.npy"), 10)
    with pytest.raises(AssertionError):
        BT.check_s2_bits(str(tmp_path / "s2.npy"), str(tmp_path / "coco.npy"), 11)


def test_iii_s2_val_from_split(tmp_path):
    import pickle
    p = tmp_path / "split.pkl"
    pickle.dump(dict(VAL=["A", "B"], forced=["A"]), open(p, "wb"))
    assert BT.s2_val_from_split(str(p)) == {"A", "B"}


# ------------------------------------------------------------------------------------------------
# build(fp_source='s2') を端から端まで（S2 のロジットは差し替え）
# ------------------------------------------------------------------------------------------------
def test_build_s2_end_to_end(tmp_path, monkeypatch):
    L, meta, I, queries, fold, calls = T._synthetic_build_inputs(monkeypatch)
    meta = dict(meta, instr=np.array(["Orbitrap"] * len(L["prec"]), dtype=object))
    banks = {1: "bank1", 2: "bank2", 3: "bank3", 4: "bank4"}
    sim_fs, rt_fs, rep_fs = BT.build(E, L, meta, I, queries, fold, banks, np.arange(64), 1, log=lambda *a: None)
    zr = np.random.default_rng(11); s2calls = []

    def fake_s2(models, L_, meta_, rows, add):
        s2calls.append((models, tuple(rows.tolist())))
        return zr.normal(size=6930).astype(np.float32)
    monkeypatch.setattr(BT, "s2_logits", fake_s2)
    n_fs_calls = len(calls)
    zl = []
    sim, rt, rep = BT.build(E, L, meta, I, queries, fold, None, None, 1, log=lambda *a: None, fp_source="s2",
                            s2_models=("S2", [], "cpu"), zout=zl)
    assert len(calls) == n_fs_calls                             # fold-safe は呼ばない
    assert len(s2calls) == rep["molecules"] == 5 and all(m == ("S2", [], "cpu") for m, _ in s2calls)
    assert rep["fp_source"] == "s2" and rep["fs_fp_failed"] is None
    assert all(np.array_equal(sim[k], sim_fs[k]) for k in sim)  # sim 表は経路によらない
    for k in ("Y", "M", "G"):
        assert np.array_equal(rt[k], rt_fs[k])
    assert np.array_equal(rt["X"][:, NON_FP], rt_fs["X"][:, NON_FP])
    assert not np.all(rt["X"][:, base_fs.FP_COLS] == 0)
    ps, pr = BT.save_tables(str(tmp_path), sim, rt)
    for path, blocks in ((ps, None), (pr, ["base"])):
        d = SCH.describe(path, blocks)
        fr = {k: v.get("frac") for k, v in d["identities"].items() if "frac" in v}
        assert fr and all(v == 1.0 for v in fr.values()), fr
    assert len(zl) == 5 and all(z.shape == (6930,) for *_, z in zl)
    with pytest.raises(AssertionError):
        BT.build(E, L, meta, I, queries, fold, None, None, 1, log=lambda *a: None, fp_source="s2")
