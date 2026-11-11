"""scripts/engine2/build_tables.py と base_fs.py の単体テスト（#72、計画 項目 1 の (a)〜(g)）。

(a)〜(e) は合成の小さなライブラリと候補で動く（numba と casmi_engine が要る。RDKit・torch・実データは要らない）。
(f)(g) は fold-safe の実モデル・RDKit 2026.03.3・v2-pool の指紋が要るので、無ければ skip（ローカルで回して PR に貼る）。
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "engine2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "engine2", "eng"))
import base_fs  # noqa: E402
import build_tables as BT  # noqa: E402

E = base_fs.engine()
E.CFG.N_ANALOG = BT.N_ANALOG

_spec = importlib.util.spec_from_file_location(
    "engine2_table_schema", os.path.join(ROOT, "scripts", "research", "engine2_table_schema.py"))
SCH = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SCH)


# ------------------------------------------------------------------------------------------------
# 合成のライブラリ・プール
# ------------------------------------------------------------------------------------------------
def _peaks(rng, base, n):
    mz = np.sort(base[rng.choice(len(base), n, replace=False)] + rng.normal(0, 0.002, n)).astype(np.float32)
    it = rng.random(n).astype(np.float32) + 0.05
    return mz, it / it.max()


def make_world(seed=0, n_struct=40, rows_per=4):
    """構造 n_struct 個（質量は 3 つの塊に集まる）、構造ごとに rows_per 行のスペクトル（極性・timsTOF が混ざる）。"""
    rng = np.random.default_rng(seed)
    mass = np.sort(np.concatenate([300 + rng.random(n_struct // 2) * 0.004, 450 + rng.random(n_struct - n_struct // 2) * 0.004]))
    keys = np.array([f"K{i:03d}XXXXXXXXXX"[:14] for i in range(n_struct)], dtype=object)
    smiles = np.array([f"C{i}" for i in range(n_struct)], dtype=object)
    fpb = (rng.random((n_struct, 6930)) < 0.08).astype(np.uint8)
    P = dict(fp=np.packbits(fpb, axis=1), mass=mass, keys=keys, smiles=smiles)
    P["k2i"] = pd.Series(np.arange(n_struct), index=keys)
    frags = [np.sort(rng.random(30) * m) for m in mass]
    mzs, its, prec, nm, pol, tims, row_p, add = [], [], [], [], [], [], [], []
    for p in range(n_struct):
        for j in range(rows_per):
            n = int(rng.integers(5, 20))
            a, b = _peaks(rng, frags[p], n)
            sign = 1 if j % 3 != 2 else -1
            mzs.append(a); its.append(b)
            pr = mass[p] + (1.00728 if sign == 1 else -1.00728)
            prec.append(pr); nm.append(mass[p]); pol.append(sign); tims.append(j % 2 == 0); row_p.append(p)
            add.append("[M+H]+" if sign == 1 else "[M-H]-")
    off = np.zeros(len(mzs) + 1, np.int64); off[1:] = np.cumsum([len(a) for a in mzs])
    L = dict(off=off, mz=np.concatenate(mzs), it=np.concatenate(its), prec=np.array(prec), nm=np.array(nm),
             pol=np.array(pol, np.int8), tims=np.array(tims), row_p=np.array(row_p, np.int64))
    return P, L, np.array(add, dtype=object), rng


def _setup(seed=0):
    P, L, add, rng = make_world(seed)
    E.POOL = P
    I = E.build_index(L)
    return P, L, add, I, rng


def _query(L, add, p):
    rows = np.flatnonzero(L["row_p"] == p)[:2]          # その構造の 2 行をクエリにする（class-1 で隠す行）
    modes = np.where(L["pol"][rows] == 1, 1.0, -1.0)
    target = float(np.median(L["nm"][rows]))
    return rows, modes, target


def _same_channels(a, b):
    (ua, *va), ana_a, ap_a = a
    (ub, *vb), ana_b, ap_b = b
    assert np.array_equal(ua, ub)
    for x, y in zip(va, vb):
        assert np.array_equal(x, y)
    assert ana_a == ana_b and ap_a == ap_b


def _removed(L, rows):
    L2 = dict(L); L2["row_p"] = L["row_p"].copy(); L2["row_p"][rows] = -1
    return L2


# ------------------------------------------------------------------------------------------------
# (a) 隠した行が lib 検索とアナログ検索に出ない（class-1・class-2）。行を実際に消した索引と完全一致
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["c1", "c2"])
def test_a_hidden_rows_absent_and_equal_to_removed_library(mode):
    P, L, add, I, rng = _setup()
    for p in (3, 25):
        rows, modes, target = _query(L, add, p)
        truth_rows = np.flatnonzero(L["row_p"] == p)
        hide = rows if mode == "c1" else truth_rows
        specs = BT.query_specs(L, rows)
        dbg = {}
        got = BT.channels_hidden(E, L, I, specs, L["prec"][rows], modes, target, hide, p, truth_rows, mode, dbg)
        assert not set(dbg["crow"]) & set(hide.tolist())
        for k in ("reps_all", "reps_pos", "reps_neg"):
            assert not set(dbg.get(k, [])) & set(hide.tolist()), k
        if mode == "c2":
            assert p not in [q for q, _ in got[1]] and p not in [q for q, _ in got[2]]
            assert p not in got[0][0].tolist()
        else:
            assert p in got[0][0].tolist()                 # 他の行の参照は残る
        L2 = _removed(L, hide); I2 = E.build_index(L2)
        ref = E.molecule_channels(L2, I2, specs, L["prec"][rows], modes, target)
        if mode == "c2":
            ref = ((ref[0][0], *ref[0][1:]), ref[1], ref[2])
        _same_channels(got, ref)


def test_a_channels_pair_equals_two_hidden_calls():
    P, L, add, I, rng = _setup()
    for p in (3, 7, 25, 31):
        rows, modes, target = _query(L, add, p)
        truth_rows = np.flatnonzero(L["row_p"] == p)
        specs = BT.query_specs(L, rows)
        c1, c2 = BT.channels_pair(E, L, I, specs, L["prec"][rows], modes, target, rows, truth_rows, p, truth_rows)
        _same_channels(c1, BT.channels_hidden(E, L, I, specs, L["prec"][rows], modes, target, rows, p, truth_rows, "c1"))
        _same_channels(c2, BT.channels_hidden(E, L, I, specs, L["prec"][rows], modes, target, truth_rows, p, truth_rows, "c2"))


def test_a_mode_none_equals_engine():
    P, L, add, I, rng = _setup()
    rows, modes, target = _query(L, add, 7)
    specs = BT.query_specs(L, rows)
    got = BT.channels_hidden(E, L, I, specs, L["prec"][rows], modes, target, rows, 7, None, None)
    ref = E.molecule_channels(L, I, specs, L["prec"][rows], modes, target)
    _same_channels(got, ref)


# ------------------------------------------------------------------------------------------------
# (b) 1 分子分の行が、同じ入力で BLOCKS を直接呼んだ結果と完全一致（base は blk_base_fs）
# ------------------------------------------------------------------------------------------------
def _rec_from_engine(L, I, rows, add, modes, target, cand, rng):
    """compute_channels と同じ組み立て（E.molecule_channels を直接呼ぶ）。"""
    specs = BT.query_specs(L, rows)
    lib, ana, anapol = E.molecule_channels(L, I, specs, L["prec"][rows], modes, target)
    u, lmax, lmean, lsh, lcnt = lib
    pos = {int(x): i for i, x in enumerate(u)}
    ci = np.array([pos.get(int(c), -1) for c in cand], np.int64); has = ci >= 0
    rec = dict(target=target, cand=cand, zlog=None, mode=float(np.mean(modes)), modes=modes, adducts=list(add[rows]),
               ana=ana, anapol=anapol)
    for nm_, arr in (("lib", lmax), ("lib_mean", lmean), ("libsh", lsh), ("lib_cnt", lcnt)):
        v = np.zeros(len(cand), np.float32); v[has] = arr[ci[has]]; rec[nm_] = v
    return rec


def _frag(rec, rng):
    n = len(rec["cand"])
    for f in ("frag", "frag_ad", "frag_mean", "frag_pol"):
        rec[f] = rng.random(n).astype(np.float32)


def test_b_rows_equal_direct_blocks():
    P, L, add, I, rng = _setup()
    rows, modes, target = _query(L, add, 25)
    cand = E.pool_window(target, E.CFG.PPM_WIN)
    assert len(cand) > 3
    rec = BT.make_rec(E, L, I, rows, add[rows], modes, target, cand, rows, 25, None, None)
    ref = _rec_from_engine(L, I, rows, add, modes, target, cand, rng)
    _frag(rec, np.random.default_rng(1)); _frag(ref, np.random.default_rng(1))
    cfp = E.pool_fps(cand)
    cfs = (np.random.default_rng(2).random((len(cand), 64)) < 0.3).astype(np.uint8)
    z = np.random.default_rng(3).normal(size=64).astype(np.float32)
    Xs, Xr = BT.rows_for_molecule(E, rec, cfp, cfs, z)
    direct = np.hstack([E.BLOCKS[b](dict(ref, zlog=None), cfp) for b in BT.BLOCKS_SIM]).astype(np.float32)
    assert Xs.shape == (len(cand), 51) and np.array_equal(Xs, direct)
    assert Xr.shape == (len(cand), 31) and np.array_equal(Xr, base_fs.blk_base_fs(ref, cfp, cfs, z))
    assert np.array_equal(Xr[:, :15], direct[:, :15]) and np.array_equal(Xr[:, 21:25], direct[:, 21:25])


# ------------------------------------------------------------------------------------------------
# (c) cfp_fs = cfp、zlog_fs = zlog のとき blk_base_fs が pv.rank_features（blk_base）と完全一致
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("zero_lib", [False, True])
def test_c_blk_base_fs_equals_rank_features(zero_lib):
    P, L, add, I, rng = _setup()
    rows, modes, target = _query(L, add, 3)
    cand = E.pool_window(target, E.CFG.PPM_WIN)
    rec = _rec_from_engine(L, I, rows, add, modes, target, cand, rng)
    _frag(rec, rng)
    if zero_lib:
        rec["lib"] = np.zeros(len(cand), np.float32)
    cfp = E.pool_fps(cand)
    zlog = rng.normal(size=6930).astype(np.float32)
    ref = E.blk_base(dict(rec, zlog=zlog), cfp)
    got = base_fs.blk_base_fs(rec, cfp, cfp, zlog)
    assert got.dtype == ref.dtype and np.array_equal(got, ref)
    assert not np.all(got[:, base_fs.FP_COLS] == 0)
    # zlog_fs が None なら FP 列は 0（use_fp=False と同じ）
    assert np.array_equal(base_fs.blk_base_fs(rec, cfp, None, None), E.blk_base(dict(rec, zlog=None), cfp))


# ------------------------------------------------------------------------------------------------
# (d) 除外集合と重なるクエリを入れると assert で止まる。選び方は fold 1–4 だけ、除外は ik・key の両方
# ------------------------------------------------------------------------------------------------
def test_d_exclusion_assert_and_selection():
    keys = [f"Q{i:013d}" for i in range(60)]
    fold = pd.DataFrame(dict(scorer_key=[k if i % 7 else "S" + k[1:] for i, k in enumerate(keys)],
                             fold_identity=[i % 5 for i in range(60)]), index=keys)
    excl = {keys[1], "S" + keys[14][1:]}            # ik で除外 1、scorer_key で除外 1
    lib_set = np.array(["gnps"] * 60, dtype=object)
    q, info = BT.select_queries(lib_set, np.array(keys, dtype=object), excl, fold, 2000)
    chosen = q["gnps"]
    assert keys[1] not in chosen and keys[14] not in chosen
    assert all(fold.loc[k, "fold_identity"] in (1, 2, 3, 4) for k in chosen)
    assert BT.assert_disjoint(q, excl, fold) == 0
    with pytest.raises(AssertionError):
        BT.assert_disjoint({"gnps": chosen + [keys[1]]}, excl, fold)
    with pytest.raises(AssertionError):
        BT.assert_disjoint({"gnps": [keys[14]]}, excl, fold)
    # 同じ入力なら同じ選び方（seed 0）、N_MOL を減らすと同じ並びの先頭
    q2, _ = BT.select_queries(lib_set, np.array(keys, dtype=object), excl, fold, 2000)
    assert q2 == q
    q3, _ = BT.select_queries(lib_set, np.array(keys, dtype=object), excl, fold, 50)   # gnps 10
    assert len(q3["gnps"]) == 10 and q3["gnps"] == q["gnps"][:10]


def test_d_allocate_50():
    assert BT.allocate(50) == dict(gnps=10, riken=7, mona=8, msdial=7, massbank=8, pluskal_ms2=10)
    assert BT.allocate(2000) == BT.QUERY_ALLOC


def test_d_guard_blocks_original_tables(tmp_path):
    p = tmp_path / "rank_train.npz"; np.savez(p, X=np.zeros(1))
    q = tmp_path / "own_rank_train.npz"; np.savez(q, X=np.zeros(1))
    s = tmp_path / "sim_rank_rows_nofp.npz"; np.savez(s, X=np.zeros(1))
    undo = BT.install_guard()
    try:
        with pytest.raises(PermissionError):
            np.load(p)
        with pytest.raises(PermissionError):
            open(s, "rb")
        assert np.load(q)["X"].shape == (1,)
        assert str(q) in BT.OPENED
    finally:
        undo()
    assert np.load(p)["X"].shape == (1,)          # 外したら元どおり


# ------------------------------------------------------------------------------------------------
# (e) 合成の表に engine2_table_schema の恒等式チェックを当てて全式 1.0（build() を端から端まで）
# ------------------------------------------------------------------------------------------------
def test_e_build_end_to_end_identities(tmp_path, monkeypatch):
    P, L, add, I, rng = _setup()
    n = len(L["prec"])
    ik = P["keys"][L["row_p"]]
    lib_set = np.where(np.arange(n) % 2 == 0, "gnps", "riken").astype(object)
    meta = dict(ingest_lib=lib_set, ik=ik, adduct=add, mode=np.where(L["pol"] == 1, "positive", "negative").astype(object),
                ce=np.array([None] * n, dtype=object))
    fold = pd.DataFrame(dict(scorer_key=P["keys"], fold_identity=[1 + i % 4 for i in range(len(P["keys"]))]), index=P["keys"])
    queries = {"gnps": list(P["keys"][[3, 10, 25]]), "riken": list(P["keys"][[4, 30]])}
    zr = np.random.default_rng(5)
    used = []

    def fake_logits(bank, sp):
        used.append(bank)
        return zr.normal(size=64).astype(np.float32)
    monkeypatch.setattr(base_fs, "fs_logits", fake_logits)
    monkeypatch.setattr(base_fs, "fs_fp", lambda smi, bits: (np.random.default_rng(len(smi)).random(64) < 0.3).astype(np.uint8))

    def fake_frag(recs, workers):
        fr = np.random.default_rng(9)
        for r in recs:
            r.pop("specs")
            _frag(r, fr)
    monkeypatch.setattr(E, "compute_frag", fake_frag)
    sim, rt, rep = BT.build(E, L, meta, I, queries, fold, {1: "bank1", 2: "bank2", 3: "bank3", 4: "bank4"}, np.arange(64), 1,
                            log=lambda *a: None, v3keys={P["keys"][4]})
    assert rep["molecules"] == 5 and rep["dropped_out_of_window"] == 0
    # fold k のクエリには fold k のモデル（bank k）だけ
    assert used == [f"bank{int(fold.loc[q['ik'], 'fold_identity'])}" for q in rep["queries"]]
    assert all(q["fold"] == int(fold.loc[q["ik"], "fold_identity"]) for q in rep["queries"])
    assert sim["X"].shape[1] == 51 and rt["X"].shape[1] == 31
    assert [str(b) for b in sim["blocks"]] == ["base", "libx", "anapol", "fragx"]
    st = BT.table_stats(sim, rt)
    assert st["sim"]["groups"] == 10 and st["sim"]["pos_per_group_min"] == 1 and st["sim"]["pos_per_group_max"] == 1
    assert st["rank_train"]["groups"] == 10 and st["rank_train"]["pos_per_group_min"] == 1
    assert set(np.unique(rt["M"]).tolist()) == {0.0, 1.0}
    ps, pr = BT.save_tables(str(tmp_path), sim, rt)
    for path, blocks in ((ps, None), (pr, ["base"])):
        d = SCH.describe(path, blocks)
        assert d["map_matches_n_feat"]
        fr = {k: v.get("frac") for k, v in d["identities"].items() if "frac" in v}
        assert fr and all(v == 1.0 for v in fr.values()), fr
    # FP 列: sim は 0、rank_train は 0 でない
    assert np.all(sim["X"][:, base_fs.FP_COLS] == 0) and not np.all(rt["X"][:, base_fs.FP_COLS] == 0)
    # 決定性: 同じ入力でもう一度作ると中身の sha256 が一致
    zr2 = np.random.default_rng(5)
    monkeypatch.setattr(base_fs, "fs_logits", lambda bank, sp: zr2.normal(size=64).astype(np.float32))
    sim2, rt2, _ = BT.build(E, L, meta, I, queries, fold, {1: None, 2: None, 3: None, 4: None}, np.arange(64), 1,
                            log=lambda *a: None)
    assert BT.content_sha256(sim) == BT.content_sha256(sim2) and BT.content_sha256(rt) == BT.content_sha256(rt2)


# ------------------------------------------------------------------------------------------------
# (f) 推論側: score_base_fs が blk_base_fs の直接呼び出しとビット一致。fs_logits が local_fold0_logits の保存値と一致
# ------------------------------------------------------------------------------------------------
class _Const:
    def __init__(self, w): self.w = w
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X.astype(np.float64) @ self.w)))
        return np.column_stack([1 - p, p])


def test_f_score_base_fs_equals_direct():
    P, L, add, I, rng = _setup()
    rows, modes, target = _query(L, add, 25)
    cand = E.pool_window(target, E.CFG.PPM_WIN)
    rec = _rec_from_engine(L, I, rows, add, modes, target, cand, rng); _frag(rec, rng); rec["mid"] = "m1"
    z = rng.normal(size=64).astype(np.float32)
    cfs = (rng.random((len(P["keys"]), 64)) < 0.3).astype(np.uint8)
    models = [_Const(rng.normal(size=31) * 0.1) for _ in range(3)]
    rep = {}
    got = base_fs.score_base_fs([rec], models, {"m1": z}, lambda c: cfs[c], rep)
    Xd = base_fs.blk_base_fs(rec, E.pool_fps(cand), cfs[cand], z).astype(np.float32)
    ref = np.mean([m.predict_proba(Xd)[:, 1] for m in models], axis=0)
    assert np.array_equal(got["m1"], ref) and rep["base_fs_molecules"] == 1


CASMI = os.path.join(ROOT, "data", "ext", "v4g", "casmi26-v3-models", "code")
FOLD0 = os.path.join(ROOT, "data", "ext", "foldsafe", "fpnet_fold0.pt")
PCL = os.path.join(ROOT, "outputs", "holdout_eval", "v3_c2n_fold0", "pc_logits.npz")
HTEST = os.path.join(ROOT, "data", "holdouts", "v3_c2n", "test.parquet")


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (os.path.isdir(CASMI) and os.path.exists(FOLD0) and os.path.exists(PCL) and os.path.exists(HTEST)
                         and _has_torch()), reason="fold-safe の実モデル・保存済みロジットが無い（CI）")
def test_f_fs_logits_matches_local_fold0_logits():
    base_fs.set_casmi_code_dir(CASMI)
    bank = base_fs.load_bank(FOLD0)
    te = pd.read_parquet(HTEST)
    zsave = np.load(PCL, allow_pickle=True)
    mids = [m for m in zsave.files if m != "__targets__"]
    mid = sorted(mids)[0]
    sub = te[te.molecule_id.astype(str) == str(mid)]
    z = base_fs.fs_logits(bank, base_fs.spectra_from_rows(sub.itertuples()))
    ref = zsave[mid]
    assert z.shape == ref.shape == (10226,)
    assert np.allclose(z, ref, rtol=1e-5, atol=1e-6), float(np.abs(z - ref).max())


# ------------------------------------------------------------------------------------------------
# (g) fs_fp の空間: v2-pool の保存済み指紋（train_fp_sel.npy、packbits）と 200 構造で array_equal
# ------------------------------------------------------------------------------------------------
V2 = os.environ.get("V2POOL_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "v2pool"))
BITS = os.path.join(ROOT, "data", "ext", "foldsafe", "fp_bits.npy")


def _rdkit_0303():
    try:
        import rdkit
        return rdkit.__version__ == base_fs.RDKIT_VERSION
    except Exception:
        return False


@pytest.mark.skipif(not (os.path.isdir(CASMI) and os.path.exists(os.path.join(V2, "train_fp_sel.npy"))
                         and os.path.exists(os.path.join(V2, "train_structs.parquet")) and os.path.exists(BITS)
                         and not os.path.exists(os.path.join(V2, "train_fp_sel.npy.kaggle-partial"))
                         and _rdkit_0303()), reason="v2-pool の指紋・RDKit 2026.03.3 が無い（CI）")
def test_g_fs_fp_matches_v2pool_train_fp_sel():
    base_fs.set_casmi_code_dir(CASMI)
    bits = np.load(BITS)
    assert len(bits) == 10226
    ts = pd.read_parquet(os.path.join(V2, "train_structs.parquet"))
    fps = np.load(os.path.join(V2, "train_fp_sel.npy"), mmap_mode="r")
    assert len(ts) == fps.shape[0]
    smi_col = "smiles" if "smiles" in ts.columns else [c for c in ts.columns if "smiles" in c.lower()][0]
    pick = np.sort(np.random.RandomState(0).choice(len(ts), 200, replace=False))
    bad = []
    for i in pick:
        got = base_fs.fs_fp(ts[smi_col].iloc[i], bits)
        ref = np.unpackbits(np.asarray(fps[i]))[:10226]
        if got is None or not np.array_equal(got, ref):
            bad.append(int(i))
    assert not bad, f"{len(bad)} / 200 mismatch: {bad[:10]}"
