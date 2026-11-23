"""Issue #196 (h): fragnn v2 の段（fragnn2_score.py の入力の作り方・塊・NaN・blind・npz、run.py の fragnn2・fragnn2blind 段）
の作り物のデータのテスト。計画 2026-11-23 昼 項目 1 の手順 5 の (i)〜(vi)。

torch が要るテストは importorskip（CI では飛ぶ）。元の .venv（torch あり）で実行して数を記録する。
v3 の集合・学習中の重みは使わない。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "tests"))
import fragnn2_score as F2  # noqa: E402
import fragnn_score as F1  # noqa: E402
import run as R  # noqa: E402
import sanpier as SP  # noqa: E402

B, T = F2.B, F2.T
COLS = ["molecule_id", "ms2_mzs", "ms2_normalized_intensities", "adduct", "ionization_mode", "precursor_mz",
        "collision_energy_ev", "instrument_type"]


def _te(rows):
    return pd.DataFrame(rows, columns=COLS)


def _pk(rng, n, lo=20.0, hi=180.0):
    return rng.random(n) * (hi - lo) + lo, rng.random(n) + 0.01


# ---------------------------------------------------------------- (i) 学習の前処理と推論の前処理が同じ

def test_i_inference_rows_equal_training_load_spectra_v2(tmp_path):
    """同じ行を、学習の T.load_spectra_v2（train.parquet の読み方）と推論の molecule_rows → B.pack_spectra_v2 に通すと
    同じ詰めた値（P5'・上位 64 本・shift・spec_feat〈mode・本数・CE・印〉・ce・tims）になる。"""
    rng = np.random.default_rng(0)
    recs = []
    adds = ["[M+H]+", "[M+Na]+", "[M-H]-", "[M+NH4]+", "[M+Cl]-", "[M-H2O+H]+"]
    for j in range(12):
        add = adds[j % len(adds)]
        mode = "negative" if add.endswith("-") else "positive"
        prec = 150.0 + 3 * j
        mz, it = _pk(rng, 30 + 7 * j, 15.0, prec + 20.0)          # prec + 2 Da より上のピークも混ぜる（P5'）
        ce = [np.array([20.0, 40.0]), np.array([]), None, np.array([35.0])][j % 4]
        inst = ["timsTOF", "Orbitrap", None][j % 3]
        recs.append(dict(molecule_id=f"m{j // 4}", inchikey14=f"IK{j:012d}"[:14], adduct=add, ionization_mode=mode,
                         precursor_mz=prec, ms2_mzs=mz, ms2_normalized_intensities=it,
                         collision_energy_ev=None if ce is None else ce, instrument_type=inst))
    df = pd.DataFrame(recs)
    tp = tmp_path / "train.parquet"
    df.drop(columns=["molecule_id"]).to_parquet(tp)
    spec_row = np.array([5, 0, 11, 3, 7, 1, 2, 4, 6, 8, 9, 10], np.int64)    # 学習の行の順は parquet の順と違ってよい
    spec_ik = np.array([df.inchikey14[i].encode() for i in spec_row], object)
    got_tr = T.load_spectra_v2(tp, spec_row, spec_ik)
    te = pd.read_parquet(tp).iloc[spec_row].reset_index(drop=True)          # 推論も parquet から読む（run.py と同じ）
    te["molecule_id"] = [f"q{k}" for k in range(len(te))]                   # 1 分子 1 本（並びを保つ）
    rows = [F2.molecule_rows(te[COLS])[f"q{k}"][0] for k in range(len(te))]
    got_inf = B.pack_spectra_v2(rows)
    for k in got_inf:
        np.testing.assert_array_equal(got_inf[k], got_tr[k], err_msg=k)
        assert got_inf[k].dtype == got_tr[k].dtype, k
    assert (got_inf["mz"][got_inf["it"] > 0] <= np.repeat(got_inf["prec"], B.TOP_PEAKS_V2).reshape(-1, B.TOP_PEAKS_V2)
            [got_inf["it"] > 0] + 2.0).all()                                 # P5'
    assert got_inf["mz"].shape[1] == 64 and got_inf["spec_feat"].shape[1] == 4
    assert set(got_inf["tims"].tolist()) == {0, 1}


def test_i_chunks_follow_infer_chunks_and_merge_equals_training_merge_batch():
    rng = np.random.default_rng(1)
    adds = ["[M+H]+", "[M+H]+", "[M+Na]+", "[M+H]+", "[M+H]+", "[M+H]+", "[M+H]+"]
    inst = ["timsTOF"] * 6 + ["Orbitrap"]
    rows = [dict(zip(["mz", "it"], _pk(rng, 40 + j))) | dict(prec=200.0, adduct=a, mode=1, ce=10.0 * (j + 1), instrument=i)
            for j, (a, i) in enumerate(zip(adds, inst))]
    spec, chunks = F2.chunk_spectra(rows)
    # (付加イオン, 印) の群を出てくる順に、各群を順に 4 本ずつ
    assert chunks == [[0, 1, 3, 4], [5], [2], [6]]
    assert chunks == B.infer_chunks(adds, [1] * 6 + [0])
    pk = B.pack_spectra_v2(rows)
    # 1 本の塊はその本のまま（merge しない）
    for r, c in enumerate(chunks):
        if len(c) == 1:
            for k in B.SPEC_KEYS:
                np.testing.assert_array_equal(spec[k][r], pk[k][c[0]], err_msg=k)
    # 4 本の塊 = 学習の merged_row（先頭の本が i）
    m = B.merged_row(pk, chunks[0], 0)
    assert spec["n"][0] == m["n"] and np.array_equal(spec["mz"][0, :m["n"]], m["mz"])
    assert np.allclose(spec["spec_feat"][0], m["spec_feat"]) and spec["prec"][0] == pk["prec"][0]
    assert spec["spec_feat"][0][2] == pytest.approx(np.mean([10.0, 20.0, 40.0, 50.0]) / 100)   # CE = 値のある本の平均
    # 2 本の群なら学習の merge_batch（p = 1、群の大きさ 2 → 2 本とも引く）と同じ行になる
    two = rows[:2]
    pk2 = B.pack_spectra_v2(two)
    gkey = np.array([7, 7], np.int64)
    bs, n_merged = B.merge_batch(0, np.array([0]), pk2, gkey, {7: np.array([0, 1])}, p=1.0)
    sp2, ch2 = F2.chunk_spectra(two)
    assert n_merged == 1 and ch2 == [[0, 1]]
    for k in B.SPEC_KEYS:
        np.testing.assert_array_equal(sp2[k][0], bs[k][0], err_msg=k)


# ---------------------------------------------------------------- (ii) CE が全部欠けた分子は CE = 0

def test_ii_all_missing_ce_gives_zero():
    rng = np.random.default_rng(2)
    te = _te([("a", *_pk(rng, 20), "[M+H]+", "positive", 200.0, np.array([]), "timsTOF"),
              ("a", *_pk(rng, 20), "[M+H]+", "positive", 200.0, None, "timsTOF"),
              ("a", *_pk(rng, 20), "[M+H]+", "positive", 200.0, np.array([]), "timsTOF")])
    rows = F2.molecule_rows(te)["a"]
    assert [r["ce"] for r in rows] == [0.0, 0.0, 0.0]
    spec, chunks = F2.chunk_spectra(rows)
    assert chunks == [[0, 1, 2]]
    assert spec["spec_feat"][0][2] == 0.0 and np.isfinite(spec["spec_feat"]).all()
    pk = B.pack_spectra_v2(rows)
    assert (pk["spec_feat"][:, 2] == 0).all() and (pk["ce"] == 0).all()


# ---------------------------------------------------------------- (iii) 有効なピーク 0 本の塊だけの分子は NaN

def _levels(n):
    rng = np.random.default_rng(3)
    return [[np.sort(rng.random(1) * 200 + 100), np.sort(rng.random(6) * 150 + 20), np.sort(rng.random(12) * 150 + 15)]
            for _ in range(n)]


def test_iii_usable_chunks_pure():
    dead = [dict(mz=np.array([10.0]), it=np.array([0.0]), prec=100.0, adduct="[M+H]+", mode=1, ce=0.0, instrument="timsTOF"),
            dict(mz=np.array([150.0, 160.0]), it=np.array([1.0, 0.5]), prec=100.0, adduct="[M+Na]+", mode=1, ce=0.0,
                 instrument="timsTOF")]                    # 2 本目は P5' で全部消える（prec + 2 より上）
    spec, chunks = F2.chunk_spectra(dead)
    assert len(chunks) == 2 and len(F2.usable_chunks(spec)) == 0
    spec0, ch0 = F2.chunk_spectra([])
    assert ch0 == [] and len(F2.usable_chunks(spec0)) == 0


def test_iii_nan_when_no_usable_chunk_and_matches_pairwise():
    torch = pytest.importorskip("torch")
    model = F2.build_v2_model(random_init=0)
    lv = _levels(5)
    lv[1] = None
    rng = np.random.default_rng(4)
    rows = [dict(zip(["mz", "it"], _pk(rng, 50))) | dict(prec=190.0, adduct="[M+H]+", mode=1, ce=20.0, instrument="timsTOF"),
            dict(zip(["mz", "it"], _pk(rng, 12))) | dict(prec=190.0, adduct="[M+H]+", mode=1, ce=40.0, instrument="timsTOF"),
            dict(zip(["mz", "it"], _pk(rng, 30))) | dict(prec=190.0, adduct="[M+Na]+", mode=1, ce=0.0, instrument="timsTOF"),
            dict(mz=np.array([10.0]), it=np.array([0.0]), prec=190.0, adduct="[M+K]+", mode=1, ce=0.0, instrument="timsTOF")]
    got = F2.score_candidates(model, lv, rows, chunk=3)
    assert np.isnan(got[1]) and np.isfinite(np.delete(got, 1)).all()
    # 独立の経路: 使える塊（[0,1] の merge と [2]）を 1 つずつ流して平均。[3] は使えない
    spec, chunks = F2.chunk_spectra(rows)
    assert chunks == [[0, 1], [2], [3]] and F2.usable_chunks(spec).tolist() == [0, 1]
    for c in (0, 2, 3, 4):
        masses, off = B.levels_to_csr([lv[c]])
        vals = []
        for s in (0, 1):
            f, fs = B.batch_features(masses, off, np.array([0]), np.array([s]), spec)
            valid = np.arange(64)[None, :] < spec["n"][[s]][:, None]
            with torch.inference_mode():
                vals.append(float(model(torch.from_numpy(np.asarray(f, np.float32)), torch.from_numpy(valid),
                                        torch.from_numpy(np.asarray(fs, np.float32)[:, None]),
                                        torch.from_numpy(np.ascontiguousarray(spec["spec_feat"][[s]], np.float32)))[0]))
        assert abs(got[c] - np.mean(vals)) < 1e-5
    assert np.isnan(F2.score_candidates(model, lv, rows[3:])).all()        # 使える塊が無い → NaN
    assert np.isnan(F2.score_candidates(model, lv, [])).all()
    assert np.isnan(F2.score_candidates(model, [None, None], rows)).all()
    assert next(model.parameters()).device.type == "cpu"


# ---------------------------------------------------------------- (iv) blind の入れ替えが v1 と同じ定義

def test_iv_blind_is_v1_definition():
    assert F2.blind_targets is F1.blind_targets and F2.first_mode is F1.first_mode
    assert F2.pair_layout is F1.pair_layout and F2.mean_logits is F1.mean_logits and F2.LevelCache is F1.LevelCache
    rng = np.random.default_rng(5)
    te = _te([("m_b", *_pk(rng, 5), "[M+H]+", "positive", 200.0, np.array([20.0]), "timsTOF"),
              ("m_a", *_pk(rng, 5), "[M-H]-", "negative", 198.0, np.array([20.0]), "timsTOF"),
              ("m_c", *_pk(rng, 5), "[M+H]+", "positive", 300.0, np.array([20.0]), "timsTOF"),
              ("m_d", *_pk(rng, 5), "[M-H]-", "negative", 298.0, np.array([20.0]), "timsTOF")])
    sp2 = F2.molecule_rows(te)
    sp1 = F1.molecule_rows(te.drop(columns=["instrument_type"]))
    m2 = {m: F2.first_mode(v) for m, v in sp2.items()}
    assert m2 == {m: F1.first_mode(v) for m, v in sp1.items()}
    assert F2.blind_targets(list(sp2), m2) == F1.blind_targets(list(sp1), m2)
    # v2 の行は v1 の行に instrument を足しただけ
    for m in sp1:
        for a, b in zip(sp1[m], sp2[m]):
            assert set(b) - set(a) == {"instrument"}
            assert all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


# ---------------------------------------------------------------- 守り（乱数の重み・学習が止まったかの確かめ）

def test_guards_random_init_only_smoke_and_weights_need_ok_report(tmp_path):
    F2.check_random_init("sub_test", True)
    for h, nt in (("v3_c1u_t10", True), ("v3_c2_t10", True), ("enveda_v1u_t10", True), ("v3_c2nu_t10", True),
                  ("sub_test", False), ("v3_c1", True)):
        with pytest.raises(SystemExit):
            F2.check_random_init(h, nt)
    w = tmp_path / "model_best.pt"; w.write_bytes(b"x")
    with pytest.raises(SystemExit):
        F2.check_weights_report(w)                                         # train_report.json が無い
    for st in ("running", "paused", "error", None):
        (tmp_path / "train_report.json").write_text(json.dumps({"status": st}))
        with pytest.raises(SystemExit):
            F2.check_weights_report(w)
    (tmp_path / "train_report.json").write_text(json.dumps({"status": "ok", "train": {"stop_reason": "stopped"}}))
    assert F2.check_weights_report(w) == {"status": "ok", "stop_reason": "stopped"}


def test_score_holdout_npz_rows_blind_nan_and_report(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rdkit")
    if not (ROOT / "data/ext/frag/frag_rescore.py").exists():
        pytest.skip("data/ext/frag が無い（ローカルだけ）")
    import library as L
    root = tmp_path
    (root / "data/ext").mkdir(parents=True)
    (root / "data/ext/frag").symlink_to((ROOT / "data/ext/frag").resolve())
    smi = ["OC1=CC=CC=C1C(O)=O", "OC(=O)C1=CC=C(O)C=C1", "not_a_smiles", "COC1=CC=CC=C1O", "CCO", "OCC(O)CO", "CCCCCC"]
    mass = np.array([138.0, 138.0003, 138.0005, 138.0008, 46.0, 92.0, 86.0])
    meta = pd.DataFrame(dict(key=[f"K{i}" for i in range(7)], smiles=smi, mass=mass, pid=np.arange(7)))
    monkeypatch.setattr(R, "load_pool", lambda root=None: R.Pool(meta, None))
    monkeypatch.setattr(L, "ROOT", root)
    zdir = root / "outputs/holdout_eval/syn_fold0"; zdir.mkdir(parents=True)
    np.savez(zdir / "pc_logits.npz", __targets__=np.array([["m_b", "138.0004"], ["m_a", "92.0"], ["m_c", "46.0"],
                                                           ["m_d", "86.0"]], object))
    tdir = root / "outputs/holdouts/syn"; tdir.mkdir(parents=True)
    (tdir / "truth.csv").write_text("molecule_id,smiles\nm_b,CCO\nm_a,CCO\nm_c,CCO\nm_d,CCO\nm_x,CCO\n")
    hdir = root / "data/holdouts/syn"; hdir.mkdir(parents=True)
    (hdir / "hide_rule.json").write_text(json.dumps(dict(rule="all", pool="remove")))
    (hdir / "drop_keys.txt").write_text("K3\n")
    rng = np.random.default_rng(5)
    pk = lambda n, hi=150.0: (rng.random(n) * (hi - 15) + 15, rng.random(n) + 0.05)  # noqa: E731
    _te([("m_b", *pk(20), "[M+H]+", "positive", 139.04, np.array([20.0]), "timsTOF"),
         ("m_b", *pk(8), "[M+H]+", "positive", 139.04, np.array([40.0]), "timsTOF"),
         ("m_b", *pk(8), "[M+Na]+", "positive", 161.03, np.array([]), "timsTOF"),
         ("m_a", *pk(10, 90.0), "[M-H]-", "negative", 91.0, np.array([30.0]), "timsTOF"),
         ("m_c", *pk(6, 47.0), "[M+H]+", "positive", 47.05, np.array([]), "timsTOF"),
         ("m_d", np.array([30.0]), np.array([0.0]), "[M-H]-", "negative", 85.0, np.array([10.0]), "timsTOF")]
        ).to_parquet(hdir / "test.parquet")
    wdir = root / "w"; wdir.mkdir()
    torch.manual_seed(1)
    torch.save(B.build_model(B.F_SPEC_V2).state_dict(), wdir / "model_best.pt")
    (wdir / "train_report.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(SystemExit):                                         # 学習の途中の重みは読まない
        F2.score_holdout("syn", root / "out", weights=wdir / "model_best.pt", root=root)
    assert not (root / "out").exists()
    (wdir / "train_report.json").write_text(json.dumps({"status": "ok"}))
    rep = F2.score_holdout("syn", root / "out", weights=wdir / "model_best.pt", root=root)
    assert rep["weights_sha256"] == hashlib.sha256((wdir / "model_best.pt").read_bytes()).hexdigest()
    assert sorted(p.name for p in (root / "out").iterdir()) == sorted([F2.NPZ_BLIND, F2.NPZ_FULL, F2.REPORT])
    full = np.load(root / "out" / F2.NPZ_FULL, allow_pickle=True)
    assert list(full["mids"]) == ["m_b", "m_a", "m_c", "m_d"]
    pool = R.load_pool()
    got, gotb = SP.load_scores(root / "out" / F2.NPZ_FULL), SP.load_scores(root / "out" / F2.NPZ_BLIND)
    idx_b = R.visible_window(pool, 138.0004, {"K3"})
    assert list(got["m_b"]) == idx_b.tolist() and "K3" not in set(pool.key[idx_b])
    assert rep["rows"] == sum(len(v) for v in got.values()) == rep[F2.NPZ_FULL]["rows"]
    assert (rep["spectra"], rep["chunks"], rep["merged_chunks"]) == (6, 5, 1)   # m_b: [H,H] の塊と [Na]
    j_bad = int(np.flatnonzero(pool.smiles == "not_a_smiles")[0])
    assert np.isnan(got["m_b"][j_bad]) and np.isnan(gotb["m_b"][j_bad])
    assert all(np.isnan(v) for v in got["m_d"].values())                    # 使える塊が無い → NaN
    assert rep["molecules_without_usable_chunks"] == 1 and rep["nonfinite_nonnan"] == 0
    assert (rep["blind_same_mode"], rep["blind_any_mode"], rep["blind_self"]) == (4, 0, 0)
    assert all(np.isnan(v) for v in gotb["m_a"].values())                   # 入れ替え先 m_d は使える塊が無い
    model = F2.build_v2_model(weights=wdir / "model_best.pt")
    fr = T.import_frag_rescore(root)
    lc = F2.LevelCache(fr, pool.smiles)
    sp = F2.molecule_rows(pd.read_parquet(hdir / "test.parquet"))
    want_b = F2.score_candidates(model, [lc(i) for i in idx_b], sp["m_c"])   # m_b の blind = m_c のスペクトル
    gb = np.array([gotb["m_b"][i] for i in idx_b])
    np.testing.assert_allclose(gb, want_b.astype(np.float32), rtol=1e-6, equal_nan=True)
    saved = json.loads((root / "out" / F2.REPORT).read_text())
    assert saved["device"] == "cpu" and "mrr25" not in saved and saved["train_status"]["status"] == "ok"


# ---------------------------------------------------------------- (v) run.py: 既定の段の出力が変わらない・fragnn2 の段

def test_v_check_stages_fragnn2():
    R.check_stages(["score", "fragnn2", "prior", "frag"])
    R.check_stages(["score", "fragnn2blind", "prior", "frag"])
    for a in ("fragnn", "fragnnblind", "fragnn2", "fragnn2blind"):
        for b in ("fragnn", "fragnnblind", "fragnn2", "fragnn2blind"):
            if a != b:
                with pytest.raises(AssertionError):
                    R.check_stages(["score", a, b, "prior", "frag"])
    for other in ("sanpier", "pmi", "nbz", "joint", "pubchem", "lib", "reflast", "elem"):
        for st in ("fragnn2", "fragnn2blind"):
            with pytest.raises(AssertionError):
                R.check_stages(["score", other, st, "prior", "frag"])
    R.check_stages(["score", "sanpier", "prior", "frag", "elem"])
    assert R.FRAGNN_STAGES[:2] == ("fragnn", "fragnnblind") and R.FRAGNN_ALLOWED == ("score", "prior", "frag")


def test_v_default_stages_unchanged_and_fragnn2_reads_its_own_npz(tmp_path, monkeypatch):
    import test_fragnn_stage as TS
    root = TS._synthetic_tree(tmp_path, monkeypatch)
    pool = R.load_pool()
    ia, ib = pool.window(100.0003), pool.window(200.0005)
    ldir = root / "outputs/local_eval/syn"
    # v2 の npz: a で候補 1（OCC(=O)O）を推す。v1 の npz（候補 3 を推す）とは別の結果になる
    np.savez_compressed(ldir / "fragnn2_scores.npz", **SP.pack({"a": (ia, np.array([0.0, 30.0, 0.0, 0.0], np.float32)),
                                                                "b": (ib, np.array([5.0, np.nan], np.float32))}))
    np.savez_compressed(ldir / "fragnn2blind_scores.npz", **SP.pack({"a": (ia, np.ones(4, np.float32)),
                                                                     "b": (ib, np.full(2, np.nan, np.float32))}))
    R.run("syn", ["score", "prior", "frag"])
    t0 = TS._top(root, "score+prior+frag")
    assert t0["a"] == [TS.SMI[0], TS.SMI[3], TS.SMI[1], TS.SMI[2]] and t0["b"] == TS.SMI[4:]   # v1 のテストと同じ値
    r1 = R.run("syn", ["score", "fragnn", "prior", "frag"])
    r2 = R.run("syn", ["score", "fragnn2", "prior", "frag"])
    r2b = R.run("syn", ["score", "fragnn2blind", "prior", "frag"])
    t1, t2 = TS._top(root, "score+fragnn+prior+frag"), TS._top(root, "score+fragnn2+prior+frag")
    assert t1["a"][0] == "c1ccccc1O" and t2["a"][0] == "OCC(=O)O" and t2["b"] == t0["b"]
    assert TS._top(root, "score+fragnn2blind+prior+frag") == t0
    h2 = hashlib.sha256((ldir / "fragnn2_scores.npz").read_bytes()).hexdigest()
    assert r2["fragnn2_npz_sha256"] == h2 and "fragnn_npz_sha256" not in r2 and "fragnn2blind_npz_sha256" in r2b
    assert r1["fragnn_npz_sha256"] != h2
    # 既定の段の出力（submission と mrr25）は、v2 の npz を置いた後も同じ
    sub = (ldir / "submission_score+prior+frag.csv").read_bytes()
    os_res = R.run("syn", ["score", "prior", "frag"])
    assert (ldir / "submission_score+prior+frag.csv").read_bytes() == sub
    assert not any(k.startswith("fragnn") for k in os_res)


# ---------------------------------------------------------------- (vi) v1 の fragnn_score の出力が持ち込みの前後でバイト一致

def test_vi_v1_fragnn_score_bytes_equal_to_golden_before_202_files():
    import fragnn_v1_golden_case as G
    z = np.load(G.GOLDEN)
    out, score = G.compute()
    for k, v in out.items():            # numpy/numba の経路（pack_spectra・batch_features）
        v = np.asarray(v)
        assert v.dtype == z[k].dtype and v.shape == z[k].shape, k
        if str(z["machine"]) == __import__("platform").machine():
            assert v.tobytes() == z[k].tobytes(), k
        else:
            np.testing.assert_allclose(v, z[k], rtol=1e-12, atol=0, err_msg=k)
    if score is None:
        pytest.skip("torch が無い（score_candidates の一致はローカルで確かめる）")
    import platform
    import torch
    if (str(z["torch"]), str(z["machine"]), str(z["system"])) != (torch.__version__, platform.machine(), platform.system()):
        pytest.skip("golden を作った環境と違う（バイト一致は同じ環境だけ）")
    assert score.tobytes() == z["score"].tobytes()
