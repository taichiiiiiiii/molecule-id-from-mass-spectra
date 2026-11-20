"""Issue #120: fragnn 段（fragnn_score.py の点の作り方・blind の入れ替え・npz の行、run.py の fragnn・fragnnblind 段）の合成データのテスト。

torch が要るテストは importorskip（CI では飛ぶ）。元の .venv（torch あり）で実行して数を記録する。
"""
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
import fragnn_score as FS  # noqa: E402
import run as R  # noqa: E402
import sanpier as SP  # noqa: E402


# ---------------------------------------------------------------- blind の入れ替え先（#120 本文 25 行）

def test_blind_next_same_mode_and_wraps():
    modes = {"m1": 1, "m2": -1, "m3": 1, "m4": -1, "m5": 1}
    bt = FS.blind_targets(["m3", "m1", "m5", "m2", "m4"], modes)   # 入力の並びは問わない（文字列の昇順で決める）
    assert bt == {"m1": ("m3", "same_mode"), "m2": ("m4", "same_mode"), "m3": ("m5", "same_mode"),
                  "m4": ("m2", "same_mode"),     # 最後の負は先頭側の負へ戻る
                  "m5": ("m1", "same_mode")}     # 最後は先頭に戻る


def test_blind_falls_back_to_next_any_mode_when_no_other_same_mode():
    bt = FS.blind_targets(["a", "b", "c"], {"a": 1, "b": 1, "c": -1})
    assert bt["a"] == ("b", "same_mode") and bt["b"] == ("a", "same_mode")
    assert bt["c"] == ("a", "any_mode")          # 負はほかに無い → 正負を問わず次（最後なので先頭）
    bt = FS.blind_targets(["x", "y"], {"x": 1, "y": -1})
    assert bt == {"x": ("y", "any_mode"), "y": ("x", "any_mode")}


def test_blind_single_molecule_is_full():
    assert FS.blind_targets(["only"], {"only": 1}) == {"only": ("only", "self")}


def test_blind_string_order_not_numeric():
    bt = FS.blind_targets(["m_10", "m_9", "m_2"], {"m_10": 1, "m_9": 1, "m_2": 1})
    # 文字列の昇順: m_10 < m_2 < m_9
    assert bt == {"m_10": ("m_2", "same_mode"), "m_2": ("m_9", "same_mode"), "m_9": ("m_10", "same_mode")}


def _te(rows):
    return pd.DataFrame(rows, columns=["molecule_id", "ms2_mzs", "ms2_normalized_intensities", "adduct",
                                       "ionization_mode", "precursor_mz", "collision_energy_ev"])


def test_mode_is_from_first_spectrum():
    pk = (np.array([50.0, 60.0]), np.array([1.0, 0.5]))
    te = _te([("a", *pk, "[M+H]+", "positive", 200.0, np.array([20.0])),
              ("a", *pk, "[M-H]-", "negative", 198.0, np.array([20.0])),
              ("a", *pk, "[M-H]-", "negative", 198.0, np.array([20.0])),
              ("b", *pk, "[M-H]-", "negative", 300.0, np.array([20.0])),
              ("c", *pk, "[M+H]+", "positive", 400.0, np.array([20.0]))])
    sp = FS.molecule_rows(te)
    modes = {m: FS.first_mode(v) for m, v in sp.items()}
    assert modes == {"a": 1, "b": -1, "c": 1}
    assert FS.blind_targets(list(sp), modes)["a"] == ("c", "same_mode")   # 2・3 本目が負でも a は正


# ---------------------------------------------------------------- スペクトルの行・使えるスペクトル

def test_molecule_rows_match_query_spectra_rows_and_ce():
    rng = np.random.default_rng(0)
    rows = []
    for j in range(20):   # 分子 m は 20 本 → 先頭 16 本
        rows.append(("m", rng.random(3) * 100, rng.random(3), "[M+H]+", "positive", 150.0 + j, np.array([10.0, 30.0])))
    rows.append(("n", np.array([10.0]), np.array([1.0]), "[M-H]-", "negative", 99.0, np.array([])))
    te = _te(rows)
    sp = FS.molecule_rows(te)
    assert len(sp["m"]) == 16 and [r["prec"] for r in sp["m"]] == [150.0 + j for j in range(16)]
    assert sp["m"][0]["ce"] == 20.0 and sp["n"][0]["ce"] == 0.0          # 平均、空なら 0（学習の load_spectra と同じ）
    assert sp["m"][0]["mode"] == 1 and sp["n"][0]["mode"] == -1
    # run.py の query_spectra と同じ行（同じ並び・同じピーク）
    q = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        q[str(mid)] = [np.asarray(r.ms2_mzs, np.float64) for r in g.itertuples()][:16]
    for m in sp:
        assert len(q[m]) == len(sp[m]) and all(np.array_equal(a, b["mz"]) for a, b in zip(q[m], sp[m]))


def test_usable_needs_one_positive_finite_peak():
    good = dict(mz=np.array([10.0, 20.0]), it=np.array([0.0, 1.0]))
    bad1 = dict(mz=np.array([10.0]), it=np.array([0.0]))
    bad2 = dict(mz=np.array([np.nan]), it=np.array([1.0]))
    bad3 = dict(mz=np.array([]), it=np.array([]))
    assert FS.usable([bad1, good, bad2, bad3]) == [good]


def test_pair_layout_and_mean_logits():
    cs, sp = FS.pair_layout(3, 2)
    assert cs.tolist() == [0, 0, 1, 1, 2, 2] and sp.tolist() == [0, 1, 0, 1, 0, 1]
    logit = np.array([1.0, 3.0, 2.0, 2.0, -1.0, 5.0])   # 組 c*2 + s
    assert FS.mean_logits(logit, 3, 2).tolist() == [2.0, 2.0, 2.0]
    assert FS.mean_logits(np.array([1.0, 2.0, 4.0]), 3, 1).tolist() == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------- run.py の段

def test_check_stages_fragnn():
    R.check_stages(["score", "fragnn", "prior", "frag"])
    R.check_stages(["score", "fragnnblind", "prior", "frag"])
    R.check_stages(["score", "fragnn"])
    with pytest.raises(AssertionError):
        R.check_stages(["score", "fragnn", "fragnnblind", "prior", "frag"])
    for other in ("sanpier", "pmi", "nbz", "joint", "pubchem", "lib", "reflast", "elem"):
        for st in ("fragnn", "fragnnblind"):
            with pytest.raises(AssertionError):
                R.check_stages(["score", other, st, "prior", "frag"])
    R.check_stages(["score", "sanpier", "prior", "frag", "elem"])   # fragnn を指定しなければ今までどおり


def test_fragnn_mix_is_sanpier_mix_and_nan_rules():
    rows = [(3.0, 10), (2.0, 11), (1.0, 12)]
    out = R.sanpier_mix(rows, {10: 0.0, 11: np.nan, 12: 9.0})      # NaN は z_n = 0
    s = np.array([3.0, 2.0, 1.0]); zs = (s - s.mean()) / s.std()
    zn = np.array([-1.0, 0.0, 1.0])
    want = dict(zip((10, 11, 12), s.mean() + s.std() * (zs + zn) / 2))
    assert all(abs(v - want[i]) < 1e-9 for v, i in out)
    assert R.sanpier_mix(rows, {10: 1.0, 11: np.nan, 12: np.nan}) == rows   # 点が 2 未満 → s のまま
    assert R.sanpier_mix(rows, {10: 1.0, 11: 1.0, 12: 1.0}) == rows         # σ = 0 → s のまま


def test_run_source_applies_fragnn_after_score_before_prior():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    a = src.index("rows = (cos_window if \"cos\" in stages else score_window)(idx, pool, stage_z(z, zbar), fpc)")   # #140 で cos を足した
    b = src.index("rows = sanpier_mix(rows, fscores[mid])")
    c = src.index("rows = rerank_with_prior(rows, pool)")
    assert a < b < c
    assert 'f"outputs/local_eval/{holdout}/{fragnn_stage}_scores.npz"' in src


# ---------------------------------------------------------------- run.py を合成の木で最後まで

SMI = ["CCO", "OCC(=O)O", "CC(C)O", "c1ccccc1O", "CCN", "CCCC"]


def _synthetic_tree(tmp_path, monkeypatch):
    """6 候補・2 分子の合成の木。FPNet の点は s = fp @ z、frag・gl_fuse は並べ替えない偽物。"""
    import library as L
    root = tmp_path
    pdir = root / "data/ext/v4g/casmi26-v2-pool"; pdir.mkdir(parents=True)
    mass = np.array([100.0, 100.0002, 100.0004, 100.0006, 200.0, 200.001])
    meta = pd.DataFrame(dict(key=[f"K{i}" for i in range(6)], smiles=SMI, mass=mass, pid=np.arange(6),
                             formula=["F1"] * 4 + ["F2"] * 2))
    meta.to_parquet(pdir / "pool_meta.parquet")
    np.save(pdir / "fp_bits.npy", np.arange(2))
    fps = {SMI[i]: np.array(v, np.float32) for i, v in enumerate([[4, 0], [3, 0], [2, 0], [3.5, 0], [2, 0], [1, 0]])}

    class FakeFP:
        def __init__(self, bits):
            self.bits = bits

        def __call__(self, smi):
            return fps.get(smi)
    monkeypatch.setattr(R, "FPCache", FakeFP)
    monkeypatch.setattr(R, "load_pool", lambda root=None: R.Pool(meta, np.zeros(6)))
    zdir = root / "outputs/holdout_eval/syn_fold0"; zdir.mkdir(parents=True)
    np.savez(zdir / "pc_logits.npz", __targets__=np.array([["a", "100.0003"], ["b", "200.0005"]], object),
             a=np.array([1.0, 0.0], np.float32), b=np.array([1.0, 0.0], np.float32))
    tdir = root / "outputs/holdouts/syn"; tdir.mkdir(parents=True)
    (tdir / "truth.csv").write_text("molecule_id,smiles\na,c1ccccc1O\nb,CCCC\n")
    hdir = root / "data/holdouts/syn"; hdir.mkdir(parents=True)
    _te([("a", np.array([50.0]), np.array([1.0]), "[M+H]+", "positive", 101.0, np.array([20.0])),
         ("b", np.array([60.0]), np.array([1.0]), "[M+H]+", "positive", 201.0, np.array([20.0]))]
        ).to_parquet(hdir / "test.parquet")
    frag = types.SimpleNamespace(warmup=lambda: True, score_molecule=lambda sp, smis, cache=None: [0.5] * len(smis))
    monkeypatch.setitem(sys.modules, "frag_rescore", frag)
    fuse = types.SimpleNamespace(rerank_multi=lambda smis, keys, sc, forms, fm, lams, top_n: list(range(len(smis))))
    monkeypatch.setattr(R, "_load", lambda name, path: fuse)
    monkeypatch.setattr(R, "ROOT", root)
    monkeypatch.setattr(L, "ROOT", root)
    # fragnn の点: a では n が s と逆向き（候補 3 を推す）、b は点が 1 つだけ（s のまま）
    pool = R.load_pool()
    ia, ib = pool.window(100.0003), pool.window(200.0005)
    assert ia.tolist() == [0, 1, 2, 3] and ib.tolist() == [4, 5]
    ldir = root / "outputs/local_eval/syn"; ldir.mkdir(parents=True)
    full = {"a": (ia, np.array([0.0, 0.0, 0.0, 30.0], np.float32)), "b": (ib, np.array([5.0, np.nan], np.float32))}
    blind = {"a": (ia, np.array([1.0, 1.0, 1.0, 1.0], np.float32)), "b": (ib, np.array([np.nan, np.nan], np.float32))}
    np.savez_compressed(ldir / "fragnn_scores.npz", **SP.pack(full))
    np.savez_compressed(ldir / "fragnnblind_scores.npz", **SP.pack(blind))
    return root


def _top(root, tag):
    df = pd.read_csv(root / f"outputs/local_eval/syn/submission_{tag}.csv")
    return {r.molecule_id: r.smiles.split(";") for r in df.itertuples()}


def test_run_end_to_end_score_fragnn_prior_frag(tmp_path, monkeypatch):
    root = _synthetic_tree(tmp_path, monkeypatch)
    base = R.run("syn", ["score", "prior", "frag"])
    res = R.run("syn", ["score", "fragnn", "prior", "frag"])
    resb = R.run("syn", ["score", "fragnnblind", "prior", "frag"])
    t0, t1, tb = _top(root, "score+prior+frag"), _top(root, "score+fragnn+prior+frag"), _top(root, "score+fragnnblind+prior+frag")
    assert t0["a"] == [SMI[0], SMI[3], SMI[1], SMI[2]] and t0["b"] == SMI[4:]
    # a: s = 4,3,2,3.5、n = 0,0,0,30 → z の和 0.61 / −0.75 / −2.10 / 2.24 で候補 3（c1ccccc1O）が 1 位
    assert t1["a"][0] == "c1ccccc1O" and t1["b"] == t0["b"]      # b は点が 1 つだけ → s のまま
    assert tb == t0                                              # blind は σ = 0 / 点なし → s のまま
    assert base["mrr25"] < res["mrr25"]
    h = hashlib.sha256((root / "outputs/local_eval/syn/fragnn_scores.npz").read_bytes()).hexdigest()
    assert res["fragnn_npz_sha256"] == h and "fragnnblind_npz_sha256" in resb
    saved = json.loads((root / "outputs/local_eval/syn/result_score+fragnn+prior+frag.json").read_text())
    assert saved["stages"] == ["score", "fragnn", "prior", "frag"]


def test_run_stops_when_molecule_missing_from_npz(tmp_path, monkeypatch):
    root = _synthetic_tree(tmp_path, monkeypatch)
    pool = R.load_pool()
    np.savez_compressed(root / "outputs/local_eval/syn/fragnn_scores.npz",
                        **SP.pack({"a": (pool.window(100.0003), np.zeros(4, np.float32))}))
    with pytest.raises(KeyError):
        R.run("syn", ["score", "fragnn", "prior", "frag"])


# ---------------------------------------------------------------- 推論（torch あり、ローカル）

def _spec(mz, it, prec=181.07, adduct="[M+H]+", mode=1, ce=20.0):
    return dict(mz=np.asarray(mz, np.float64), it=np.asarray(it, np.float64), prec=prec, adduct=adduct, mode=mode, ce=ce)


def _levels(n):
    rng = np.random.default_rng(3)
    return [[np.sort(rng.random(1) * 200 + 100), np.sort(rng.random(6) * 150 + 20), np.sort(rng.random(12) * 150 + 15)]
            for _ in range(n)]


def test_score_candidates_matches_pairwise_forward_and_nan_rules():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = FS.B.build_model().eval()
    lv = _levels(5)
    lv[2] = None                                              # 読めない候補 → NaN
    rng = np.random.default_rng(1)
    sps = [_spec(rng.random(40) * 180 + 15, rng.random(40)), _spec(rng.random(5) * 100 + 20, rng.random(5) + 0.1),
           _spec([10.0], [0.0])]                              # 3 本目は使えない（強度 0）→ 平均に入れない
    got = FS.score_candidates(model, lv, sps, chunk=3)        # chunk を小さくして分割の境目も通す
    assert np.isnan(got[2]) and np.isfinite(np.delete(got, 2)).all()
    # 独立の経路: 組を 1 つずつ学習の forward と同じ形で流して平均
    spec = FS.B.pack_spectra(sps[:2])
    for c in (0, 1, 3, 4):
        masses, off = FS.B.levels_to_csr([lv[c]])
        vals = []
        for s in range(2):
            f, fs = FS.B.batch_features(masses, off, np.array([0]), np.array([s]), spec)
            valid = np.arange(FS.B.TOP_PEAKS)[None, :] < spec["n"][[s]][:, None]
            with torch.inference_mode():
                vals.append(float(model(torch.from_numpy(f), torch.from_numpy(valid), torch.from_numpy(fs[:, None]),
                                        torch.from_numpy(spec["spec_feat"][[s]]))[0]))
        assert abs(got[c] - np.mean(vals)) < 1e-5
    assert np.isnan(FS.score_candidates(model, lv, [_spec([10.0], [0.0])])).all()   # 使えるスペクトルが無い → NaN
    assert np.isnan(FS.score_candidates(model, [None, None], sps)).all()


def test_score_holdout_writes_npz_rows_blind_and_sha(tmp_path, monkeypatch):
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
    order = np.argsort(mass, kind="stable")
    meta = pd.DataFrame(dict(key=[f"K{i}" for i in range(7)], smiles=smi, mass=mass, pid=np.arange(7)))
    monkeypatch.setattr(R, "load_pool", lambda root=None: R.Pool(meta, None))
    monkeypatch.setattr(L, "ROOT", root)
    zdir = root / "outputs/holdout_eval/syn_fold0"; zdir.mkdir(parents=True)
    np.savez(zdir / "pc_logits.npz", __targets__=np.array([["m_b", "138.0004"], ["m_a", "92.0"], ["m_c", "46.0"],
                                                           ["m_d", "86.0"]], object))
    tdir = root / "outputs/holdouts/syn"; tdir.mkdir(parents=True)
    (tdir / "truth.csv").write_text("molecule_id,smiles\nm_b,CCO\nm_a,CCO\nm_c,CCO\nm_d,CCO\nm_x,CCO\n")  # m_x はロジット無し
    hdir = root / "data/holdouts/syn"; hdir.mkdir(parents=True)
    (hdir / "hide_rule.json").write_text(json.dumps(dict(rule="all", pool="remove")))
    (hdir / "drop_keys.txt").write_text("K3\n")              # pool: remove → K3 は窓から消える
    rng = np.random.default_rng(5)
    pk = lambda n: (rng.random(n) * 120 + 15, rng.random(n) + 0.05)  # noqa: E731
    _te([("m_b", *pk(20), "[M+H]+", "positive", 139.04, np.array([20.0])),
         ("m_b", *pk(8), "[M+H]+", "positive", 139.04, np.array([40.0])),
         ("m_a", *pk(10), "[M-H]-", "negative", 91.0, np.array([30.0])),
         ("m_c", *pk(6), "[M+H]+", "positive", 47.05, np.array([])),
         ("m_d", np.array([30.0]), np.array([0.0]), "[M-H]-", "negative", 85.0, np.array([10.0]))]
        ).to_parquet(hdir / "test.parquet")
    torch.manual_seed(1)
    wpath = root / "w.pt"
    torch.save(FS.B.build_model().state_dict(), wpath)
    rep = FS.score_holdout("syn", wpath, root / "out", root=root)
    assert rep["weights_sha256"] == hashlib.sha256(wpath.read_bytes()).hexdigest()
    full = np.load(root / "out/fragnn_scores.npz", allow_pickle=True)
    assert list(full["mids"]) == ["m_b", "m_a", "m_c", "m_d"]          # truth の並び、ロジットの無い m_x は無い
    pool = R.load_pool()
    got = SP.load_scores(root / "out/fragnn_scores.npz")
    gotb = SP.load_scores(root / "out/fragnnblind_scores.npz")
    idx_b = R.visible_window(pool, 138.0004, {"K3"})
    assert list(got["m_b"]) == idx_b.tolist() and "K3" not in set(pool.key[idx_b])   # 行 = run.py の窓の Pool の添字
    assert rep["rows"] == sum(len(v) for v in got.values()) == rep["fragnn_scores.npz"]["rows"]
    j_bad = int(np.flatnonzero(pool.smiles == "not_a_smiles")[0])
    assert np.isnan(got["m_b"][j_bad]) and np.isnan(gotb["m_b"][j_bad])   # 読めない候補は NaN
    assert all(np.isnan(v) for v in got["m_d"].values())                  # 使えるスペクトルが無い → NaN
    # blind: 昇順 m_a(-) m_b(+) m_c(+) m_d(-) → m_b→m_c、m_c→m_b、m_a→m_d、m_d→m_a
    assert (rep["blind_same_mode"], rep["blind_any_mode"], rep["blind_self"]) == (4, 0, 0)
    assert all(np.isnan(v) for v in gotb["m_a"].values())                 # 入れ替え先 m_d は使えるスペクトルが無い
    model = FS.load_model(wpath)
    fr = FS.T.import_frag_rescore(root)
    lc = FS.LevelCache(fr, pool.smiles)
    te = pd.read_parquet(hdir / "test.parquet")
    sp = FS.molecule_rows(te)
    want_b = FS.score_candidates(model, [lc(i) for i in idx_b], sp["m_c"])
    want_f = FS.score_candidates(model, [lc(i) for i in idx_b], sp["m_b"])
    gb = np.array([gotb["m_b"][i] for i in idx_b]); gf = np.array([got["m_b"][i] for i in idx_b])
    np.testing.assert_allclose(gb, want_b.astype(np.float32), rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(gf, want_f.astype(np.float32), rtol=1e-6, equal_nan=True)
    assert not np.allclose(gf[np.isfinite(gf)], gb[np.isfinite(gb)])     # 入れ替えで点が変わる
    saved = json.loads((root / "out/fragnn_scores.json").read_text())
    assert saved["weights_sha256"] == rep["weights_sha256"] and "mrr25" not in saved
