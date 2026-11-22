"""Issue #196: fragnn v2（P1・P5'・P2・P8(a)・P4）の純粋な関数・学習の核・前処理・起動の関門（合成データ）。

v1 の既定の動きが変わらないことのテストは tests/test_fragnn_train.py・tests/test_fragnn_bench.py（そのまま通ること）と、
ここの「v1 の経路は merge を呼ばない・形が v1 のまま」のテストで見る。
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train"))
import fragnn_bench as B  # noqa: E402

_spec = importlib.util.spec_from_file_location("fragnn_train", ROOT / "scripts/train/fragnn_train.py")
ft = importlib.util.module_from_spec(_spec)
sys.modules["fragnn_train"] = ft
_spec.loader.exec_module(ft)
_gs = importlib.util.spec_from_file_location("fragnn_gate", ROOT / "scripts/train/fragnn_gate.py")
gate = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(gate)
LOOP = ROOT / "scripts/train/fragnn_loop.sh"
GATE_WAIT = ROOT / "scripts/train/fragnn_gate_wait.sh"


# ---------------------------------------------------------------- 定数（事前登録のとおり）

def test_v2_constants_match_prereg():
    assert B.TEST_ADDUCTS == ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+", "[M-H]-",
                              "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
    assert (B.ABOVE_PREC_DA, B.TOP_PEAKS_V2, B.F_SPEC_V2) == (2.0, 64, 4)
    assert (B.MERGE_P, B.MERGE_KMAX, B.MERGE_TOL, B.MERGE_SEED_TAG, B.TIMSTOF) == (0.3, 4, 0.005, 2, "timsTOF")
    c = ft.CONFIG_V2
    for k in ("LR", "WD", "WARMUP", "CLIP", "BATCH", "K_CAND", "SEED", "EVAL_EVERY", "STOP_REL", "STOP_PATIENCE",
              "MAX_STEPS", "PLATEAU_FACTOR", "PLATEAU_PATIENCE", "MIN_LR", "VAL_FRAC", "VAL_SEED", "VAL_CAND_SEED",
              "PPM", "TOL", "H_SHIFTS", "LEVEL_W", "F_PEAK", "F_PAIR", "PAD_ROWS", "negatives", "CAP_FACTOR",
              "CAP_MAX_H"):
        assert c[k] == ft.CONFIG[k], k                       # 止め方・最適化・特徴は v1 と同じ（#196 (d)）
    assert (c["ISSUE"], c["TOP_PEAKS"], c["F_SPEC"], c["P4_val_merge"]) == (196, 64, 4, False)
    assert (ft.EXPECTED_TRAIN_SPECTRA_V2, ft.EXPECTED_VAL_SPECTRA_V2) == (1_658_625, 33_811)
    assert ft.JUDGE_SETS == ("v3_c1u_t10", "v3_c2_t10", "enveda_v1u_t10", "v3_c2nu_t10")
    assert ft.V1_PREP_SHA256.startswith("b59350da") and ft.V2_SPEED_REPORT_SHA256.startswith("32c51f56")
    assert ft.CONFIG["ISSUE"] == 120 and ft.CONFIG["TOP_PEAKS"] == 32 and ft.CONFIG["F_SPEC"] == 3   # v1 は変えない


def test_v2_wallclock_cap():
    rep = {"ratio": {"v1_real_h": 10.946777777777779}, "per_pair_ratio": 1.6744377000494446}
    assert ft.wallclock_cap_hours_v2(rep) == pytest.approx(1.5 * 10.946777777777779 * 1.6744377000494446)
    assert ft.wallclock_cap_hours_v2(rep) == pytest.approx(27.49, abs=0.01)
    assert ft.wallclock_cap_hours_v2({"ratio": {"v1_real_h": 100.0}, "per_pair_ratio": 1.0}) == 96.0
    p = ROOT / ft.V2_SPEED_REPORT_REL
    if p.exists():
        assert ft.sha256_file(p) == ft.V2_SPEED_REPORT_SHA256
        assert ft.wallclock_cap_hours_v2(json.loads(p.read_text())) == pytest.approx(27.49, abs=0.01)


# ---------------------------------------------------------------- P1・P5'・P2・P8(a)

def test_p1_keeps_only_the_ten_adducts():
    rows = ["[M+H]+", "[2M+H]+", "[2M+Na]+", "[M-H]-", "[3M+H]+", "[M+Cl]-", "[2M-H]-", "[M+H-NH3]+", None]
    keep = [B.keep_adduct(a) for a in rows]
    assert keep == [True, False, False, True, False, True, False, False, False]
    kept = [a for a, k in zip(rows, keep) if k]
    assert not [a for a in kept if a.startswith("[2M") or a.startswith("[3M")]     # [2M…] の行が 0


def test_p5_drop_above_precursor():
    mz, it = B.drop_above_precursor([100.0, 301.9, 302.0, 302.01, 500.0], [1, 2, 3, 4, 5], 300.0)
    assert mz.tolist() == [100.0, 301.9, 302.0] and it.tolist() == [1, 2, 3]
    mz, it = B.drop_above_precursor([100.0, 500.0], [1, 2], float("nan"))
    assert mz.tolist() == [100.0, 500.0]


def test_pack_v2_top64_tims_bit_and_spec_feat():
    rng = np.random.default_rng(0)
    rows = [dict(mz=rng.uniform(50, 400, 120), it=rng.uniform(0.01, 1, 120), prec=300.0, adduct="[M+H]+", mode=1,
                 ce=40.0, instrument="timsTOF"),
            dict(mz=[100.0, 200.0, 350.0], it=[0.5, 1.0, 0.9], prec=300.0, adduct="[M-H]-", mode=-1, ce=0.0,
                 instrument="Orbitrap"),
            dict(mz=[100.0], it=[1.0], prec=150.0, adduct="[M+Na]+", mode=1, ce=None, instrument=None)]
    pk = B.pack_spectra_v2(rows)
    assert pk["mz"].shape == (3, 64) and pk["spec_feat"].shape == (3, 4)
    assert pk["n"].tolist()[1:] == [2, 1] and pk["n"][0] == 64      # 350 > 302 を除いてから上位 64 本
    assert (pk["mz"][0][:64] <= 302.0).all()
    assert pk["tims"].tolist() == [1, 0, 0] and pk["spec_feat"][:, 3].tolist() == [1.0, 0.0, 0.0]  # 印は timsTOF だけ 1
    assert pk["spec_feat"][0, 1] == pytest.approx(np.log1p(64) / np.log(65))
    assert pk["spec_feat"][1].tolist()[:3] == pytest.approx([-1.0, np.log1p(2) / np.log(65), 0.0])
    assert pk["ce"].tolist() == [40.0, 0.0, 0.0]
    # 上位 64 本は v1 の top_peaks と同じ選び方（P5' の後）
    m5, i5 = B.drop_above_precursor(rows[0]["mz"], rows[0]["it"], 300.0)
    ref = B.top_peaks(m5, i5, k=64)
    assert np.array_equal(pk["mz"][0], ref[0]) and np.array_equal(pk["it"][0], ref[1])
    # shift は v1 と同じ表
    v, c, d = B.shift_table("[M-H]-", -1)
    assert np.array_equal(pk["sh_v"][1, :len(v)], v) and pk["sh_n"][1] == len(v)


# ---------------------------------------------------------------- P4

def test_merge_peak_lists_rule():
    a = ([100.0, 200.0, 300.0], [10.0, 5.0, 2.5])          # 最大 1 に正規化 → 1, 0.5, 0.25
    b = ([100.003, 250.0], [1.0, 4.0])                      # → 0.25, 1
    mz, it = B.merge_peak_lists([a, b])
    assert mz.tolist() == [100.0, 200.0, 250.0, 300.0]     # 100.003 は 0.005 Da 未満の隣で弱い方 → 落ちる
    assert it.tolist() == [1.0, 0.5, 1.0, 0.25]
    mz, it = B.merge_peak_lists([([100.0], [1.0]), ([100.004], [3.0])])
    assert mz.tolist() == [100.004]                         # 同じ強さ（1 と 1）なら後ろ（強い方 >=）
    mz, it = B.merge_peak_lists([(np.arange(100, 200, 1.0), np.ones(100))], k=64)
    assert len(mz) == 64


def test_peers_draw():
    rng = np.random.default_rng(0)
    assert B.peers(rng, [5], 5) == [5]
    for _ in range(200):
        p = B.peers(rng, np.arange(10), 3)
        assert 3 in p and 2 <= len(p) <= 4 and len(set(p)) == len(p)
    sizes = {len(B.peers(rng, [1, 2], 1)) for _ in range(50)}
    assert sizes == {2}


def test_group_keys_and_members():
    pos = np.array([0, 0, 0, 1, 1, 0, 0])
    add = np.array([0, 0, 6, 0, 0, 0, 0])
    tims = np.array([1, 1, 1, 1, 0, 0, 1])
    g = B.merge_group_keys(pos, add, tims)
    assert len({g[0], g[2], g[3], g[4], g[5]}) == 5 and g[0] == g[1] == g[6]
    m = B.group_members(g, np.array([0, 1, 2, 3, 4, 6]))      # 5 は学習の行でない
    assert sorted(map(sorted, m.values())) == [[0, 1, 6], [2], [3], [4]]


def _v2_spec(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        k = int(rng.integers(3, 90))
        rows.append(dict(mz=rng.uniform(20, 400, k), it=rng.uniform(0.01, 1, k), prec=380.0 + (i % 3),
                         adduct="[M+H]+" if i % 3 else "[M-H]-", mode=1 if i % 3 else -1,
                         ce=float((i % 4) * 20), instrument="timsTOF" if i % 2 else "QTOF"))
    return B.pack_spectra_v2(rows), rows


def test_merge_batch_deterministic_and_rules():
    spec, rows = _v2_spec()
    pos = np.arange(60) % 5
    add = np.array([B.TEST_ADDUCTS.index(r["adduct"]) for r in rows])
    gkey = B.merge_group_keys(pos, add, spec["tims"])
    groups = B.group_members(gkey, np.arange(60))
    bi = np.arange(0, 60, 2)
    a, na = B.merge_batch(7, bi, spec, gkey, groups)
    b, nb = B.merge_batch(7, bi, spec, gkey, groups)
    assert na == nb and all(np.array_equal(a[k], b[k]) for k in a)          # ステップ番号から決まる
    c, _ = B.merge_batch(8, bi, spec, gkey, groups)
    assert not all(np.array_equal(a[k], c[k]) for k in a)
    # merge しない行は元の行そのもの、precursor・shift はいつも i のもの
    for r, i in enumerate(bi):
        assert a["prec"][r] == spec["prec"][i] and np.array_equal(a["sh_v"][r], spec["sh_v"][i])
        assert a["spec_feat"][r][3] == spec["spec_feat"][i][3] and a["spec_feat"][r][0] == spec["spec_feat"][i][0]
    # p = 0 なら全部元のまま、p = 1 なら群が 2 本以上の行は全部 merge
    z, nz = B.merge_batch(7, bi, spec, gkey, groups, p=0.0)
    assert nz == 0 and all(np.array_equal(z[k], spec[k][bi]) for k in B.SPEC_KEYS)
    o, no = B.merge_batch(7, bi, spec, gkey, groups, p=1.0)
    assert no == sum(len(groups[int(gkey[i])]) > 1 for i in bi)
    # 確率はおよそ 0.3（多数のステップで）
    rate = np.mean([B.merge_batch(s, bi, spec, gkey, groups)[1] / len(bi) for s in range(200)])
    assert 0.25 < rate < 0.35
    # merge した行の CE は値のある本（0 でない）の平均
    m = B.merged_row(spec, [1, 3], 1)
    ces = [x for x in (spec["ce"][1], spec["ce"][3]) if x]
    assert m["ce"] == pytest.approx(np.mean(ces) if ces else 0.0) and m["n"] <= 64


def test_infer_chunks():
    add = ["[M+H]+"] * 6 + ["[M-H]-"] + ["[M+H]+"] * 3
    tims = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    ch = B.infer_chunks(add, tims)
    assert ch == [[0, 1, 3, 4], [5, 7, 8, 9], [2], [6]]
    assert B.infer_chunks([], []) == []
    assert sorted(j for c in ch for j in c) == list(range(10)) and all(1 <= len(c) <= 4 for c in ch)


# ---------------------------------------------------------------- v1 の経路は変わらない

def test_v1_shapes_unchanged():
    torch = pytest.importorskip("torch")
    m = B.build_model()
    assert m.rho[0].in_features == 256 + 1 + 3
    m2 = B.build_model(B.F_SPEC_V2)
    assert m2.rho[0].in_features == 256 + 1 + 4
    torch.manual_seed(42); a = B.build_model()
    torch.manual_seed(42); b = B.build_model(3)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    mask = np.ones((2, 3), bool)
    n, p, cs, sp, slot, valid = ft.assemble(np.zeros((2, 3), np.int64), mask, np.array([0, 1]), np.array([5, 40]))
    assert valid.shape == (p, 32)


def _synthetic_v2(seed=0):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_fragnn_train import _synthetic   # v1 の合成データ（同じ構造・負例）に v2 のスペクトルを載せる
    D = _synthetic(seed=seed)
    rng = np.random.default_rng(seed + 10)
    rows = []
    for i in range(len(D["pos_struct"])):
        k = int(rng.integers(5, 90))
        rows.append(dict(mz=rng.uniform(20, 420, k), it=rng.uniform(0.01, 1, k), prec=410.0,
                         adduct="[M+H]+" if i % 3 else "[M-H]-", mode=1 if i % 3 else -1, ce=20.0,
                         instrument="timsTOF" if i % 2 else "QTOF"))
    D["spec"] = B.pack_spectra_v2(rows)
    add = np.array([B.TEST_ADDUCTS.index(r["adduct"]) for r in rows])
    gkey = B.merge_group_keys(D["pos_struct"], add, D["spec"]["tims"])
    D["v2"] = dict(gkey=gkey, groups=B.group_members(gkey, D["tr_i"]))
    D["config"] = ft.CONFIG_V2
    return D


META = {"keys_json_sha256": "k", "frag_rescore_sha256": "f", "wallclock_cap_s": 1e9}


def test_v1_path_never_merges(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from test_fragnn_train import _synthetic

    def boom(*a, **k):
        raise AssertionError("v1 で merge が呼ばれた")
    monkeypatch.setattr(ft.B, "merge_batch", boom)
    r = ft.run_training(_synthetic(), tmp_path, META, dev="cpu", max_steps=3, eval_every=3,
                        mem_check=lambda: {"avail_gb": None})
    assert r["status"] == "ok" and r["config"] == ft.CONFIG and r["issue"] == 120
    rows = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert "merged_spectra" not in rows[0]


def test_v2_training_runs_resumes_in_order_and_val_is_not_merged(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    D = _synthetic_v2()
    calls = []
    real = B.merge_batch
    monkeypatch.setattr(ft.B, "merge_batch", lambda step, bi, *a, **k: (calls.append((step, tuple(bi))),
                                                                       real(step, bi, *a, **k))[1])
    kw = dict(dev="cpu", max_steps=6, eval_every=3, ckpt_every=2, mem_check=lambda: {"avail_gb": None})
    ra = ft.run_training(D, tmp_path / "a", META, **kw)
    assert ra["status"] == "ok" and ra["issue"] == 196 and ra["config"]["TOP_PEAKS"] == 64
    assert [c[0] for c in calls] == list(range(6))            # merge は学習のバッチだけ（6 ステップ、評価では呼ばない）
    tr = set(D["tr_i"].tolist())
    assert all(set(c[1]) <= tr for c in calls)
    rows = [json.loads(x) for x in (tmp_path / "a/train_log.jsonl").read_text().splitlines()]
    assert all("merged_spectra" in r for r in rows) and sum(r["merged_spectra"] for r in rows) > 0
    st = torch.load(tmp_path / "a/state.pt", map_location="cpu", weights_only=False)
    assert st["model"]["rho.0.weight"].shape[1] == 256 + 1 + 4
    with pytest.raises(ft._Abort):
        ft.run_training(D, tmp_path / "b", META, abort_at=5, **kw)
    rb = ft.run_training(D, tmp_path / "b", META, resume=True, **kw)
    assert rb["resumed_from_step"] == 4 and rb["status"] == "ok"
    pa = torch.load(tmp_path / "a/state.pt", map_location="cpu", weights_only=False)["model"]
    pb = torch.load(tmp_path / "b/state.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(pa[k], pb[k]) for k in pa)          # 再開しても同じ列（merge の乱数もステップから決まる）
    lb = [json.loads(x) for x in (tmp_path / "b/train_log.jsonl").read_text().splitlines()]
    assert [r["val_loss"] for r in rows] == [r["val_loss"] for r in lb]
    cfg = json.loads((tmp_path / "a/fragnn_config.json").read_text())
    assert cfg["ISSUE"] == 196 and cfg["P4_merge_p"] == 0.3


# ---------------------------------------------------------------- 前処理（偽のデータの root、端から端まで）

@pytest.fixture
def isolated_frag_import(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    prev = sys.modules.pop("frag_rescore", None)
    yield
    sys.modules.pop("frag_rescore", None)
    if prev is not None:
        sys.modules["frag_rescore"] = prev


def _v2_roots(tmp_path, monkeypatch):
    """v1 の偽の root で v1 の prep.npz を作り、train.parquet の付加イオンの一部を [2M+H]+ に、装置の列を足す。"""
    pytest.importorskip("torch")
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from test_fragnn_train import _fake_roots
    data, code, sp = _fake_roots(tmp_path)
    for k, v in dict(EXPECTED_DROP=7, EXPECTED_DROP_KEY=9, EXPECTED_POOL_ALLOWED=150, EXPECTED_TRAIN_STRUCTS=98,
                     EXPECTED_VAL_STRUCTS=2, EXPECTED_TRAIN_SPECTRA=490, EXPECTED_VAL_SPECTRA=10,
                     EXPECTED_STRUCTS_DESIGN=100, CODE_ROOT=code, SPEED_REPORT_SHA256=ft.sha256_file(sp)).items():
        monkeypatch.setattr(ft, k, v)      # test_fragnn_train._patch_expected と同じ値（このファイルの ft に当てる）
    v1out = data / "outputs/fragnn_train"
    with pytest.raises(ft._Abort):
        ft.train(data, v1out, device="cpu", resume=True, speed_src=sp, abort_at=1, max_steps=2, eval_every=2,
                 mem_check=lambda: {"avail_gb": None})
    assert (v1out / "prep.npz").exists()
    t = pq.read_table(data / "data/train.parquet").to_pydict()
    n = len(t["adduct"])
    t["adduct"] = ["[2M+H]+" if i % 7 == 0 else a for i, a in enumerate(t["adduct"])]
    t["instrument_type"] = ["timsTOF" if i % 3 == 0 else "QTOF" for i in range(n)]
    t["precursor_mz"] = [150.0 if i % 5 == 0 else p for i, p in enumerate(t["precursor_mz"])]   # P5' が効く行
    pq.write_table(pa.table(t), data / "data/train.parquet", row_group_size=50)
    z = np.load(v1out / "prep.npz")
    keep = np.array([t["adduct"][r] != "[2M+H]+" for r in z["spec_row"]])
    n_tr, n_va = int((keep & ~z["is_val"]).sum()), int((keep & z["is_val"]).sum())
    monkeypatch.setattr(ft, "V1_PREP_SHA256", ft.sha256_file(v1out / "prep.npz"))
    monkeypatch.setattr(ft, "EXPECTED_TRAIN_SPECTRA_V2", n_tr)
    monkeypatch.setattr(ft, "EXPECTED_VAL_SPECTRA_V2", n_va)
    monkeypatch.setattr(ft, "JUDGE_SETS", ("j1", "j2"))
    for h, ik in (("j1", ["R105", "R106"]), ("j2", ["R100", "ZZZZZZZZZZZZZZ"])):
        (data / "outputs/local_eval" / h).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"molecule_id": ["a", "b"], "inchikey14": ik, "smiles": ["C", "C"]}).to_csv(
            data / "outputs/local_eval" / h / "truth.csv", index=False)
        (data / "data/holdouts" / h).mkdir(parents=True, exist_ok=True)
        (data / "data/holdouts" / h / "drop_keys.txt").write_text("T105\nT100\n")
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps({"ratio": {"v1_real_h": 2.0}, "per_pair_ratio": 1.5}))
    monkeypatch.setattr(ft, "V2_SPEED_REPORT_SHA256", ft.sha256_file(bench))
    return data, code, bench, t, z, keep


def test_v2_train_end_to_end_on_fake_roots(tmp_path, monkeypatch, isolated_frag_import):
    data, code, bench, t, z, keep = _v2_roots(tmp_path, monkeypatch)
    prep_sha = ft.sha256_file(data / "outputs/fragnn_train/prep.npz")
    out = tmp_path / "v2out"
    kw = dict(max_steps=4, eval_every=2, mem_check=lambda: {"avail_gb": None})
    with pytest.raises(ft._Abort):
        ft.train(data, out, device="cpu", resume=True, speed_src=bench, v2=True, abort_at=3, **kw)
    r = ft.train(data, out, device="cpu", resume=True, speed_src=bench, v2=True, **kw)
    assert r["status"] == "ok", r.get("error")
    assert r["resumed_from_step"] == 2 and r["train"]["steps"] == 4 and r["issue"] == 196
    assert r["wallclock_cap_h"] == pytest.approx(1.5 * 2.0 * 1.5)
    assert r["meta"]["v1_prep_sha256"] == prep_sha and r["meta"]["config"]["variant"] == "v2"
    assert ft.sha256_file(data / "outputs/fragnn_train/prep.npz") == prep_sha      # v1 の prep は書き換えない
    d = json.loads((out / "prep_v2.json").read_text())
    assert d["n_dropped_p1"] == int((~keep).sum()) > 0
    assert d["n_train_spectra"] + d["n_val_spectra"] == int(keep.sum())
    assert all(v == 0 for v in d["leak"].values()) and len(d["leak"]) == 7
    assert set(d["judge_sets"]) == {"j1", "j2"}
    assert all(v == 0 for j in d["judge_sets"].values() for k, v in j.items() if k.endswith(("_pos", "_neg")))
    assert d["merge_groups_train"]["n_groups"] > 0 and 0 < d["tims_share"] < 1
    # P1 の後のスペクトルの読み込みを直接確かめる: [2M…] は 0 行、P5' の後、印は timsTOF だけ 1
    kept = np.flatnonzero(keep)
    rows = z["spec_row"][kept]
    sp = ft.load_spectra_v2(data / "data/train.parquet", rows, z["spec_ik"][kept])
    assert not any(t["adduct"][r].startswith("[2M") for r in rows)
    assert sp["tims"].tolist() == [1 if t["instrument_type"][r] == "timsTOF" else 0 for r in rows]
    for j, r in enumerate(rows[:40]):
        assert (sp["mz"][j, :sp["n"][j]] <= t["precursor_mz"][r] + 2.0).all()
    with pytest.raises(AssertionError, match="outside the 10"):
        ft.load_spectra_v2(data / "data/train.parquet", z["spec_row"][~keep][:3], z["spec_ik"][~keep][:3])


def test_v2_prepare_stops_on_judge_set_leak(tmp_path, monkeypatch, isolated_frag_import):
    import pandas as pd
    data, code, bench, t, z, keep = _v2_roots(tmp_path, monkeypatch)
    # 学習の正解の構造（R000）を判定の集合の正解に入れる → 止まる
    pd.DataFrame({"molecule_id": ["a"], "inchikey14": ["R000"], "smiles": ["C"]}).to_csv(
        data / "outputs/local_eval/j1/truth.csv", index=False)
    with pytest.raises(AssertionError, match="judge-set"):
        ft.prepare_v2(data, code / ft.KEYS_REL, data / "outputs/fragnn_train/prep.npz")
    r = ft.train(data, tmp_path / "o", device="cpu", resume=True, speed_src=bench, v2=True, max_steps=2,
                 eval_every=2, mem_check=lambda: {"avail_gb": None})
    assert r["status"] == "error" and "judge-set" in r["error"] and not (tmp_path / "o/state.pt").exists()


def test_v2_prepare_checks_prep_sha_and_counts(tmp_path, monkeypatch, isolated_frag_import):
    data, code, bench, t, z, keep = _v2_roots(tmp_path, monkeypatch)
    prep = data / "outputs/fragnn_train/prep.npz"
    monkeypatch.setattr(ft, "V1_PREP_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="prep.npz sha256"):
        ft.prepare_v2(data, code / ft.KEYS_REL, prep)
    monkeypatch.setattr(ft, "V1_PREP_SHA256", ft.sha256_file(prep))
    monkeypatch.setattr(ft, "EXPECTED_TRAIN_SPECTRA_V2", 1)
    with pytest.raises(AssertionError):
        ft.prepare_v2(data, code / ft.KEYS_REL, prep)


def test_judge_set_overlap():
    o = ft.judge_set_overlap(["A", "B"], ["kX"], {"A": "kA"}, pos_raw={"B"}, pos_key={"kA"}, neg_key={"kX"},
                             neg_raw=set())
    assert o == dict(raw_in_pos=1, raw_in_neg=0, key_in_pos=1, key_in_neg=1)
    assert ft.judge_set_overlap(["A"], [], {}, set(), set(), set(), set()) == dict(raw_in_pos=0, raw_in_neg=0,
                                                                                    key_in_pos=0, key_in_neg=0)


def test_cli_has_v2_flag():
    import inspect
    src = inspect.getsource(ft.main)
    assert '"--v2"' in src and "v2=a.v2" in src


# ---------------------------------------------------------------- 再開ループ（memguard の 99 で止める）と起動の関門

FAKE_GUARD = """#!/bin/bash
LOG="$1"; shift
if [ -n "${FAKE_GUARD_KILL:-}" ]; then echo "memguard: free 1% < 15% -> kill" >> "$LOG"; exit 99; fi
"$@" > "$LOG" 2>&1; rc=$?
echo "2026-01-01 00:00:00 exit=$rc" >> "$LOG"
exit $rc
"""


def _guard(tmp_path):
    g = tmp_path / "guard.sh"
    g.write_text(FAKE_GUARD); g.chmod(g.stat().st_mode | stat.S_IEXEC)
    return g


def test_loop_stop_on_memguard(tmp_path):
    g = _guard(tmp_path)
    base = dict(os.environ, MEMGUARD=str(g), MAX_RESTARTS="2", WAIT_S="0", LOOP_PY=sys.executable, FAKE_GUARD_KILL="1")
    p = subprocess.run(["bash", str(LOOP), str(tmp_path / "s"), str(tmp_path / "r.json"), "--", sys.executable, "-c",
                        "pass"], env=dict(base, STOP_ON_MEMGUARD="1"), capture_output=True, text=True, timeout=60)
    loop = (tmp_path / "s.loop").read_text()
    assert p.returncode == 2 and loop.count("round") == 2 and "decision=stop memguard_kill" in loop
    assert "stopped (stop memguard_kill)" in loop
    p = subprocess.run(["bash", str(LOOP), str(tmp_path / "d"), str(tmp_path / "r.json"), "--", sys.executable, "-c",
                        "pass"], env=base, capture_output=True, text=True, timeout=60)
    assert p.returncode == 3 and "resume memguard_kill" in (tmp_path / "d.loop").read_text()   # 既定は v1 のまま


def test_gate_parsers_and_judge():
    assert gate.parse_memory_pressure("System-wide memory free percentage: 48%") == 48
    assert gate.parse_memory_pressure("") is None
    assert gate.parse_swap_free_gb("vm.swapusage: total = 25600.00M  used = 24626.50M  free = 973.50M  (encrypted)") \
        == pytest.approx(973.5 / 1024)
    assert gate.parse_swap_free_gb("total = 1G used = 0G free = 2.00G") == pytest.approx(2.0)
    vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:  100000.\n"
          "Pages active: 5.\nPages inactive:  500000.\nPages speculative:  10000.\n")
    assert gate.parse_vm_stat_avail_gb(vm) == pytest.approx(round(610000 * 16384 / 1e9, 2))
    ok = gate.judge(30, 11.0, 0.6, [], [])
    assert ok["ok"] and all(ok["checks"].values())
    assert (gate.MIN_MP_FREE_PCT, gate.MIN_AVAIL_GB, gate.MIN_SWAP_FREE_GB, gate.BIG_RSS_GIB) == (25, 10.9, 0.5, 3.0)
    for args in ((24, 11.0, 0.6, [], []), (30, 10.8, 0.6, [], []), (30, 11.0, 0.4, [], []), (30, 11.0, 0.6, ["x"], []),
                 (30, 11.0, 0.6, [], ["y"]), (None, 11.0, 0.6, [], []), (30, None, 0.6, [], []), (30, 11.0, None, [], [])):
        assert not gate.judge(*args)["ok"], args


def test_gate_classify():
    lines = [
        "  101 7000000 /Users/x/Generate_picture/.venv/bin/python /Users/x/.venv/bin/moss-mlx --batch a",
        "  102 4000000 /Users/x/Molecule ID From Mass Spectra/.venv/bin/python scripts/train/fragnn_train.py --v2",
        "  103 4000000 bash fragnn_loop.sh a b -- /x/.venv/bin/python fragnn_train.py",
        "  104 9000000 /Applications/Google Chrome Helper (Renderer)",
        "  105 100000 /usr/bin/python3 other_small.py",
        "  106 5000000 /usr/bin/time -l /x/bin/python fragnn_train.py",
        "  107 3984588 /opt/homebrew/Cellar/python@3.12/3.12.7/Frameworks/Python.framework/Versions/3.12/Resources/"
        "Python.app/Contents/MacOS/Python tools/generate.py",
        "  108 4000000 /Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python "
        "/x/fragnn_train.py --v2",
        "  109 4000000 /Users/x/My Env/.venv/bin/Python3 big.py",
    ]
    c = gate.classify(lines, me=-1)
    assert [x.split()[0] for x in c["own_heavy"]] == ["102", "108"]
    assert [x.split()[0] for x in c["big_other"]] == ["101", "107", "109"]   # framework 版の Python も数える
    assert [x.split()[0] for x in gate.classify(lines, me=101)["big_other"]] == ["107", "109"]


def _fake_gate(tmp_path, rc):
    p = tmp_path / f"gate_{rc}.py"
    p.write_text(f"import sys; print('GATE fake'); sys.exit({rc})\n")
    return p


def test_gate_wait_starts_loop_when_ok(tmp_path):
    g = _guard(tmp_path)
    rep = tmp_path / "r.json"
    W = "import json,sys; json.dump({'status': 'ok', 'train': {'stop_reason': 'converged'}}, open(sys.argv[1], 'w'))"
    env = dict(os.environ, MEMGUARD=str(g), WAIT_S="0", LOOP_PY=sys.executable, GATE_PY=sys.executable,
               GATE_SCRIPT=str(_fake_gate(tmp_path, 0)), GATE_EVERY_S="1", GATE_MAX_S="5")
    p = subprocess.run(["bash", str(GATE_WAIT), str(tmp_path / "t"), str(rep), "--", sys.executable, "-c", W,
                        str(rep)], env=env, capture_output=True, text=True, timeout=60)
    loop = (tmp_path / "t.loop").read_text()
    assert p.returncode == 0 and "gate ok -> start" in loop and "finished (done converged)" in loop
    assert "GATE fake" in (tmp_path / "t.gate").read_text()


def test_gate_wait_times_out_without_starting(tmp_path):
    env = dict(os.environ, GATE_PY=sys.executable, GATE_SCRIPT=str(_fake_gate(tmp_path, 1)), GATE_EVERY_S="1",
               GATE_MAX_S="2")
    p = subprocess.run(["bash", str(GATE_WAIT), str(tmp_path / "t"), str(tmp_path / "r.json"), "--", sys.executable,
                        "-c", "open('SHOULD_NOT_RUN','w')"], env=env, capture_output=True, text=True, timeout=60,
                       cwd=tmp_path)
    loop = (tmp_path / "t.loop").read_text()
    assert p.returncode == 5 and "gate_timeout" in loop and "round" not in loop
    assert not (tmp_path / "SHOULD_NOT_RUN").exists() and (tmp_path / "t.gate").read_text().count("GATE fake") >= 1
