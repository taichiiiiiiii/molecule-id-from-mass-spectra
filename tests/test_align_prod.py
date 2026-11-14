"""整列チャネルの本番（scripts/train/align_{query_manifest,embed,train}.py、make_align_prod_kernels.py、
scripts/local_eval/align.py、run.py の align 段、scripts/cv/align_v3_check.py）を小さな合成データで確かめる。

torch の要る部分は importorskip（CI は torch を入れない。ローカルの .venv では走る）。
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts/train", "scripts/local_eval", "scripts/cv", "scripts/validate", "scripts/ranker"):
    sys.path.insert(0, str(ROOT / p))

import align_query_manifest as QM  # noqa: E402
import align_embed as E  # noqa: E402
import align_train as T  # noqa: E402
import make_align_prod_kernels as MK  # noqa: E402
import align as AL  # noqa: E402
import align_v3_check as AC  # noqa: E402


# ---------------------------------------------------------------- 問い合わせの表

def test_spectrum_sig_is_exact():
    a = QM.spectrum_sig(100.0, "[M+H]+", [1.0, 2.0], [0.5, 1.0])
    assert a == QM.spectrum_sig(100.0, "[M+H]+", np.array([1.0, 2.0]), np.array([0.5, 1.0]))
    assert a != QM.spectrum_sig(100.0, "[M+H]+", [1.0, 2.0000001], [0.5, 1.0])
    assert a != QM.spectrum_sig(100.0, "[M+Na]+", [1.0, 2.0], [0.5, 1.0])


def test_assign_rows_prefers_truth_key_and_unused_rows():
    q = [["h1", "m1", "s1", None, "TRUE", "sigA"], ["h1", "m1", "s2", None, "TRUE", "sigA"],
         ["h1", "m2", "s3", None, "CANON", "sigB"], ["h2", "m9", "s4", None, "X", "sigA"]]
    found = {"sigA": [(7, "TRUE"), (3, "TRUE"), (1, "OTHER")], "sigB": [(5, "RAW")]}
    drop = {"h1": {"TRUE", "CANON", "RAW", "OTHER"}, "h2": {"Y"}}
    assert QM.assign_rows(q, found, drop) == 1          # h2 の行は drop に無いキーしか無い
    assert q[0][3:5] == [3, "TRUE"] and q[1][3:5] == [7, "TRUE"]   # 同じキーの未使用の行を順に
    assert q[2][3:5] == [5, "RAW"]                     # 互変異性体の生キー（drop にある）でも取る


def test_committed_manifest_shape():
    m = json.loads(MK.MANIFEST.read_text())
    keys = json.loads(MK.KEYS.read_text())
    assert m["train_rows"] == E.TRAIN_ROWS and m["n"] == E.N_QUERY == len(m["rows"])
    per = pd.Series([r[0] for r in m["rows"]]).value_counts().to_dict()
    assert per == {"v3_c1": 1252, "v3_c2": 1215, "enveda_v1": 614, "v3_c2n": 1675}
    for h, mid, sid, row, ik, sig in m["rows"]:
        assert isinstance(row, int) and 0 <= row < E.TRAIN_ROWS and len(sig) == 16
        assert ik in set(keys[h])   # 問い合わせの行は学習から外れる（DROP に入る）


# ---------------------------------------------------------------- 埋め込み（純粋な部分）

def test_ce_value_and_classify():
    assert E.ce_value([20.0, 40.0]) == 20.0 and math.isnan(E.ce_value(None)) and math.isnan(E.ce_value([]))
    assert math.isnan(E.ce_value([float("nan")])) and E.ce_value(35) == 35.0
    q = {5: [("v3_c1", "m", "s", "K", "g")]}
    assert E.classify(5, "K", q, {"K"})[0] == 1
    assert E.classify(6, "K", q, {"K"}) == (None, None)
    assert E.classify(7, "Z", q, {"K"}) == (0, None)


def test_offsets_shards_and_query_rows():
    assert E.row_group_offsets([3, 4, 5]) == [0, 3, 7]
    assert E.assign_shards(5, 2, 0) == [0, 2, 4] and E.assign_shards(5, 2, 1) == [1, 3]
    qr = E.query_rows({"rows": [["a", "m", "s1", 9, "K", "x"], ["b", "n", "s2", 9, "K", "x"]]})
    assert list(qr) == [9] and len(qr[9]) == 2


def test_config_hash_and_done_marker(tmp_path):
    rows = [["v3_c1", "m", "s", 1, "K", "abc"]]
    h1 = E.config_hash(rows, 10)
    assert h1 == E.config_hash(rows, 10) and h1 != E.config_hash(rows, 11)
    assert not E.shard_is_done(tmp_path, 0, h1)
    n = E.shard_names(0)
    np.save(tmp_path / n["emb"], np.zeros((1, 4), np.float16))
    (tmp_path / n["idx"]).write_bytes(b"x")
    (tmp_path / n["done"]).write_text(json.dumps({"status": "done", "config_hash": h1}))
    assert E.shard_is_done(tmp_path, 0, h1) and not E.shard_is_done(tmp_path, 0, "other")
    assert E.find_resume_dirs(tmp_path) == [tmp_path] and E.find_resume_dirs(tmp_path / "nope") == []


# ---------------------------------------------------------------- 学習（純粋な部分）

def test_config_is_massspecgym_formula_yaml():
    assert (T.D_SHARED, T.D_HIDDEN, T.D_META) == (512, 2048, 64)
    assert (T.N_MS_PRE, T.N_MS_POST, T.N_MOL) == (0, 2, 3)
    assert (T.DROPOUT, T.TEMP_INIT, T.K_CAND, T.BATCH) == (0.30, 0.07, 128, 128)
    assert (T.LR, T.WD, T.WARMUP, T.CLIP, T.SEED) == (1e-4, 1e-6, 2000, 5.0, 42)
    # #109 の止め方
    assert (T.MAX_STEPS, T.EVAL_EVERY, T.VAL_FRAC, T.VAL_SEED, T.STOP_REL, T.STOP_PATIENCE) == (300_000, 2000, 0.02, 0, 0.005, 5)
    assert (T.PLATEAU_FACTOR, T.PLATEAU_PATIENCE, T.MIN_LR) == (0.5, 3, 1e-6)


def test_epoch_batch_is_step_indexed():
    a = [T.epoch_batch(s, 10, 4, 42) for s in range(5)]
    assert all(len(b) == 4 for b in a)
    assert len(set(np.concatenate(a[:2]))) == 8 and not np.array_equal(a[0], a[2])   # 1 エポック = 2 バッチ、端数は捨てる
    T._PERM_CACHE.clear()
    assert np.array_equal(T.epoch_batch(3, 10, 4, 42), a[3])                          # 再開しても同じ
    assert T.step_rng(7).integers(1 << 30) == T.step_rng(7).integers(1 << 30)
    with pytest.raises(ValueError):
        T.epoch_batch(0, 3, 4)


def test_split_val_keys():
    keys = [f"K{i:04d}" for i in range(1000)] * 3
    v = T.split_val_keys(keys)
    assert len(v) == 20 and v == T.split_val_keys(list(reversed(keys))) and v <= set(keys)


def test_early_stop_rule():
    es = T.EarlyStop()
    seq = [5.0, 4.0, 3.99, 3.985, 3.99, 3.98, 3.979]   # 4.0 のあと最良からの相対改善 < 0.5% が 5 回
    out = [es.update(x, i) for i, x in enumerate(seq)]
    assert [o["stop"] for o in out] == [False] * 6 + [True]
    assert es.best == 3.979 and es.best_step == 6               # 重みは最小の値の評価
    es2 = T.EarlyStop(); es2.update(5.0, 0); es2.update(4.9, 1)   # 2% の改善は数を戻す
    assert es2.bad == 0 and es2.update(4.88, 2)["bad"] == 1
    es3 = T.EarlyStop(); es3.load(es.state()); assert es3.state() == es.state()


def test_warmup_lr():
    assert T.warmup_lr(0) == 0.0 and T.warmup_lr(1000) == 0.5e-4 and T.warmup_lr(5000) == 1e-4


def test_free_gb_parses_vm_stat(monkeypatch):
    import subprocess
    txt = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      100000.\n"
           "Pages active:   5.\nPages inactive:  200000.\nPages speculative:  50000.\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"stdout": txt})())
    assert T.free_gb() == {"free_gb": round(150000 * 16384 / 1e9, 2), "inactive_gb": round(200000 * 16384 / 1e9, 2)}
    assert T.FP_BITS == 10_226 and T.PPM == 10.0 and E.PEAKS == 100


def test_adduct_vocab_and_ids():
    vocab, signs = T.adduct_vocab(["[M+H]+", "[M+HCOO]-", "[M-H]-", None, "[M+H]+", "odd"])
    assert vocab == ["[M+HCOOH-H]-", "[M+H]+", "[M-H]-", "odd"] and signs == [-1, 1, -1, 0, 0]
    assert T.adduct_ids(["[M+FA-H]-", "[M+K]+", "[M+H]+"], vocab).tolist() == [0, 4, 1]


def test_drop_key_space():
    out = T.drop_key_space({"RAW1", "RAW2"}, {"RAW1": "CAN1", "RAW3": "CAN3"}, {"h": ["HK"]}, ["SK"])
    assert out == {"SK", "HK", "CAN1"}


def _brute(pos, mass, allowed, ppm):
    out = []
    for p in pos:
        tol = mass[p] * ppm * 1e-6
        out.append([i for i in range(len(mass)) if abs(mass[i] - mass[p]) <= tol + 1e-12 and allowed[i] and i != p])
    return out


def test_negative_csr_matches_bruteforce():
    rng = np.random.default_rng(0)
    mass = np.sort(rng.uniform(100, 101, 300))
    allowed = rng.random(300) > 0.2
    pos = np.array([0, 10, 150, 299])
    off, flat = T.negative_csr(pos, mass, allowed, ppm=2000.0)
    for j, want in enumerate(_brute(pos, mass, allowed, 2000.0)):
        assert sorted(flat[off[j]:off[j + 1]].tolist()) == want


def test_sample_candidates_shape_mask_and_determinism():
    off = np.array([0, 3, 3, 203])
    flat = np.concatenate([[11, 12, 13], np.arange(1000, 1200)])
    pos = np.array([1, 2, 3]); cid = np.array([0, 1, 2])
    c1, m1 = T.sample_candidates(np.random.default_rng(5), pos, cid, off, flat, k=128)
    c2, m2 = T.sample_candidates(np.random.default_rng(5), pos, cid, off, flat, k=128)
    assert (c1 == c2).all() and (m1 == m2).all()
    assert (c1[:, 0] == pos).all() and m1[:, 0].all()
    assert m1.sum(1).tolist() == [4, 1, 128]                       # 3 個 / 0 個 / 200 個から 127 個
    assert set(c1[0, 1:4]) == {11, 12, 13} and len(set(c1[2, 1:])) == 127 and set(c1[2, 1:]) <= set(range(1000, 1200))
    assert (c1[~m1] == 0).all()


def test_unpack_bits_np_matches_pool_fps():
    bits = np.random.default_rng(1).integers(0, 2, (5, 21)).astype(np.uint8)
    assert (T.unpack_bits_np(np.packbits(bits, axis=1), 21) == bits).all()


# ---------------------------------------------------------------- カーネルの生成物

@pytest.mark.parametrize("build,kdir,meta,mods", [
    (MK.build_embed_notebook, MK.EMBED_KDIR, MK.embed_metadata, MK.EMBED_MODULES)])
def test_kernels_deterministic_compile_and_committed(build, kdir, meta, mods):
    nb = build()
    assert nb == build()
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    scope: dict = {}
    exec(compile("".join(code[0]["source"]), "cell0", "exec"), scope)
    assert sorted(scope["SRC"]) == sorted(mods)
    for n, s in scope["SRC"].items():
        assert s == (ROOT / "scripts/train" / n).read_text()
        compile(s, n, "exec")
    assert scope["HOLDOUT_KEYS"] == json.loads(MK.KEYS.read_text())
    for c in code[1:]:
        compile("".join(c["source"]), "cell", "exec")
    assert json.loads((kdir / "notebook.ipynb").read_text()) == nb
    assert json.loads((kdir / "kernel-metadata.json").read_text()) == meta()


def test_metadata_sources():
    e = MK.embed_metadata()
    assert e["enable_internet"] is True and e["dataset_sources"] == ["dmitriigluzdov/casmi26-fold-safe-fpnet"]
    assert e["kernel_sources"] == [] and MK.embed_metadata(resume=True)["kernel_sources"] == [MK.EMBED_SLUG]
    assert not (ROOT / "kernels/casmi26-align-train").exists()     # 学習はローカルだけ（#109）
    for m in (e,):
        assert m["is_private"] and m["enable_gpu"] and m["machine_shape"] == "NvidiaTeslaT4" and "id_no" not in m


# ---------------------------------------------------------------- ローカルの align 段と判定

def test_molecule_vectors_and_window_scores():
    g = np.array([[1, 0], [0, 1], [1, 0], [0.6, 0.8]], np.float32)
    vec = AL.molecule_vectors(g, ["a", "a", "b", "a"], ["m1", "m1", "m1", "m2"], "a")
    assert np.allclose(vec["m1"], [2 ** -0.5, 2 ** -0.5]) and np.allclose(vec["m2"], [0.6, 0.8]) and len(vec) == 2
    h = np.array([[1, 0], [0, 1], [0.6, 0.8]], np.float16)
    s = AL.window_scores(np.array([2, 0]), vec["m2"], h)
    assert np.allclose(s, [1.0, 0.6], atol=1e-3)
    assert np.isnan(AL.window_scores(np.array([0]), None, h)).all()


def test_run_has_align_stage_using_sanpier_mix():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    assert 'if "align" in stages:' in src and "align_scores.npz" in src
    assert "rows = sanpier_mix(rows, ascores.get(mid, {}))" in src
    import run as R
    assert R.sanpier_mix([(3.0, 1), (2.0, 2)], {}) == [(3.0, 1), (2.0, 2)]   # 点の無い分子は s のまま


def test_align_judge_rule():
    ok = {h: {"lo": 0.001} for h in AC.REQUIRED}
    assert AC.judge(ok)["adopt"]
    no_c1 = dict(ok, v3_c1={"lo": 0.0}, v3_c2={"lo": 0.05})
    assert not AC.judge(no_c1)["adopt"]                       # v3_c2 の勝ちでは採用しない
    assert not AC.judge(dict(ok, enveda_v1={"lo": -0.021}))["adopt"]
    assert AC.judge(dict(ok, v3_c2={"lo": -0.02}))["adopt"]    # ちょうど −0.02 は可
    part = {h: v for h, v in ok.items() if h != "v3_c2n"}
    assert not AC.judge(part)["adopt"] and AC.judge(part)["missing"] == ["v3_c2n"]
    assert AC.NEW_TAG == "score+align+prior+frag" and AC.CTRL_TAG == "score+prior+frag"


# ---------------------------------------------------------------- torch の要る部分（ローカル）

def test_unpack_bits_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    bits = np.random.default_rng(2).integers(0, 2, (7, 10_226)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    got = T.unpack_bits_torch(torch.from_numpy(packed)).numpy()
    assert (got == bits).all()


def test_model_scores_mask_and_param_count():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    m = T.build_model(4, [1, -1, 0, 0], d_ms=16, d_mol=40).eval()
    mask = torch.tensor([[True, True, False], [True, False, False]])
    mols = torch.rand(int(mask.sum()), 40)
    s = m(torch.rand(2, 16), torch.tensor([20.0, float("nan")]), torch.tensor([0, 3]), mols, mask)
    assert s.shape == (2, 3) and torch.isinf(s[~mask]).all() and torch.isfinite(s[mask]).all()
    assert abs(float(m.log_temperature.detach().exp()) - 0.07) < 1e-6
    names = {n for n, _ in m.named_parameters()}
    assert "mol.network.1.weight" in names                       # Linear → LayerNorm → GELU → Dropout
    assert len([x for x in m.mol.network if isinstance(x, torch.nn.Linear)]) == 4
    assert len([x for x in m.ms_post.network if isinstance(x, torch.nn.Linear)]) == 3
    assert len([x for x in m.ms_pre.network if isinstance(x, torch.nn.Linear)]) == 1


class _FakeSP:
    def __init__(self, dformat=None, n_highest_peaks=100):
        self.n = n_highest_peaks

    def __call__(self, arr, prec_mz, high_form=False):
        if prec_mz < 0:
            raise ValueError("bad")
        out = np.zeros((self.n + 1, 2))
        out[0] = [prec_mz, 1.0]
        k = min(self.n, arr.shape[1])
        out[1:1 + k] = arr[:, :k].T
        return out


def _fake_dreams(torch):
    class FF(torch.nn.Module):
        def forward(self, x):
            return x.sum(-1, keepdim=True).repeat(1, 1, E.D_EMB)

    class M(torch.nn.Module):
        n_layers = 1

        def __init__(self):
            super().__init__()
            self.transformer_encoder = torch.nn.Module()
            self.transformer_encoder.ffs = torch.nn.ModuleList([FF()])

        def forward(self, x):
            return self.transformer_encoder.ffs[0](x)
    return M()


def _train_parquet(path, iks, n_rg):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(iks)
    tab = pa.table({"inchikey14": iks, "ms2_mzs": [[50.0 + i, 60.0] if i % 7 else [] for i in range(n)],
                    "ms2_normalized_intensities": [[1.0, 0.5] if i % 7 else [] for i in range(n)],
                    "precursor_mz": [100.0 + i for i in range(n)],
                    "collision_energy_ev": [[20.0] if i % 2 else None for i in range(n)],
                    "adduct": ["[M+H]+" if i % 3 else "[M-H]-" for i in range(n)]})
    pq.write_table(tab, path, row_group_size=math.ceil(n / n_rg))


def test_embed_shard_writes_shards_and_marks_queries(tmp_path):
    torch = pytest.importorskip("torch")
    import pyarrow.parquet as pq
    iks = [f"K{i % 9}" for i in range(30)]
    _train_parquet(tmp_path / "train.parquet", iks, 2)
    pf = pq.ParquetFile(tmp_path / "train.parquet")
    offs = E.row_group_offsets([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    sig = QM.spectrum_sig(103.0, "[M-H]-", [53.0, 60.0], [1.0, 0.5])   # 行 3: 3 % 3 == 0 → [M-H]-
    q = E.query_rows({"rows": [["v3_c1", "m1", "s1", 3, "K3", sig], ["v3_c2", "m2", "s2", 3, "K3", sig]]})
    out = tmp_path / "out"; out.mkdir()
    infos = [E.embed_shard(k, _fake_dreams(torch), _FakeSP, object, pf, offs[k], q, {"K3", "K4"}, out, "cpu", "H", time.time())
             for k in range(2)]
    assert all(E.shard_is_done(out, k, "H") for k in range(2))
    idx = pd.concat([pd.read_parquet(out / E.shard_names(k)["idx"]) for k in range(2)])
    emb = np.concatenate([np.load(out / E.shard_names(k)["emb"]) for k in range(2)])
    assert len(idx) == len(emb) and emb.dtype == np.float16 and emb.shape[1] == E.D_EMB
    assert sorted(idx[idx.kind == 1].holdout) == ["v3_c1", "v3_c2"] and (idx[idx.kind == 1].row == 3).all()
    tr = idx[idx.kind == 0]
    assert not tr.inchikey14.isin({"K3", "K4"}).any()
    keep = [i for i in range(30) if i % 7 and iks[i] not in {"K3", "K4"}]
    assert sorted(tr.row) == keep
    assert sum(i["query_sig_mismatch"] for i in infos) == 0 and sum(i["query_ik_mismatch"] for i in infos) == 0
    assert sum(i["empty"] for i in infos) == len([i for i in range(30) if i % 7 == 0 and iks[i] not in {"K3", "K4"}])
    r = idx.set_index("row").loc[1]
    assert keep[0] == 1 and r.ce == 20.0 and r.precursor_mz == 101.0 and r.adduct == "[M+H]+"
    assert math.isnan(idx.set_index("row").loc[2].ce)


def _synthetic_train_inputs(tmp_path, monkeypatch):
    import hashlib
    n_pool = 120
    keys = [f"P{i:03d}" for i in range(n_pool)]
    mass = np.sort(np.random.default_rng(0).uniform(300, 300.05, n_pool))
    root = tmp_path / "repo"
    pool = root / T.DEFAULTS["pool_dir"]; pool.mkdir(parents=True)
    pd.DataFrame({"key": keys, "mass": mass}).to_parquet(pool / "pool_meta.parquet")
    nb = 40
    bits = np.random.default_rng(1).integers(0, 2, (n_pool, nb)).astype(np.uint8)
    np.save(pool / "pool_fp.npy", np.packbits(bits, axis=1))
    np.save(pool / "fp_bits.npy", np.arange(nb))
    raw = [f"R{i:03d}" for i in range(n_pool)]
    pd.DataFrame({"inchikey14": raw, "key": keys}).to_parquet(pool / "train_structs.parquet")
    (root / T.DEFAULTS["split"]).parent.mkdir(parents=True)
    pd.DataFrame({"inchikey14": raw[:5], "key": keys[:5], "fold": ["hold"] * 4 + ["np"]}).to_parquet(root / T.DEFAULTS["split"])
    hk = {"v3_c1": ["P005"], "v3_c2": ["P006"], "enveda_v1": ["R007", "P007"], "v3_c2n": ["P008"]}
    emb_dir = root / T.DEFAULTS["embed_dir"]; emb_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = []
    for i in range(9, n_pool):
        for _ in range(4):
            rows.append((len(rows), 0, raw[i], "[M+H]+" if i % 2 else "[M-H]-", 20.0, 301.0, None, None, None))
    for j, (h, k) in enumerate([("v3_c1", 5), ("v3_c2", 6), ("enveda_v1", 7)]):
        rows.append((10_000 + j, 1, raw[k], "[M+H]+", float("nan"), 301.0, h, f"m{j}", f"s{j}"))
    idx = pd.DataFrame(rows, columns=list(E.IDX_COLS))
    half = len(idx) // 2
    for k, part in enumerate((idx.iloc[:half], idx.iloc[half:])):
        n = E.shard_names(k)
        part.to_parquet(emb_dir / n["idx"], index=False)
        np.save(emb_dir / n["emb"], rng.normal(size=(len(part), T.D_EMB)).astype(np.float16))
        (emb_dir / n["done"]).write_text(json.dumps({"status": "done", "config_hash": "CH", "dtype": "float16"}))
    (emb_dir / "embed_report.json").write_text(json.dumps({"status": "ok", "config_hash": "CH", "n_shards": 2,
                                                           "n_train_spectra": int((idx.kind == 0).sum())}))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    monkeypatch.setattr(T, "POOL_META_SHA256", sha(pool / "pool_meta.parquet"))
    monkeypatch.setattr(T, "TRAIN_STRUCTS_SHA256", sha(pool / "train_structs.parquet"))
    monkeypatch.setattr(T, "FP_BITS_SHA256", sha(pool / "fp_bits.npy"))
    monkeypatch.setattr(T, "POOL_FP_SHA256", sha(pool / "pool_fp.npy"))
    monkeypatch.setattr(T, "POOL_ROWS", n_pool)
    monkeypatch.setattr(T, "FP_BITS", nb)
    monkeypatch.setattr(T, "EXPECTED_DROP", len(set(raw[:5]) | {"P005", "P006", "R007", "P007", "P008"}))
    monkeypatch.setattr(T, "BATCH", 8)
    monkeypatch.setattr(T, "K_CAND", 16)
    monkeypatch.setattr(T, "D_HIDDEN", 32)
    monkeypatch.setattr(T, "WARMUP", 2)
    monkeypatch.setattr(T, "VAL_FRAC", 0.1)
    return root, hk


def test_train_end_to_end_resume_and_rule(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    root, hk = _synthetic_train_inputs(tmp_path, monkeypatch)
    kw = dict(root=root, holdout_keys=hk, device="cpu", eval_every=2)
    rep = T.train(out_dir=tmp_path / "a", max_steps=12, **kw)
    assert rep["status"] == "ok", rep.get("traceback")
    la = rep["leak_audit"]
    assert la["raw_inchikey14"]["total"] == 0 and la["canonical_key"]["total"] == 0
    assert all(la[k] == 0 for k in ("negatives_in_drop_key", "positives_in_drop_key", "train_negatives_in_val",
                                    "train_positives_in_val"))
    d = rep["data"]
    assert d["n_val_structures"] == 11 and d["n_val_spectra"] == 44 and d["n_train_spectra"] == (111 - 11) * 4
    log = [json.loads(x) for x in (tmp_path / "a/train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == list(range(2, 13, 2)) or rep["train"]["stop_reason"] == "converged"
    assert rep["train"]["best_step"] == min(log, key=lambda r: r["val_loss"])["step"]
    g = np.load(tmp_path / "a/g_query.npy"); h = np.load(tmp_path / "a/h_pool.npy")
    assert g.shape == (3, T.D_SHARED) and h.shape == (120, T.D_SHARED) and h.dtype == np.float16
    assert np.allclose(np.linalg.norm(g, axis=1), 1, atol=1e-4)
    assert (tmp_path / "a/learning_curve.csv").exists() and (tmp_path / "a/model_best.pt").exists()
    cfg = json.loads((tmp_path / "a/align_config.json").read_text())
    assert cfg["weights"] == "best internal-val loss" and cfg["pool_meta_sha256"] == T.POOL_META_SHA256
    # 途中で止めて再開しても、同じ列になる
    T._PERM_CACHE.clear()
    T.train(out_dir=tmp_path / "b", max_steps=6, **kw)
    T._PERM_CACHE.clear()
    rep_b = T.train(out_dir=tmp_path / "b", max_steps=12, resume=True, **kw)
    assert rep_b["resumed_from_step"] == 6
    log_b = [json.loads(x) for x in (tmp_path / "b/train_log.jsonl").read_text().splitlines()]
    n = len(log)
    assert [round(r["val_loss"], 5) for r in log_b[:n]] == [round(r["val_loss"], 5) for r in log]


def test_own_heavy_jobs_ignores_shell_and_time_wrappers(monkeypatch):
    import subprocess
    txt = ("  11 /bin/zsh -lic cd x && .venv/bin/python scripts/train/align_train.py\n"
           "  12 /usr/bin/time -l .venv/bin/python scripts/train/align_train.py --bench 30\n"
           "  13 .venv/bin/python scripts/train/align_local_embed.py\n"
           "  14 /opt/python3 other_project/train.py\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("R", (), {"stdout": txt})())
    got = T.own_heavy_jobs()
    assert len(got) == 1 and got[0].startswith("13 ")
