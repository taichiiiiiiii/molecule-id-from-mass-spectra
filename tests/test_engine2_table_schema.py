"""engine2_table_schema の単体テスト（合成 npz、numba・casmi_engine 不要）。"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "engine2_table_schema", os.path.join(HERE, "..", "scripts", "research", "engine2_table_schema.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def test_column_map_widths():
    assert len(S.BLOCK_COLS["base"]) == 31
    assert sum(len(S.BLOCK_COLS[b]) for b in ["base", "libx", "anapol", "fragx"]) == 51
    cmap = S.column_map(["base", "libx", "anapol", "fragx"])
    assert [i for i, *_ in cmap] == list(range(51))
    fp = [i for i, _, _, c in cmap if c == "fp"]
    assert fp == list(range(15, 21)) + list(range(25, 31))
    names = [n for _, _, n, _ in cmap]
    assert len(set(names)) == len(names)


def _group_cols(rng, n, cmap):
    """推論コードと同じ式で 1 グループの列を作る（FP 列は 0）。"""
    X = rng.random((n, len(cmap))).astype(np.float32) + 0.1   # 式の無い列は乱数（0 にならない）
    idx = {name: i for i, _, name, _ in cmap}
    X[:, [i for i, _, _, c in cmap if c == "fp"]] = 0
    for name in ("top_sim", "anapol_top_w"):   # 分子ごとの定数の列
        if name in idx: X[:, idx[name]] = X[0, idx[name]]
    for base in ("lib", "ap", "frag", "lib_mean", "libsh", "anapol_ap", "frag_ad", "frag_mean"):
        if base not in idx: continue
        v = rng.random(n).astype(np.float32); v[rng.random(n) < 0.3] = 0
        X[:, idx[base]] = v
        if base + "_rank" in idx: X[:, idx[base + "_rank"]] = S._rank_norm(v)
        if base + "_minus_max" in idx: X[:, idx[base + "_minus_max"]] = v - v.max()
        if base + "_max" in idx: X[:, idx[base + "_max"]] = v.max()
    X[:, idx["lib_pos"]] = (X[:, idx["lib"]] > 0)
    X[:, idx["log_nc"]] = np.log(n)
    f = X[:, idx["frag"]]; X[:, idx["frag_z"]] = (f - f.mean()) / f.std()
    if "log_cnt" in idx:
        c = np.log1p(rng.integers(0, 4, n)).astype(np.float32)
        X[:, idx["log_cnt"]] = c; X[:, idx["log_cnt_minus_max"]] = c - c.max()
    return X


def _write(tmp_path, broken=False):
    rng = np.random.default_rng(0)
    blocks = ["base", "libx", "anapol", "fragx"]
    cmap = S.column_map(blocks)
    Xs, G, Y, Sv = [], [], [], []
    for g in range(6):
        n = int(rng.integers(5, 12))
        Xs.append(_group_cols(rng, n, cmap)); G += [g] * n; Sv += [g % 2] * n
        y = np.zeros(n, np.int8); y[0] = 1; Y.append(y)
    X = np.vstack(Xs)
    if broken: X[0, 1] = 0.5; X[1, 1] = 0.5  # lib_rank を壊す（グループ 0）
    p = tmp_path / "sim.npz"
    np.savez(p, X=X, Y=np.concatenate(Y), S=np.array(Sv, np.int8), G=np.array(G, np.int32),
             Q=np.zeros(len(G), np.int8), blocks=np.array(blocks), sets=np.array(["gnps"]))
    return str(p)


def test_describe_synthetic(tmp_path):
    d = S.describe(_write(tmp_path))
    assert d["n_feat"] == 51 and d["map_matches_n_feat"]
    assert d["n_groups"] == 6 and d["pos_per_group"] == {"1": 6}
    assert d["rows_by_S"].keys() == {"0", "1"}
    assert d["all_zero_cols"] == list(range(15, 21)) + list(range(25, 31))
    assert d["category_counts"] == {"own": 39, "fp": 12}
    ids = d["identities"]
    for name, v in ids.items():
        if v.get("status") == "all_zero":
            assert name.startswith(("fp_", "x_")); continue
        assert v["frac"] == 1.0, (name, v)
    assert ids["log_nc"]["frac"] == 1.0
    json.dumps(d)  # JSON に書ける


def test_identity_detects_break(tmp_path):
    d = S.describe(_write(tmp_path, broken=True))
    assert d["identities"]["lib_rank"]["frac"] < 1.0


def test_rank_consistent_ties():
    x = np.array([0.5, 0.5, 0.2, 0.0])
    assert S._rank_consistent(x, np.array([0, 1, 2, 3]) / 3)
    assert S._rank_consistent(x, np.array([1, 0, 2, 3]) / 3)   # 同値の入れ替えは許す
    assert not S._rank_consistent(x, np.array([2, 1, 0, 3]) / 3)


def test_main_writes_json(tmp_path):
    p = _write(tmp_path)
    rng = np.random.default_rng(1)
    q = tmp_path / "rt.npz"
    z = np.load(p)
    np.savez(q, X=z["X"][:, :31], Y=z["Y"].astype(float), M=z["S"].astype(float), G=z["G"].astype(np.int64))
    out = tmp_path / "o" / "schema.json"
    S.main(["--rank-train", str(q), "--sim-rows", p, "--out", str(out)])
    d = json.load(open(out))
    assert d["rank_train"]["n_feat"] == 31 and d["rank_train"]["blocks"] == ["base"]
    assert d["shape_matches_registered"] == {"rank_train": False, "sim_rank_rows": False}
    del rng


def test_multi_formulas():
    lib = np.array([0.0, 0.9, 0.3]); mr = np.array([0.5, 1.0, 0.0]); ap = np.array([0.2, 0.0, 0.1])
    c = dict(lib=lib, fp_raw_rank=mr, ap=ap, lib_max=np.full(3, 0.9))
    assert np.allclose(S.MULTI["x_lib_mr"][1](c), lib * (1 - mr))
    assert np.allclose(S.MULTI["x_agree_lib"][1](c), 0.0)        # argmax lib = 1 → 1 − mr[1] = 0
    assert np.allclose(S.MULTI["x_agree_ap"][1](c), 0.5)         # argmax ap = 0 → 1 − 0.5
    c["x_agree_lib"] = np.full(3, 0.25)
    assert np.allclose(S.MULTI["x_agree_lib_lvmax"][1](c), 0.225)
    assert np.allclose(S.MULTI["fp_raw_is_max"][1](dict(fp_raw_minus_max=np.array([0.0, -1.0]))), [1.0, 0.0])
