"""Engine 2 の 2 表（rank_train.npz / sim_rank_rows_nofp.npz）の中身と列の対応を記録する（読むだけ）。

- 各 npz のキー・形・dtype、Y の正例率、S/M/Q 別の行数、グループ数、グループあたりの正例数、全 0 の列
- 列の対応表: 推論側 `casmi_engine.BLOCKS`（base = `pv.rank_features` の 31 列、libx 8、anapol 5、fragx 7、mass 3）
  の列名と、作り直しの区分（own = 自前で作れる / fp = FP ロジットが要る / unknown = 定義不明）
- 列の定義が表の値と整合するかの恒等式チェック（グループ内で再計算: rank 列 = _rank_norm(元の列)、
  max 列 = グループ内最大、minus_max 列 = 元 − 最大、pos 列 = 元 > 0、log_nc 列 = log(グループの行数)）
- --engine-dir を渡すと casmi_engine を読み込み、合成の 1 分子で BLOCKS の各関数を呼んで列数を数える

使い方:
  .venv/bin/python scripts/research/engine2_table_schema.py \
      --rank-train data/ext/engine2_ref/rank_train.npz \
      --sim-rows data/ext/engine2_ref/sim_rank_rows_nofp.npz \
      --engine-dir outputs/holdout_eval/np_v1/eng --out outputs/engine2_ref/schema.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

# ------------------------------------------------------------------------------------------------
# 列の対応表（推論側のコードの順。casmi_engine.py 299-341 行、pv.py rank_features 210-255 行）
# ------------------------------------------------------------------------------------------------
BASE_COLS = [
    ("lib", "own"), ("lib_rank", "own"), ("lib_max", "own"), ("lib_minus_max", "own"), ("lib_pos", "own"),
    ("ap", "own"), ("ap_rank", "own"), ("ap_max", "own"), ("ap_minus_max", "own"),
    ("a1", "own"), ("best_tan", "own"), ("top_tan", "own"), ("mean_tan", "own"), ("top_sim", "own"),
    ("log_nc", "own"),
    ("fp_raw_z", "fp"), ("fp_raw_rank", "fp"), ("fp_raw_minus_max", "fp"),
    ("fp_nrm_z", "fp"), ("fp_nrm_rank", "fp"), ("fp_raw_is_max", "fp"),
    ("frag", "own"), ("frag_rank", "own"), ("frag_minus_max", "own"), ("frag_z", "own"),
    ("x_lib_mr", "fp"), ("x_ap_mr", "fp"), ("x_agree_lib", "fp"), ("x_agree_ap", "fp"),
    ("x_agree_lib_lvmax", "fp"), ("x_corr_lib_mr", "fp"),
]
BLOCK_COLS = {
    "base": BASE_COLS,
    "mass": [("ppm", "own"), ("abs_ppm", "own"), ("abs_ppm_p1_5", "own")],
    "libx": [("lib_mean", "own"), ("lib_mean_rank", "own"), ("lib_mean_minus_max", "own"),
             ("libsh", "own"), ("libsh_rank", "own"), ("libsh_minus_max", "own"),
             ("log_cnt", "own"), ("log_cnt_minus_max", "own")],
    "anapol": [("anapol_ap", "own"), ("anapol_ap_rank", "own"), ("anapol_ap_minus_max", "own"),
               ("anapol_ap6", "own"), ("anapol_top_w", "own")],
    "fragx": [("frag_ad", "own"), ("frag_ad_rank", "own"), ("frag_ad_minus_max", "own"),
              ("frag_mean", "own"), ("frag_mean_rank", "own"), ("frag_mean_minus_max", "own"),
              ("frag_pol", "own")],
}

# 恒等式チェック: (検査する列名, 種類, 元の列名)
IDENTITIES = [
    ("lib_rank", "rank", "lib"), ("lib_max", "max", "lib"), ("lib_minus_max", "minus_max", "lib"),
    ("lib_pos", "pos", "lib"),
    ("ap_rank", "rank", "ap"), ("ap_max", "max", "ap"), ("ap_minus_max", "minus_max", "ap"),
    ("log_nc", "log_nc", None),
    ("fp_raw_rank", "rank", "fp_raw_z"), ("fp_raw_minus_max", "minus_max_scaled", "fp_raw_z"),
    ("frag_rank", "rank", "frag"), ("frag_minus_max", "minus_max", "frag"), ("frag_z", "z", "frag"),
    ("lib_mean_rank", "rank", "lib_mean"), ("lib_mean_minus_max", "minus_max", "lib_mean"),
    ("libsh_rank", "rank", "libsh"), ("libsh_minus_max", "minus_max", "libsh"),
    ("log_cnt_minus_max", "minus_max", "log_cnt"),
    ("anapol_ap_rank", "rank", "anapol_ap"), ("anapol_ap_minus_max", "minus_max", "anapol_ap"),
    ("frag_ad_rank", "rank", "frag_ad"), ("frag_ad_minus_max", "minus_max", "frag_ad"),
    ("frag_mean_rank", "rank", "frag_mean"), ("frag_mean_minus_max", "minus_max", "frag_mean"),
]


def column_map(blocks):
    """blocks の順に [(index, block, name, category)] を返す。"""
    out, i = [], 0
    for b in blocks:
        for name, cat in BLOCK_COLS[b]:
            out.append((i, b, name, cat)); i += 1
    return out


def _rank_norm(x):
    o = np.argsort(-x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)


def _rank_consistent(x, r, tol=1e-6):
    """同順位の扱いに依らない検査: x_i > x_j なら r_i < r_j、x_i == x_j の組は問わない。
    実装: x の降順に並べたとき r が（x の同値ブロックをまとめた上で）非減少か。"""
    if len(x) < 2: return True
    if not np.allclose(np.sort(r), np.arange(len(x)) / max(1, len(x) - 1), atol=1e-5): return False
    o = np.argsort(-x, kind="mergesort"); xs = x[o]; rs = r[o]
    # 同値ブロックごとの r の最小と最大
    starts = np.r_[0, np.where(np.diff(xs) != 0)[0] + 1]; ends = np.r_[starts[1:], len(xs)]
    mins = np.array([rs[a:b].min() for a, b in zip(starts, ends)])
    maxs = np.array([rs[a:b].max() for a, b in zip(starts, ends)])
    return bool(np.all(maxs[:-1] < mins[1:] + tol))


def check_identity(kind, col, src, tol=1e-4):
    """1 グループ分の列 col が src から kind の式で得られるか。"""
    n = len(col)
    if kind == "rank":
        return _rank_consistent(src.astype(np.float64), col.astype(np.float64))
    if kind == "max":
        return bool(np.allclose(col, src.max(), atol=tol))
    if kind == "minus_max":
        return bool(np.allclose(col, src - src.max(), atol=tol))
    if kind == "minus_max_scaled":   # raw − max(raw) は z の定数倍（std 倍）: 符号と 0 の位置だけ見る
        d = col.astype(np.float64)
        return bool(np.all(d <= tol) and np.isclose(d.max(), 0.0, atol=tol)
                    and _rank_consistent(src.astype(np.float64), _rank_norm(d)))
    if kind == "pos":
        return bool(np.array_equal(col > 0.5, src > 0))
    if kind == "log_nc":
        return bool(np.allclose(col, np.log(max(n, 1)), atol=tol))
    if kind == "z":
        s = src.std()
        ref = (src - src.mean()) / s if s > 1e-9 else np.zeros_like(src)
        return bool(np.allclose(col, ref, atol=1e-3))
    raise ValueError(kind)


def identity_report(X, G, cmap):
    """各恒等式について、成り立つグループの割合（全 0 の列は 'all_zero'）。"""
    name2i = {name: i for i, _, name, _ in cmap}
    order = np.argsort(G, kind="mergesort"); Gs = G[order]
    cuts = np.r_[0, np.where(np.diff(Gs) != 0)[0] + 1, len(Gs)]
    res = {}
    for name, kind, srcname in IDENTITIES:
        if name not in name2i or (srcname is not None and srcname not in name2i): continue
        ci = name2i[name]; col_all = X[:, ci]
        if np.all(col_all == 0):
            res[name] = dict(kind=kind, src=srcname, status="all_zero"); continue
        ok = 0; ng = len(cuts) - 1
        for a, b in zip(cuts[:-1], cuts[1:]):
            rows = order[a:b]
            src = X[rows, name2i[srcname]] if srcname else None
            ok += check_identity(kind, X[rows, ci], src)
        res[name] = dict(kind=kind, src=srcname, groups=int(ng), holds=int(ok), frac=round(ok / ng, 4))
    return res


def _counts(v):
    u, c = np.unique(v, return_counts=True)
    return {str(k.item() if hasattr(k, "item") else k): int(n) for k, n in zip(u, c)}


def describe(path, blocks=None):
    z = np.load(path, allow_pickle=False)
    d = dict(file=os.path.basename(path), bytes=os.path.getsize(path),
             sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), keys={})
    for k in z.files:
        a = z[k]
        d["keys"][k] = dict(shape=list(a.shape), dtype=str(a.dtype))
    X, Y, G = z["X"], z["Y"], z["G"]
    d["n_rows"], d["n_feat"] = int(X.shape[0]), int(X.shape[1])
    d["pos_rate"] = round(float(Y.mean()), 6)
    d["n_groups"] = int(len(np.unique(G)))
    pos = np.bincount(np.unique(G, return_inverse=True)[1], weights=Y.astype(np.float64))
    d["pos_per_group"] = _counts(pos.astype(int))
    for k in ("S", "M", "Q"):
        if k in z.files: d[f"rows_by_{k}"] = _counts(z[k])
    if "sets" in z.files: d["sets"] = [str(s) for s in z["sets"]]
    if "Q" in z.files and "sets" in z.files:
        d["groups_by_set"] = {str(z["sets"][q]): int(len(np.unique(G[z["Q"] == q]))) for q in np.unique(z["Q"])}
    if "blocks" in z.files: blocks = [str(b) for b in z["blocks"]]
    d["blocks"] = blocks
    d["all_zero_cols"] = [int(i) for i in range(X.shape[1]) if np.all(X[:, i] == 0)]
    d["nan_cols"] = [int(i) for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]
    cmap = column_map(blocks)
    d["map_width"] = len(cmap)
    d["map_matches_n_feat"] = len(cmap) == X.shape[1]
    if len(cmap) == X.shape[1]:
        d["columns"] = [dict(i=i, block=b, name=n, category=c,
                             all_zero=i in d["all_zero_cols"]) for i, b, n, c in cmap]
        cats = {}
        for i, b, n, c in cmap:
            cats[c] = cats.get(c, 0) + 1
        d["category_counts"] = cats
        # 恒等式は M/S 別のサブグループ（同じ G が M=0/1 で 2 回出るため）
        sub = z["M"] if "M" in z.files else (z["S"] if "S" in z.files else np.zeros(len(G)))
        key = G.astype(np.int64) * 2 + np.asarray(sub).astype(np.int64)
        d["identities"] = identity_report(X, key, cmap)
    return d


def count_block_widths(engine_dir):
    """casmi_engine を読み込み、合成の 1 分子で BLOCKS の各関数の列数を数える。"""
    sys.path.insert(0, os.path.abspath(engine_dir))
    import casmi_engine as E
    rng = np.random.default_rng(0)
    npool, nbits = 50, 6930
    bits = (rng.random((npool, nbits)) < 0.05).astype(np.uint8)
    E.POOL = dict(fp=np.packbits(bits, axis=1), mass=np.sort(rng.uniform(300, 301, npool)))
    cand = np.arange(10, 22); nc = len(cand)
    f = lambda: rng.random(nc).astype(np.float32)
    r = dict(cand=cand, target=300.5, lib=f(), lib_mean=f(), libsh=f(), lib_cnt=rng.integers(0, 4, nc).astype(np.float32),
             ana=[(1, 0.9), (2, 0.5), (3, 0.3)], anapol=[(4, 0.8), (5, 0.4)], zlog=rng.normal(size=nbits).astype(np.float32),
             frag=f(), frag_ad=f(), frag_mean=f(), frag_pol=f())
    cfp = E.pool_fps(cand)
    widths = {}
    for name, fn in E.BLOCKS.items():
        widths[name] = int(fn(r, cfp).shape[1])
    # use_fp=False（Engine 2 の「ours」）では zlog=None
    widths["base_nofp_zero_cols"] = [int(i) for i in np.where(np.all(E.BLOCKS["base"](dict(r, zlog=None), cfp) == 0, axis=0))[0]]
    return widths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-train", required=True)
    ap.add_argument("--sim-rows", required=True)
    ap.add_argument("--engine-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = dict(rank_train=describe(a.rank_train, blocks=["base"]), sim_rank_rows=describe(a.sim_rows))
    out["registered_shapes"] = {"rank_train": [142762, 31], "sim_rank_rows": [399790, 51]}
    out["shape_matches_registered"] = {
        "rank_train": [out["rank_train"]["n_rows"], out["rank_train"]["n_feat"]] == [142762, 31],
        "sim_rank_rows": [out["sim_rank_rows"]["n_rows"], out["sim_rank_rows"]["n_feat"]] == [399790, 51]}
    if a.engine_dir:
        w = count_block_widths(a.engine_dir)
        blocks = out["sim_rank_rows"]["blocks"]
        out["engine_block_widths"] = w
        out["engine_width_sum_sim_blocks"] = int(sum(w[b] for b in blocks))
        out["engine_width_matches_sim_n_feat"] = out["engine_width_sum_sim_blocks"] == out["sim_rank_rows"]["n_feat"]
        out["engine_width_matches_rank_train"] = w["base"] == out["rank_train"]["n_feat"]
        out["map_widths_match_engine"] = {b: len(BLOCK_COLS[b]) == w[b] for b in BLOCK_COLS}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in out if k not in ("rank_train", "sim_rank_rows")}, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
