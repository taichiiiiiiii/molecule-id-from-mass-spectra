"""Engine 2 の rank_train 表で恒等式 `fp_raw_minus_max` が成り立たないグループを型に分ける（読むだけ、#165）。

検査（`scripts/research/engine2_table_schema.py` の check_identity(kind="minus_max_scaled")、そのまま呼ぶ）:
  d = fp_raw_minus_max（列 17）、z = fp_raw_z（列 15）、tol 1e-4、グループ = G×2+M。
  all(d ≤ tol) かつ isclose(max(d), 0, atol=tol) かつ _rank_consistent(z, _rank_norm(d))。

不成立のグループの型（#165 に npz を開く前に事前登録した読み方。順に 1 つ選ぶ。旗は重ねて全部記録する）:
  (i)   max(d) ≠ 0（isclose(max(d), 0, atol=tol) が偽）
  (ii)  d > tol の行がある（|max(d)| ≤ tol なら d > tol は起きないので、(ii) の旗は常に (i) と一緒に立つ）
  (iii) 順の食い違いが d の同値の塊の中だけ（z_i > z_j かつ d_i < d_j の組が無く、d が同値で z が違う塊がある）
  (iv)  それ以外（z_i > z_j かつ d_i < d_j の組がある、または上のどれにも当たらない）

数える量: 「d が同値（保存値の完全一致）で、その塊の中で z が違う（完全一致でない）」塊の行。表全体（通過グループも含む）。

判定（事前登録、変えない）: 通過 = 不成立のグループがすべて (iii) かつ 塊の行 ≤ 表全体の行の 1%。それ以外は停止。

使い方:
  .venv/bin/python scripts/engine2/identity_breakdown.py \
      --table orig=data/ext/engine2_ref/rank_train.npz \
      --table run2=outputs/engine2_build/run2/own/own_rank_train.npz \
      --table b=outputs/engine2_build_b/run1_v2/own/own_rank_train.npz \
      --out outputs/engine2_build_b/run1_v2/identity_breakdown.json
"""
import argparse
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "engine2_table_schema", os.path.join(HERE, "..", "research", "engine2_table_schema.py"))
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)

_NAME2I = {name: i for i, _, name, _ in S.column_map(["base"])}
COL_Z = _NAME2I["fp_raw_z"]            # 15
COL_RANK = _NAME2I["fp_raw_rank"]      # 16
COL_D = _NAME2I["fp_raw_minus_max"]    # 17
TOL = 1e-4
TIE_FRAC_MAX = 0.01


def tie_rows_z_differs(d, z):
    """d が完全一致で同値の塊（2 行以上）のうち、塊の中で z が完全一致でない塊の行の数。"""
    o = np.argsort(d, kind="mergesort"); ds = d[o]; zs = z[o]
    starts = np.r_[0, np.where(np.diff(ds) != 0)[0] + 1]; ends = np.r_[starts[1:], len(ds)]
    n = 0
    for a, b in zip(starts, ends):
        if b - a >= 2 and np.any(zs[a:b] != zs[a]):
            n += b - a
    return int(n)


def has_strict_reversal(d, z):
    """z_i > z_j かつ d_i < d_j の組があるか（z の同値の組は問わない）。"""
    if len(d) < 2: return False
    o = np.argsort(-z, kind="mergesort"); zs = z[o]; ds = d[o]
    starts = np.r_[0, np.where(np.diff(zs) != 0)[0] + 1]; ends = np.r_[starts[1:], len(zs)]
    run_min = np.inf   # z が大きい側の塊すべての d の最小
    for a, b in zip(starts, ends):
        if ds[a:b].max() > run_min: return True
        run_min = min(run_min, ds[a:b].min())
    return False


def classify_group(d, z, tol=TOL):
    """1 グループの型（'pass' / 'i' / 'ii' / 'iii' / 'iv'）と旗、塊の行。d・z は float64。"""
    ties = tie_rows_z_differs(d, z)
    ok = S.check_identity("minus_max_scaled", d, z, tol=tol)
    flags = dict(i=not bool(np.isclose(d.max(), 0.0, atol=tol)), ii=bool(np.any(d > tol)),
                 reversal=has_strict_reversal(d, z), tie_rows=ties)
    if ok: return "pass", flags
    if flags["i"]: return "i", flags
    if flags["ii"]: return "ii", flags
    if flags["reversal"]: return "iv", flags
    if ties > 0: return "iii", flags
    return "iv", flags   # どれにも当たらない不成立（知らない型）は安全側で (iv)


def _q(v, qs=(0, 5, 50, 95, 100)):
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0: return None
    return {f"p{q}": round(float(np.percentile(v, q)), 4) for q in qs}


def _groups(G, M):
    key = G.astype(np.int64) * 2 + np.asarray(M).astype(np.int64)
    order = np.argsort(key, kind="mergesort"); ks = key[order]
    cuts = np.r_[0, np.where(np.diff(ks) != 0)[0] + 1, len(ks)]
    return order, cuts


def breakdown(X, G, M, tol=TOL):
    """表 1 本の分類・塊の行・raw の std・d の範囲（通過/不通過のグループ別）と判定。"""
    order, cuts = _groups(G, M)
    counts = dict(pass_=0, i=0, ii=0, iii=0, iv=0); flag_ii = 0
    tie_total = 0; groups_with_ties = 0
    std = {"pass": [], "fail": []}; dmin = {"pass": [], "fail": []}; examples = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        rows = order[a:b]
        d = X[rows, COL_D].astype(np.float64); z = X[rows, COL_Z].astype(np.float64)
        t, f = classify_group(d, z, tol)
        counts["pass_" if t == "pass" else t] += 1
        flag_ii += f["ii"]
        tie_total += f["tie_rows"]; groups_with_ties += f["tie_rows"] > 0
        side = "pass" if t == "pass" else "fail"
        std[side].append(d.std()); dmin[side].append(d.min())
        if t != "pass" and len(examples) < 5:
            examples.append(dict(type=t, n=int(len(d)), tie_rows=f["tie_rows"], d_max=float(d.max()),
                                 d_min=float(d.min()), d_std=float(d.std())))
    n_rows = int(X.shape[0]); n_groups = int(len(cuts) - 1)
    n_fail = n_groups - counts["pass_"]
    tie_frac = tie_total / max(1, n_rows)
    verdict = "pass" if (counts["i"] == 0 and counts["ii"] == 0 and counts["iv"] == 0
                         and tie_frac <= TIE_FRAC_MAX) else "stop"
    return dict(n_rows=n_rows, n_groups_GxM=n_groups, n_fail=int(n_fail),
                types={"pass": counts["pass_"], "i": counts["i"], "ii": counts["ii"],
                       "iii": counts["iii"], "iv": counts["iv"]},
                flag_ii_groups=int(flag_ii),
                tie_rows_z_differs=int(tie_total), tie_groups=int(groups_with_ties),
                tie_frac=round(tie_frac, 8), tie_frac_max=TIE_FRAC_MAX,
                raw_std={k: _q(v) for k, v in std.items()}, d_min={k: _q(v) for k, v in dmin.items()},
                fail_examples=examples, verdict=verdict)


def table_stats(X, Y, G, M):
    """表の統計（判定には使わない）: グループ数・正例率・候補数・M=0 の正解の FP 1 位の割合。"""
    Y = np.asarray(Y); M = np.asarray(M)
    _, inv, cnt = np.unique(G, return_inverse=True, return_counts=True)
    key = G.astype(np.int64) * 2 + M.astype(np.int64)
    _, cnt_gm = np.unique(key, return_counts=True)
    pos0 = (Y > 0.5) & (M == 0)
    top1 = X[pos0, COL_RANK] == 0
    return dict(n_groups_G=int(len(cnt)), n_groups_GxM=int(len(cnt_gm)), pos_rate=round(float(Y.mean()), 6),
                rows_by_M={str(int(k)): int(v) for k, v in zip(*np.unique(M, return_counts=True))},
                cand_per_GxM=_q(cnt_gm), n_pos_M0=int(pos0.sum()),
                fp_top1_rate_M0_pos=round(float(top1.mean()), 4) if pos0.any() else None)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="append", required=True, help="name=path（rank_train の npz）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = dict(check="minus_max_scaled", col_d=COL_D, col_z=COL_Z, tol=TOL, tables={})
    for spec in a.table:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=False)
        X, Y, G, M = z["X"], z["Y"], z["G"], z["M"]
        assert X.shape[1] == 31, X.shape
        r = dict(path=path, bytes=os.path.getsize(path), sha256=sha256(path))
        r.update(breakdown(X, G, M)); r["stats"] = table_stats(X, Y, G, M)
        out["tables"][name] = r
        print(name, json.dumps({k: r[k] for k in ("n_rows", "n_groups_GxM", "types", "tie_rows_z_differs",
                                                    "tie_frac", "verdict")}, ensure_ascii=False))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    main()
