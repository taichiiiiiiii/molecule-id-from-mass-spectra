#!/usr/bin/env python3
"""
§5.2 の E（v3 の 4 集合の MRR@25 を等しい重みで平均したもの）の差の区間（Issue #114、設計 docs/design/lineage487-foldsafe-eval.md §5.2）。

scripts/ranker/train_eval.py の bootstrap_diff（分子を再抽出、2,000 回、seed 0、2.5〜97.5 パーセンタイル）と同じ定義を、
集合ごとの層別にしたもの:
  - 各回、集合ごとに、その集合の分子を復元抽出する（集合の大きさは保つ）
  - その回の値 = 集合ごとの (a − b) の平均を、集合の数で等しく平均したもの
  - 乱数は 1 本（np.random.default_rng(seed)）。集合は与えた順（既定は SETS の順）に、bootstrap_diff と同じ
    rng.integers(0, n_s, size=(n, n_s)) で引く。集合が 1 つなら bootstrap_diff と同じ値になる
  - 除外（MSG と重なる分子など）は exclude で渡す。事前に決めた除外だけを渡し、結果を見て変えない
集合ごとの区間は bootstrap_diff そのもの（同じ seed）で出す。

    from stratified_bootstrap import stratified_bootstrap_diff
    r = stratified_bootstrap_diff({"v3_c1": rr_new_c1, ...}, {"v3_c1": rr_ctrl_c1, ...}, exclude={"v3_c2n": msg_ids})
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
N_BOOT = 2000


def _bootstrap_diff():
    sys.path.insert(0, str(ROOT / "scripts/ranker"))
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    from train_eval import bootstrap_diff
    return bootstrap_diff


def paired(a: dict, b: dict, exclude: set | None = None) -> tuple[list[str], np.ndarray]:
    """分子ごとの a − b（分子の並びは sorted、bootstrap_diff と同じ）。a と b の分子は一致していなければならない。"""
    if set(a) != set(b):
        raise ValueError(f"分子が一致しない: a だけ {len(set(a) - set(b))}、b だけ {len(set(b) - set(a))}")
    ex = exclude or set()
    unknown = ex - set(a)
    if unknown:
        raise ValueError(f"除外に、集合に無い分子がある: {sorted(unknown)[:5]}")
    mids = sorted(m for m in a if m not in ex)
    if not mids:
        raise ValueError("除外のあとに分子が残らない")
    return mids, np.array([a[m] - b[m] for m in mids], dtype=float)


def stratified_bootstrap_diff(a_sets: dict[str, dict], b_sets: dict[str, dict], exclude: dict[str, set] | None = None,
                              n: int = N_BOOT, seed: int = 0, order: tuple[str, ...] | None = None) -> dict:
    """E(a) − E(b) の 95% 区間。E は集合ごとの平均の等しい重みの平均。"""
    if set(a_sets) != set(b_sets):
        raise ValueError(f"集合が一致しない: {sorted(a_sets)} vs {sorted(b_sets)}")
    order = tuple(order) if order else tuple(s for s in SETS if s in a_sets) + tuple(sorted(set(a_sets) - set(SETS)))
    if set(order) != set(a_sets):
        raise ValueError(f"order が集合と一致しない: {order}")
    exclude = exclude or {}
    if set(exclude) - set(a_sets):
        raise ValueError(f"除外に未知の集合: {sorted(set(exclude) - set(a_sets))}")
    rng = np.random.default_rng(seed)
    bd = _bootstrap_diff()
    xs, boot, per, ma, mb = {}, np.zeros(n), {}, [], []
    for s in order:
        mids, x = paired(a_sets[s], b_sets[s], exclude.get(s))
        xs[s] = x
        boot += x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
        a_kept = {m: a_sets[s][m] for m in mids}
        b_kept = {m: b_sets[s][m] for m in mids}
        ma.append(float(np.mean(list(a_kept.values())))); mb.append(float(np.mean(list(b_kept.values()))))
        d = bd(a_kept, b_kept, n=n, seed=seed)
        d.update(n=len(mids), n_excluded=len(exclude.get(s, ())), a=round(float(np.mean(list(a_kept.values()))), 4),
                 b=round(float(np.mean(list(b_kept.values()))), 4))
        per[s] = d
    k = len(order)
    boot /= k
    diff = float(np.mean([x.mean() for x in xs.values()]))
    E_a, E_b = float(np.mean(ma)), float(np.mean(mb))
    return dict(diff=round(diff, 4), lo=round(float(np.percentile(boot, 2.5)), 4), hi=round(float(np.percentile(boot, 97.5)), 4),
                E_a=round(E_a, 4), E_b=round(E_b, 4), sets=list(order), n_boot=n, seed=seed,
                better=int(sum((x > 0).sum() for x in xs.values())), worse=int(sum((x < 0).sum() for x in xs.values())),
                per_set=per)


def per_set_rr(truth_paths: dict[str, Path], sub_paths: dict[str, Path]) -> dict[str, dict]:
    """集合ごとに、分子ごとの逆数順位（mrr25.py と同じ）。"""
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "scripts/ranker"))
    import mrr25
    from train_eval import per_molecule_rr
    return {s: per_molecule_rr(mrr25.read_truth(truth_paths[s]), mrr25.read_submission(sub_paths[s])) for s in truth_paths}


def main(argv=None) -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="E の差（層別ブートストラップ）")
    ap.add_argument("--new", required=True, help="outputs/local_eval/<set>/submission_<tag>.csv の <tag>（新）")
    ap.add_argument("--ctrl", required=True, help="同（対照）")
    ap.add_argument("--exclude-json", default=None, help="{集合: [molecule_id, ...]}（事前に決めた除外）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    truth = {s: ROOT / f"outputs/holdouts/{s}/truth.csv" for s in SETS}
    A = per_set_rr(truth, {s: ROOT / f"outputs/local_eval/{s}/submission_{a.new}.csv" for s in SETS})
    B = per_set_rr(truth, {s: ROOT / f"outputs/local_eval/{s}/submission_{a.ctrl}.csv" for s in SETS})
    ex = {s: set(v) for s, v in json.loads(Path(a.exclude_json).read_text()).items()} if a.exclude_json else None
    r = stratified_bootstrap_diff(A, B, exclude=ex)
    r.update(new=a.new, ctrl=a.ctrl)
    txt = json.dumps(r, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == "__main__":
    main()
