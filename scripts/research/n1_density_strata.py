#!/usr/bin/env python3
"""
Issue #292（2026-11-28 夜 計画 項目 3、記録だけ）: 版 A（N1+GLACIER、exp016〜018）と対照（exp007・014・015）の
3 seed 平均の差を、正解の行の nbr_tani_mean（近傍の多さ）の型ごとの 3 分位＋「欠け」で層に分けて書く。
採否・関門・列・定数の選択には使わない。

定義（#292 の本文で値を見る前に固定）:
- 対象の分子: msg0_eval の分子（6 本の per_molecule.csv で分子の集合と型が同じこと、feature_groups._check_triplet）
- 正解の行: exp016 の meta の表（sha256 一致）で lambdarank_oof.make_labels が 1 の行
- v: 正解の行の nbr_tani_mean（2 行以上なら最大、数を記録）
- 層（型ごと）: 境 q1, q2 = 正解の行がある分子の v の 1/3・2/3 分位（numpy 既定の線形補間、有限の値だけ）。
  低 = v ≤ q1、中 = q1 < v ≤ q2、高 = v > q2、欠け = 正解の行が無い。v が NaN（近傍 0 件）は低（数を記録）
- 差: d = mean(rr 候補 3 seed) − mean(rr 対照 3 seed)、層の平均と evaluate.paired_bootstrap（2,000 回、seed 0）
- 検算: 4 層の n の和 = 型の n、n 重みの層の差の平均 = seed_mean.json の delta（|差| ≤ 1e-12）
- 読み: 低い層の下限 > 0 → not_only_density、区間が 0 を含む → near_zero、上限 < 0 → low_loses

    .venv/bin/python scripts/research/n1_density_strata.py \\
        --out docs/research/neighbour-density/n1_density_strata.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ISSUE = 292
ROOT = Path(__file__).resolve().parents[2]
LR = ROOT / "outputs/local_eval/ranker/lambdarank"
CAND = ("exp016", "exp017", "exp018")
CTRL = ("exp007", "exp014", "exp015")
SEED_MEAN = ROOT / "docs/experiments/version_a_combo_282/seed_mean.json"
COL = "nbr_tani_mean"
STRATA = ("low", "mid", "high", "absent")
TYPES = ("a", "b", "c", "d", "e")
REF_KEPT_TYPES = ("a", "e")   # 正解の他装置の参照が残る型（正解自身が近傍に入り得る）
TOL = 1e-12


# ---------------------------------------------------------------- 小さな道具（単体テストの対象）

def tertile_edges(v) -> tuple[float, float]:
    """正解の行がある分子の v（有限の値だけ）の 1/3・2/3 分位（numpy 既定の線形補間）。有限の値が無ければ止める。"""
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("有限の値が無い")
    return float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3))


def assign_stratum(v: float | None, has_truth: bool, q1: float, q2: float) -> str:
    """分子の層。正解の行が無ければ absent、v が NaN（近傍 0 件）なら low。"""
    if not has_truth:
        return "absent"
    if v is None or not np.isfinite(v):
        return "low"
    if v <= q1:
        return "low"
    return "mid" if v <= q2 else "high"


def reading(lo: float | None, hi: float | None) -> str:
    """低い層の 95% 区間 → 読み（#292 の規則）。n = 0 なら no_data。"""
    if lo is None or hi is None:
        return "no_data"
    if lo > 0:
        return "not_only_density"
    if hi < 0:
        return "low_loses"
    return "near_zero"


def molecule_values(groups, labels, values) -> tuple[dict, dict]:
    """行の群 id・ラベル・値 → (群 id → 正解の行の値の最大, 群 id → 正解の行の数)。正解の行が無い群は入らない。"""
    g = np.asarray(groups)
    y = np.asarray(labels)
    x = np.asarray(values, dtype=np.float64)
    vmax, count = {}, {}
    for gid, val in zip(g[y == 1], x[y == 1]):
        count[gid] = count.get(gid, 0) + 1
        prev = vmax.get(gid)
        if prev is None or (np.isfinite(val) and (not np.isfinite(prev) or val > prev)):
            vmax[gid] = float(val)
    return vmax, count


def summarize_type(ids, d, cand, ctrl, vmax: dict, q1: float, q2: float, boot) -> dict:
    """1 型の層ごとの n・差・区間・候補と対照の MRR。boot(a, b) -> (lo, hi)。"""
    d = np.asarray(d, float)
    cand = np.asarray(cand, float)
    ctrl = np.asarray(ctrl, float)
    st = np.array([assign_stratum(vmax.get(i), i in vmax, q1, q2) for i in ids])
    out = {}
    for s in STRATA:
        m = st == s
        n = int(m.sum())
        if n == 0:
            out[s] = dict(n=0, delta=None, lo=None, hi=None, cand_mrr=None, ctrl_mrr=None)
            continue
        lo, hi = boot(cand[m], ctrl[m])
        out[s] = dict(n=n, delta=float(d[m].mean()), lo=lo, hi=hi, cand_mrr=float(cand[m].mean()),
                      ctrl_mrr=float(ctrl[m].mean()))
    return out


def weighted_delta(strata: dict) -> float:
    n = sum(v["n"] for v in strata.values())
    return sum(v["n"] * v["delta"] for v in strata.values() if v["n"]) / n


# ---------------------------------------------------------------- 本体

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "scripts/ranker"))
    import pandas as pd
    from evaluate import paired_bootstrap
    from feature_groups import _check_triplet
    from lambdarank_oof import group_ids, load_truth, make_labels

    cand_runs = _check_triplet("候補", [LR / e for e in CAND])
    ctrl_runs = _check_triplet("対照", [LR / e for e in CTRL])
    idx = ctrl_runs[0][1].index
    types = ctrl_runs[0][1]["type"]
    for _, pm in cand_runs + ctrl_runs:
        if set(pm.index) != set(idx) or len(pm) != len(idx) or not (pm.loc[idx, "type"] == types).all():
            raise RuntimeError("per_molecule.csv の分子・型が 6 本で同じでない")
    rr_c = np.stack([pm.loc[idx, "rr"].to_numpy(float) for _, pm in cand_runs]).mean(axis=0)
    rr_k = np.stack([pm.loc[idx, "rr"].to_numpy(float) for _, pm in ctrl_runs]).mean(axis=0)
    ids = np.array([f"{s}\t{m}" for s, m in idx])

    meta = cand_runs[0][0]
    table = Path(meta["table"])
    tsha = sha256_file(table)
    if tsha != meta["table_sha256"]:
        raise RuntimeError("表の sha256 が exp016 の meta と違う")
    feats = meta["features"] if isinstance(meta["features"], list) else ast.literal_eval(meta["features"])
    if COL not in feats:
        raise RuntimeError(f"{COL} が exp016 の特徴に無い")
    df = pd.read_parquet(table, columns=["set", "type", "molecule_id", "cand_smiles", COL])
    truth = load_truth(df, Path(a.eval_root))
    y = make_labels(df, truth)
    g = group_ids(df)
    vmax, count = molecule_values(g, y, df[COL].to_numpy(np.float64))

    sm = json.loads(SEED_MEAN.read_text())
    res, checks = {}, {}
    for t in TYPES:
        m = (types == t).to_numpy()
        tid = ids[m]
        vt = [vmax[i] for i in tid if i in vmax]
        q1, q2 = tertile_edges(vt)
        strata = summarize_type(tid, rr_c[m] - rr_k[m], rr_c[m], rr_k[m], vmax, q1, q2, paired_bootstrap)
        n_sum = sum(v["n"] for v in strata.values())
        wd = weighted_delta(strata)
        ref = sm["types"][t]["delta"]
        checks[t] = dict(n_sum=n_sum, n_type=int(m.sum()), weighted_delta=wd, seed_mean_delta=ref,
                         ok=bool(n_sum == int(m.sum()) and abs(wd - ref) <= TOL))
        if not checks[t]["ok"]:
            raise RuntimeError(f"型 {t}: 検算が合わない {checks[t]}")
        res[t] = dict(group="ref_kept" if t in REF_KEPT_TYPES else "ref_hidden", q1=q1, q2=q2,
                      n_nan_v=int(sum(1 for i in tid if i in vmax and not np.isfinite(vmax[i]))),
                      n_multi_truth_rows=int(sum(1 for i in tid if count.get(i, 0) > 1)),
                      v_quantiles_truth=[float(x) for x in np.quantile(np.asarray(vt), [0.1, 0.5, 0.9])],
                      strata=strata, reading_low=reading(strata["low"]["lo"], strata["low"]["hi"]))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = dict(issue=ISSUE, cand=list(CAND), ctrl=list(CTRL), column=COL, table=str(table), table_sha256=tsha,
               eval_molecules_sha256=meta["eval_molecules"]["sha256"], n_molecules=int(len(ids)),
               n_boot=2000, boot_seed=0, git_commit=commit, types=res, checks=checks,
               secs=round(time.time() - t0, 1),
               max_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for t in TYPES:
        s = res[t]["strata"]
        print(t, res[t]["reading_low"], {k: (v["n"], None if v["delta"] is None else round(v["delta"], 4),
                                            None if v["lo"] is None else round(v["lo"], 4),
                                            None if v["hi"] is None else round(v["hi"], 4)) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
