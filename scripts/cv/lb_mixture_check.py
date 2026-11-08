#!/usr/bin/env python3
"""
LB 2 点と検証区分の混合比の整合性点検（Issue #42、分析のみ。パイプラインの定数は動かさない）。

縮小パイプラインの LB 2 点（score+prior 0.282、model_c2p 0.246）を、区分ごとの MRR@25 の混合
  予測 = Σ 比_s × MRR_s
で同時に説明できる比を、0.05 刻みの単体格子で全部数える（網羅のためで、値を選ぶのではない）。
|予測 − LB| ≤ k × SE（SE = 0.04、k = 1, 2）を 2 提出とも満たす比を「整合」とする。

区分（MRR はすべて scripts/validate/mrr25.py で計算）:
  C1       v3_c1        outputs/local_eval/v3_c1/submission_{score+prior,ranker_c2p}.csv
  C2       v3_c2        同上
  C2p_env  enveda_v1    同上（プール内・参照なし）
  C2p_n    v3_c2n       同上（プール内・参照なし、COCONUT 構造）
  strong   見える test  outputs/submit/submission_score_prior.csv / submission.csv。lib_sim ≥ 0.99 のプール候補を正解とみなす推定
  unreach  未到達       両方 0
格子の値は 0.05 刻みの上での範囲（解像度 ±0.05）。連続の比での最小・最大も線形計画で併記する（k*_continuous、scipy があるときだけ）。
格子は 3 通り: (a) C2p_env と C2p_n を別の区分として両方、(b) C2p_env だけ、(c) C2p_n だけ。

    python scripts/cv/lb_mixture_check.py
出力: outputs/lb_mixture/result.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))

LB = {"score_prior": 0.282, "ranker": 0.246}    # 提出 58119572 / 58111373（#37 / #34）
SE = 0.04
KS = (1, 2)
STEP_N = 20                                       # 0.05 刻み
STRONG_SIM = 0.99
HOLDOUT_SEGMENTS = {"C1": "v3_c1", "C2": "v3_c2", "C2p_env": "enveda_v1", "C2p_n": "v3_c2n"}
SUBS = {"score_prior": "submission_score+prior.csv", "ranker": "submission_ranker_c2p.csv"}
VARIANTS = {"a_both": ["C1", "C2", "C2p_env", "C2p_n", "strong", "unreach"],
            "b_enveda_v1_only": ["C1", "C2", "C2p_env", "strong", "unreach"],
            "c_v3_c2n_only": ["C1", "C2", "C2p_n", "strong", "unreach"]}


def simplex_grid(n_parts: int, steps: int = STEP_N) -> np.ndarray:
    """和が 1 になる 1/steps 刻みの比をすべて（行 = 1 つの比）。"""
    rows = []
    for bars in itertools.combinations(range(steps + n_parts - 1), n_parts - 1):
        prev, parts = -1, []
        for b in bars:
            parts.append(b - prev - 1); prev = b
        parts.append(steps + n_parts - 2 - prev)
        rows.append(parts)
    return np.asarray(rows, dtype=np.float64) / steps


def predict(grid: np.ndarray, mrr: np.ndarray) -> np.ndarray:
    return grid @ mrr


def consistent(grid: np.ndarray, mrr_by_sub: dict, lb: dict, tol: float) -> np.ndarray:
    ok = np.ones(len(grid), bool)
    for name, m in mrr_by_sub.items():
        ok &= np.abs(predict(grid, m) - lb[name]) <= tol + 1e-12
    return ok


def summarize(grid: np.ndarray, ok: np.ndarray, segs: list[str]) -> dict:
    out = dict(n_grid=int(len(grid)), n_consistent=int(ok.sum()))
    if ok.any():
        g = grid[ok]
        out["share_range"] = {s: [round(float(g[:, i].min()), 2), round(float(g[:, i].max()), 2)] for i, s in enumerate(segs)}
    return out


def lp_bounds(mrr_by_sub: dict, lb: dict, tol: float, segs: list[str]) -> dict | None:
    """格子の解像度（±0.05）を除いた連続の比での各区分の最小・最大（線形計画、scipy があるときだけ）。"""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None
    n = len(segs)
    A_ub, b_ub = [], []
    for name, m in mrr_by_sub.items():
        A_ub += [m, -m]; b_ub += [lb[name] + tol, -(lb[name] - tol)]
    out = {}
    for i, s in enumerate(segs):
        lo_hi = []
        for sign in (1, -1):
            c = np.zeros(n); c[i] = sign
            r = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.ones((1, n)), b_eq=[1.0], bounds=[(0, 1)] * n)
            lo_hi.append(round(float(sign * r.fun), 3) if r.status == 0 else None)
        out[s] = lo_hi
    return out


def strong_mrr(sub_path: Path, feat_path: Path) -> tuple[float, int]:
    """見える test で lib_sim ≥ STRONG_SIM のプール候補（キーの集合）を正解とみなした MRR@25（推定）と対象分子数。"""
    import pandas as pd
    from mrr25 import read_submission, to_key
    ft = pd.read_parquet(feat_path, columns=["molecule_id", "key", "lib_sim", "src_pubchem"])
    ex = ft[(ft.src_pubchem == 0) & (ft.lib_sim >= STRONG_SIM)].groupby("molecule_id").key.apply(set).to_dict()
    sub = read_submission(sub_path)
    rr = []
    for m, keys in ex.items():
        r = 0.0
        for i, s in enumerate(sub.get(m, [])[:25], 1):
            if to_key(s) in keys:
                r = 1.0 / i
                break
        rr.append(r)
    return float(np.mean(rr)), len(rr)


def main() -> None:
    from mrr25 import calculate_mrr25, read_submission, read_truth
    seg_mrr = {"score_prior": {}, "ranker": {}}
    for seg, h in HOLDOUT_SEGMENTS.items():
        truth = read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
        for name, fn in SUBS.items():
            seg_mrr[name][seg] = float(calculate_mrr25(truth, read_submission(ROOT / f"outputs/local_eval/{h}/{fn}")))
    feat = ROOT / "outputs/ranker/features/sub_test.parquet"
    seg_mrr["score_prior"]["strong"], n_strong = strong_mrr(ROOT / "outputs/submit/submission_score_prior.csv", feat)
    seg_mrr["ranker"]["strong"], _ = strong_mrr(ROOT / "outputs/submit/submission.csv", feat)
    for name in seg_mrr:
        seg_mrr[name]["unreach"] = 0.0
    res = dict(lb=LB, se=SE, step=1 / STEP_N, strong_sim=STRONG_SIM, strong_molecules=n_strong,
               segment_mrr={n: {s: round(v, 4) for s, v in d.items()} for n, d in seg_mrr.items()}, variants={})
    for vname, segs in VARIANTS.items():
        grid = simplex_grid(len(segs))
        m = {n: np.array([seg_mrr[n][s] for s in segs]) for n in seg_mrr}
        res["variants"][vname] = dict(segments=segs, **{f"k{k}": summarize(grid, consistent(grid, m, LB, k * SE), segs) for k in KS},
                                      **{f"k{k}_continuous": lp_bounds(m, LB, k * SE, segs) for k in KS})
    out = ROOT / "outputs/lb_mixture/result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
