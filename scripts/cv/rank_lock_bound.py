#!/usr/bin/env python3
"""
Issue #174: 最終 25 件の中で「上位 k 位を固定して k+1〜25 位だけを並べ替える」段（例: 0.495 系統の FR 3-25 は k = 2）が
v3 の 4 集合で届く MRR の伸びの上限を、対照 score+prior+frag の分子ごとの rr から数える。採点はしない（既にある
submission_<対照>.csv と truth.csv を読むだけ）。計画の目安で、採否には使わない。

上限の定義: 正解が k+2 位以下（rr < 1/(k+1)）にある分子は、並べ替えで最善でも k+1 位に上がる → 伸び 1/(k+1) − rr。
k+1 位以内の正解と圏外（rr = 0、25 件に無い）は並べ替えでは伸びない。上限 = Σ max(0, 1/(k+1) − rr) / n（rr > 0 の分子だけ）。
best_case_ci: 最善の場合（動かせる分子が全部 k+1 位に上がり、他は変わらない）の「新 − 対照」を、採否と同じ
`scripts/ranker/train_eval.bootstrap_diff`（2,000 回・seed 0、molecule_id の昇順）に通した 95% 区間。負の側（k+1 位以内の
正解が下がる）を入れない楽観の値で、検出できるかの目安。

    .venv/bin/python scripts/cv/rank_lock_bound.py [--lock 2] [--out outputs/local_eval/rank_lock_bound.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))

HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
CTRL = "score+prior+frag"


def lock_bound(rr: list[float], lock: int) -> dict:
    """rr（1/順位、圏外 0）のリストから、上位 lock 位を固定した並べ替えで届く伸びの上限と内訳を返す。"""
    assert lock >= 0
    best = 1.0 / (lock + 1)
    movable = [x for x in rr if 0 < x < best]
    n = len(rr)
    gain = sum(best - x for x in movable)
    return dict(n=n, lock=lock, n_movable=len(movable),
                n_top=sum(1 for x in rr if x >= best), n_absent=sum(1 for x in rr if x == 0),
                gain_sum=round(gain, 4), gain_bound=round(gain / n, 4) if n else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/rank_lock_bound.json"))
    args = ap.parse_args()
    import mrr25
    sys.path.insert(0, str(ROOT / "scripts/ranker"))
    from train_eval import bootstrap_diff
    best = 1.0 / (args.lock + 1)
    res = {}
    for h in HOLDOUTS:
        truth = mrr25.read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
        pred = mrr25.read_submission(ROOT / f"outputs/local_eval/{h}/submission_{CTRL}.csv")
        ctrl = {m: mrr25.reciprocal_rank(mrr25.to_key(s), pred.get(m, [])) for m, s in truth.items()}
        rr = [ctrl[m] for m in sorted(ctrl)]
        res[h] = lock_bound(rr, args.lock)
        new = {m: (best if 0 < x < best else x) for m, x in ctrl.items()}
        res[h]["best_case_ci"] = bootstrap_diff(new, ctrl)
        res[h]["mrr_ctrl"] = round(sum(rr) / len(rr), 4)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for h, r in res.items():
        print(h, r)


if __name__ == "__main__":
    main()
