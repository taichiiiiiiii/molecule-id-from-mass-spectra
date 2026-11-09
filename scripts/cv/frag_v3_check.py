#!/usr/bin/env python3
"""
Issue #48: MetFrag-lite（frag）を score+prior に足した効果を検証 v3 で判定する（事前登録どおり）。

前提: 各ホールドアウトで先に
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,prior
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,prior,frag
を流してあること（FRAG_LAM=0.5・TOPN は run.py の既定値のまま。動かさない）。

分子ごとの rr は result JSON に無いので、outputs/local_eval/<h>/submission_<tag>.csv と
outputs/holdouts/<h>/truth.csv から mrr25.reciprocal_rank で計算する。差は train_eval.bootstrap_diff
（分子ブートストラップ 2,000 回、seed 0）の「新 − 対照」の 95% 区間。

採否（#48 で事前固定）: v3_c1 と v3_c2 の両方で下限 ≥ −0.02、かつ少なくとも片方で下限 > 0。
加えて enveda_v1 / v3_c2n でも下限 ≥ −0.02。必須の集合が 1 つでも欠けたら採用しない。

    .venv/bin/python scripts/cv/frag_v3_check.py [--out outputs/local_eval/frag_v3_check.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))

NONINF = -0.02
PRIMARY = ("v3_c1", "v3_c2")          # 下限 ≥ −0.02、かつどちらかで下限 > 0
SECONDARY = ("enveda_v1", "v3_c2n")   # 下限 ≥ −0.02
REQUIRED = PRIMARY + SECONDARY
NEW_TAG = "score+prior+frag"
CTRL_TAG = "score+prior"


def per_molecule_rr(truth: dict[str, str], pred: dict[str, list[str]]) -> dict[str, float]:
    """truth の全分子について rr（予測が無い分子は 0）。"""
    import mrr25
    return {m: mrr25.reciprocal_rank(mrr25.to_key(s), pred.get(m, [])) for m, s in truth.items()}


def judge(diffs: dict[str, dict]) -> dict:
    """diffs[h] = bootstrap_diff の結果（lo を持つ）。事前固定の規則だけで採否を出す。"""
    missing = [h for h in REQUIRED if h not in diffs]
    failing = [h for h in REQUIRED if h in diffs and diffs[h]["lo"] < NONINF]
    win = [h for h in PRIMARY if h in diffs and diffs[h]["lo"] > 0]
    adopt = not missing and not failing and bool(win)
    return dict(adopt=adopt, missing=missing, failing=failing, significant_win=win, noninf=NONINF)


def main() -> None:
    import mrr25
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/frag_v3_check.json"))
    args = ap.parse_args()

    res: dict[str, dict] = {}
    for h in REQUIRED:
        truth_p = ROOT / f"outputs/holdouts/{h}/truth.csv"
        new_p = ROOT / f"outputs/local_eval/{h}/submission_{NEW_TAG}.csv"
        ctl_p = ROOT / f"outputs/local_eval/{h}/submission_{CTRL_TAG}.csv"
        if not (truth_p.exists() and new_p.exists() and ctl_p.exists()):
            print(f"[warn] {h}: 入力が無い（{truth_p.exists()=}, {new_p.exists()=}, {ctl_p.exists()=}）")
            continue
        truth = mrr25.read_truth(truth_p)
        a = per_molecule_rr(truth, mrr25.read_submission(new_p))
        b = per_molecule_rr(truth, mrr25.read_submission(ctl_p))
        d = bootstrap_diff(a, b)
        d.update(n=len(a), new=round(sum(a.values()) / len(a), 4), ctrl=round(sum(b.values()) / len(b), 4))
        res[h] = d
        print(h, d)
    out = dict(issue=48, new=NEW_TAG, ctrl=CTRL_TAG, diffs=res, judge=judge(res))
    print(json.dumps(out["judge"], ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
