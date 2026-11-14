#!/usr/bin/env python3
"""整列チャネル（凍結 DreaMS ＋ 自前学習の整列 MLP）の段を score+prior+frag に足した効果を検証 v3 で判定する（事前登録どおり）。

前提: 各ホールドアウトで先に
    .venv/bin/python scripts/local_eval/align.py --holdout <h>
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,prior,frag         # 対照
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,align,prior,frag   # 新
を流してあること。

差は #48・#60 と同じ per_molecule_rr ＋ train_eval.bootstrap_diff（2,000 回、seed 0）の「新 − 対照」の 95% 区間。
採否（文書 §(e) で事前固定、#48 の judge とは勝ちの条件が違う）: 4 集合すべてで下限 ≥ −0.02、かつ **v3_c1** の下限 > 0。
v3_c2 は ≥ −0.02 の確認にだけ使う。必須の集合が 1 つでも欠けたら採用しない。

    .venv/bin/python scripts/cv/align_v3_check.py [--out outputs/local_eval/align_v3_check.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cv"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
from frag_v3_check import per_molecule_rr  # noqa: E402  同じ rr の数え方

NONINF = -0.02
REQUIRED = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
WIN = "v3_c1"
NEW_TAG = "score+align+prior+frag"
CTRL_TAG = "score+prior+frag"


def judge(diffs: dict[str, dict]) -> dict:
    """diffs[h] = bootstrap_diff の結果（lo を持つ）。事前固定の規則だけで採否を出す。"""
    missing = [h for h in REQUIRED if h not in diffs]
    failing = [h for h in REQUIRED if h in diffs and diffs[h]["lo"] < NONINF]
    win = WIN in diffs and diffs[WIN]["lo"] > 0
    adopt = not missing and not failing and win
    return dict(adopt=bool(adopt), missing=missing, failing=failing, v3_c1_lo_gt_0=bool(win), noninf=NONINF)


def main() -> None:
    import mrr25
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/align_v3_check.json"))
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
    out = dict(new=NEW_TAG, ctrl=CTRL_TAG, diffs=res, judge=judge(res))
    print(json.dumps(out["judge"], ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
