#!/usr/bin/env python3
"""
Issue #60: fold-safe sanpier の段を score+prior+frag に足した効果を検証 v3 で判定する（事前登録どおり）。

前提: 各ホールドアウトで先に
    PYTHONPATH=<rdkit 2026.03.3> .venv/bin/python scripts/local_eval/sanpier.py --holdout <h>
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,prior,frag           # 対照
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,sanpier,prior,frag   # 新
を流してあること。

採否の判定コードは #48 と同じもの（scripts/cv/frag_v3_check.py の judge と per_molecule_rr を import する）:
4 集合すべてで下限 ≥ −0.02、かつ v3_c1・v3_c2 の少なくとも片方で下限 > 0。集合が欠けたら採用しない。

    .venv/bin/python scripts/cv/sanpier_v3_check.py [--out outputs/local_eval/sanpier_v3_check.json]
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
from frag_v3_check import REQUIRED, judge, per_molecule_rr  # noqa: E402  同じ判定コード

NEW_TAG = "score+sanpier+prior+frag"
CTRL_TAG = "score+prior+frag"


def main() -> None:
    import mrr25
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/sanpier_v3_check.json"))
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
    out = dict(issue=60, new=NEW_TAG, ctrl=CTRL_TAG, diffs=res, judge=judge(res))
    print(json.dumps(out["judge"], ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
