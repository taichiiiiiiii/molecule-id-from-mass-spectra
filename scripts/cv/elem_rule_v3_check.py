#!/usr/bin/env python3
"""
Issue #90: 自前の規則 `elem`（天然物らしくない元素を含む候補を後ろへ、定数なし）を検証 v3 の 4 集合で判定する。

前提: 各ホールドアウトで先に
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,prior,frag        # 対照
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,prior,frag,elem   # 新
を流してあること。

採否は #90 で事前固定した規則（frag_v3_check.judge をそのまま import）:
4 集合すべてで bootstrap_diff（分子 2,000 回、seed 0）の (新 − 対照) 95% 区間の下限 ≥ −0.02、
かつ v3_c1・v3_c2 の少なくとも片方で下限 > 0。1 集合でも欠ければ不採用。

数え上げ（記録用、採否には使わない）: 正解に対象元素を含む分子数 / 対照の上位 25 に対象元素の候補を含む分子数 /
対照で正解（上位 25 内）より上に対象元素の候補がある分子数 / elem で上位 25 の並びが変わった分子数。

    .venv/bin/python scripts/cv/elem_rule_v3_check.py [--out outputs/local_eval/elem_rule_v3_check.json]
    .venv/bin/python scripts/cv/elem_rule_v3_check.py --visible <submission.csv> --out <json>   # 0.487 写しの目安
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/cv"))

from elem_rule import has_unusual_element  # noqa: E402
from frag_v3_check import REQUIRED, judge, per_molecule_rr  # noqa: E402

NEW_TAG = "score+prior+frag+elem"
CTRL_TAG = "score+prior+frag"


def count_molecule(truth_smi: str, ctrl: list[str], new: list[str], to_key) -> dict[str, bool]:
    """1 分子の数え上げ。ctrl / new は上位 25 の SMILES。"""
    top = ctrl[:25]
    tk = to_key(truth_smi)
    keys = [to_key(s) for s in top]
    unusual = [has_unusual_element(s) for s in top]
    pos = keys.index(tk) if tk is not None and tk in keys else None
    return dict(truth_unusual=has_unusual_element(truth_smi),
                ctrl_top25_has_unusual=any(unusual),
                unusual_above_truth=pos is not None and any(unusual[:pos]),
                elem_changed_top25=top != new[:25])


def counts(truth: dict[str, str], ctrl: dict, new: dict, to_key) -> dict[str, int]:
    tot: dict[str, int] = {}
    for m, s in truth.items():
        for k, v in count_molecule(s, ctrl.get(m, []), new.get(m, []), to_key).items():
            tot[k] = tot.get(k, 0) + int(v)
    tot["n"] = len(truth)
    return tot


def visible_counts(sub: dict[str, list[str]]) -> dict[str, int]:
    """visible test の提出ファイルで、上位 25 に対象元素を含む分子数と 1 位が対象元素の分子数。"""
    return dict(n=len(sub),
                top25_has_unusual=sum(any(has_unusual_element(s) for s in c[:25]) for c in sub.values()),
                top1_unusual=sum(bool(c) and has_unusual_element(c[0]) for c in sub.values()),
                n_candidates=sum(len(c[:25]) for c in sub.values()),
                n_unusual_candidates=sum(has_unusual_element(s) for c in sub.values() for s in c[:25]))


def main() -> None:
    import mrr25
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/elem_rule_v3_check.json"))
    ap.add_argument("--visible", default=None, help="0.487 写しの submission.csv（読むだけの目安）")
    args = ap.parse_args()

    if args.visible:
        out = dict(issue=90, visible=args.visible, counts=visible_counts(mrr25.read_submission(Path(args.visible))),
                   note="visible test は train の写しなので、隠れテストでの効果の推定ではない")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        return

    res: dict[str, dict] = {}
    cnt: dict[str, dict] = {}
    for h in REQUIRED:
        truth_p = ROOT / f"outputs/holdouts/{h}/truth.csv"
        new_p = ROOT / f"outputs/local_eval/{h}/submission_{NEW_TAG}.csv"
        ctl_p = ROOT / f"outputs/local_eval/{h}/submission_{CTRL_TAG}.csv"
        if not (truth_p.exists() and new_p.exists() and ctl_p.exists()):
            print(f"[warn] {h}: 入力が無い（{truth_p.exists()=}, {new_p.exists()=}, {ctl_p.exists()=}）")
            continue
        truth = mrr25.read_truth(truth_p)
        new, ctl = mrr25.read_submission(new_p), mrr25.read_submission(ctl_p)
        a = per_molecule_rr(truth, new)
        b = per_molecule_rr(truth, ctl)
        d = bootstrap_diff(a, b)
        d.update(n=len(a), new=round(sum(a.values()) / len(a), 4), ctrl=round(sum(b.values()) / len(b), 4))
        res[h] = d
        cnt[h] = counts(truth, ctl, new, mrr25.to_key)
        print(h, d, cnt[h])
    out = dict(issue=90, new=NEW_TAG, ctrl=CTRL_TAG, diffs=res, counts=cnt, judge=judge(res))
    print(json.dumps(out["judge"], ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
