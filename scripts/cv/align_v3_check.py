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

blind 対照（#109、2026-11-15 事前登録。judge には使わない）: submission_score+alignblind+prior+frag.csv があれば、
Δs = full − blind の bootstrap_diff（2,000 回、seed 0）と参考の blind − 対照 を json の blind 節に書く。
v3_c1 の Δs の下限 > 0 のときだけ「スペクトル由来の情報を含む」（0.487 系統の診断提出に進める）。v3_c1 が無ければ進まない。

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
BLIND_TAG = "score+alignblind+prior+frag"   # #109 blind 対照（2026-11-15 事前登録）。judge には使わない


def judge(diffs: dict[str, dict]) -> dict:
    """diffs[h] = bootstrap_diff の結果（lo を持つ）。事前固定の規則だけで採否を出す。"""
    missing = [h for h in REQUIRED if h not in diffs]
    failing = [h for h in REQUIRED if h in diffs and diffs[h]["lo"] < NONINF]
    win = WIN in diffs and diffs[WIN]["lo"] > 0
    adopt = not missing and not failing and win
    return dict(adopt=bool(adopt), missing=missing, failing=failing, v3_c1_lo_gt_0=bool(win), noninf=NONINF)


def g_blind_info(src: Path | None = None) -> dict:
    """b を作った本数と ‖b‖（train_report.json の blind 節）。無ければ missing。"""
    p = Path(src or ROOT / "outputs/align_train") / "train_report.json"
    if not p.exists():
        return {"missing": True}
    return json.loads(p.read_text()).get("blind") or {"missing": True}


def blind_reading(blind_diffs: dict[str, dict]) -> dict:
    """blind_diffs[h] = bootstrap_diff(full, blind)。v3_c1 の下限 > 0 のときだけ「スペクトル由来の情報を含む」。
    v3_c1 が無い（測れない）ときは下限 ≤ 0 と同じ扱い。0.487 系統への診断提出に進めるのは spectral のときだけ。"""
    missing = [h for h in REQUIRED if h not in blind_diffs]
    spectral = WIN in blind_diffs and blind_diffs[WIN]["lo"] > 0
    return dict(spectral=bool(spectral), proceed_to_0487_diagnostic=bool(spectral), missing=missing,
                note="伸びはスペクトル由来の情報を含む" if spectral else "伸びはスペクトル由来と言えない")


def main() -> None:
    import mrr25
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/align_v3_check.json"))
    args = ap.parse_args()
    res: dict[str, dict] = {}
    blind: dict[str, dict] = {}
    blind_ref: dict[str, dict] = {}
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
        bl_p = ROOT / f"outputs/local_eval/{h}/submission_{BLIND_TAG}.csv"
        if bl_p.exists():
            c = per_molecule_rr(truth, mrr25.read_submission(bl_p))
            ds = bootstrap_diff(a, c)
            ds.update(n=len(c), full=d["new"], blind=round(sum(c.values()) / len(c), 4))
            blind[h] = ds
            blind_ref[h] = bootstrap_diff(c, b)
            print(h, "full-blind", ds)
        else:
            print(f"[warn] {h}: blind の CSV が無い（{bl_p}）")
    out = dict(new=NEW_TAG, ctrl=CTRL_TAG, diffs=res, judge=judge(res),
               blind=dict(tag=BLIND_TAG, full_minus_blind=blind, blind_minus_ctrl_reference=blind_ref,
                          reading=blind_reading(blind), g_blind=g_blind_info()))
    print(json.dumps(out["judge"], ensure_ascii=False))
    print(json.dumps(out["blind"]["reading"], ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
