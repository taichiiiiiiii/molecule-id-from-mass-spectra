#!/usr/bin/env python3
"""
Issue #150: lfus 段（ライブラリの類縁体から点を伝える、NAP の Fusion 型）を score+prior+frag に足した効果を、検証 v3 の
4 集合で判定する（事前登録どおり。nbz_v3_check.py と同じ型。frag_v3_check.py の judge は使わない）。

前提: 各ホールドアウトで
    scripts/local_eval/lfus.py build --holdout <h>                        （類縁検索・印・混入の検査・段 0）
    scripts/local_eval/run.py --holdout <h> --stages score,prior,frag      （対照を変更後のコードで流し直し、
                                                                            前のファイル submission_score+prior+frag.before.csv と cmp）
    scripts/local_eval/run.py --holdout <h> --stages score,prior,lfus,frag （新、1 回だけ）

採否（#150 で事前固定）: 4 集合 v3_c1・v3_c2・enveda_v1・v3_c2n のすべてで「新 − 対照」の 95% 下限 ≥ −0.02、
かつ v3_c1 の下限 > 0。必須の集合が 1 つでも欠けたら採用しない。v3_c2 は新・対照の rr がすべて 0 であることを assert する
（正解がプールに無い）。対照の MRR は今の値（下の CTRL_EXPECTED）と一致を assert、drop_keys の sha256 は
drop_keys_manifest.json の after.sha256 と一致を assert。MDE は記述のためだけで、採否には使わない。

    python scripts/cv/lfus_v3_check.py [--out outputs/local_eval/lfus_v3_check.json]
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("LOCAL_EVAL_ROOT", CODE))
sys.path.insert(0, str(CODE / "scripts/validate"))
sys.path.insert(0, str(CODE / "scripts/ranker"))
sys.path.insert(0, str(CODE / "scripts/cv"))

NONINF = -0.02
WIN_REQUIRED = "v3_c1"
REQUIRED = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
NEW_TAG = "score+prior+lfus+frag"
CTRL_TAG = "score+prior+frag"
CTRL_EXPECTED = {"v3_c1": 0.8535, "v3_c2": 0.0, "enveda_v1": 0.8376, "v3_c2n": 0.6284}
MDE = {"v3_c1": 0.0176, "v3_c2": None, "enveda_v1": 0.0175, "v3_c2n": 0.0171}   # #150 本文（v3_c2 は該当なし）
LEAK_KEYS = ("leak_rep_drop_keys", "leak_rep_source_keys", "leak_rep_meta_lib", "leak_analog_hits")


def judge(diffs: dict[str, dict]) -> dict:
    """diffs[h] = bootstrap_diff の結果（lo を持つ）。#150 の事前固定の規則だけで採否を出す。"""
    missing = [h for h in REQUIRED if h not in diffs]
    failing = [h for h in REQUIRED if h in diffs and diffs[h]["lo"] < NONINF]
    win = WIN_REQUIRED in diffs and diffs[WIN_REQUIRED]["lo"] > 0
    adopt = not missing and not failing and win
    below_mde = bool(adopt and MDE[WIN_REQUIRED] is not None and diffs[WIN_REQUIRED]["diff"] < MDE[WIN_REQUIRED])
    return dict(adopt=bool(adopt), missing=missing, failing=failing, v3_c1_lo_gt_0=bool(win), noninf=NONINF,
                adopted_below_planning_mde=below_mde)


def split_summary(a: dict, b: dict, applied: set) -> dict:
    """記録用: 類縁体で点を変えた分子だけ・変えなかった分子だけの差（採否には使わない）。"""
    out = {}
    for name, ms in (("applied", [m for m in a if m in applied]), ("not_applied", [m for m in a if m not in applied])):
        d = [a[m] - b[m] for m in ms]
        out[name] = dict(n=len(ms), diff=round(sum(d) / len(d), 4) if d else None,
                         better=sum(x > 0 for x in d), worse=sum(x < 0 for x in d))
    return out


def check_inputs(h: str, rj: dict, aj: dict, manifest: dict, ctrl_mrr: float, analogs_sha256: str | None = None) -> dict:
    """事前登録の前提の assert（対照の値・drop_keys の sha256・行の一致・混入 0 件・新が読んだ類縁体の json）。記録を返す。"""
    assert round(ctrl_mrr, 4) == CTRL_EXPECTED[h], f"{h}: 対照 {ctrl_mrr} が事前登録の {CTRL_EXPECTED[h]} と違う"
    assert rj["lfus_drop_keys_sha256"] == manifest["after"]["sha256"] == aj["drop_keys_sha256"], f"{h}: drop_keys の sha256 が違う"
    assert aj["align"]["all_ok"], f"{h}: 行の一致の確認が通っていない"
    assert all(aj["audit"][k] == 0 for k in LEAK_KEYS), f"{h}: 混入 {aj['audit']}"
    assert aj["limit"] is None and rj["lfus_analogs_sha256"], f"{h}: 全分子の build でない"
    if analogs_sha256 is not None:
        assert rj["lfus_analogs_sha256"] == analogs_sha256, f"{h}: 新が読んだ lfus_analogs.json と今のファイルが違う"
    return dict(align=aj["align"], mask=aj["mask"], audit=aj["audit"], drop_keys_sha256=aj["drop_keys_sha256"],
                stage0=aj["stage0"], build_timing=aj["timing"], build_secs=aj["secs_total"])


def main() -> None:
    import mrr25
    from frag_v3_check import per_molecule_rr
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/lfus_v3_check.json"))
    args = ap.parse_args()

    res: dict[str, dict] = {}
    inputs: dict[str, dict] = {}
    for h in REQUIRED:
        d_ = ROOT / f"outputs/local_eval/{h}"
        truth_p = ROOT / f"outputs/holdouts/{h}/truth.csv"
        new_p, ctl_p = d_ / f"submission_{NEW_TAG}.csv", d_ / f"submission_{CTRL_TAG}.csv"
        before_p = d_ / f"submission_{CTRL_TAG}.before.csv"
        rj_p, aj_p = d_ / f"result_{NEW_TAG}.json", d_ / "lfus_analogs.json"
        need = [truth_p, new_p, ctl_p, before_p, rj_p, aj_p]
        if not all(p.exists() for p in need):
            print(f"[warn] {h}: 入力が無い {[str(p) for p in need if not p.exists()]}")
            continue
        truth = mrr25.read_truth(truth_p)
        a = per_molecule_rr(truth, mrr25.read_submission(new_p))
        b = per_molecule_rr(truth, mrr25.read_submission(ctl_p))
        ctrl_same = filecmp.cmp(ctl_p, before_p, shallow=False)
        assert ctrl_same, f"{h}: 変更後のコードの対照が前の submission と違う"
        rj, aj = json.loads(rj_p.read_text()), json.loads(aj_p.read_text())
        manifest = json.loads((ROOT / f"data/holdouts/{h}/drop_keys_manifest.json").read_text())
        import hashlib
        inputs[h] = check_inputs(h, rj, aj, manifest, sum(b.values()) / len(b), hashlib.sha256(aj_p.read_bytes()).hexdigest()) \
            | dict(ctrl_rerun_cmp_identical=ctrl_same)
        if h == "v3_c2":
            assert all(v == 0 for v in a.values()) and all(v == 0 for v in b.values()), "v3_c2 で rr が 0 でない分子がある"
        d = bootstrap_diff(a, b)
        d.update(n=len(a), new=round(sum(a.values()) / len(a), 4), ctrl=round(sum(b.values()) / len(b), 4),
                 mde_planning=MDE[h], lfus_n_applied=rj.get("lfus_n_applied"), lfus_n_fp_fail=rj.get("lfus_n_fp_fail"),
                 split=split_summary(a, b, set(rj.get("lfus_applied", []))))
        res[h] = d
        print(h, {k: v for k, v in d.items() if k != "split"})
    out = dict(issue=150, new=NEW_TAG, ctrl=CTRL_TAG, diffs=res, judge=judge(res), inputs=inputs,
               C1=res.get("v3_c1"), C2=res.get("v3_c2"))
    print(json.dumps(out["judge"], ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
