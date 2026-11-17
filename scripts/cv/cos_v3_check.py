#!/usr/bin/env python3
"""
Issue #140: cos 段（候補の点を cos(σ(z), fp) にする）を score+prior+frag の score の代わりに使った効果を、
検証 v3 の 4 集合で判定する（事前登録どおり。規則は #48 の frag_v3_check.judge をそのまま import する）。

前提: 各ホールドアウトで
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages score,prior,frag   # 対照（流し直して 4 値の一致を確認）
    .venv/bin/python scripts/local_eval/run.py --holdout <h> --stages cos,prior,frag     # 新（1 回だけ）
を流してあること。

採否（#140 で事前固定）: judge = 4 集合 v3_c1・v3_c2・enveda_v1・v3_c2n のすべてで「新 − 対照」の 95% 下限 ≥ −0.02、
かつ v3_c1・v3_c2 の少なくとも片方で下限 > 0。欠けたら採用しない。あわせて C1（v3_c1 の下限 > 0）と
C2（v3_c2 の下限 ≥ −0.02）を報告する。

記録だけ（採否に使わない）: top-1 が変わった分子の数、flip_counts（#118 と同じ定義）、同じ分子式の中の順位
（iso_rank_diag と同じ窓・キーのまとめ方・G・悲観的な順位で、(A) fp @ z と (E) cos）。

    .venv/bin/python scripts/cv/cos_v3_check.py [--out outputs/local_eval/cos_v3_check.json] [--no-iso]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/cv"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

from frag_v3_check import NONINF, REQUIRED, judge, per_molecule_rr  # noqa: E402,F401  規則はコピーしない

NEW_TAG = "cos+prior+frag"
CTRL_TAG = "score+prior+frag"


def conditions(diffs: dict[str, dict]) -> dict:
    """C1: v3_c1 の下限 > 0。C2: v3_c2 の下限 ≥ −0.02（集合が無ければ False）。"""
    return dict(C1=bool("v3_c1" in diffs and diffs["v3_c1"]["lo"] > 0),
                C2=bool("v3_c2" in diffs and diffs["v3_c2"]["lo"] >= NONINF))


def top1_changed(ctrl: dict[str, list[str]], new: dict[str, list[str]], mids) -> int:
    """1 位の InChIKey14 が対照と新で違う分子の数（どちらも空なら同じとみなす）。"""
    from mrr25 import to_key

    def first(p, m):
        c = p.get(m) or []
        return to_key(c[0]) if c else None
    return sum(1 for m in mids if first(ctrl, m) != first(new, m))


def iso_diag(holdout: str) -> dict:
    """同じ分子式の中の (A) fp @ z と (E) cos の順位（iso_rank_diag と同じ定義。E は aggregate_by_key の n 欄に入れる）。"""
    import iso_rank_diag as I
    import library as L
    import run as R
    from mrr25 import read_truth, to_key
    from train_eval import bootstrap_diff
    zf = np.load(ROOT / f"outputs/holdout_eval/{holdout}_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = read_truth(ROOT / f"outputs/holdouts/{holdout}/truth.csv")
    hidden = L.drop_keys(holdout) if L.hide_rule(holdout).get("pool") == "remove" else set()
    per, excl = {}, {}
    for mid in truth:
        if mid not in targets:
            continue
        idx = R.visible_window(pool, targets[mid], hidden)
        z = R.stage_z(zf[mid], None)
        cos = {i: s for s, i in R.cos_window(idx, pool, z, fpc)}
        recs = [dict(key=pool.key[i], row=i, smiles=pool.smiles[i], s=s, n=cos[i]) for s, i in R.score_window(idx, pool, z, fpc)]
        cd = I.aggregate_by_key(recs)
        tk = to_key(truth[mid]) if truth[mid] else None
        G, info = I.build_group(cd, tk, I.formula_of_smiles)
        if G is None:
            excl[mid] = info["excluded"]
            continue
        ti = G.index(tk)
        per[mid] = dict(A=I.pessimistic_rank([cd[k]["s"] for k in G], ti), E=I.pessimistic_rank([cd[k]["n"] for k in G], ti),
                        g_size=len(G))
    out = dict(n=len(per), excluded={r: sum(1 for v in excl.values() if v == r) for r in ("truth_absent", "truth_formula_none", "g_lt2")})
    if per:
        rr = {c: {m: 1.0 / p[c] for m, p in per.items()} for c in ("A", "E")}
        out.update(mean_rr=dict(A_dot=round(float(np.mean(list(rr["A"].values()))), 4), E_cos=round(float(np.mean(list(rr["E"].values()))), 4)),
                   top1_rate=dict(A_dot=round(float(np.mean([p["A"] == 1 for p in per.values()])), 4),
                                  E_cos=round(float(np.mean([p["E"] == 1 for p in per.values()])), 4)),
                   g_size_median=float(np.median([p["g_size"] for p in per.values()])),
                   E_minus_A=bootstrap_diff(rr["E"], rr["A"]))
    return out


def flips(holdout: str, ctrl: dict, new: dict, truth: dict) -> dict:
    import csv
    import iso_rank_diag as I
    from mrr25 import to_key
    trows = {r["molecule_id"]: r for r in csv.DictReader(open(ROOT / f"outputs/holdouts/{holdout}/truth.csv"))}
    cls = {}
    for tag, preds in (("control", ctrl), ("new", new)):
        cls[tag] = {}
        for mid, t in truth.items():
            c = (preds.get(mid) or [])[:25]
            tsmi = trows[mid].get("smiles") or ""
            cls[tag][mid] = I.classify_top1(to_key(t), [to_key(s) for s in c],
                                            I.formula_of_smiles(tsmi) if tsmi else None, [I.formula_of_smiles(s) for s in c[:1]])
    return I.flip_counts(cls["control"], cls["new"])


def main() -> None:
    import mrr25
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/cos_v3_check.json"))
    ap.add_argument("--no-iso", action="store_true", help="同じ分子式の中の順位（記録だけ）を省く")
    args = ap.parse_args()

    t0 = time.time()
    res: dict[str, dict] = {}
    rec: dict[str, dict] = {}
    for h in REQUIRED:
        truth_p = ROOT / f"outputs/holdouts/{h}/truth.csv"
        new_p = ROOT / f"outputs/local_eval/{h}/submission_{NEW_TAG}.csv"
        ctl_p = ROOT / f"outputs/local_eval/{h}/submission_{CTRL_TAG}.csv"
        if not (truth_p.exists() and new_p.exists() and ctl_p.exists()):
            print(f"[warn] {h}: 入力が無い（{truth_p.exists()=}, {new_p.exists()=}, {ctl_p.exists()=}）")
            continue
        truth = mrr25.read_truth(truth_p)
        pn, pc = mrr25.read_submission(new_p), mrr25.read_submission(ctl_p)
        a, b = per_molecule_rr(truth, pn), per_molecule_rr(truth, pc)
        d = bootstrap_diff(a, b)
        d.update(n=len(a), new=round(sum(a.values()) / len(a), 4), ctrl=round(sum(b.values()) / len(b), 4))
        res[h] = d
        rec[h] = dict(top1_changed=top1_changed(pc, pn, list(truth)), flip=flips(h, pc, pn, truth))
        if not args.no_iso:
            rec[h]["iso_rank"] = iso_diag(h)
        print(h, d, rec[h], flush=True)
    out = dict(issue=140, new=NEW_TAG, ctrl=CTRL_TAG, diffs=res, judge=judge(res), conditions=conditions(res),
               record_only=rec, secs=round(time.time() - t0, 1))
    print(json.dumps(dict(judge=out["judge"], conditions=out["conditions"]), ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
