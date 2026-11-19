#!/usr/bin/env python3
"""
Issue #160: v3 の 4 集合（v3_c1・v3_c2・enveda_v1・v3_c2n）ごとに、対照 score+prior+frag の分子ごとの rr と、
既に採点済みの変種（nbz・lfus・fragnn など）との分子ごとの差 Δrr から、計画用の MDE ≈ 2.8·sd(Δrr)/√n を数える。
採点はしない（既にある submission_<tag>.csv と truth.csv を読むだけ）。結果は計画の目安で、採否には使わない。

    .venv/bin/python scripts/cv/net_mde.py [--out outputs/local_eval/net_mde.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))

HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
CTRL = "score+prior+frag"
# 対照と同じ候補・同じ後段で 1 つの段だけを足した、既に採点済みの変種（#103 nbz、#150 lfus、#120 fragnn、#60 sanpier、
# #127 align、#97 pmi、#90 elem）。変種を選んで MDE を小さく見せないよう、ある物は全部使う
VARIANTS = ("score+nbz+prior+frag", "score+prior+lfus+frag", "score+fragnn+prior+frag", "score+sanpier+prior+frag",
            "score+align+prior+frag", "score+pmi+prior+frag", "score+prior+frag+elem")
Z = 2.8   # 両側 5%・検出力 80% の (1.96 + 0.84)


def sd(x: list[float]) -> float:
    """標本標準偏差（n−1）。要素が 2 未満なら 0。"""
    n = len(x)
    if n < 2:
        return 0.0
    m = sum(x) / n
    return math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))


def mde(delta: list[float]) -> float:
    return Z * sd(delta) / math.sqrt(len(delta)) if delta else float("nan")


def per_molecule_rr(truth: dict, pred: dict) -> dict:
    import mrr25
    return {m: mrr25.reciprocal_rank(mrr25.to_key(s), pred.get(m, [])) for m, s in truth.items()}


def summarize(ctrl: dict, variants: dict) -> dict:
    """ctrl = {mid: rr}、variants = {tag: {mid: rr}}（同じ分子の集合）。集合 1 つ分の数を返す。"""
    mids = sorted(ctrl)
    rr = [ctrl[m] for m in mids]
    out = dict(n=len(mids), mrr_ctrl=round(sum(rr) / len(rr), 4), sd_rr_ctrl=round(sd(rr), 4),
               mde_indep_upper=round(Z * math.sqrt(2) * sd(rr) / math.sqrt(len(rr)), 4), variants={})
    for tag, v in variants.items():
        assert sorted(v) == mids, f"{tag}: 分子の集合が対照と違う"
        d = [v[m] - ctrl[m] for m in mids]
        out["variants"][tag] = dict(diff=round(sum(d) / len(d), 4), sd_delta=round(sd(d), 4), mde=round(mde(d), 4),
                                    n_changed=sum(1 for x in d if x != 0))
    nz = [x["mde"] for x in out["variants"].values() if x["n_changed"] > 0]
    out["mde_min"] = round(min(nz), 4) if nz else None
    out["mde_median"] = round(statistics.median(nz), 4) if nz else None
    out["mde_max"] = round(max(nz), 4) if nz else None
    return out


def rank_profile(rr: dict) -> dict:
    """rr（1/順位、圏外 0）から、正解の順位が 1 位・2〜25 位・圏外の分子数と、最終 25 件の中の並べ替えだけで届く
    MRR の伸びの上限（集合の n で割る前の和）。unlocked = 2〜25 位の正解を全部 1 位にした場合 Σ(1 − rr)、
    lock1 = 1 位を動かさない場合（届くのは 2 位まで）Σ max(0, 1/2 − rr)。"""
    v = list(rr.values())
    inside = [x for x in v if 0 < x < 1.0]
    return dict(n=len(v), rank1=sum(1 for x in v if x == 1.0), rank2_25=len(inside),
                absent=sum(1 for x in v if x == 0),
                gain_sum_unlocked=round(sum(1.0 - x for x in inside), 4),
                gain_sum_lock1=round(sum(max(0.0, 0.5 - x) for x in inside), 4))


def ctrl_since(path: Path) -> float:
    """今の対照のファイルと中身（sha256）が同じものを、本体と退避（<name>.bak_*）の中から探し、その最も古い mtime を返す。
    これより前に書かれた変種は、別の中身の対照と組で採点されたとみなして除く（mtime による目安）。"""
    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    cur = sha(path)
    same = [p.stat().st_mtime for p in [path, *path.parent.glob(path.name + ".bak_*")] if sha(p) == cur]
    return min(same)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/net_mde.json"))
    args = ap.parse_args()
    import mrr25
    res = {}
    for h in HOLDOUTS:
        truth = mrr25.read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
        sub = ROOT / f"outputs/local_eval/{h}"
        cpath = sub / f"submission_{CTRL}.csv"
        since = ctrl_since(cpath)
        ctrl = per_molecule_rr(truth, mrr25.read_submission(cpath))
        vs, stale = {}, []
        for t in VARIANTS:
            p = sub / f"submission_{t}.csv"
            if not p.exists():
                continue
            if p.stat().st_mtime < since:   # 今の対照の中身ができる前に採点された変種（対照が違うので組にしない）
                stale.append(t)
                continue
            vs[t] = per_molecule_rr(truth, mrr25.read_submission(p))
        res[h] = summarize(ctrl, vs)
        res[h]["stale_excluded"] = stale
        qn = ROOT / f"outputs/local_eval/query_network_{h}.json"
        if qn.exists():   # #101 の相互最近傍でつながる分子の、対照での正解の順位の内訳（網の段が動かせる余地の目安）
            conn = {p[k] for p in json.loads(qn.read_text())["pairs"] for k in ("a", "b")}
            res[h]["connected_ctrl_rank"] = rank_profile({m: ctrl[m] for m in ctrl if m in conn})
        res[h]["all_ctrl_rank"] = rank_profile(ctrl)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for h, r in res.items():
        print(h, {k: r[k] for k in ("n", "mrr_ctrl", "sd_rr_ctrl", "mde_indep_upper", "mde_min", "mde_median", "mde_max")})
        for t, x in r["variants"].items():
            print("   ", t, x)


if __name__ == "__main__":
    main()
