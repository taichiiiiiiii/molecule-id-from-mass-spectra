#!/usr/bin/env python3
"""
Issue #49: #46（0.479 版の FP バンクを fold-safe 5 本に差し替え）の見える test の中間出力を、元の版と記述的に比べる。

入力（どちらも Kaggle の見える test 実行の出力。smoke モードで frozen 段が処理したのは 12 分子）:
    outputs/kaggle/lb479_repro_v1/   taichiiiii/casmi26-lb479-repro v1（提出 57877948 = Public 0.479）
    outputs/kaggle/foldsafe5_v1/     #46 の差し替え版（Public 0.416）
各ディレクトリの eng_lists.json / pc_lists.json / merge_report.json / run_manifest.json /
submission_pipeline.log / submission.csv を読む。

出す数（定義はこのファイルの関数どおり。n=12 の記述で、帯の判定はしない。結果を見て定数は動かさない）:
  (i)   eng_lists の上位 40 のキー集合の Jaccard（分子ごと）、1 位のキーが変わる分子数、上位 40 の並びまで同じ分子数
  (ii)  pc_lists の S・fz_top・best_pool_fz と d = fz_top − best_pool_fz の分子ごとの値
        （d は scripts/local_eval/run.py の rel に相当する差の生の値。Kaggle 側の rel と同一定義かは未確認）
  (iii) submission_pipeline.log の「merge stats」行と run_manifest.json の merge の内訳
  (iv)  merge_report.json の overlay（changed_rows・top1_changed・adjacent_swaps）
  (v)   submission.csv で候補が 2 つ以上ある分子（順位付けされた分子）の 1 位が同じ数、と 1 候補だけの行数

    .venv/bin/python scripts/cv/bank_swap_visible.py [--a DIR] [--b DIR] [--out outputs/kaggle/bank_swap_visible.json]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP_ENG = 40


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare_eng(ea: dict, eb: dict, top: int = TOP_ENG) -> dict:
    mids = sorted(set(ea) & set(eb))
    jac = {m: round(jaccard(ea[m]["keys"][:top], eb[m]["keys"][:top]), 4) for m in mids}
    top1 = [m for m in mids if ea[m]["keys"][:1] != eb[m]["keys"][:1]]
    order_same = sum(ea[m]["keys"][:top] == eb[m]["keys"][:top] for m in mids)
    return dict(n=len(mids), only_a=sorted(set(ea) - set(eb)), only_b=sorted(set(eb) - set(ea)), jaccard=jac,
                jaccard_median=round(statistics.median(jac.values()), 4) if jac else None,
                top1_changed=len(top1), top1_changed_ids=top1, order_same=order_same)


def compare_pc(pa: dict, pb: dict) -> dict:
    mids = sorted(set(pa) & set(pb))
    rows = {}
    for m in mids:
        r = {}
        for tag, p in (("a", pa[m]), ("b", pb[m])):
            r[f"S_{tag}"] = round(p["S"], 4)
            r[f"fz_top_{tag}"] = round(p["fz_top"], 3)
            r[f"best_pool_fz_{tag}"] = round(p["best_pool_fz"], 3)
            r[f"d_{tag}"] = round(p["fz_top"] - p["best_pool_fz"], 3)
        r["pc_jaccard"] = round(jaccard(pa[m]["pc_keys"], pb[m]["pc_keys"]), 4)
        r["pc_top1_same"] = pa[m]["pc_keys"][:1] == pb[m]["pc_keys"][:1]
        rows[m] = r

    def med(k):
        return round(statistics.median(r[k] for r in rows.values()), 3) if rows else None
    summ = {k: med(k) for k in ("S_a", "S_b", "fz_top_a", "fz_top_b", "best_pool_fz_a", "best_pool_fz_b", "d_a", "d_b", "pc_jaccard")}
    summ["pc_top1_same"] = sum(r["pc_top1_same"] for r in rows.values())
    return dict(n=len(mids), median=summ, per_molecule=rows)


_MERGE = re.compile(r"merge stats (\{.*\})")


def merge_stats_from_log(text: str) -> dict | None:
    hits = _MERGE.findall(text)
    return ast.literal_eval(hits[-1]) if hits else None


def read_sub(path: Path) -> dict[str, list[str]]:
    with open(path, newline="") as f:
        return {r["molecule_id"]: [s for s in r["smiles"].split(";") if s] for r in csv.DictReader(f)}


def compare_sub(sa: dict, sb: dict) -> dict:
    ranked_a = {m for m, c in sa.items() if len(c) >= 2}
    ranked_b = {m for m, c in sb.items() if len(c) >= 2}
    both = sorted(ranked_a & ranked_b)
    return dict(rows_a=len(sa), rows_b=len(sb), ranked_a=len(ranked_a), ranked_b=len(ranked_b), ranked_both=len(both),
                top1_same=sum(sa[m][:1] == sb[m][:1] for m in both),
                single_a=sum(len(c) == 1 for c in sa.values()), single_b=sum(len(c) == 1 for c in sb.values()),
                single_both=sum(len(sa[m]) == 1 and len(sb.get(m, [])) == 1 for m in sa))


def load(d: Path) -> dict:
    j = lambda n: json.loads((d / n).read_text())  # noqa: E731
    return dict(eng=j("eng_lists.json"), pc=j("pc_lists.json"), merge_report=j("merge_report.json"),
                manifest=j("run_manifest.json"), log_merge=merge_stats_from_log((d / "submission_pipeline.log").read_text()),
                sub=read_sub(d / "submission.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(ROOT / "outputs/kaggle/lb479_repro_v1"))
    ap.add_argument("--b", default=str(ROOT / "outputs/kaggle/foldsafe5_v1"))
    ap.add_argument("--out", default=str(ROOT / "outputs/kaggle/bank_swap_visible.json"))
    args = ap.parse_args()
    A, B = load(Path(args.a)), load(Path(args.b))
    ov = lambda r: {k: r.get("overlay", {}).get(k) for k in ("changed_rows", "top1_changed", "adjacent_swaps")}  # noqa: E731
    out = dict(
        a=args.a, b=args.b,
        i_eng=compare_eng(A["eng"], B["eng"]),
        ii_pc=compare_pc(A["pc"], B["pc"]),
        iii_merge=dict(log_a=A["log_merge"], log_b=B["log_merge"], manifest_a=A["manifest"].get("merge"), manifest_b=B["manifest"].get("merge")),
        iv_overlay=dict(a=ov(A["merge_report"]), b=ov(B["merge_report"])),
        v_sub=compare_sub(A["sub"], B["sub"]),
    )
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "ii_pc"} | {"ii_pc_median": out["ii_pc"]["median"]}, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
