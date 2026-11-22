#!/usr/bin/env python3
"""
Issue #196 手順 3（記録だけ）: fragnn v1 の (v1 − 対照) を問い合わせの形で層に分けて数える。

v2 の事前登録が確定した後（#196 の 2026-11-22T16:20Z のコメント）に作った。v2 の中身・定数・集合の
選択には使わない（記録だけ）。新しい採点はせず、#120 の判定で書かれた既存の submission CSV の
分子ごとの rr だけを読む。

- 対照 = score+prior+frag、v1 = score+fragnn+prior+frag（#120 と同じタグ）
- 集合: v3_c1・enveda_v1（計画の手順 3 の 2 つ）
- 層（この docstring で値を見る前に固定）:
  * multimer: 分子の問い合わせに [2M…]（adduct が ^\\[\\s*([2-9]|\\d{2,})\\s*M）が 1 本でもあるか
  * n_spec: 問い合わせのスペクトルの本数 1–2 / 3–5 / 6 以上（test.parquet の分子ごとの行数）
  * peaks: 分子の問い合わせのピーク数（ms2_mzs の長さ）の中央値 <50 / 50–199 / 200 以上
- rr は scripts/cv/frag_v3_check.per_molecule_rr（このブランチの mrr25、#120 の判定と同じ版）

    python scripts/research/fragnn_v1_strata.py [--out outputs/local_eval/shift_1123/v1_strata.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CTRL_TAG = "score+prior+frag"
NEW_TAG = "score+fragnn+prior+frag"
HOLDOUTS = ("v3_c1", "enveda_v1")
MULTIMER = re.compile(r"^\[\s*([2-9]|\d{2,})\s*M")


def is_multimer(adduct: str | None) -> bool:
    return bool(adduct) and bool(MULTIMER.match(str(adduct)))


def n_spec_bin(n: int) -> str:
    return "1-2" if n <= 2 else ("3-5" if n <= 5 else "6+")


def peaks_bin(med: float) -> str:
    return "<50" if med < 50 else ("50-199" if med < 200 else "200+")


def molecule_strata(rows: Sequence[Mapping]) -> dict[str, dict[str, str]]:
    """rows: molecule_id・adduct・n_peaks を持つ問い合わせの行。分子ごとの層の名前。"""
    by: dict[str, list[Mapping]] = {}
    for r in rows:
        by.setdefault(r["molecule_id"], []).append(r)
    out = {}
    for m, rs in by.items():
        out[m] = dict(
            multimer="with_multimer" if any(is_multimer(r["adduct"]) for r in rs) else "no_multimer",
            n_spec=n_spec_bin(len(rs)),
            peaks=peaks_bin(statistics.median(r["n_peaks"] for r in rs)),
        )
    return out


def summarize(rr_new: Mapping[str, float], rr_ctl: Mapping[str, float],
              strata: Mapping[str, Mapping[str, str]]) -> dict:
    """層ごとの n・mean(新 − 対照)・mean 対照・mean 新。対照の全分子が対象。層の無い分子は 'missing'。"""
    res: dict = {}
    mols = list(rr_ctl)
    res["all"] = _agg(mols, rr_new, rr_ctl)
    for dim in ("multimer", "n_spec", "peaks"):
        groups: dict[str, list[str]] = {}
        for m in mols:
            groups.setdefault(strata.get(m, {}).get(dim, "missing"), []).append(m)
        res[dim] = {g: _agg(ms, rr_new, rr_ctl) for g, ms in sorted(groups.items())}
    return res


def _agg(ms, rr_new, rr_ctl) -> dict:
    d = [rr_new.get(m, 0.0) - rr_ctl[m] for m in ms]
    n = len(ms)
    return dict(n=n,
                mean_diff=round(sum(d) / n, 4) if n else None,
                mean_ctrl=round(sum(rr_ctl[m] for m in ms) / n, 4) if n else None,
                mean_new=round(sum(rr_new.get(m, 0.0) for m in ms) / n, 4) if n else None,
                n_better=sum(x > 0 for x in d), n_worse=sum(x < 0 for x in d))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/shift_1123/v1_strata.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    import pyarrow.parquet as pq
    import mrr25
    from frag_v3_check import per_molecule_rr

    out = {}
    for h in HOLDOUTS:
        tp = ROOT / f"data/holdouts/{h}/test.parquet"
        truth_p = ROOT / f"outputs/holdouts/{h}/truth.csv"
        new_p = ROOT / f"outputs/local_eval/{h}/submission_{NEW_TAG}.csv"
        ctl_p = ROOT / f"outputs/local_eval/{h}/submission_{CTRL_TAG}.csv"
        t = pq.read_table(tp, columns=["molecule_id", "adduct", "ms2_mzs"]).to_pydict()
        rows = [dict(molecule_id=m, adduct=ad, n_peaks=len(mz or []))
                for m, ad, mz in zip(t["molecule_id"], t["adduct"], t["ms2_mzs"])]
        truth = mrr25.read_truth(truth_p)
        rr_new = per_molecule_rr(truth, mrr25.read_submission(new_p))
        rr_ctl = per_molecule_rr(truth, mrr25.read_submission(ctl_p))
        out[h] = dict(files={str(p.relative_to(ROOT)): sha256(p) for p in (tp, truth_p, new_p, ctl_p)},
                      summary=summarize(rr_new, rr_ctl, molecule_strata(rows)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({h: out[h]["summary"] for h in out}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
