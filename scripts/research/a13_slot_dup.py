#!/usr/bin/env python3
"""#230（#195 A13）: 提出の 25 枠の中の「指標の上で同じ答え」の候補（重複）を数える（読むだけ）。

定義（#230 の本文、数える前に固定）:
- 同じ答え = `scripts/validate/mrr25.py` の `to_key`（立体を外す → 互変異性体の正準化 → InChIKey14）が同じ候補。
  `to_key` が None の候補はどれとも同じにしない（metric では必ず不一致）
- 重複 = 順位順に見て、それより上に同じ `to_key` の候補がある候補（2 つ目以降）
- 分子ごと: 上位 25 の重複の数、正解の最初の一致の順位（rr の順位）、その順位より上の重複の数、
  深い順位を `to_key` で重複除去した後の正解の順位
- 集合ごと: 重複が 1 件以上の分子の割合（Wilson 95%）、正解より上に重複がある分子の数、
  重複を除くと 25 位以内に入る（今は上位 25 に無い）分子の数、正解が深い順位のどこにも無い分子の数
- MRR の差は出さない（判定は #208 の R1 で別に行う）

入力: `run.py --dump-ranks` の CSV（molecule_id, rank, pool_row, key, smiles）と truth.csv。
`--sub` を渡すと、深い順位の先頭 25 がその submission の並びと同じことを確かめる（違えば終了コード 2）。
正解の無い CSV（見える test の submission）は `--submission-only` で重複の数だけを出す。

    .venv/bin/python scripts/research/a13_slot_dup.py --set v3_c1u_t10 \\
        --ranks outputs/local_eval/a13_1124/ranks_v3_c1u_t10.csv --truth outputs/local_eval/v3_c1u_t10/truth.csv \\
        --sub "outputs/local_eval/v3_c1u_t10/submission_score+prior+frag.csv" --out-dir outputs/local_eval/a13_1124
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
from mrr25 import parse_candidates, read_truth, to_key, truth_keys  # noqa: E402

K = 25


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def dup_flags(keys: list) -> list[bool]:
    """順位順のキーの列で、上に同じキーがある位置を True。None はどれとも同じにしない。"""
    seen, out = set(), []
    for k in keys:
        out.append(k is not None and k in seen)
        if k is not None:
            seen.add(k)
    return out


def molecule_stats(cands: list[str], tkeys: set, k: int = K, key_fn=to_key) -> dict:
    """1 分子分。cands は深い順位の SMILES（先頭 k が提出の並び）。tkeys が空なら正解なし（重複の数だけ）。

    26 位以降のキーは、正解が上位 k に無いときだけ、正解が見つかるまで計算する。
    """
    top = [key_fn(c) for c in cands[:k]]
    fl = dup_flags(top)
    out = dict(n_cand=len(cands), n_top=len(top), dup_top=sum(fl), none_top=sum(x is None for x in top),
               rank=None, dup_above=None, rank_dedup=None, found=None)
    if not tkeys:
        return out
    keys = list(top)
    hit = next((i for i, x in enumerate(keys) if x is not None and x in tkeys), None)
    j = len(keys)
    while hit is None and j < len(cands):
        x = key_fn(cands[j]); keys.append(x)
        if x is not None and x in tkeys:
            hit = j
        j += 1
    if hit is None:
        out.update(found=False)
        return out
    fl_all = dup_flags(keys[:hit + 1])
    above = sum(fl_all[:hit])
    out.update(found=True, rank=hit + 1, dup_above=above, rank_dedup=hit + 1 - above)
    return out


def summarize(per: list[dict], k: int = K) -> dict:
    n = len(per)
    m_dup = sum(1 for p in per if p["dup_top"] > 0)
    labelled = [p for p in per if p["found"] is not None]
    above = [p for p in labelled if p["found"] and p["rank"] <= k and p["dup_above"] > 0]
    enter = [p for p in labelled if p["found"] and p["rank"] > k and p["rank_dedup"] <= k]
    lo, hi = wilson(m_dup, n)
    return dict(n=n, dup_total=sum(p["dup_top"] for p in per), none_total=sum(p["none_top"] for p in per),
                mol_with_dup=m_dup, mol_with_dup_share=round(m_dup / n, 4) if n else None,
                mol_with_dup_wilson95=[round(lo, 4), round(hi, 4)],
                dup_per_mol_max=max((p["dup_top"] for p in per), default=0),
                mol_dup_above_truth=len(above) if labelled else None,
                mol_enter_top25_after_dedup=len(enter) if labelled else None,
                truth_in_top25=sum(1 for p in labelled if p["found"] and p["rank"] <= k) if labelled else None,
                truth_deep_only=sum(1 for p in labelled if p["found"] and p["rank"] > k) if labelled else None,
                truth_absent=sum(1 for p in labelled if p["found"] is False) if labelled else None)


def read_ranks(path) -> dict[str, list[str]]:
    out: dict[str, list[tuple[int, str]]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["molecule_id"], []).append((int(r["rank"]), r["smiles"]))
    res = {}
    for m, rows in out.items():
        rows.sort()
        assert [r for r, _ in rows] == list(range(1, len(rows) + 1)), m
        res[m] = [s for _, s in rows]
    return res


def read_sub(path) -> dict[str, list[str]]:
    with open(path, newline="") as f:
        return {r["molecule_id"]: parse_candidates(r["smiles"]) for r in csv.DictReader(f)}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--ranks", default=None, help="run.py --dump-ranks の CSV")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--sub", default=None, help="突き合わせる submission（--ranks あり）か、数える submission（--submission-only）")
    ap.add_argument("--submission-only", action="store_true", help="正解なしで submission の上位 25 の重複だけを数える")
    ap.add_argument("--skip-rows", default=None, help="--submission-only で除く行の SMILES セル（置き換えの行）と同じ行を除く")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    inputs, summ_empty = {}, 0
    if a.submission_only:
        sub = read_sub(a.sub); inputs["sub"] = sha(a.sub)
        if a.skip_rows:
            sub = {m: c for m, c in sub.items() if ";".join(c) != a.skip_rows}
        lists, truth = {m: c[:K] for m, c in sub.items()}, {}
    else:
        lists = read_ranks(a.ranks); truth = read_truth(a.truth)
        inputs.update(ranks=sha(a.ranks), truth=sha(a.truth))
        if a.sub:
            sub = read_sub(a.sub); inputs["sub"] = sha(a.sub)
            bad = [m for m in truth if lists.get(m, [])[:K] != sub.get(m, [])]
            if bad:
                print(f"深い順位の先頭 {K} が submission と違う: {len(bad)} 分子（例 {bad[:3]}）", file=sys.stderr)
                sys.exit(2)
        missing = [m for m in truth if m not in lists]
        # 窓に候補が 1 件も無い分子は深い順位に行が無い。submission も空のときだけ空のリストとして数える
        if missing and (not a.sub or any(sub.get(m) for m in missing)):
            print(f"深い順位に無い分子: {len(missing)}（submission が空でないか、--sub が無い）", file=sys.stderr)
            sys.exit(2)
        summ_empty = len(missing)
        lists = {m: lists.get(m, []) for m in truth}
    per = []
    for m, c in lists.items():
        st = molecule_stats(c, truth_keys(truth[m])) if truth else molecule_stats(c, set())
        per.append(dict(molecule_id=m, **st))
    summ = dict(set=a.set, **summarize(per), empty_window=summ_empty, inputs=inputs)
    with open(out_dir / f"a13_{a.set}_per_molecule.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys())); w.writeheader(); w.writerows(per)
    (out_dir / f"a13_{a.set}.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
