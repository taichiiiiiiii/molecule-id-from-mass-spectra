#!/usr/bin/env python3
"""
Issue #178: 分子式だけの証拠（断片木・同位体・元素など、候補の分子式にしか依らない点）で届く上限を、対照の
リスト（score+prior+frag）から出す（採否には使わない。新しい段を作る前の見込みの確認）。

定義（docs/research/train-independent-evidence.md §4.1 のとおり。ここで替えない）:
- 入力: outputs/local_eval/<h>/submission_score+prior+frag.csv（対照、上位 25）と outputs/holdouts/<h>/truth.csv
- 正解の照合は scripts/validate/mrr25.py の to_key（互変異性体の正準化 → InChIKey14）。分子式は RDKit の
  CalcMolFormula（候補・正解とも SMILES から。正解の SMILES が無ければ式は None）
- 神託（いちばん都合のよい分子式の証拠）: 正解より上にある「正解と分子式が違う候補」をすべて正解の下へ動かし、
  同じ式の候補の順は変えない。正解の新しい順位 r' = 1 + (正解より上の同じ式の候補の数)
- 上限 = Σ (1/r' − 1/r) / n（n は truth の分子数。正解が上位 25 に無い分子は 0 として数え、数だけ別に記録する。
  分子式だけの証拠は同じ式の異性体を分けられないので、上位 25 の外の正解を何位に上げるかはこのリストからは
  決まらない）
- 先に、このスクリプトで数えた MRR が run.py の値（v3_c1 0.8535・v3_c2 0.0000・enveda_v1 0.8376・
  v3_c2n 0.6284）と 4 桁で一致することを確かめる。一致しなければ終了コード 2

    .venv/bin/python scripts/research/formula_oracle_bound.py

出力: outputs/local_eval/formula_oracle_bound.json（gitignore。数値は文書に写す）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
EXPECTED_MRR = {"v3_c1": 0.8535, "v3_c2": 0.0, "enveda_v1": 0.8376, "v3_c2n": 0.6284}
CONTROL = "submission_score+prior+frag.csv"
MISMATCH_EXIT = 2


def oracle_rank(truth_key, truth_formula, cand_keys, cand_formulas) -> tuple[int | None, int | None]:
    """(r, r')。正解が上位 25 に無ければ (None, None)。式の分からない候補は「正解と違う式」とは数えない
    （神託でも動かさない＝上限を大きく見せない側）。正解の式が None なら r' = r。"""
    r = next((i + 1 for i, k in enumerate(cand_keys[:25]) if truth_key is not None and k == truth_key), None)
    if r is None:
        return None, None
    if truth_formula is None:
        return r, r
    above_same_or_unknown = sum(1 for f in cand_formulas[: r - 1] if f is None or f == truth_formula)
    return r, 1 + above_same_or_unknown


def bound(recs: list[tuple[int | None, int | None]], n: int) -> dict:
    gain = sum(1.0 / rp - 1.0 / r for r, rp in recs if r is not None)
    mrr = sum(1.0 / r for r, _ in recs if r is not None) / n
    return dict(n=n, mrr=round(mrr, 4), mrr_raw=mrr, oracle_mrr=round(mrr + gain / n, 4), bound=round(gain / n, 4),
                bound_raw=gain / n, n_hit=sum(1 for r, _ in recs if r == 1),
                n_absent=sum(1 for r, _ in recs if r is None),
                n_moved=sum(1 for r, rp in recs if r is not None and rp < r),
                n_top1_other_formula=sum(1 for r, rp in recs if r is not None and r > 1 and rp == 1))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/formula_oracle_bound.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    from mrr25 import read_submission, read_truth, to_key
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")

    fcache: dict[str, str | None] = {}
    kcache: dict[str, str | None] = {}

    def formula(smi: str) -> str | None:
        if smi not in fcache:
            m = Chem.MolFromSmiles(smi)
            fcache[smi] = CalcMolFormula(m) if m is not None else None
        return fcache[smi]

    def key(smi: str) -> str | None:
        if smi not in kcache:
            kcache[smi] = to_key(smi)
        return kcache[smi]

    import rdkit
    res: dict = dict(issue=178, control=CONTROL, rdkit=rdkit.__version__, inputs={})
    bad = []
    for h in SETS:
        sub_p, tr_p = ROOT / f"outputs/local_eval/{h}/{CONTROL}", ROOT / f"outputs/holdouts/{h}/truth.csv"
        res["inputs"][h] = dict(submission_sha256=sha256_file(sub_p), truth_sha256=sha256_file(tr_p))
        preds, truth = read_submission(sub_p), read_truth(tr_p)
        recs = []
        for mid, t in truth.items():
            cands = preds.get(mid, [])
            tf = None if len(t.strip()) == 27 and t.strip()[14] == "-" else formula(t)
            recs.append(oracle_rank(key(t), tf, [key(c) for c in cands], [formula(c) for c in cands]))
        res[h] = bound(recs, len(truth))
        if res[h]["mrr"] != EXPECTED_MRR[h]:
            bad.append(h)
    if bad:
        print(f"MRR が run.py の値と食い違う: {bad}", file=sys.stderr)
        sys.exit(MISMATCH_EXIT)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for h in SETS:
        r = res[h]
        print(f"{h}: n {r['n']} mrr {r['mrr']:.4f} -> oracle {r['oracle_mrr']:.4f} (bound +{r['bound']:.4f}), "
              f"moved {r['n_moved']}, top1 by oracle {r['n_top1_other_formula']}, absent {r['n_absent']}")


if __name__ == "__main__":
    main()
