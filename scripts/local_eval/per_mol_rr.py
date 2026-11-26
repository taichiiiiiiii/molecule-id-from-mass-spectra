"""submission CSV と truth CSV から分子ごとの rr（MRR@25 の 1 分子分）を書く（#249）。

12/6 の holdout の判定（docs/research/type-weights.md §3.2）は (版 A − 基準 score+prior) の分子単位の対応のある
bootstrap の区間で行うので、基準の分子ごとの rr を固定しておく。採点は scripts/validate/mrr25.py の
reciprocal_rank・truth_keys をそのまま使い、rr の平均が calculate_mrr25 の値と一致することを確かめる。

使い方:
    python scripts/local_eval/per_mol_rr.py --truth <truth.csv> --pred <submission.csv> --out <rr.csv>
出力: molecule_id,rr（truth の順、rr は repr の 17 桁）。標準出力に n・mrr25・出力の sha256 の JSON。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _mrr25():
    if "mrr25" in sys.modules:
        return sys.modules["mrr25"]
    spec = importlib.util.spec_from_file_location("mrr25", ROOT / "scripts/validate/mrr25.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mrr25"] = mod
    spec.loader.exec_module(mod)
    return mod


def per_molecule_rr(truth: dict[str, str], preds: dict[str, list[str]]) -> list[tuple[str, float]]:
    M = _mrr25()
    if not truth:
        raise ValueError("truth が空です")
    out = []
    for mid, t in truth.items():
        cands = preds.get(mid, [])
        if len(cands) > M.TOP_K:
            raise M.SubmissionError(f"{mid}: 候補が {len(cands)} 件（上限 {M.TOP_K}）")
        out.append((mid, M.reciprocal_rank(M.truth_keys(t), cands, M.TOP_K)))
    mean = sum(r for _, r in out) / len(out)
    ref = M.calculate_mrr25(truth, preds)
    if abs(mean - ref) >= 1e-12:
        raise RuntimeError(f"rr の平均 {mean} が calculate_mrr25 {ref} と一致しない")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    M = _mrr25()
    truth = M.read_truth(a.truth)
    preds = M.read_submission(a.pred)
    extra = set(preds) - set(truth)
    if extra:
        raise SystemExit(f"truth に無い molecule_id が {len(extra)} 件")
    rows = per_molecule_rr(truth, preds)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule_id", "rr"])
        for mid, r in rows:
            w.writerow([mid, repr(float(r))])
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps(dict(n=len(rows), mrr25=round(sum(r for _, r in rows) / len(rows), 4),
                          n_hit=sum(r > 0 for _, r in rows), out=str(out), sha256=sha), indent=2))


if __name__ == "__main__":
    main()
