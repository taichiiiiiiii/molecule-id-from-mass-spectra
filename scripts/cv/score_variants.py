#!/usr/bin/env python3
"""Kaggle の出力（submission.csv と variants/*.csv など）を、ホールドアウトの正解で一括採点する。

    python scripts/cv/score_variants.py --truth data/holdout_v1/truth.csv --dir outputs/holdout_eval_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validate"))
import mrr25  # noqa: E402


def score(truth: pd.DataFrame, path: Path) -> dict:
    preds = mrr25.read_submission(path)
    keys = dict(zip(truth.molecule_id, truth.inchikey14))
    rr = {m: mrr25.reciprocal_rank(k, preds.get(m, [])) for m, k in keys.items()}
    s = pd.Series(rr)
    out = {"file": str(path.name), "mrr25": round(float(s.mean()), 4), "top1": round(float((s == 1).mean()), 4),
           "hit25": round(float((s > 0).mean()), 4), "n": int(len(s))}
    if "scenario" in truth:
        for scn, g in truth.groupby("scenario"):
            out[f"mrr25_{scn}"] = round(float(s[g.molecule_id].mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    truth = pd.read_csv(args.truth, dtype=str)
    files = sorted(p for p in Path(args.dir).rglob("*.csv") if p.name.startswith(("submission", "baseline")))
    rows = []
    for p in files:
        try:
            rows.append(score(truth, p))
        except mrr25.SubmissionError as e:
            rows.append({"file": p.name, "error": str(e)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = Path(args.out or Path(args.dir) / "scores.json")
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
