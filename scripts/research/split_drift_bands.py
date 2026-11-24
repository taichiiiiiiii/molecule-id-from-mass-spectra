#!/usr/bin/env python3
"""#222 §4 の案の費用見積もり: v3 の _t10 4 集合の正解を、重原子数の帯（G5 = fold-0 の学習側の第 1・第 3 四分位で
3 つに分ける。境界は学習側だけで決まり、保留側の値は使わない）に分けたときの分子数と、MDE の粗い見積もり
MDE_帯 ≈ MDE_集合 × √(n / n_帯)（帯の間で sd(Δrr) が同じと仮定。未確認）を出す。rr・Δrr は読まない。

    .venv/bin/python scripts/research/split_drift_bands.py \
        --drift outputs/local_eval/split_drift/split_drift.json --out outputs/local_eval/split_drift/bands.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_drift import HOLDOUTS, heavy_atoms  # noqa: E402

# 集合ごとの MDE の既存の見積もり = 2.8/1.96 × fragnn v1 − 対照の 95% bootstrap 区間の半幅
# （outputs/local_eval/a0i_1123/a0i_result.json、#195 A0 (i)）。v3_c2_t10 は対照が 0（正解が手元の候補に無い）なので無し
MDE_SET = {"v3_c1u_t10": 0.0249, "v3_c2_t10": None, "enveda_v1u_t10": 0.0279, "v3_c2nu_t10": 0.0279}


def band_of(h: np.ndarray, q1: float, q3: float) -> np.ndarray:
    """0: h ≤ q1、1: q1 < h < q3、2: h ≥ q3。"""
    h = np.asarray(h)
    return np.where(h <= q1, 0, np.where(h >= q3, 2, 1))


def band_mde(mde: float | None, n: int, n_band: int) -> float | None:
    if mde is None or n_band == 0:
        return None
    return round(mde * math.sqrt(n / n_band), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", default=str(ROOT / "outputs/local_eval/split_drift/split_drift.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/split_drift/bands.json"))
    args = ap.parse_args()
    d = json.loads(Path(args.drift).read_text())
    g5 = d["groups"]["fold0_train"]["heavy"]
    q1, q3 = g5["q1"], g5["q3"]
    res = dict(issue=222, rule="G5 の重原子数の第 1・第 3 四分位で 3 帯（≤q1 / q1<h<q3 / ≥q3）", q1=q1, q3=q3,
               mde_assumption="MDE_帯 ≈ MDE_集合 × √(n/n_帯)、帯の間で sd(Δrr) が同じと仮定（未確認）", sets={})
    for h in HOLDOUTS:
        with open(ROOT / f"outputs/local_eval/{h}/truth.csv", newline="") as f:
            sm = [r["smiles"] for r in csv.DictReader(f)]
        b = band_of(heavy_atoms(sm), q1, q3)
        n = len(sm)
        counts = [int((b == k).sum()) for k in range(3)]
        res["sets"][h] = dict(n=n, n_band=counts, mde_set=MDE_SET[h],
                              mde_band=[band_mde(MDE_SET[h], n, c) for c in counts])
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
