#!/usr/bin/env python3
"""
リーク無しのローカル縮小パイプライン（Issue #20）。改善ループの評価環境。

段:
  1. candidates : 候補プール（pool_meta）の ±PPM 窓
  2. score      : fold-0 FPNet ロジット z で fp[bits] @ z（ホールドアウト分子を学習で見ていないモデル）
  3. prior      : 上位 TOPN を z 標準化 + POOLPOP_MU * (log1p 物質数 + log1p PubMed)（0.479 版 v19 と同じ定数）
  4. output     : InChIKey14 で重複除去した上位 25 → submission 形式 → MRR@25

    python scripts/local_eval/run.py --holdout np_v2
    python scripts/local_eval/run.py --holdout enveda_v1 --stages score      # 文献 prior 無し

ロジットが無ければ scripts/cv/local_fold0_logits.py で作る（CPU で数秒）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
sys.path.insert(0, str(ROOT / "scripts/validate"))

PPM = 10.0          # 0.479 版の PubChem チャネルと同じ
TOPN = 60           # 0.479 版の base リスト長
POOLPOP_MU = 0.15   # 0.479 版 v19
K = 25


# ---------------------------------------------------------------- 段 1: 候補
class Pool:
    def __init__(self, meta: pd.DataFrame, pop: np.ndarray | None):
        order = np.argsort(meta["mass"].values, kind="stable")
        self.key = meta["key"].values[order]
        self.smiles = meta["smiles"].values[order]
        self.mass = meta["mass"].values[order].astype(np.float64)
        self.pop = pop[meta["pid"].values[order]] if pop is not None else np.zeros(len(order))

    def window(self, target: float, ppm: float = PPM) -> np.ndarray:
        tol = target * ppm * 1e-6
        a = np.searchsorted(self.mass, target - tol, "left")
        b = np.searchsorted(self.mass, target + tol, "right")
        return np.arange(a, b)


def load_pool(root: Path = ROOT) -> Pool:
    meta = pd.read_parquet(root / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet",
                           columns=["key", "smiles", "mass", "pid"])
    pp = pd.read_csv(root / "data/ext/pop/pool_popularity.csv", usecols=["pool_row", "log1p_substances", "log1p_pubmed"])
    pop = np.zeros(len(meta))
    pop[pp.pool_row.values] = pp.log1p_substances.fillna(0).values + pp.log1p_pubmed.fillna(0).values
    return Pool(meta, pop)


# ---------------------------------------------------------------- 段 2: 採点
class FPCache:
    """SMILES -> 選択ビットの指紋。同じ構造を何度も計算しない。"""

    def __init__(self, bits: np.ndarray):
        from casmi import chem
        self.chem, self.bits, self.c = chem, bits, {}

    def __call__(self, smi: str):
        if smi not in self.c:
            fp = self.chem.raw_fingerprint(smi)
            self.c[smi] = None if fp is None else fp[self.bits].astype(np.float32)
        return self.c[smi]


def score_window(idx: np.ndarray, pool: Pool, z: np.ndarray, fpc: FPCache):
    rows = []
    for i in idx:
        fp = fpc(pool.smiles[i])
        if fp is not None:
            rows.append((float(fp @ z), int(i)))
    rows.sort(key=lambda r: -r[0])
    return rows


# ---------------------------------------------------------------- 段 3: 文献 prior（v19 と同じ形）
def rerank_with_prior(rows, pool: Pool, mu: float = POOLPOP_MU, topn: int = TOPN):
    head = dedup(rows, pool, topn)
    if len(head) <= 1 or mu == 0:
        return head
    s = np.array([r[0] for r in head]); zs = (s - s.mean()) / (s.std() + 1e-9)
    f = zs + mu * np.array([pool.pop[r[1]] for r in head])
    order = np.argsort(-f, kind="stable")
    return [(float(f[o]), head[o][1]) for o in order]


def dedup(rows, pool: Pool, n: int):
    out, seen = [], set()
    for sc, i in rows:
        k = pool.key[i]
        if k in seen:
            continue
        seen.add(k); out.append((sc, i))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 実行
def run(holdout: str, stages: list[str], logits_run: str | None = None) -> dict:
    from mrr25 import calculate_mrr25, read_truth
    t0 = time.time()
    run_name = logits_run or f"{holdout}_fold0"
    zf = np.load(ROOT / f"outputs/holdout_eval/{run_name}/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = load_pool()
    fpc = FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = read_truth(ROOT / f"outputs/holdouts/{holdout}/truth.csv")
    preds = {}
    for mid in truth:
        if mid not in targets:
            preds[mid] = []
            continue
        rows = score_window(pool.window(targets[mid]), pool, zf[mid].astype(np.float32), fpc)
        if "prior" in stages:
            rows = rerank_with_prior(rows, pool)
        preds[mid] = [pool.smiles[i] for _, i in dedup(rows, pool, K)]
    tag = "+".join(stages)
    out_dir = ROOT / f"outputs/local_eval/{holdout}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"submission_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["molecule_id", "smiles"])
        for mid, c in preds.items():
            w.writerow([mid, ";".join(c)])
    res = dict(holdout=holdout, stages=stages, logits=run_name, n=len(truth),
               mrr25=round(calculate_mrr25(truth, preds), 4), secs=round(time.time() - t0, 1))
    (out_dir / f"result_{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--stages", default="score,prior", help="カンマ区切り: score[,prior]")
    ap.add_argument("--logits-run", default=None, help="outputs/holdout_eval/<run>/pc_logits.npz（既定 <holdout>_fold0）")
    args = ap.parse_args()
    print(json.dumps(run(args.holdout, args.stages.split(","), args.logits_run), indent=2))


if __name__ == "__main__":
    main()
