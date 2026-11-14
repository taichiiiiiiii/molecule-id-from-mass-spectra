#!/usr/bin/env python3
"""整列チャネル（凍結 DreaMS ＋ 自前学習の整列 MLP）の点 a を、縮小版の候補窓の各候補に付けてキャッシュする。

    .venv/bin/python scripts/local_eval/align.py --holdout v3_c1 [--src outputs/align_train]

- 入力: casmi26-align-train の出力（--src に置く）: g_query.npy（問い合わせごとの正規化した g、fp32）、query_index.parquet、
  h_pool.npy（pool_meta の行順の正規化した h、fp16）、align_config.json（pool_meta の sha256 を照合する）
- 候補窓は run.py と同じ（±10 ppm、pool:'remove' の集合は正解を窓から消す）
- 点 a = cos( normalize(その分子の全スペクトルの g の平均), h[候補] )（文書 §(e)）。torch も RDKit も要らない（CPU、数秒）
- 問い合わせのスペクトルが 1 本も無い分子は点無し（NaN）→ run.py の align 段では s のまま
- 出力: outputs/local_eval/<holdout>/align_scores.npz（mids, offsets, idx, n。sanpier.py と同じ形）と align_scores.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/align_train"
POOL_META = ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet"


def molecule_vectors(g: np.ndarray, holdout_col, mid_col, holdout: str) -> dict[str, np.ndarray]:
    """その集合の分子ごとに、g の平均を正規化したもの（g は行ごとに正規化済みの前提でも、平均のあとにもう一度正規化する）。"""
    out: dict[str, list] = {}
    for j, (h, m) in enumerate(zip(holdout_col, mid_col)):
        if h == holdout:
            out.setdefault(str(m), []).append(j)
    vec = {}
    for m, rows in out.items():
        v = g[rows].astype(np.float64).mean(0)
        nv = np.linalg.norm(v)
        if nv > 0 and np.isfinite(nv):
            vec[m] = v / nv
    return vec


def window_scores(idx: np.ndarray, v: np.ndarray | None, h_sorted) -> np.ndarray:
    """候補（ソート後の pool 行）の点。v が無ければ NaN。"""
    if v is None or len(idx) == 0:
        return np.full(len(idx), np.nan, np.float32)
    return (np.asarray(h_sorted[idx], np.float64) @ v).astype(np.float32)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--src", default=str(SRC), help="casmi26-align-train の出力を置いたディレクトリ")
    args = ap.parse_args()
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    import run as R
    import library as L
    import sanpier as SP
    from mrr25 import read_truth
    t0 = time.time()
    src = Path(args.src)
    cfg = json.loads((src / "align_config.json").read_text())
    if cfg["pool_meta_sha256"] != sha256(POOL_META):
        raise SystemExit("pool_meta.parquet differs from the one the kernel used")
    g = np.load(src / "g_query.npy")
    qi = pd.read_parquet(src / "query_index.parquet")
    assert len(qi) == len(g)
    h_pool = np.load(src / "h_pool.npy", mmap_mode="r")
    meta = pd.read_parquet(POOL_META, columns=["mass"])
    order = np.argsort(meta["mass"].values, kind="stable")   # run.Pool と同じ並べ替え
    assert h_pool.shape[0] == len(order)
    zf = np.load(ROOT / f"outputs/holdout_eval/{args.holdout}_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    rule = L.hide_rule(args.holdout)
    hidden = L.drop_keys(args.holdout) if rule.get("pool") == "remove" else set()
    truth = read_truth(ROOT / f"outputs/holdouts/{args.holdout}/truth.csv")
    vec = molecule_vectors(g, qi.holdout.values, qi.molecule_id.values, args.holdout)
    per_mol, n_cand, n_nospec = {}, 0, 0
    for mid in [m for m in truth if m in targets]:
        idx = pool.window(targets[mid])
        if hidden:
            idx = idx[~np.isin(pool.key[idx], list(hidden))]
        v = vec.get(mid)
        n_nospec += v is None
        per_mol[mid] = (idx, window_scores(order[idx], v, h_pool))
        n_cand += len(idx)
    out = ROOT / f"outputs/local_eval/{args.holdout}/align_scores.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **SP.pack(per_mol))
    rep = dict(holdout=args.holdout, molecules=len(per_mol), candidates=n_cand, molecules_without_spectra=n_nospec,
               query_spectra=int((qi.holdout == args.holdout).sum()), hidden_keys=len(hidden), src=str(src),
               g_sha256=sha256(src / "g_query.npy"), secs=round(time.time() - t0, 1))
    (out.parent / "align_scores.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
