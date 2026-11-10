#!/usr/bin/env python3
"""
Issue #60: fold-safe sanpier（#54 の final_model.pt）で、縮小版の候補窓の各候補に点 n を付けてキャッシュする。

    PYTHONPATH=<rdkit 2026.03.3 の --target 先> .venv/bin/python scripts/local_eval/sanpier.py --holdout v3_c1 [--limit 10]

- RDKit は学習時（Kaggle）と同じ 2026.03.3 を使う（違えば止まる）。用意するには
  `uv pip install --python .venv/bin/python --target <dir> --no-deps rdkit==2026.3.3`
- 候補窓は run.py と同じ（±10 ppm、pool:'remove' の集合は正解を窓から消す）
- n はキットの inference.Scorer（CPU）で、その分子の全スペクトルをまとめて採点したもの（スペクトルごとの点の平均と同じ）
- キットの min_peaks 未満のスペクトルは使わない（学習の prepare.py と同じ）。変換できない候補は NaN
- 出力: outputs/local_eval/<holdout>/sanpier_scores.npz（mids, offsets, idx, n）。run.py の `sanpier` 段が読む
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "outputs/sanpier_fold0/casmi_spectrum_model_fold0"
RDKIT_VERSION = "2026.03.3"


def stage_bundle(src: Path = BUNDLE, dst: Path = ROOT / "outputs/local_eval/_sanpier_bundle") -> Path:
    """キットの Scorer は <bundle>/config.json を読むので、config_used.json を config.json として置いた写しを作る。"""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("features.py", "model.py", "inference.py", "final_model.pt"):
        if not (dst / name).exists():
            shutil.copy2(src / name, dst / name)
    shutil.copy2(src / "config_used.json", dst / "config.json")
    return dst


def molecule_spectra(te: pd.DataFrame, features, min_peaks: int, max_peaks: int) -> dict[str, list[dict]]:
    """分子ごとのスペクトル（キットの Scorer の形）。min_peaks 未満・ピークの無いスペクトルは除く。"""
    out: dict[str, list[dict]] = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        keep = []
        for r in g.itertuples():
            s = dict(mzs=np.asarray(r.ms2_mzs, np.float64), intensities=np.asarray(r.ms2_normalized_intensities, np.float64),
                     precursor_mz=float(r.precursor_mz), adduct=r.adduct, ionization_mode=r.ionization_mode,
                     collision_energy=r.collision_energy_ev)   # 学習時の ALIASES['ce'] と同じ列
            try:
                _, n = features.peaks(s["mzs"], s["intensities"], s["precursor_mz"], max_peaks)
            except (ValueError, TypeError):
                continue
            if n >= min_peaks:
                keep.append(s)
        out[str(mid)] = keep
    return out


def pack(per_mol: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    mids = list(per_mol)
    offs = np.zeros(len(mids) + 1, np.int64)
    for j, m in enumerate(mids):
        offs[j + 1] = offs[j] + len(per_mol[m][0])
    idx = np.concatenate([per_mol[m][0] for m in mids]).astype(np.int64) if mids else np.zeros(0, np.int64)
    n = np.concatenate([per_mol[m][1] for m in mids]).astype(np.float32) if mids else np.zeros(0, np.float32)
    return dict(mids=np.array(mids, dtype=object), offsets=offs, idx=idx, n=n)


def load_scores(path: Path) -> dict[str, dict[int, float]]:
    """npz → {mid: {pool 行: n}}（NaN は点無し）。"""
    z = np.load(path, allow_pickle=True)
    out = {}
    for j, m in enumerate(z["mids"]):
        a, b = z["offsets"][j], z["offsets"][j + 1]
        out[str(m)] = dict(zip(z["idx"][a:b].tolist(), z["n"][a:b].tolist()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 分子だけ（費用の測定用。キャッシュは書かない）")
    args = ap.parse_args()
    from rdkit import rdBase
    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise SystemExit(f"RDKit {rdBase.rdkitVersion} != {RDKIT_VERSION}（学習時と同じ版を PYTHONPATH で前に置く）")
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import run as R
    import library as L
    bundle = stage_bundle()
    sys.path.insert(0, str(bundle))
    import features
    from inference import Scorer
    scorer = Scorer(bundle, checkpoint="final_model.pt", device="cpu")
    cfg = scorer.cfg
    t0 = time.time()
    zf = np.load(ROOT / f"outputs/holdout_eval/{args.holdout}_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    rule = L.hide_rule(args.holdout)
    hidden = L.drop_keys(args.holdout) if rule.get("pool") == "remove" else set()
    te = pd.read_parquet(ROOT / f"data/holdouts/{args.holdout}/test.parquet")
    spec = molecule_spectra(te, features, cfg["min_peaks"], cfg["max_peaks"])
    from mrr25 import read_truth   # run.py が scripts/validate を sys.path に入れている
    truth = read_truth(ROOT / f"outputs/holdouts/{args.holdout}/truth.csv")
    mids = [m for m in truth if m in targets]
    if args.limit:
        mids = mids[: args.limit]
    per_mol, n_cand, n_nospec, n_inv = {}, 0, 0, 0
    for mid in mids:
        idx = pool.window(targets[mid])
        if hidden:
            idx = idx[~np.isin(pool.key[idx], list(hidden))]
        sp = spec.get(mid, [])
        if not sp or len(idx) == 0:
            n_nospec += not sp
            per_mol[mid] = (idx, np.full(len(idx), np.nan, np.float32))
            continue
        sc = scorer.score_candidates(sp, [str(pool.smiles[i]) for i in idx])
        sc = np.where(np.isfinite(sc), sc, np.nan).astype(np.float32)
        n_inv += int(np.isnan(sc).sum()); n_cand += len(idx)
        per_mol[mid] = (idx, sc)
    secs = round(time.time() - t0, 1)
    rep = dict(holdout=args.holdout, molecules=len(mids), candidates=n_cand, invalid_candidates=n_inv,
               molecules_without_spectra=n_nospec, hidden_keys=len(hidden), rdkit=rdBase.rdkitVersion,
               torch=torch.__version__, checkpoint=str(BUNDLE / "final_model.pt"), secs=secs)
    if not args.limit:
        out = ROOT / f"outputs/local_eval/{args.holdout}/sanpier_scores.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **pack(per_mol))
        (out.parent / "sanpier_scores.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
