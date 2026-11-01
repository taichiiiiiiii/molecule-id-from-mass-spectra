#!/usr/bin/env python3
"""
テスト分子のうち、学習ライブラリとのスペクトル照合で解けそうな割合（クラス 1 の見積もり）を出す。

各テストスペクトルについて、同じイオンモード・同じ付加体で、プリカーサー m/z が ±PPM 以内の学習スペクトルとの
最大コサイン類似度（m/z 許容 TOL Da、強度は平方根）を計算し、分子ごとに最大値をとる。

    python scripts/eda/library_coverage.py            # data/test.parquet
    python scripts/eda/library_coverage.py --panel    # 検算: パネル 250 分子（enveda-np-examples）を自分以外と照合
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PPM = 10.0
TOL = 0.01
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
COLS = ["precursor_mz", "ionization_mode", "adduct", "ms2_mzs", "ms2_normalized_intensities"]


def load_flat(path: str, extra: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """メタ情報の DataFrame と、ピークを平らにした (offsets, mz, sqrt 強度で L2 正規化) を返す。"""
    t = pq.read_table(path, columns=COLS + extra)
    mz_list = t.column("ms2_mzs").combine_chunks()
    it_list = t.column("ms2_normalized_intensities").combine_chunks()
    offsets = mz_list.offsets.to_numpy().astype(np.int64)
    mz = mz_list.values.to_numpy(zero_copy_only=False).astype(np.float64)
    it = np.sqrt(np.clip(it_list.values.to_numpy(zero_copy_only=False).astype(np.float64), 0, None))
    # スペクトルごとに L2 正規化
    sq = np.add.reduceat(it * it, offsets[:-1]) if len(it) else np.zeros(0)
    lens = np.diff(offsets)
    norm = np.sqrt(np.where(lens > 0, sq, 1.0))
    it = it / np.repeat(norm, lens)
    meta = t.drop(["ms2_mzs", "ms2_normalized_intensities"]).to_pandas()
    meta["n_peaks"] = lens
    return meta, offsets, mz, it


def cosine(q_mz, q_it, r_mz, r_it) -> float:
    """各クエリピークを、許容内で最も近い参照ピークに対応させる近似コサイン。"""
    if len(q_mz) == 0 or len(r_mz) == 0:
        return 0.0
    idx = np.searchsorted(r_mz, q_mz)
    lo = np.clip(idx - 1, 0, len(r_mz) - 1)
    hi = np.clip(idx, 0, len(r_mz) - 1)
    dlo, dhi = np.abs(r_mz[lo] - q_mz), np.abs(r_mz[hi] - q_mz)
    j = np.where(dlo <= dhi, lo, hi)
    ok = np.minimum(dlo, dhi) <= TOL
    if not ok.any():
        return 0.0
    # 参照ピークの二重使用を防ぐため、参照ピークごとに最大の積だけ数える
    prod = q_it[ok] * r_it[j[ok]]
    jj = j[ok]
    order = np.argsort(-prod)
    _, first = np.unique(jj[order], return_index=True)
    return float(min(1.0, prod[order][first].sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.parquet")
    ap.add_argument("--test", default="data/test.parquet")
    ap.add_argument("--panel", action="store_true", help="enveda-np-examples を自分以外の学習スペクトルと照合する")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()

    tr, tr_off, tr_mz, tr_it = load_flat(args.train, ["inchikey14", "ingest_lib"])
    print(f"train {len(tr):,} spectra loaded in {time.time() - t0:.0f}s", flush=True)

    if args.panel:
        is_q = (tr.ingest_lib == "enveda-np-examples").to_numpy()
        q = tr[is_q].copy()
        q["molecule_id"] = q.inchikey14
        q_idx = np.flatnonzero(is_q)
        q_off, q_mz, q_it = tr_off, tr_mz, tr_it
        tag = "panel"
    else:
        q, q_off, q_mz, q_it = load_flat(args.test, ["molecule_id", "spectrum_id"])
        q_idx = np.arange(len(q))
        tag = "test"

    # 同じモード・付加体ごとに、m/z でソートした索引を作る
    ik14 = tr.inchikey14.to_numpy().astype(str)
    tr["row"] = np.arange(len(tr))
    groups = {}
    for key, g in tr.groupby(["ionization_mode", "adduct"], sort=False):
        g = g.sort_values("precursor_mz")
        groups[key] = (g.precursor_mz.to_numpy(), g.row.to_numpy())

    rows = []
    for k, (qi, r) in enumerate(zip(q_idx, q.itertuples(index=False))):
        a, b = q_off[qi], q_off[qi + 1]
        qm, qv = q_mz[a:b], q_it[a:b]
        best, best_key, n_win, n_id = 0.0, None, 0, 0
        key = (r.ionization_mode, r.adduct)
        if key in groups:
            pmz, prow = groups[key]
            d = r.precursor_mz * PPM * 1e-6
            lo, hi = np.searchsorted(pmz, r.precursor_mz - d), np.searchsorted(pmz, r.precursor_mz + d)
            cand = prow[lo:hi]
            if args.panel:  # パネル自身のスペクトルは除き、他ライブラリの参照だけ残す（「他の参照あり」の状況）
                cand = cand[~is_q[cand]]
            n_win = len(cand)
            n_id = len(set(ik14[cand])) if n_win else 0
            for c in cand:
                s = cosine(qm, qv, tr_mz[tr_off[c]:tr_off[c + 1]], tr_it[tr_off[c]:tr_off[c + 1]])
                if s > best:
                    best, best_key = s, ik14[c]
        rows.append(dict(molecule_id=r.molecule_id, adduct=r.adduct, precursor_mz=r.precursor_mz,
                         n_window=n_win, n_window_ids=n_id, best_cos=best, best_key=best_key))
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(q_idx)} spectra, {time.time() - t0:.0f}s", flush=True)

    per_spec = pd.DataFrame(rows)
    per_mol = per_spec.sort_values("best_cos", ascending=False).groupby("molecule_id", sort=False).agg(
        best_cos=("best_cos", "first"), best_key=("best_key", "first"), n_spectra=("best_cos", "size"),
        n_window_ids=("n_window_ids", "max"))
    # 同じ分子の複数スペクトルが同じ構造を指しているか（照合の一貫性）
    agree = per_spec[per_spec.best_cos >= 0.7].groupby("molecule_id").best_key.nunique()
    per_mol["consistent_top"] = per_mol.index.map(lambda m: agree.get(m, 0) == 1)

    summary = {
        "tag": tag, "n_molecules": int(len(per_mol)), "n_spectra": int(len(per_spec)), "ppm": PPM, "tol": TOL,
        "share_best_cos_ge": {str(th): round(float((per_mol.best_cos >= th).mean()), 4) for th in THRESHOLDS},
        "share_empty_window": round(float((per_mol.n_window_ids == 0).mean()), 4),
        "median_window_ids": float(per_mol.n_window_ids.median()),
        "seconds": round(time.time() - t0, 1),
    }
    if args.panel:
        # 検算: 照合の最上位が正解構造か。しきい値ごとの適合率と、全体に対する正解率
        hit = per_mol.index.to_numpy() == per_mol.best_key.to_numpy()
        per_mol["top_is_truth"] = hit
        summary["precision_at"] = {str(th): round(float(hit[per_mol.best_cos >= th].mean()), 4)
                                   if (per_mol.best_cos >= th).any() else None for th in THRESHOLDS}
        summary["top_is_truth_share"] = round(float(hit.mean()), 4)

    out = Path(args.out or f"outputs/{tag}_library_match.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    per_mol.reset_index().to_parquet(out, index=False)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/library_coverage_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
