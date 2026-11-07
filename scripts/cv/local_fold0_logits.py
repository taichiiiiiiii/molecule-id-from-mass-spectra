#!/usr/bin/env python3
"""
リークの無い FPNet ロジットをローカルで作る（fold-safe FPNet の fold-0 モデルだけを使う）。

ホールドアウト（enveda_v1 / np_v2）の分子はほぼ全部が fold-safe FPNet の split.parquet で fold=hold（fold-0 モデルが学習で見ていない）。
Kaggle のパイプラインの z は A+B（全学習データで学習、ホールドアウトを見ている）と fold0 の平均なので甘く出る。
ここでは fold0 だけで z を作り、pc_logits.npz と同じ形式で保存する（スケールは順位に影響しない）。

    python scripts/cv/local_fold0_logits.py --holdout np_v2
出力: outputs/holdout_eval/<holdout>_fold0/pc_logits.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "data/ext/v4g/casmi26-v3-models/code")
from casmi import chem, fpnet  # noqa: E402
from casmi.spectra import merge_spectra  # noqa: E402


def molecule_logits(bank, spectra):
    """0.479 版 probe_core の molecule_logits と同じ前処理（単スペクトル平均 + 極性ごとの合成スペクトル平均）。"""
    items_s = []
    for s in spectra:
        pm, pi = fpnet.prep_peaks(s["mz"], s["it"], s["prec"])
        ce_ok = s["ce"] is not None and not np.isnan(s["ce"])
        items_s.append(dict(mz=pm, it=pi, prec=float(s["prec"]), adduct_ix=chem.adduct_index(s["adduct"]),
                            ce=float(s["ce"]) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                            n_merged=min(int(s.get("ce_n", 1)), 8), mode=float(s["mode"])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s["mode"] == md]
        if not grp:
            continue
        mm, ii = merge_spectra([(s["mz"], s["it"]) for s in grp])
        prec = float(np.median([s["prec"] for s in grp]))
        pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s["ce"] for s in grp if s["ce"] is not None and not np.isnan(s["ce"])]
        adducts = [s["adduct"] for s in grp]
        ad = max(adducts, key=adducts.count)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get("ce_n", 1))) for s in grp), 8), mode=float(md)))
    if not items_s:
        return None
    zs, _ = bank.logits_el(items_s)
    zm, _ = bank.logits_el(items_m)
    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--model", default="data/ext/foldsafe/fpnet_fold0.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run", default=None, help="出力先 outputs/holdout_eval/<run>/（既定 <holdout>_fold0）。fold k のモデルなら <holdout>_fold<k> など")
    args = ap.parse_args()
    te = pd.read_parquet(f"data/holdouts/{args.holdout}/test.parquet")
    te["nm"] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
    bank = fpnet.ModelBank([args.model], device=args.device)
    out, targets = {}, []
    t0 = time.time()
    for mid, sub in te.groupby("molecule_id", sort=False):
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms):
            continue
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == "positive" else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        z = molecule_logits(bank, spectra[:16])
        if z is None:
            continue
        out[mid] = z
        targets.append([mid, target])
    d = Path(f"outputs/holdout_eval/{args.run or args.holdout + '_fold0'}")
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "pc_logits.npz", **out, __targets__=np.array(targets, dtype=object))
    print(f"wrote {d}/pc_logits.npz: {len(out)} molecules in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
