#!/usr/bin/env python3
"""
検証 v3（Issue #25）: 隠れテストに似せた 2 種のホールドアウトを fold-0（fold-safe FPNet の未学習側）から作る。

  C2（クラス 2 型）: enveda-180 にしかスペクトルの無い構造。問い合わせ = timsTOF スペクトル。
                     正解は学習データの参照からも候補プールからも消す（drop_keys.txt、hide_rule = all）。
  C1（クラス 1 型）: enveda（timsTOF）と公開ライブラリの両方にスペクトルのある構造。問い合わせ = timsTOF。
                     enveda 系の参照だけ隠し、公開側は残す（hide_rule = enveda_libs）。候補プールには残る。

    python scripts/cv/build_holdout_v3.py --n 200 --seed 20261107

出力: data/holdouts/v3_c{1,2}/{test.parquet, drop_keys.txt, hide_rule.json, HOLDOUT.json}
      outputs/holdouts/v3_c{1,2}/truth.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = {"pluskal_ms2", "riken", "gnps", "massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"}
ENVEDA = {"enveda-180", "enveda-np-examples"}
MASS_LO, MASS_HI, MAX_SPECTRA = 157.0, 1159.0, 16
TEST_COLS = ["ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct", "ionization_mode",
             "instrument_type", "precursor_mz", "collision_energy_orig", "collision_energy_ev", "collision_energy_orig_units"]


def hidden_mask(libs: pd.Series, keys: pd.Series, drop: set, rule: str) -> np.ndarray:
    """学習データの行のうち、参照から隠す行。rule = all（その構造の全行）| enveda_libs（enveda 系の行だけ）。"""
    in_drop = keys.isin(drop).values
    if rule == "all":
        return in_drop
    if rule == "enveda_libs":
        return in_drop & libs.astype(str).isin(ENVEDA).values
    raise ValueError(rule)


def select(meta: pd.DataFrame, hold: set, n: int, seed: int):
    g = meta.groupby("inchikey14", observed=True).ingest_lib.agg(lambda x: set(x.astype(str)))
    mass = meta.groupby("inchikey14", observed=True).neutral_mass.median()
    ok_mass = mass[(mass >= MASS_LO) & (mass <= MASS_HI)].index
    cand = g[g.index.isin(hold) & g.index.isin(ok_mass)]
    c2 = sorted(k for k, v in cand.items() if "enveda-180" in v and not (v & PUBLIC))
    c1 = sorted(k for k, v in cand.items() if (v & ENVEDA) and (v & PUBLIC))
    rng = np.random.default_rng(seed)
    pick = lambda ks: sorted(rng.choice(ks, size=min(n, len(ks)), replace=False).tolist())
    return pick(c1), pick(c2), len(c1), len(c2)


def write_holdout(name: str, keys: list, rule: str, meta: pd.DataFrame, rng, seed: int, train: Path, out_root: Path = ROOT) -> dict:
    """keys の構造から擬似ホールドアウトを書き出す（問い合わせ = enveda 系の timsTOF、最大 MAX_SPECTRA 本）。HOLDOUT.json の中身を返す。"""
    keyset = set(keys)
    t = pq.read_table(train, filters=pc.field("inchikey14").isin(keys)).to_pandas()
    q = t[(t.instrument_type == "timsTOF") & t.ingest_lib.isin(ENVEDA)].copy()
    q = q.iloc[rng.permutation(len(q))].groupby("inchikey14", sort=False).head(MAX_SPECTRA).sort_values(["inchikey14"], kind="stable")
    ids = {k: f"m_{i:05d}" for i, k in enumerate(sorted(q.inchikey14.unique()))}
    q["molecule_id"] = q.inchikey14.map(ids)
    q["spectrum_id"] = [f"s_{i:06d}" for i in range(len(q))]
    out = out_root / "data/holdouts" / name
    out.mkdir(parents=True, exist_ok=True)
    q[["molecule_id", "spectrum_id"] + TEST_COLS].to_parquet(out / "test.parquet", index=False)
    pd.DataFrame({"molecule_id": sorted(ids.values()), "smiles": "CCO"}).to_csv(out / "sample_submission.csv", index=False)
    (out / "drop_keys.txt").write_text("\n".join(sorted(keyset)) + "\n")
    (out / "hide_rule.json").write_text(json.dumps(dict(rule=rule, enveda_libs=sorted(ENVEDA),
                                                        pool="remove" if rule == "all" else "keep"), indent=2))
    truth = t.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles"]].rename(columns={"normalized_smiles": "smiles"})
    truth["molecule_id"] = truth.inchikey14.map(ids)
    truth = truth.dropna(subset=["molecule_id"]).sort_values("molecule_id")
    tdir = out_root / "outputs/holdouts" / name
    tdir.mkdir(parents=True, exist_ok=True)
    truth[["molecule_id", "inchikey14", "smiles"]].to_csv(tdir / "truth.csv", index=False)
    hid = hidden_mask(meta.ingest_lib, meta.inchikey14, keyset, rule)
    info = dict(name=name, seed=seed, rule=rule, n_molecules=len(ids), n_spectra=len(q),
                spectra_per_molecule=q.groupby("molecule_id").size().describe().round(2).to_dict(),
                hidden_train_rows=int(hid.sum()),
                remaining_ref_rows_of_truth=int((meta.inchikey14.isin(keyset).values & ~hid).sum()),
                libs_of_queries=q.ingest_lib.value_counts().to_dict())
    (out / "HOLDOUT.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--out-root", default=str(ROOT), help="出力先の根（再現確認用）")
    args = ap.parse_args()
    train = ROOT / "data/train.parquet"
    meta = pq.read_table(train, columns=["inchikey14", "ingest_lib", "instrument_type", "precursor_mz", "adduct"],
                         read_dictionary=["inchikey14", "ingest_lib", "instrument_type", "adduct"]).to_pandas()  # メモリ節約
    import sys
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem
    meta["neutral_mass"] = chem.neutral_mass(meta.precursor_mz.values.astype(np.float64), meta.adduct.astype(str).values)
    split = pd.read_parquet(ROOT / "data/ext/foldsafe/split.parquet")
    hold = set(split.inchikey14[split.fold == "hold"])
    c1, c2, n1, n2 = select(meta, hold, args.n, args.seed)
    print(f"eligible C1 {n1}, C2 {n2}; picked {len(c1)}, {len(c2)}")
    rng = np.random.default_rng(args.seed + 1)
    for name, keys, rule in (("v3_c1", c1, "enveda_libs"), ("v3_c2", c2, "all")):
        print(json.dumps(write_holdout(name, keys, rule, meta, rng, args.seed, train, Path(args.out_root))))

if __name__ == "__main__":
    main()
