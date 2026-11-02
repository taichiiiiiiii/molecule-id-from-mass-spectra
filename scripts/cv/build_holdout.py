#!/usr/bin/env python3
"""
正解付きホールドアウト（隠れテストと同じ形式）を作る。Kaggle 上で公開パイプラインを丸ごと実行して採点するため。

- 構造: identity-disjoint の fold_identity == FOLD から、Enveda スペクトルを持つ構造を N 個選ぶ
- 選んだ構造（同じ identity group の別キーも含む）の全スペクトルを train から除く（参照なしの状況）
- テスト: 各構造の Enveda スペクトル（--np のときは全ライブラリ）から 1〜6 本を選び、test.parquet と同じ列で並べる
- Kaggle 側で同じ除外を再現できるよう、除くキーを drop_keys.txt に書く（train.parquet は --write-train のときだけ書く）

    python scripts/cv/build_holdout.py --out data/holdout_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FOLDS = "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
TEST_COLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct",
             "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_orig", "collision_energy_ev",
             "collision_energy_orig_units"]


def hid(prefix: str, s: str, n: int) -> str:
    return prefix + hashlib.sha1(s.encode()).hexdigest()[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.parquet")
    ap.add_argument("--out", default="data/holdout_v1")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--np", action="store_true", help="天然物（LOTUS/NPAtlas）の構造から選び、全ライブラリのスペクトルをテストに使う")
    ap.add_argument("--np-table", default="data/ext/npk/descriptors.parquet")
    ap.add_argument("--write-train", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st = pd.read_parquet(FOLDS)
    base = (st.fold_identity == args.fold) & st.mono_mass.between(157, 1159)
    if args.np:
        npk = set(pd.read_parquet(args.np_table, columns=["ik14"]).ik14)
        pool = st[base & (st.scorer_key.isin(npk) | st.inchikey14.isin(npk))]
    else:
        pool = st[base & (st.n_enveda_spectra > 0)]
    pool = pool.drop_duplicates("identity_group")
    pick = pool.sample(n=args.n, random_state=args.seed).copy()
    # 参照ありの状況は作らない（Enveda 構造で他ライブラリに参照があるのは 1.2% だけで、
    # --np のときはテストに他ライブラリのスペクトルを使うため）
    pick["scenario"] = "no_reference"
    groups = set(pick.identity_group)
    # 同じ identity group の別キーもまとめて除く
    grp = st[st.identity_group.isin(groups)][["inchikey14", "scorer_key", "identity_group"]]
    g_scn = dict(zip(pick.identity_group, pick.scenario))
    keys_all = {}
    for r in grp.itertuples():
        for k in (r.inchikey14, r.scorer_key):
            keys_all[k] = g_scn[r.identity_group]

    meta = pq.read_table(args.train, columns=["inchikey14", "ingest_lib"]).to_pandas()
    scn = meta.inchikey14.map(keys_all)
    is_enveda = meta.ingest_lib.str.startswith("enveda")
    drop = scn.notna()
    (out / "drop_keys.txt").write_text("\n".join(sorted(keys_all)) + "\n")
    test_cand = scn.notna() & (True if args.np else is_enveda)
    print(f"holdout structures {len(pick)} ({pick.scenario.value_counts().to_dict()}), "
          f"train rows dropped {int(drop.sum()):,} of {len(meta):,}")

    # テスト用スペクトルを選ぶ（構造ごとに 1〜6 本）
    rows = np.flatnonzero(test_cand.to_numpy())
    cand = meta.iloc[rows].assign(row=rows)
    chosen = []
    for k, g in cand.groupby("inchikey14"):
        m = int(rng.integers(1, 7))
        chosen.extend(g.row.sample(n=min(m, len(g)), random_state=int(rng.integers(1 << 31))).tolist())
    chosen = np.sort(np.array(chosen))

    # 学習データは行グループごとに流して処理する（全体を読むと 10 GB 超になるため）
    pf = pq.ParquetFile(args.train)
    writer = pq.ParquetWriter(out / "train.parquet", pf.schema_arrow, compression="zstd") if args.write_train else None
    drop_np, chosen_set = drop.to_numpy(), set(chosen.tolist())
    parts, start = [], 0
    for b in pf.iter_batches(batch_size=125_000):
        n = b.num_rows
        idx = np.arange(start, start + n)
        if writer:
            writer.write_table(pa.Table.from_batches([b]).filter(pa.array(~drop_np[start:start + n])))
        sel = [i - start for i in idx if i in chosen_set]
        if sel:
            parts.append(pa.Table.from_batches([b]).take(pa.array(sel)).to_pandas().assign(_row=idx[sel]))
        start += n
    if writer:
        writer.close()
    te = pd.concat(parts, ignore_index=True).sort_values("_row")
    chosen = te._row.to_numpy()
    k2g = dict(zip(grp.inchikey14, grp.identity_group))
    te["group"] = te.inchikey14.map(k2g)
    te["molecule_id"] = te.group.map(lambda g: hid("m_", f"{args.seed}-{g}", 6))
    te["spectrum_id"] = [hid("s_", f"{args.seed}-{i}", 8) for i in chosen]
    truth = (te.groupby("molecule_id").agg(inchikey14=("inchikey14", "first"), smiles=("normalized_smiles", "first"),
                                           group=("group", "first"), n_spectra=("spectrum_id", "size"))
             .reset_index())
    truth["scenario"] = truth.group.map(g_scn)
    # 採点キーは scorer key（RDKit の標準互変異性体から作った InChIKey14）
    truth["inchikey14"] = truth.group.map(dict(zip(pick.identity_group, pick.scorer_key)))
    te = te.sample(frac=1, random_state=args.seed)[TEST_COLS].reset_index(drop=True)
    te.to_parquet(out / "test.parquet", index=False)
    pd.DataFrame({"molecule_id": sorted(truth.molecule_id), "smiles": "CCO"}).to_csv(out / "sample_submission.csv", index=False)
    truth.drop(columns="group").to_csv(out / "truth.csv", index=False)

    summary = dict(seed=args.seed, fold=args.fold, np=args.np, n_molecules=len(truth), n_spectra=len(te),
                   scenario=truth.scenario.value_counts().to_dict(),
                   spectra_per_molecule=truth.n_spectra.describe().round(2).to_dict(),
                   train_rows=int((~drop).sum()), train_rows_dropped=int(drop.sum()), n_drop_keys=len(keys_all))
    (out / "HOLDOUT.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
