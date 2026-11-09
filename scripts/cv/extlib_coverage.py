#!/usr/bin/env python3
"""
Issue #52: 外部ライブラリ（ExtLib・MassBank harmonized）の構造が、train・fold-safe の hold 構造・timsTOF の問い合わせと
どれだけ重なるかを数える（検証集合を作れるかの判断材料）。スペクトル本体は読まない（メタデータ列だけ）。

入力（Kaggle から取得して置く。リポジトリには入れない）:
    --extlib DIR   takumuhata/casmi26-extlib の ext_meta.parquet
    --mbh DIR      samartalwar/casmi-2026-spectral-library-massbankharmonized の spectra.parquet
train は data/train.parquet の 4 列（inchikey14, ingest_lib, instrument_type, adduct）だけ辞書型で読む。
hold は data/ext/foldsafe/split.parquet の fold == "hold"（fold-safe fold 0 モデルが学習で見ていない構造）。

数（定義はこの関数どおり）:
  n_struct / in_train / not_in_train     : 外部ライブラリの構造（InChIKey14）と train との重なり
  instrument                             : 外部ライブラリのスペクトルの装置の内訳（上位 10）
  tims_spectra                           : 外部ライブラリで装置名に "tims" を含むスペクトル数
  hold / hold_tims / hold_public_query   : hold 構造との重なり、そのうち train に timsTOF スペクトルがある数、
                                           train に公開ライブラリ（PUBLIC）の test と同じ付加イオンのスペクトルがある数
  not_in_train_tims_query                : train に無い構造で timsTOF の問い合わせを作れる数（train に無いので定義上 0）

    .venv/bin/python scripts/cv/extlib_coverage.py --extlib <dir> --mbh <dir> [--out outputs/extlib_coverage.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = {"pluskal_ms2", "riken", "gnps", "massbank", "mona", "spectraverse", "msdial", "drug_plus", "masaryk"}


def coverage(ext_keys: set, ext_instr: pd.Series, train_keys: set, hold: set, tims_keys: set, public_q_keys: set) -> dict:
    h = ext_keys & hold
    return dict(n_struct=len(ext_keys), in_train=len(ext_keys & train_keys), not_in_train=len(ext_keys - train_keys),
                not_in_train_tims_query=len((ext_keys - train_keys) & tims_keys),
                n_spectra=int(len(ext_instr)), instrument=ext_instr.astype(str).value_counts().head(10).to_dict(),
                tims_spectra=int(ext_instr.astype(str).str.contains("tims", case=False).sum()),
                hold=len(h), hold_tims=len(h & tims_keys), hold_public_query=len(h & public_q_keys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extlib", required=True)
    ap.add_argument("--mbh", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs/extlib_coverage.json"))
    a = ap.parse_args()
    meta = pq.read_table(ROOT / "data/train.parquet", columns=["inchikey14", "ingest_lib", "instrument_type", "adduct"],
                         read_dictionary=["inchikey14", "ingest_lib", "instrument_type", "adduct"]).to_pandas()
    meta["inchikey14"] = meta.inchikey14.astype(str)
    test_adducts = set(pd.read_parquet(ROOT / "data/test.parquet", columns=["adduct"]).adduct.astype(str))
    train_keys = set(meta.inchikey14)
    tims_keys = set(meta.loc[meta.instrument_type.astype(str) == "timsTOF", "inchikey14"])
    public_q = set(meta.loc[meta.ingest_lib.astype(str).isin(PUBLIC) & meta.adduct.astype(str).isin(test_adducts), "inchikey14"])
    sp = pd.read_parquet(ROOT / "data/ext/foldsafe/split.parquet")
    hold = set(sp.loc[sp.fold.astype(str) == "hold", "inchikey14"])

    ext = pd.read_parquet(Path(a.extlib) / "ext_meta.parquet", columns=["inchikey14", "instrument_type"])
    mb = pd.read_parquet(Path(a.mbh) / "spectra.parquet", columns=["inchikey", "instrument_type", "license"])
    mb_keys = set(mb.inchikey.astype(str).str[:14])
    out = dict(train_struct=len(train_keys), hold=len(hold), test_adducts=sorted(test_adducts),
               extlib=coverage(set(ext.inchikey14), ext.instrument_type, train_keys, hold, tims_keys, public_q),
               mbh=coverage(mb_keys, mb.instrument_type, train_keys, hold, tims_keys, public_q)
               | dict(license=mb.license.astype(str).value_counts().to_dict()),
               extlib_and_mbh=len(set(ext.inchikey14) & mb_keys))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
