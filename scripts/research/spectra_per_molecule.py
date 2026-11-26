#!/usr/bin/env python3
"""
1 分子あたりのスペクトル本数の分布（Issue #276、ラベルを使わない数え）。

- dev: 特徴表 dev_b8.parquet の `set` 列にある集合（B8 の 20 集合）の data/holdouts/<set>/test.parquet を
  molecule_id ごとに数える。型（集合名の末尾の a〜e）ごとと全体
- 見える test: data/test.parquet（400 分子）
- 参考: train.parquet の inchikey14 ごと（全体と ingest_lib ごと）。列は inchikey14・ingest_lib だけを
  row group ごとに読む（3 GB を全部は読まない）

正解の構造・MRR は使わない（train の inchikey14 は train の側の構造の数え）。結果は JSON で標準出力と --out に。

    .venv/bin/python scripts/research/spectra_per_molecule.py --out outputs/local_eval/spectra_per_molecule.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
QUERY_CAP = 16  # run.query_spectra と同じ「分子ごとに先頭 16 本」


def summarize(counts) -> dict:
    c = np.asarray(list(counts), np.int64)
    if len(c) == 0:
        return dict(n_molecules=0)
    hist = Counter(int(x) for x in c)
    return dict(
        n_molecules=int(len(c)), n_spectra=int(c.sum()),
        mean=round(float(c.mean()), 3), median=float(np.median(c)),
        p10=float(np.percentile(c, 10)), p90=float(np.percentile(c, 90)), max=int(c.max()),
        share_1=round(float((c == 1).mean()), 4), share_le3=round(float((c <= 3).mean()), 4),
        share_gt16=round(float((c > QUERY_CAP).mean()), 4),
        mean_capped16=round(float(np.minimum(c, QUERY_CAP).mean()), 3),
        hist={str(k): hist[k] for k in sorted(hist)[:20]},
    )


def per_molecule(path: Path) -> Counter:
    t = pq.read_table(path, columns=["molecule_id"]).column("molecule_id").to_pylist()
    return Counter(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "outputs/local_eval/features/dev_b8.parquet"))
    ap.add_argument("--test", default=str(ROOT / "data/test.parquet"))
    ap.add_argument("--train", default=str(ROOT / "data/train.parquet"))
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sets = sorted(set(pq.read_table(a.table, columns=["set"]).column("set").to_pylist()))
    out: dict = dict(dev_sets=sets)
    by_type: dict[str, list[int]] = {}
    allc: list[int] = []
    for s in sets:
        c = per_molecule(ROOT / "data/holdouts" / s / "test.parquet")
        t = s.split("_")[2]  # b8_f1_a_t10 → a
        by_type.setdefault(t, []).extend(c.values())
        allc.extend(c.values())
    out["dev_all"] = summarize(allc)
    out["dev_by_type"] = {t: summarize(v) for t, v in sorted(by_type.items())}

    vt = pq.read_table(a.test, columns=["molecule_id", "adduct", "ionization_mode"]).to_pandas()
    out["visible_test"] = summarize(vt.groupby("molecule_id").size().to_numpy())
    out["visible_test_adducts"] = {k: int(v) for k, v in vt["adduct"].value_counts().items()}
    out["visible_test_modes_per_molecule"] = {
        str(k): int(v) for k, v in vt.groupby("molecule_id")["ionization_mode"].nunique().value_counts().items()}

    if not a.no_train:
        f = pq.ParquetFile(a.train, buffer_size=1 << 20, pre_buffer=False)
        tot: Counter = Counter()
        by_lib: dict[str, Counter] = {}
        for b in f.iter_batches(batch_size=65536, columns=["inchikey14", "ingest_lib"]):
            k = b.column(0).to_pylist()
            lib = b.column(1).to_pylist()
            tot.update(k)
            for kk, ll in zip(k, lib):
                by_lib.setdefault(ll, Counter())[kk] += 1
        out["train_per_inchikey14"] = summarize(tot.values())
        out["train_per_inchikey14_by_lib"] = {l: summarize(c.values()) for l, c in sorted(by_lib.items())}

    s = json.dumps(out, ensure_ascii=False, indent=1)
    print(s)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(s + "\n")


if __name__ == "__main__":
    main()
