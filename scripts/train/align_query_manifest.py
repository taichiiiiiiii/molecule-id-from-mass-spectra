#!/usr/bin/env python3
"""v3 の 4 集合の問い合わせスペクトルが train.parquet の何行目かを突き合わせ、埋め込みカーネル用の表にする（整列チャネルの本番）。

    .venv/bin/python scripts/train/align_query_manifest.py

v3 の問い合わせ（data/holdouts/<h>/test.parquet）は train.parquet の行の写しで、行番号の列を持たない。
そこで、その集合の drop_keys.txt（正解の inchikey14。enveda_v1 は互変異性体の正準キーと train の生キーの両方を含む）の
行だけを train.parquet から流し読みし、(precursor_mz, adduct, ms2_mzs, ms2_normalized_intensities) が完全に同じ行を探す。
候補が複数あれば、train の inchikey14 が truth.csv のキーと同じ行を優先し、その中でまだ使っていない最初の行を取る
（内容が同じなので埋め込みも同じ）。

出力: scripts/train/align_query_manifest.json
  {"train_rows": 2539608, "n": 4756, "rows": [[holdout, molecule_id, spectrum_id, row, inchikey14, sig], ...]}
  inchikey14 = その行の train.parquet の inchikey14（カーネルが同じ行を読んだことの照合に使う）、sig = spectrum_sig()。
メモリ: 5 列を 65,536 行ずつ読む（ピーク約 4.7 GB）。
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
OUT = ROOT / "scripts/train/align_query_manifest.json"
COLS = ["inchikey14", "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"]


def spectrum_sig(prec, adduct, mzs, its) -> str:
    """スペクトルの内容の指紋（sha1 の先頭 16 桁）。float64 のバイト列で比べるので丸めの差も別物になる。"""
    h = hashlib.sha1()
    h.update(struct.pack("<d", float(prec)))
    h.update(str(adduct).encode())
    h.update(struct.pack(f"<{len(mzs)}d", *map(float, mzs)))
    h.update(struct.pack(f"<{len(its)}d", *map(float, its)))
    return h.hexdigest()[:16]


def assign_rows(queries: list, found: dict, drop_by_holdout: dict) -> int:
    """queries の各行 [h, mid, sid, None, truth_ik, sig] の 4 番目に train の行、5 番目に train の inchikey14 を入れる。
    found[sig] = [(row, train_ik), ...]。見つからない件数を返す。"""
    used: set = set()
    missing = 0
    for q in queries:
        h, truth_ik, sig = q[0], q[4], q[5]
        cands = sorted(c for c in found.get(sig, []) if c[1] in drop_by_holdout[h])
        if not cands:
            missing += 1
            continue
        cands.sort(key=lambda c: (c[1] != truth_ik, c[0] in used, c[0]))
        q[3], q[4] = cands[0]
        used.add(cands[0][0])
    return missing


def main() -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    queries, drop_by_holdout = [], {}
    for h in HOLDOUTS:
        drop_by_holdout[h] = set((ROOT / f"data/holdouts/{h}/drop_keys.txt").read_text().split())
        tr = pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv")
        ik = dict(zip(tr.molecule_id.astype(str), tr.inchikey14.astype(str)))
        te = pd.read_parquet(ROOT / f"data/holdouts/{h}/test.parquet",
                             columns=["molecule_id", "spectrum_id", "precursor_mz", "adduct", "ms2_mzs",
                                      "ms2_normalized_intensities"])
        for r in te.itertuples():
            queries.append([h, str(r.molecule_id), str(r.spectrum_id), None, ik[str(r.molecule_id)],
                            spectrum_sig(r.precursor_mz, r.adduct, r.ms2_mzs, r.ms2_normalized_intensities)])
    all_drop = set().union(*drop_by_holdout.values())
    sigs = {q[5] for q in queries}
    pf = pq.ParquetFile(ROOT / "data/train.parquet")
    n_rows, row0 = pf.metadata.num_rows, 0
    found: dict[str, list] = {}
    for b in pf.iter_batches(batch_size=65536, columns=COLS):
        ik = b.column("inchikey14").to_pylist()
        hit = [i for i, k in enumerate(ik) if k in all_drop]
        if hit:
            sub = b.take(hit).to_pydict()
            for j, i in enumerate(hit):
                s = spectrum_sig(sub["precursor_mz"][j], sub["adduct"][j], sub["ms2_mzs"][j],
                                 sub["ms2_normalized_intensities"][j])
                if s in sigs:
                    found.setdefault(s, []).append((row0 + i, sub["inchikey14"][j]))
        row0 += len(ik)
        print(f"\r{row0:,}/{n_rows:,}", end="", file=sys.stderr)
    print(file=sys.stderr)
    assert row0 == n_rows
    truth_keys = [q[4] for q in queries]
    missing = assign_rows(queries, found, drop_by_holdout)
    if missing:
        raise SystemExit(f"{missing} query spectra not found in train.parquet")
    OUT.write_text(json.dumps({"train_rows": n_rows, "n": len(queries), "holdouts": list(HOLDOUTS),
                               "columns": ["holdout", "molecule_id", "spectrum_id", "row", "inchikey14", "sig"],
                               "rows": queries}, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(queries), "per_holdout": {h: sum(q[0] == h for q in queries) for h in HOLDOUTS},
                      "unique_rows": len({q[3] for q in queries}),
                      "train_key_differs_from_truth_key": sum(q[4] != t for q, t in zip(queries, truth_keys))}))


if __name__ == "__main__":
    main()
