#!/usr/bin/env python3
"""GLACIER の群の漏れの規則（#242・#259）の分子の一覧を molecules.parquet から作る。

- msg1_exclude.csv: msg_overlap = 1 の分子（528）。GLACIER の群を含む行の学習から外す（exp007・exp008 の両方）
- msg0_eval.csv:    msg_overlap = 0 の分子（3,373）。評価はこの分子で読む

    python scripts/ranker/make_msg_lists.py --molecules outputs/local_eval/glacier_l60/molecules.parquet \
        --out-dir docs/experiments
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECT_N1 = 528        # #242 の manifest（msg_overlap.n1）
EXPECT_N = 3901


def make_lists(mols: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = mols[["set", "molecule_id", "msg_overlap"]].copy()
    m["molecule_id"] = m.molecule_id.astype(str)
    assert not m.duplicated(["set", "molecule_id"]).any()
    assert set(m.msg_overlap.unique()) <= {0, 1}
    m = m.sort_values(["set", "molecule_id"], kind="stable")
    one = m[m.msg_overlap == 1][["set", "molecule_id"]].reset_index(drop=True)
    zero = m[m.msg_overlap == 0][["set", "molecule_id"]].reset_index(drop=True)
    return one, zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    mols = pd.read_parquet(a.molecules)
    assert len(mols) == EXPECT_N, len(mols)
    one, zero = make_lists(mols)
    assert len(one) == EXPECT_N1 and len(one) + len(zero) == EXPECT_N, (len(one), len(zero))
    rec = {}
    for name, df in (("msg1_exclude.csv", one), ("msg0_eval.csv", zero)):
        p = a.out_dir / name
        df.to_csv(p, index=False)
        rec[name] = dict(n=len(df), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    rec["by_type_msg0"] = mols[mols.msg_overlap == 0].type.value_counts().sort_index().to_dict()
    rec["molecules_sha256"] = hashlib.sha256(a.molecules.read_bytes()).hexdigest()
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
