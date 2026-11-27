#!/usr/bin/env python3
"""GLACIER の群の漏れの規則（#242・#259）の分子の一覧を molecules.parquet から作る。

- msg1_exclude.csv: msg_overlap = 1 の分子（528）。GLACIER の群を含む行の学習から外す（exp007・exp008 の両方）
- msg0_eval.csv:    msg_overlap = 0 の分子（3,373）。評価はこの分子で読む

    python scripts/ranker/make_msg_lists.py --molecules outputs/local_eval/glacier_l60/molecules.parquet \
        --out-dir docs/experiments

Issue #290（判定用 4 集合、§4.2 (d)）: `--which holdout` で holdout_msg1_exclude.csv・holdout_msg0_eval.csv と、集合ごとの
分子数（全体・除く・残り）と「除いた後 150 分子未満の集合は全集合で判定」の印を holdout_msg_lists.json に書く。
数えるのは正解の鍵の重なりだけ（候補の点・順位・MRR は使わない）。規則は分子数だけで決まる。

    python scripts/ranker/make_msg_lists.py --which holdout \
        --molecules outputs/local_eval/glacier_holdout/pool/molecules.parquet --out-dir docs/experiments
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECT_N1 = 528        # #242 の manifest（msg_overlap.n1）
EXPECT_N = 3901
EXPECT_N_HOLD = 1358
HOLD_MIN_REMAIN = 150  # #290 (d): 除いた後の分子数がこれ未満の集合は全集合（重なる分子を含む）で判定する
HOLD_SETS = ("v3_c1u_t10", "enveda_v1u_t10", "v3_c2_t10", "v3_c2nu_t10")


def holdout_rule(mols: pd.DataFrame) -> dict:
    """集合ごとの分子数と (d) の印（分子数だけで決まる）。"""
    out = {}
    for s in HOLD_SETS:
        m = mols[mols.set == s]
        n, n1 = int(len(m)), int((m.msg_overlap == 1).sum())
        out[s] = dict(n=n, n_msg1=n1, n_remain=n - n1, judge_on="subset" if n - n1 >= HOLD_MIN_REMAIN else "full_set")
    return out


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
    ap.add_argument("--which", choices=["dev", "holdout"], default="dev")
    a = ap.parse_args()
    mols = pd.read_parquet(a.molecules)
    if a.which == "holdout":
        assert len(mols) == EXPECT_N_HOLD and sorted(mols.set.unique()) == sorted(HOLD_SETS), len(mols)
        one, zero = make_lists(mols)
        assert len(one) + len(zero) == EXPECT_N_HOLD
        rec = dict(issue=290, min_remain=HOLD_MIN_REMAIN)
        for name, df in (("holdout_msg1_exclude.csv", one), ("holdout_msg0_eval.csv", zero)):
            p = a.out_dir / name
            df.to_csv(p, index=False)
            rec[name] = dict(n=len(df), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        rec["by_set"] = holdout_rule(mols)
        rec["molecules_sha256"] = hashlib.sha256(a.molecules.read_bytes()).hexdigest()
        (a.out_dir / "holdout_msg_lists.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(json.dumps(rec, ensure_ascii=False, indent=1))
        return
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
