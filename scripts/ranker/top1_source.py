#!/usr/bin/env python3
"""
OOF の 1 位の候補源（PubChem 層 / プール）の内訳を型ごとに数える（Issue #266）。記述だけ（採否・列の足し引きには使わない）。

1 位 = evaluate.ranked_lists と同じ並び（分子ごとに score の降順、同点は row の順）の先頭の行。is_tier は表の
`row` 番目の行（meta.json の table）から取り、set・molecule_id が OOF と一致することを確かめる。
正しいかどうかは per_molecule.csv の rr == 1（1 位が正解）で数える。

    python scripts/ranker/top1_source.py outputs/local_eval/ranker/lambdarank/exp003 outputs/local_eval/ranker/lambdarank/exp009
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ("a", "b", "c", "d", "e")


def top1_rows(oof: pd.DataFrame) -> pd.DataFrame:
    """分子ごとの 1 位の行（evaluate.ranked_lists と同じ並びの先頭）。列 set, molecule_id, type, row。"""
    d = oof.sort_values(["set", "molecule_id", "score", "row"], ascending=[True, True, False, True], kind="mergesort")
    return d.groupby(["set", "molecule_id"], sort=False).head(1)[["set", "molecule_id", "type", "row"]]


def count(top: pd.DataFrame, is_tier: np.ndarray, per_mol: pd.DataFrame) -> dict:
    t = top.assign(tier=is_tier[top["row"].to_numpy()].astype(int))
    pm = per_mol[["set", "molecule_id", "rr"]]
    t = t.merge(pm, on=["set", "molecule_id"], how="left", validate="one_to_one")
    t["hit"] = (t["rr"] == 1.0).astype(int)
    out = {}
    for ty in TYPES:
        g = t[t["type"] == ty]
        out[ty] = dict(n_molecules_with_rows=int(len(g)), top1_tier=int(g["tier"].sum()),
                       top1_pool=int((g["tier"] == 0).sum()), top1_tier_correct=int((g["tier"] * g["hit"]).sum()),
                       top1_pool_correct=int(((1 - g["tier"]) * g["hit"]).sum()))
    return out


def run(exp_dir: Path) -> dict:
    meta = json.loads((exp_dir / "meta.json").read_text())
    oof = pd.read_parquet(exp_dir / "oof.parquet", columns=["row", "set", "molecule_id", "type", "score"])
    tab = pd.read_parquet(meta["table"], columns=["set", "molecule_id", "is_tier"])
    if meta["rows"] == "pool":
        tab = tab[tab["is_tier"] == 0].reset_index(drop=True)
    if len(tab) != len(oof) or not ((tab["set"].values == oof["set"].values).all()
                                    and (tab["molecule_id"].values == oof["molecule_id"].values).all()):
        raise RuntimeError(f"{exp_dir}: OOF と表の行が揃わない")
    if not (oof["row"].to_numpy() == np.arange(len(oof))).all():
        raise RuntimeError(f"{exp_dir}: row が 0..n−1 でない")
    pm = pd.read_csv(exp_dir / "per_molecule.csv", dtype={"molecule_id": str})
    return count(top1_rows(oof), tab["is_tier"].to_numpy(), pm)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(json.dumps({Path(d).name: run(Path(d)) for d in argv}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
