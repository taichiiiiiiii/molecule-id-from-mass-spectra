#!/usr/bin/env python3
"""
Issue #236 補助（事後の読み、#236 の定義の外。recall の値は変えない）: プールの窓 (i) で正解が見つからなかった dev の分子について、
窓の中心と正解の構造の質量（RDKit の ExactMolWt）の差を数える。中心が ±10 ppm の外にずれているか（付加イオンの読みの違いなど）を見る。
dev の分子だけ（candidate_recall.py の per_molecule.parquet を読む）。

    .venv/bin/python scripts/bench/recall_miss_center.py --out docs/bench/recall_miss_center.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def bins(delta_da: np.ndarray, target: np.ndarray) -> dict:
    """差（Da）→ ±10 ppm の内、10 ppm〜0.5 Da、0.5 Da より大の数。"""
    ok = np.isfinite(delta_da)
    ppm = np.abs(delta_da[ok]) / target[ok] * 1e6
    return dict(n_mass=int(ok.sum()), le10ppm=int((ppm <= 10).sum()),
                gt10ppm_le0p5Da=int(((ppm > 10) & (np.abs(delta_da[ok]) <= 0.5)).sum()),
                gt0p5Da=int((np.abs(delta_da[ok]) > 0.5).sum()))


def main() -> None:
    import pandas as pd
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-molecule", default=str(ROOT / "outputs/local_eval/i236_recall/per_molecule.parquet"))
    ap.add_argument("--out", default=str(ROOT / "docs/bench/recall_miss_center.json"))
    a = ap.parse_args()
    pm = pd.read_parquet(a.per_molecule)
    ts = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet", columns=["inchikey14", "smiles"])
    smi = dict(zip(ts.inchikey14, ts.smiles))
    res = {}
    for t in ("a", "b", "d", "none"):   # プールを隠さない型だけ（(c)(e) は (i) から正解を除いている）
        sel = pm[(pm.type == t) & (~pm.found_i) & pm.has_center]
        d = []
        for r in sel.itertuples():
            m = Chem.MolFromSmiles(smi.get(r.inchikey14, ""))
            d.append(np.nan if m is None else r.target - ExactMolWt(m))
        res[t] = dict(n_miss_i=len(sel), found_ii_among_miss=int(sel.found_ii.sum()),
                      **bins(np.array(d, np.float64), sel.target.values.astype(np.float64)))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
