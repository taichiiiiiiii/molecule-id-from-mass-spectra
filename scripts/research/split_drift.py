#!/usr/bin/env python3
"""#222: 構造で分けた検証の分布のずれ。v3 の _t10 4 集合の正解・fold-0 の学習側・見える test の
precursor_mz と重原子数を、Issue #222 の事前登録どおり 1 回だけ数える。学習・推論・採点はしない。

群:
- G1〜G4: outputs/local_eval/<h>/truth.csv（v3_c1u_t10・v3_c2_t10・enveda_v1u_t10・v3_c2nu_t10）。
  precursor_mz は data/holdouts/<h>/test.parquet の全行
- G5: identity-disjoint folds の structures.parquet のうち data/ext/foldsafe/split.parquet（fold-0 の FPNet から
  外した hold・np）に入らない構造。precursor_mz は train.parquet の行のうち inchikey14 が G5 にあり、付加イオンが
  t10 の 10 種（data/holdouts/v3_c2_t10/HOLDOUT.json の adducts）に入るもの
- G6: 見える test（train の写し）。precursor_mz は data/test.parquet の全行、構造は
  outputs/test_library_match.parquet の best_key を structures.parquet で引く

指標: 件数・中央値・第 1/第 3 四分位（numpy linear）、G5 との 2 標本 KS の D（ここの numpy 実装）と p 値（scipy、
実データの実行のときだけ）、補助に P(群 > G5)（U/(n1·n2)、同値 0.5）。読み方は記述だけ（KS の値で何も決めない）。

train.parquet は `pq.ParquetFile(p, buffer_size=1<<20, pre_buffer=False)` で 3 列だけ行群ごとに読み、
`pc.is_in` で絞る。純粋な関数（CI でテストする）は numpy だけを使う。

    .venv/bin/python scripts/research/split_drift.py --out outputs/local_eval/split_drift/split_drift.json
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ISSUE = 222
ROOT = Path(__file__).resolve().parents[2]
HOLDOUTS = ("v3_c1u_t10", "v3_c2_t10", "enveda_v1u_t10", "v3_c2nu_t10")
FOLDS = Path("/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra/data/ext/panel/"
             "casmi26-identity-disjoint-validation-folds/structures.parquet")


# ---------- 純粋な関数（tests/test_split_drift.py） ----------

def quartiles(x) -> dict:
    """件数・第 1 四分位・中央値・第 3 四分位（numpy の linear）。"""
    a = np.asarray(x, dtype=float)
    assert a.size > 0 and np.isfinite(a).all()
    q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    return dict(n=int(a.size), q1=float(q1), median=float(med), q3=float(q3))


def ks_d(x, y) -> float:
    """2 標本 KS 統計量 D = sup |F_x − F_y|（経験分布、同値は両方の値の右連続で評価）。"""
    a = np.sort(np.asarray(x, dtype=float))
    b = np.sort(np.asarray(y, dtype=float))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def p_greater(x, y) -> float:
    """P(X > Y) + 0.5·P(X = Y)（Mann–Whitney の U/(n1·n2)）。y が大きくても速いようにソートで数える。"""
    a = np.asarray(x, dtype=float)
    b = np.sort(np.asarray(y, dtype=float))
    lt = np.searchsorted(b, a, side="left")     # b < a の数
    le = np.searchsorted(b, a, side="right")    # b <= a の数
    return float((lt + 0.5 * (le - lt)).sum() / (a.size * b.size))


# ---------- 実データ ----------

def heavy_atoms(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = np.empty(len(smiles), dtype=np.int64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"RDKit が読めない SMILES: {s}")
        out[i] = m.GetNumHeavyAtoms()
    return out


def read_truth(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def train_precursor(train: Path, keys: set[str], adducts: list[str]) -> tuple[np.ndarray, dict]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(train, buffer_size=1 << 20, pre_buffer=False)
    kset = pa.array(sorted(keys))
    aset = pa.array(adducts)
    parts, n_all, n_key = [], 0, 0
    for b in pf.iter_batches(batch_size=8192, columns=["inchikey14", "adduct", "precursor_mz"]):
        n_all += b.num_rows
        mk = pc.is_in(b.column("inchikey14"), value_set=kset)
        n_key += int(pc.sum(mk).as_py() or 0)
        m = pc.and_(mk, pc.is_in(b.column("adduct"), value_set=aset))
        parts.append(b.column("precursor_mz").filter(m).to_numpy(zero_copy_only=False))
    x = np.concatenate(parts).astype(float)
    return x, dict(train_rows=n_all, g5_rows_all_adducts=n_key, g5_rows_t10=int(x.size))


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/split_drift/split_drift.json"))
    ap.add_argument("--folds", default=str(FOLDS))
    args = ap.parse_args()
    t0 = time.time()
    import rdkit
    import scipy
    from scipy.stats import ks_2samp

    st = pd.read_parquet(args.folds, columns=["inchikey14", "smiles", "n_heavy"]).drop_duplicates("inchikey14")
    split = pd.read_parquet(ROOT / "data/ext/foldsafe/split.parquet", columns=["inchikey14", "fold"])
    out_keys = set(split.inchikey14)
    g5 = st[~st.inchikey14.isin(out_keys)].reset_index(drop=True)
    adducts = json.loads((ROOT / "data/holdouts/v3_c2_t10/HOLDOUT.json").read_text())["adducts"]
    assert len(adducts) == 10

    groups: dict[str, dict] = {}
    for h in HOLDOUTS:
        tr = read_truth(ROOT / f"outputs/local_eval/{h}/truth.csv")
        te = pd.read_parquet(ROOT / f"data/holdouts/{h}/test.parquet", columns=["molecule_id", "precursor_mz"])
        assert set(te.molecule_id) == {r["molecule_id"] for r in tr}, h
        k = [r["inchikey14"] for r in tr]
        groups[h] = dict(pmz=te.precursor_mz.to_numpy(float), heavy=heavy_atoms([r["smiles"] for r in tr]),
                         n_in_g5=int(pd.Series(k).isin(set(g5.inchikey14)).sum()))
    # G5
    g5_heavy = heavy_atoms(g5.smiles.tolist())
    g5_pmz, g5_rows = train_precursor(ROOT / "data/train.parquet", set(g5.inchikey14), adducts)
    groups["fold0_train"] = dict(pmz=g5_pmz, heavy=g5_heavy, rows=g5_rows,
                                 n_heavy_mismatch_vs_folds=int((g5_heavy != g5.n_heavy.to_numpy()).sum()))
    # G6
    lm = pd.read_parquet(ROOT / "outputs/test_library_match.parquet", columns=["molecule_id", "best_cos", "best_key"])
    vt = pd.read_parquet(ROOT / "data/test.parquet", columns=["molecule_id", "precursor_mz", "adduct"])
    assert set(lm.molecule_id) == set(vt.molecule_id) and len(lm) == 400
    sm = st.set_index("inchikey14").reindex(lm.best_key)
    assert sm.smiles.notna().all()
    groups["visible_test"] = dict(pmz=vt.precursor_mz.to_numpy(float), heavy=heavy_atoms(sm.smiles.tolist()),
                                  n_best_cos_ge_099=int((lm.best_cos >= 0.99).sum()),
                                  adducts_outside_t10=int((~vt.adduct.isin(adducts)).sum()),
                                  n_in_g5=int(lm.best_key.isin(set(g5.inchikey14)).sum()))

    ref = groups["fold0_train"]
    res = dict(issue=ISSUE, rdkit=rdkit.__version__, scipy=scipy.__version__, adducts_t10=adducts,
               n_g5_structures=int(len(g5)), n_split_excluded=int(len(out_keys)), groups={})
    for name, g in groups.items():
        r = {k: v for k, v in g.items() if k not in ("pmz", "heavy")}
        for m in ("pmz", "heavy"):
            r[m] = quartiles(g[m])
            if name != "fold0_train":
                d = ks_d(g[m], ref[m])
                s = ks_2samp(g[m], ref[m])
                assert abs(d - s.statistic) < 1e-12, (name, m, d, s.statistic)
                r[m].update(ks_d=round(d, 4), ks_p=float(s.pvalue), p_greater_than_g5=round(p_greater(g[m], ref[m]), 4))
        res["groups"][name] = r
    res["secs"] = round(time.time() - t0, 1)
    res["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)   # macOS は bytes
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
