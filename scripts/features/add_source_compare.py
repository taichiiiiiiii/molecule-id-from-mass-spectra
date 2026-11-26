#!/usr/bin/env python3
"""
dev 表に群 N2（候補源の比べ: プール vs PubChem 層、Issue #266）の 16 列を足して別名で書く。run.py・pubchem.py は変えない。
正解のラベル・MRR は使わない（記録は行数・sha256・NaN/inf・件数だけ）。

    PY=".../.venv/bin/python"
    $PY scripts/features/add_source_compare.py \
        --in  /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8_cc.parquet \
        --out /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8_cc_n2.parquet

定義（#266 の本文、値を見る前に固定）:
- (i)   log_n_tier_window = log1p(pc_mass.npy の ±10 ppm の窓の件数)。pubchem.Tier.window と同じ searchsorted、隠さない
- (ii)  log_n_pool_window = log1p(run.visible_window（hidden = pool=remove の集合の drop_keys）の窓の重複を除いた鍵の数)
- (iii) src_gap_fz・src_gap_tani = 分子ごとの プールの行の最大 − 層の行の最大（片側なしは NaN）
- (iv)  opp_gap_fz・opp_gap_tani = 候補の値 − 反対側の候補源の行の最大（反対側なしは NaN）
- (v)   fz_src_*・tani_pred_src_* = 群 set × molecule_id × is_tier の中の relativize.py と同じ 5 種
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/features", "scripts/local_eval"):
    sys.path.insert(0, str(ROOT / _p))

import relativize as RZ  # noqa: E402
import run as R  # noqa: E402

PPM = R.PPM                     # 10 ppm（pubchem.PPM と同じ）
TOPN = R.TOPN                   # 60（プールの行の上限）
MOL = ["set", "molecule_id"]
EXPECT = dict(n_mol=3901, n_rows=430979)
COUNT_COLS = ("log_n_tier_window", "log_n_pool_window")
GAP_COLS = ("src_gap_fz", "src_gap_tani", "opp_gap_fz", "opp_gap_tani")
REL_COLS = tuple(f"{v}_src_{s}" for v in ("fz", "tani_pred") for s in RZ.SUFFIXES)
NEW_COLS = COUNT_COLS + GAP_COLS + REL_COLS


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- (i)(ii) 窓の件数
def tier_window_count(mass: np.ndarray, target: float, ppm: float = PPM) -> int:
    """pubchem.Tier.window と同じ境界（left / right の searchsorted）の件数。SMILES は読まない。正解は隠さない。"""
    tol = target * ppm * 1e-6
    a = int(np.searchsorted(mass, target - tol, "left"))
    b = int(np.searchsorted(mass, target + tol, "right"))
    return max(b - a, 0)


def pool_window_keys(pool, target: float, hidden: set, ppm: float = PPM) -> np.ndarray:
    """run.visible_window と同じ（hidden の鍵を除いた）窓の、重複を除いた鍵。"""
    idx = R.visible_window(pool, target, hidden, ppm)
    return np.unique(pool.key[idx])


# ---------------------------------------------------------------- (iii)(iv) 候補源の差
def source_gaps(df: pd.DataFrame) -> dict:
    """{列名: float64 配列}。片側（反対側）の行が無ければ NaN。行の順は df のまま。"""
    tier = df["is_tier"].to_numpy() == 1
    keys = [df[c] for c in MOL]
    out = {}
    for v, name in (("fz", "fz"), ("tani_pred", "tani")):
        x = df[v].astype(np.float64)
        pmax = x.where(~tier).groupby(keys, sort=False).transform("max").to_numpy()
        tmax = x.where(tier).groupby(keys, sort=False).transform("max").to_numpy()
        out[f"src_gap_{name}"] = pmax - tmax
        out[f"opp_gap_{name}"] = np.where(tier, x.to_numpy() - pmax, x.to_numpy() - tmax)
    return out


# ---------------------------------------------------------------- (v) 候補源の中の相対化
def source_relative(df: pd.DataFrame) -> dict:
    tmp = pd.DataFrame({"set": df["set"].to_numpy(), "molecule_id": df["molecule_id"].to_numpy(),
                        "is_tier": df["is_tier"].to_numpy(), "fz_src": df["fz"].to_numpy(np.float64),
                        "tani_pred_src": df["tani_pred"].to_numpy(np.float64)})
    rel = RZ.add_relative(tmp, MOL + ["is_tier"], ["fz_src", "tani_pred_src"])
    return {c: rel[c].to_numpy() for c in REL_COLS}


def add_columns(df: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """counts（set, molecule_id, n_tier_window, n_pool_window）から 16 列を足した表。元の列・行の順は変えない。"""
    for c in NEW_COLS:
        if c in df.columns:
            raise ValueError(f"列が既にある: {c}")
    m = df[MOL].merge(counts, on=MOL, how="left", validate="many_to_one")
    if m[["n_tier_window", "n_pool_window"]].isna().any().any():
        raise ValueError("窓の件数の無い分子がある")
    new = {"log_n_tier_window": np.log1p(m["n_tier_window"].to_numpy(np.float64)),
           "log_n_pool_window": np.log1p(m["n_pool_window"].to_numpy(np.float64))}
    new |= source_gaps(df)
    new |= source_relative(df)
    return df.assign(**{c: new[c] for c in NEW_COLS})


# ---------------------------------------------------------------- 実データ
def set_counts(h: str, mols: list, pool, tier_mass, cand_keys: dict) -> tuple[pd.DataFrame, dict]:
    """1 集合の分子ごとの窓の件数と点検。cand_keys = {molecule_id: 表のプールの行の cand_key のリスト}。"""
    import library as L
    zf = np.load(ROOT / f"outputs/local_eval/{h}/pc_logits.npz", allow_pickle=True)
    targets = {str(m): float(t) for m, t in zf["__targets__"]}
    rule = L.hide_rule(h)
    hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
    rows, chk = [], dict(set=h, pool_rule=rule.get("pool"), n_hidden_keys=len(hidden), n_molecules=len(mols),
                         n_lt60=0, n_lt60_mismatch=0, n_ge60_rows_ne60=0, n_pool_rows_gt_window=0,
                         n_keys_not_in_window=0, n_hidden_counted=0, n_mol_hidden_removed=0)
    for mid in mols:
        t = targets[mid]
        vk = pool_window_keys(pool, t, hidden)
        raw = np.unique(pool.key[pool.window(t)])
        nv = int(len(vk))
        if hidden:
            chk["n_hidden_counted"] += int(np.isin(vk, list(hidden)).sum())
            chk["n_mol_hidden_removed"] += int(len(raw) > nv)
        ck = cand_keys.get(mid, [])
        npool = len(ck)
        if nv < TOPN:
            chk["n_lt60"] += 1
            chk["n_lt60_mismatch"] += int(npool != nv)
        elif npool != TOPN:
            chk["n_ge60_rows_ne60"] += 1
        chk["n_pool_rows_gt_window"] += int(npool > nv)
        chk["n_keys_not_in_window"] += int(len(set(ck) - set(vk.tolist())))
        rows.append((h, mid, tier_window_count(tier_mass, t), nv))
    chk["ok"] = chk["n_hidden_counted"] == 0 and chk["n_pool_rows_gt_window"] == 0 and chk["n_keys_not_in_window"] == 0
    return pd.DataFrame(rows, columns=MOL + ["n_tier_window", "n_pool_window"]), chk


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier-dir", default=str(ROOT / "data/ext/pubchem_tier"))
    a = ap.parse_args(argv)
    t0 = time.time()
    inp, out = Path(a.inp), Path(a.out)
    df = pd.read_parquet(inp)
    pool = R.load_pool()
    tier_mass = np.load(Path(a.tier_dir) / "pc_mass.npy", mmap_mode="r")
    assert bool(np.all(tier_mass[1:] >= tier_mass[:-1])), "pc_mass.npy が昇順でない"
    pool_rows = df[df["is_tier"].to_numpy() == 0]
    counts, checks = [], []
    for h, g in df.groupby("set", sort=False):
        mols = list(pd.unique(g["molecule_id"].astype(str)))
        pg = pool_rows[pool_rows["set"] == h]
        ck = {str(m): list(x) for m, x in pg.groupby("molecule_id", sort=False)["cand_key"]}
        c, chk = set_counts(h, mols, pool, tier_mass, ck)
        counts.append(c); checks.append(chk)
        print(json.dumps(chk, ensure_ascii=False), flush=True)
    counts = pd.concat(counts, ignore_index=True)
    new = add_columns(df, counts)
    # 点検: 元の列は値も順も同じ・行数
    same = all(new[c].equals(df[c]) for c in df.columns)
    num = new[list(NEW_COLS)].to_numpy(np.float64)
    tier = df["is_tier"].to_numpy() == 1
    mol_has = df.assign(t=tier.astype(int)).groupby(MOL, sort=False)["t"].agg(["min", "max"])
    n_mol_no_tier = int((mol_has["max"] == 0).sum()); n_mol_no_pool = int((mol_has["min"] == 1).sum())
    rows_one_side = df.assign(t=tier.astype(int)).groupby(MOL, sort=False)["t"].transform(
        lambda s: int(s.min() == s.max())).to_numpy() == 1
    nan = {c: int(np.isnan(num[:, j]).sum()) for j, c in enumerate(NEW_COLS)}
    expect_nan = {c: 0 for c in NEW_COLS} | {c: int(rows_one_side.sum()) for c in GAP_COLS}
    tmp = out.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    tmp.replace(out)
    rep = dict(issue=266, input=str(inp), input_sha256=sha256_file(inp), output=str(out),
               output_sha256=sha256_file(out), output_is_symlink=out.is_symlink(),
               output_realpath=str(out.resolve()), parent_is_symlink=out.parent.is_symlink(),
               n_rows=int(len(new)), n_molecules=int(len(new[MOL].drop_duplicates())),
               n_rows_ok=len(new) == EXPECT["n_rows"], n_molecules_ok=len(new[MOL].drop_duplicates()) == EXPECT["n_mol"],
               added=list(NEW_COLS), original_columns_identical=bool(same), n_inf=int(np.isinf(num).sum()),
               nan=nan, expect_nan=expect_nan, nan_ok=nan == expect_nan,
               n_mol_no_tier_rows=n_mol_no_tier, n_mol_no_pool_rows=n_mol_no_pool,
               n_tier_window_total=int(counts.n_tier_window.sum()), n_pool_window_total=int(counts.n_pool_window.sum()),
               window_checks=checks, window_checks_ok=all(c["ok"] for c in checks),
               n_lt60=sum(c["n_lt60"] for c in checks), n_lt60_mismatch=sum(c["n_lt60_mismatch"] for c in checks),
               n_ge60_rows_ne60=sum(c["n_ge60_rows_ne60"] for c in checks),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
               secs=round(time.time() - t0, 1))
    rep["all_ok"] = bool(rep["n_rows_ok"] and rep["n_molecules_ok"] and same and rep["n_inf"] == 0 and rep["nan_ok"]
                         and rep["window_checks_ok"] and not rep["output_is_symlink"] and not rep["parent_is_symlink"])
    rp = out.with_name(out.stem + "_report.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "window_checks"}, ensure_ascii=False), flush=True)
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
