#!/usr/bin/env python3
"""
ランカー学習グループ（Issue #29）: fold_identity 1〜4 の構造から、検証 v3 と同じ規則で C1 型・C2 型の擬似ホールドアウトを作る。

  除外: (i) np（enveda-np-examples の構造と同じ identity_group の全構造）、(ii) split.parquet の全構造（hold / np）、
        (iii) v3_c1 / v3_c2 の構造。
  選び方: scripts/cv/build_holdout_v3.py の select（質量 157〜1159、C1 = enveda 系と公開ライブラリの両方、C2 = enveda-180 だけ）を
          fold k の残りの構造に当てる。各 fold から C1 125・C2 125（seed 20261108 + k）。足りなければ C1 と C2 を少ない方に揃える。
  出力: v3 と同じ形式の擬似ホールドアウト data/holdouts/rk_f{k}_c{1,2}/ と outputs/holdouts/rk_f{k}_c{1,2}/truth.csv。
        C1 は enveda 系の参照だけ隠す（hide_rule = enveda_libs、プールに残す）、C2 は全参照とプールから消す（hide_rule = all）。
        fold k のグループの z は fpnet_fold{k}.pt（fold k を学習していないモデル）で作る。

    python scripts/ranker/build_groups.py --n 125 --seed 20261108

np の起点は train.parquet の ingest_lib == 'enveda-np-examples'（250 キー）。structures.parquet の libs を起点にすると split とずれる
（計画改訂 2 のレビューでの実測: np 265 / hold 55,102 で一致 vs np 255 / hold 55,105）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/cv"))
FOLDS = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
SPLIT = ROOT / "data/ext/foldsafe/split.parquet"
NP_LIB = "enveda-np-examples"


def np_keys(np_seed_keys: set, st: pd.DataFrame) -> set:
    """np = np_seed_keys と同じ identity_group に属する全構造（make_split.py の規則）。"""
    groups = set(st.identity_group[st.inchikey14.isin(np_seed_keys)])
    return set(st.inchikey14[st.identity_group.isin(groups)])


def reproduce_split(train_keys: set, st: pd.DataFrame, npk: set, fold: int) -> dict:
    """train に載る構造を np / hold / train に分ける（make_split.py と同じ）。key -> 区分。"""
    s = st[st.inchikey14.isin(train_keys)]
    return {k: ("np" if k in npk else "hold" if f == fold else "train") for k, f in zip(s.inchikey14, s.fold_identity)}


def training_pool(train_keys: set, st: pd.DataFrame, npk: set, split_keys: set, v3_keys: set, fold: int) -> set:
    """fold の学習グループに使ってよい構造: fold_identity == fold で、np・split・v3 のどれにも入らないもの。"""
    s = st[st.inchikey14.isin(train_keys) & (st.fold_identity == fold)]
    return set(s.inchikey14) - npk - split_keys - v3_keys


def balance(c1: list, c2: list) -> tuple[list, list]:
    """C1:C2 = 1:1 を保つ（少ない方に合わせる。各リストは既にランダム抽出済みで、先頭から切る）。"""
    n = min(len(c1), len(c2))
    return c1[:n], c2[:n]


def main() -> None:
    import pyarrow.parquet as pq
    import build_holdout_v3 as V3
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=125)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--check-only", action="store_true", help="fold 0 の split 再現だけ確かめる")
    args = ap.parse_args()
    train = ROOT / "data/train.parquet"
    meta = pq.read_table(train, columns=["inchikey14", "ingest_lib", "instrument_type", "precursor_mz", "adduct"],
                         read_dictionary=["inchikey14", "ingest_lib", "instrument_type", "adduct"]).to_pandas()
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem
    meta["neutral_mass"] = chem.neutral_mass(meta.precursor_mz.values.astype(np.float64), meta.adduct.astype(str).values)
    st = pd.read_parquet(FOLDS, columns=["inchikey14", "identity_group", "fold_identity"]).drop_duplicates("inchikey14")
    train_keys = set(meta.inchikey14.astype(str).unique())
    seed_np = set(meta.inchikey14[meta.ingest_lib.astype(str) == NP_LIB].astype(str))
    npk = np_keys(seed_np, st)
    split = pd.read_parquet(SPLIT)
    split_hold, split_np = set(split.inchikey14[split.fold == "hold"]), set(split.inchikey14[split.fold == "np"])
    rep = reproduce_split(train_keys, st, npk, 0)
    rep_hold = {k for k, v in rep.items() if v == "hold"}
    rep_np = {k for k, v in rep.items() if v == "np"}
    check = dict(np_seed=len(seed_np), rep_np=len(rep_np), rep_hold=len(rep_hold), split_np=len(split_np), split_hold=len(split_hold),
                 hold_missing_from_rep=len(split_hold - rep_hold), hold_extra_in_rep=len(rep_hold - split_hold),
                 np_missing_from_rep=len(split_np - rep_np), np_extra_in_rep=len(rep_np - split_np))
    print("fold-0 split check:", json.dumps(check))
    if check["hold_missing_from_rep"]:
        raise SystemExit("再現した hold が split.parquet の hold を含まない（必須条件）")
    if args.check_only:
        return
    v3 = set()
    for h in ("v3_c1", "v3_c2"):
        v3 |= set(pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv").inchikey14)
    split_keys = split_hold | split_np
    summary = dict(seed=args.seed, n_per_class=args.n, fold0_check=check, folds={})
    for k in map(int, args.folds.split(",")):
        allowed = training_pool(train_keys, st, npk, split_keys, v3, k)
        c1, c2, n1, n2 = V3.select(meta, allowed, args.n, args.seed + k)
        # select は sorted で返すので、件数を揃えるときは seed で並べ替えてから切る（偏りを避ける）
        rng = np.random.default_rng(args.seed + 100 + k)
        c1 = [c1[i] for i in rng.permutation(len(c1))]; c2 = [c2[i] for i in rng.permutation(len(c2))]
        c1, c2 = balance(c1, c2)
        info = dict(allowed=len(allowed), eligible_c1=n1, eligible_c2=n2, picked=len(c1))
        qrng = np.random.default_rng(args.seed + 200 + k)
        for cls, keys, rule in (("c1", sorted(c1), "enveda_libs"), ("c2", sorted(c2), "all")):
            name = f"rk_f{k}_{cls}"
            info[name] = V3.write_holdout(name, keys, rule, meta, qrng, args.seed + k, train)
        summary["folds"][k] = info
        print(json.dumps({k: {a: b for a, b in info.items() if not a.startswith("rk_")}}), flush=True)
    out = ROOT / "outputs/ranker"
    out.mkdir(parents=True, exist_ok=True)
    (out / "GROUPS.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}/GROUPS.json")


if __name__ == "__main__":
    main()
