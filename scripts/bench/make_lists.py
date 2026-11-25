#!/usr/bin/env python3
"""
Issue #236（playbook §5 11/26 の前倒し）: 自前の実験の場の dev / holdout の分子の一覧を作り、4 鍵の重なりを点検する。

定義は #236 issuecomment-7150781116（数える前に固定）のとおり。要点:
  - dev = fold k = 1〜4 の hold の構造の和（b8_fold_counts.split_labels と同じ規則。np の群は除く）。1 行 = 1 構造
  - holdout = fold-0 の 4 集合 v3_c1u_t10・v3_c2_t10・enveda_v1u_t10・v3_c2nu_t10（1 行 = 集合 × 分子）。重なりの鍵には
    _t10 の前の 7 集合（v3_c1・v3_c2・enveda_v1・v3_c2n・v3_c1b・v3_c2nb・enveda_v1b）の正解と drop_keys も入れる
  - 4 鍵: 生 InChIKey14・正規化キー（train_structs.key と正解 SMILES の mrr25.to_key）・scorer_key・identity_group。
    どれかの鍵で重なる dev の構造は dev から除き（数を記録）、除いた後に 4 鍵すべてで 0 を assert する
  - used_b8 = 1: B8 の判定に使った 20 集合 b8_f{1..4}_{a..e}_t10 の正解の構造（_t10 で落ちた分子は 0、数を別に記録）

このスクリプトは一覧と重なりの数だけを作る。holdout の recall・MRR は数えない（holdout の正解はキーの集合にするだけ）。

    .venv/bin/python scripts/bench/make_lists.py --out-dir docs/bench
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
SPLIT0 = ROOT / "data/ext/foldsafe/split.parquet"
DEV_FOLDS = (1, 2, 3, 4)
B8_TYPES = ("a", "b", "c", "d", "e")
B8_SETS = tuple(f"b8_f{k}_{t}_t10" for k in DEV_FOLDS for t in B8_TYPES)
HOLDOUT_SETS = ("v3_c1u_t10", "v3_c2_t10", "enveda_v1u_t10", "v3_c2nu_t10")
HOLDOUT_TYPE = {"v3_c1u_t10": "c1", "v3_c2_t10": "c2", "enveda_v1u_t10": "enveda_v1", "v3_c2nu_t10": "c2n"}
# 重なりの鍵に入れる _t10 の前の集合（truth の置き場所）
ORIG = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")          # outputs/holdouts/<h>/truth.csv
A3A = ("v3_c1b", "v3_c2nb", "enveda_v1b")                 # outputs/local_eval/<h>/truth.csv
KEYS4 = ("raw_inchikey14", "canonical_key", "scorer_key", "identity_group")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 純粋な関数（テストあり）

def key_sets(rows: pd.DataFrame) -> dict:
    """列 inchikey14・key・scorer_key・identity_group を持つ表 → 4 鍵それぞれの集合（None・NaN は除く）。"""
    out = {}
    for name, col in zip(KEYS4, ("inchikey14", "key", "scorer_key", "identity_group")):
        out[name] = {str(v) for v in rows[col] if v is not None and not (isinstance(v, float) and np.isnan(v))}
    return out


def overlap4(dev: pd.DataFrame, hold_keys: dict) -> dict:
    """dev の行ごとに 4 鍵の重なりを数える。返り値: 鍵ごとの重なる dev の行数と、どれかの鍵で重なる行の mask。
    正規化キーと scorer_key はどちらもタウトマー正準の InChIKey14 なので、互いの集合とも比べる（別名の取りこぼしを防ぐ）。"""
    canon = hold_keys["canonical_key"] | hold_keys["scorer_key"] | hold_keys["raw_inchikey14"]
    m_raw = dev.inchikey14.astype(str).isin(hold_keys["raw_inchikey14"] | canon).values
    m_key = dev.key.astype(str).isin(canon).values
    m_sc = dev.scorer_key.astype(str).isin(canon).values
    m_grp = dev.identity_group.astype(str).isin(hold_keys["identity_group"]).values
    any_ = m_raw | m_key | m_sc | m_grp
    return dict(counts={"raw_inchikey14": int(m_raw.sum()), "canonical_key": int(m_key.sum()),
                        "scorer_key": int(m_sc.sum()), "identity_group": int(m_grp.sum())}, any=any_)


def holdout_key_rows(truth: pd.DataFrame, drop: set, st: pd.DataFrame, ts_key: dict, to_key) -> pd.DataFrame:
    """holdout の正解（列 inchikey14・smiles）と drop_keys → 4 鍵の行。正解の SMILES の to_key も正規化キーに入れ、
    鍵のどれかが structures.parquet の構造に当たればその構造の scorer_key・identity_group も足す。"""
    raw = set(truth.inchikey14.astype(str)) | set(drop)
    canon = {k for k in (to_key(s) for s in truth.smiles.astype(str)) if k}
    canon |= {ts_key[k] for k in raw if k in ts_key}
    allk = raw | canon
    hit = st[st.inchikey14.astype(str).isin(allk) | st.scorer_key.astype(str).isin(allk)]
    rows = [dict(inchikey14=k, key=None, scorer_key=None, identity_group=None) for k in sorted(raw)]
    rows += [dict(inchikey14=None, key=k, scorer_key=None, identity_group=None) for k in sorted(canon)]
    rows += [dict(inchikey14=None, key=None, scorer_key=str(r.scorer_key), identity_group=str(r.identity_group))
             for r in hit.itertuples()]
    return pd.DataFrame(rows, columns=["inchikey14", "key", "scorer_key", "identity_group"])


def mark_b8(dev: pd.DataFrame, b8: pd.DataFrame) -> pd.DataFrame:
    """b8 = 列 set・type・fold・molecule_id・inchikey14（_t10 の後の正解）。dev に used_b8・b8_set・b8_type・b8_molecule_id を付ける。
    1 構造は高々 1 集合、正解は dev にあり、fold が一致することを assert する。"""
    assert b8.inchikey14.is_unique, "B8 の正解の構造が 2 つの集合にある"
    # (d) 型の truth.csv の inchikey14 は群の scorer_key（別名の構造では生の鍵と違う）。生の鍵で当たらなければ scorer_key で、
    # ちょうど 1 行に当たることを要する
    raw = set(dev.inchikey14)
    sc_rows = dev.groupby("scorer_key").inchikey14.apply(list).to_dict()
    b8 = b8.copy()
    b8["truth_key"] = b8.inchikey14
    mapped = []
    for k in b8.inchikey14:
        if k in raw:
            mapped.append(k)
            continue
        rows = sc_rows.get(k, [])
        assert len(rows) == 1, f"B8 の正解 {k} が dev の生の鍵にも 1 つの scorer_key にも当たらない（{len(rows)} 行）"
        mapped.append(rows[0])
    b8["inchikey14"] = mapped
    assert b8.inchikey14.is_unique, "B8 の 2 つの正解が dev の同じ構造に当たる"
    d = dev.merge(b8.rename(columns={"set": "b8_set", "type": "b8_type", "molecule_id": "b8_molecule_id", "fold": "b8_fold",
                                     "truth_key": "b8_truth_key"}),
                  on="inchikey14", how="left")
    assert len(d) == len(dev)
    used = d.b8_set.notna().values
    assert (d.fold[used].astype(int).values == d.b8_fold[used].astype(int).values).all(), "B8 の集合の fold と dev の fold が違う"
    d["used_b8"] = used.astype(int)
    assert int(used.sum()) == len(b8)
    return d.drop(columns=["b8_fold"])


# ---------------------------------------------------------------- 実データ

def load_dev(st: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "scripts/research"))
    import b8_fold_counts as B8
    parts = []
    for k in DEV_FOLDS:
        lab = B8.split_labels(ts, st, k)
        h = lab[lab.fold == "hold"][["inchikey14", "key", "sid"]].copy()
        h["fold"] = k
        parts.append(h)
    dev = pd.concat(parts, ignore_index=True)
    assert dev.inchikey14.is_unique, "1 つの構造が 2 つの fold の hold にある"
    s = st[["inchikey14", "scorer_key", "identity_group", "fold_identity"]].drop_duplicates("inchikey14")
    dev = dev.merge(s, on="inchikey14", how="left")
    assert (dev.fold_identity.astype(int) == dev.fold).all()
    g = dev.groupby("identity_group").fold.nunique()
    assert (g == 1).all(), "identity_group が 2 つの fold にまたがる"
    dev["identity_group"] = dev.identity_group.astype(str)
    return dev.drop(columns=["fold_identity"]).sort_values(["fold", "inchikey14"]).reset_index(drop=True)


def load_b8() -> pd.DataFrame:
    rows = []
    for h in B8_SETS:
        tr = pd.read_csv(ROOT / f"outputs/local_eval/{h}/truth.csv")
        k, t = int(h[4]), h[6]
        rows.append(pd.DataFrame(dict(set=h, type=t, fold=k, molecule_id=tr.molecule_id.astype(str),
                                      inchikey14=tr.inchikey14.astype(str))))
    return pd.concat(rows, ignore_index=True)


def b8_t10_dropped() -> dict:
    """_t10 で落ちた分子（集合には引かれたが判定に入っていない）の構造。"""
    out = {}
    for h in B8_SETS:
        base = h[:-4]
        full = pd.read_csv(ROOT / f"data/holdouts/{base}/sample_submission.csv").molecule_id.astype(str)
        t10 = pd.read_csv(ROOT / f"outputs/local_eval/{h}/truth.csv").molecule_id.astype(str)
        out[h] = sorted(set(full) - set(t10))
    return out


def holdout_truth(h: str) -> pd.DataFrame:
    p = ROOT / (f"outputs/holdouts/{h}/truth.csv" if h in ORIG else f"outputs/local_eval/{h}/truth.csv")
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "docs/bench"))
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    import mrr25
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    st = pd.read_parquet(FOLDS, columns=["inchikey14", "scorer_key", "identity_group", "fold_identity"])
    ts = pd.read_parquet(TSTRUCT, columns=["inchikey14", "key", "libmask", "sid"])
    ts_key = dict(zip(ts.inchikey14, ts.key))
    dev = load_dev(st, ts)
    n_dev_raw = len(dev)

    # holdout の一覧（集合 × 分子）と重なりの鍵
    hold_rows, key_frames = [], []
    for h in HOLDOUT_SETS:
        tr = holdout_truth(h)
        hold_rows.append(pd.DataFrame(dict(set=h, type=HOLDOUT_TYPE[h], molecule_id=tr.molecule_id.astype(str),
                                           inchikey14=tr.inchikey14.astype(str))))
    for h in HOLDOUT_SETS + ORIG + A3A:
        tr = holdout_truth(h)
        dp = ROOT / f"data/holdouts/{h}/drop_keys.txt"
        drop = set(dp.read_text().split()) if dp.is_file() else set()
        key_frames.append(holdout_key_rows(tr, drop, st, ts_key, mrr25.to_key))
    hold = pd.concat(hold_rows, ignore_index=True)
    hk = key_sets(pd.concat(key_frames, ignore_index=True))
    hold_struct = st[st.inchikey14.isin(set(hold.inchikey14))].drop_duplicates("inchikey14")
    assert set(hold_struct.fold_identity.astype(int)) == {0}, "holdout の正解に fold 0 以外の構造がある"

    ov_before = overlap4(dev, hk)
    dev_excluded = dev[ov_before["any"]].inchikey14.tolist()
    dev = dev[~ov_before["any"]].reset_index(drop=True)
    ov_after = overlap4(dev, hk)
    assert all(v == 0 for v in ov_after["counts"].values()), ov_after["counts"]
    # fold-0 の hold（split.parquet）とも重ならない（独立の確かめ）
    sp = pd.read_parquet(SPLIT0)
    sp_keys = set(sp.inchikey14.astype(str)) | set(sp.key.astype(str))
    assert not (set(dev.inchikey14) & sp_keys) and not (set(dev.key.astype(str)) & sp_keys), "dev に fold-0 の構造がある"

    b8 = load_b8()
    dev = mark_b8(dev, b8)
    dropped = b8_t10_dropped()
    n_t10_dropped = sum(len(v) for v in dropped.values())

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["inchikey14", "key", "scorer_key", "identity_group", "fold", "sid", "used_b8", "b8_set", "b8_type", "b8_molecule_id", "b8_truth_key"]
    dev[cols].to_csv(out / "dev_list.csv.gz", index=False, compression=dict(method="gzip", mtime=0))
    hold.to_csv(out / "holdout_list.csv", index=False)
    by_type = dev[dev.used_b8 == 1].groupby("b8_type").size().to_dict()
    summ = dict(
        issue=236, definitions="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/236#issuecomment-7150781116",
        dev=dict(n_before_overlap_exclusion=n_dev_raw, n_excluded_overlap=len(dev_excluded), excluded=dev_excluded,
                 n=len(dev), by_fold={int(k): int(v) for k, v in dev.groupby("fold").size().items()},
                 n_groups=int(dev.identity_group.nunique()), used_b8=int(dev.used_b8.sum()),
                 used_b8_by_type={k: int(v) for k, v in by_type.items()},
                 used_b8_by_set={k: int(v) for k, v in dev[dev.used_b8 == 1].groupby("b8_set").size().items()},
                 no_type=int((dev.used_b8 == 0).sum()), b8_t10_dropped_molecules=n_t10_dropped),
        holdout=dict(n_rows=len(hold), n_structures=int(hold.inchikey14.nunique()),
                     by_set={k: int(v) for k, v in hold.groupby("set").size().items()},
                     key_set_sizes={k: len(v) for k, v in hk.items()},
                     key_sources=list(HOLDOUT_SETS + ORIG + A3A)),
        overlap_dev_vs_holdout=dict(before_exclusion=ov_before["counts"], after_exclusion=ov_after["counts"]),
        fold0_split_overlap=0,
        inputs={p.name: sha256(p) for p in (FOLDS, TSTRUCT, SPLIT0)},
        outputs={"dev_list.csv.gz": sha256(out / "dev_list.csv.gz"), "holdout_list.csv": sha256(out / "holdout_list.csv")},
        secs=round(time.time() - t0, 1))
    assert summ["dev"]["used_b8"] + summ["dev"]["no_type"] == summ["dev"]["n"]
    (out / "lists_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summ.items() if k != "inputs"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
