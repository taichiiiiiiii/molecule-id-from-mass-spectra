#!/usr/bin/env python3
"""
自前の LambdaRank（既定のハイパラ）の当てはめと OOF の点（Issue #247、親 #195 B9・#228、playbook §4・§4.2(e)）。

入力は #244 の候補ごとの特徴表（dev: B8 の 20 集合 `b8_f{1..4}_{a..e}_t10`）。このコードは表の**列名と fold 列**だけに依る。

- fold = 表の `query_fold`（B8 の集合の fold 1〜4 = FPNet の分割）。fold k ごとに、学習 = `query_fold != k` かつ型 ≠ (e)、
  予測 = `query_fold == k`（型 (e) を含む全部）。群 = 分子（`set` と `molecule_id` の組）
- ラベル = 候補の SMILES の metric のキー（`mrr25.to_key`）が正解のキーの集合に入れば 1、それ以外 0
- モデル = `lightgbm.train({"objective": "lambdarank", "seed": seed, "verbose": -1}, ..., num_boost_round=100)`。
  決めるのは seed だけ（`num_boost_round=100` は LGBMRanker の既定 `n_estimators` と同じ）。早期停止・重み・検証集合なし
- 特徴 = 表の列 71 から識別の列 8 を引いた 63 列（FEATURES に固定。表の列と完全に一致しなければ止める）

G1（playbook §4）の承認の前は `--check-only` だけを使う（列名・行数・fold・型・ファイルの有無の点検。学習も点も作らず、
正解の SMILES も読まない）。当てはめは `--approval <承認のコメントの URL>` が無いと実行しない。

    python scripts/ranker/lambdarank_oof.py --table outputs/local_eval/features/dev_b8.parquet --check-only
    # 承認の後:
    python scripts/ranker/lambdarank_oof.py --table outputs/local_eval/features/dev_b8.parquet \
        --exp-id e001_lambdarank_base --approval <URL>
    python scripts/ranker/evaluate.py --exp-dir outputs/ranker/lambdarank/e001_lambdarank_base
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
sys.path.insert(0, str(ROOT / "scripts/validate"))

ID_COLS = ("set", "type", "molecule_id", "query_fold", "model_fold", "np_model_fold", "cand_key", "cand_smiles")
_REL = ("dmean", "z", "dmax", "softmax", "sd")
_MF = ("n_matched", "frac_matched", "log_edges", "props_per_matched", "unambig_frac", "int_cov", "excl_int_cov",
       "excl_peak_frac", "mean_matched_mz", "low_int_cov", "mid_int_cov", "high_int_cov", "nearprec_int_cov")
FEATURES = (
    ("fz", "tani_pred", "lib_max_sim", "has_ref", "is_tier", "prior_unverified_pop", "ctrl_rank", "q_rank",
     "metfrag_lite", "has_frag", "np_score", "has_np")
    + tuple(f"mf_{m}" for m in _MF) + tuple(f"mf_{m}_dmean" for m in _MF)
    + tuple(f"{v}_{r}" for v in ("fz", "tani_pred", "lib_max_sim", "metfrag_lite", "np_score") for r in _REL)
)
# 表にあれば止める列（#244「入れない列」と計画: 型・B8 の印・正解のラベル・train 由来の印）
FORBIDDEN = ("used_b8", "msg_overlap", "src", "train_sid", "np_coconut", "p", "y", "is_truth", "label", "in_train")
TYPES = ("a", "b", "c", "d", "e")
TRAIN_TYPES = ("a", "b", "c", "d")   # 型 (e) は評価だけ（#216）
FOLDS = (1, 2, 3, 4)
NUM_BOOST_ROUND = 100
CHECK_READ_COLS = ("set", "type", "molecule_id", "query_fold", "model_fold", "np_model_fold")


class HarnessError(RuntimeError):
    """点検に通らない入力。ここで止める。"""


def features_sha256(features=FEATURES) -> str:
    return hashlib.sha256("\n".join(features).encode()).hexdigest()


def check_columns(columns) -> dict:
    """表の列名を点検する。入れない列がある・期待の列が欠けている・知らない列があるなら止める。"""
    cols = list(columns)
    forbidden = [c for c in cols if c in FORBIDDEN]
    expected = list(ID_COLS) + list(FEATURES)
    missing = [c for c in expected if c not in cols]
    extra = [c for c in cols if c not in expected and c not in FORBIDDEN]
    dup = sorted({c for c in cols if cols.count(c) > 1})
    rep = dict(n_columns=len(cols), n_expected=len(expected), n_features=len(FEATURES), missing=missing, extra=extra,
               forbidden=forbidden, duplicated=dup)
    if forbidden:
        raise HarnessError(f"入れない列が表にある: {forbidden}")
    if missing or extra or dup:
        raise HarnessError(f"列が定義と一致しない: missing={missing} extra={extra} duplicated={dup}")
    return rep


def group_ids(df: pd.DataFrame) -> np.ndarray:
    return (df["set"].astype(str) + "\t" + df["molecule_id"].astype(str)).to_numpy()


def check_rows(df: pd.DataFrame, folds=FOLDS, types=TYPES) -> dict:
    """fold・型・群の点検（ラベルも特徴の値も見ない）。"""
    qf = df["query_fold"].to_numpy()
    fold_values = sorted(int(v) for v in np.unique(qf))
    if fold_values != sorted(folds):
        raise HarnessError(f"query_fold の値が {sorted(folds)} でない: {fold_values}")
    for c in ("model_fold", "np_model_fold"):
        bad = int((df[c].to_numpy() != qf).sum())
        if bad:
            raise HarnessError(f"{c} != query_fold の行が {bad}（§4.2(a)）")
    type_values = sorted(str(v) for v in df["type"].unique())
    if type_values != sorted(types):
        raise HarnessError(f"型の値が {sorted(types)} でない: {type_values}")
    per_set = df.groupby("set", sort=True).agg(n_type=("type", "nunique"), n_fold=("query_fold", "nunique"))
    if (per_set.n_type != 1).any() or (per_set.n_fold != 1).any():
        raise HarnessError("1 つの集合に型か fold が 2 つ以上ある")
    g = group_ids(df)
    per_group_fold = pd.Series(qf).groupby(g).nunique()
    if (per_group_fold != 1).any():
        raise HarnessError("1 つの分子に fold が 2 つ以上ある")
    # 群の行が連続しているか（LightGBM の group は連続した行の数で渡す）
    starts = np.r_[True, g[1:] != g[:-1]]
    if int(starts.sum()) != len(pd.unique(g)):
        raise HarnessError("同じ分子の行が連続していない")
    return dict(n_rows=int(len(df)), n_molecules=int(len(pd.unique(g))), n_sets=int(df["set"].nunique()),
                folds=fold_values, types=type_values,
                molecules_per_type={t: int(df.loc[df["type"] == t, "molecule_id"].groupby(
                    df.loc[df["type"] == t, "set"]).nunique().sum()) for t in type_values},
                molecules_per_fold={int(k): int(len(pd.unique(g[qf == k]))) for k in fold_values})


def fold_masks(df: pd.DataFrame, k: int, train_types=TRAIN_TYPES) -> tuple[np.ndarray, np.ndarray]:
    """fold k の学習の行と予測の行。同じ分子がまたがれば止める。"""
    qf = df["query_fold"].to_numpy()
    pred = qf == k
    train = (qf != k) & df["type"].isin(list(train_types)).to_numpy()
    g = group_ids(df)
    both = set(g[train]) & set(g[pred])
    if both:
        raise HarnessError(f"fold {k}: 学習と予測に同じ分子が {len(both)}")
    return train, pred


def group_sizes(g: np.ndarray) -> np.ndarray:
    """連続した同じ群の行の数（順番どおり）。"""
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return np.diff(np.r_[starts, len(g)])


def make_labels(df: pd.DataFrame, truth_by_group: dict[str, str]) -> np.ndarray:
    """候補の SMILES の metric のキーが正解のキーの集合に入れば 1。"""
    from mrr25 import to_key, truth_keys
    g = group_ids(df)
    keys_by_group: dict[str, set] = {}
    y = np.zeros(len(df), dtype=np.int32)
    for i, (gid, smi) in enumerate(zip(g, df["cand_smiles"].astype(str))):
        if gid not in keys_by_group:
            keys_by_group[gid] = truth_keys(truth_by_group.get(gid))
        if keys_by_group[gid] and to_key(smi) in keys_by_group[gid]:
            y[i] = 1
    return y


def check_truth_disjoint(df: pd.DataFrame, truth_by_group: dict[str, str]) -> dict:
    """fold の間で正解の構造（metric のキー）が重ならないこと。"""
    from mrr25 import truth_keys
    g = group_ids(df)
    fold_of = dict(zip(g, df["query_fold"].to_numpy()))
    by_fold: dict[int, set] = {}
    for gid, f in fold_of.items():
        by_fold.setdefault(int(f), set()).update(truth_keys(truth_by_group.get(gid)))
    overlaps = {f"{a}-{b}": len(by_fold[a] & by_fold[b]) for a in by_fold for b in by_fold if a < b}
    if any(overlaps.values()):
        raise HarnessError(f"fold の間で正解の構造が重なる: {overlaps}")
    return overlaps


def lgb_params(seed: int) -> dict:
    return {"objective": "lambdarank", "seed": int(seed), "verbose": -1}


def fit_predict_oof(df: pd.DataFrame, y: np.ndarray, seed: int = 0, folds=FOLDS, features=FEATURES):
    """leave-one-fold-out の OOF の点。戻り値: (点, {fold: Booster}, fold ごとの情報)。"""
    import lightgbm as lgb
    X = df[list(features)].to_numpy(dtype=np.float64)
    g = group_ids(df)
    oof = np.full(len(df), np.nan)
    models, info = {}, {}
    for k in folds:
        train, pred = fold_masks(df, k)
        ds = lgb.Dataset(X[train], label=y[train], group=group_sizes(g[train]), feature_name=list(features),
                         free_raw_data=True)
        bst = lgb.train(lgb_params(seed), ds, num_boost_round=NUM_BOOST_ROUND)
        oof[pred] = bst.predict(X[pred])
        models[k] = bst
        info[k] = dict(n_train_rows=int(train.sum()), n_train_molecules=int(len(pd.unique(g[train]))),
                       n_pred_rows=int(pred.sum()), n_pred_molecules=int(len(pd.unique(g[pred]))),
                       n_train_pos=int(y[train].sum()))
    if np.isnan(oof).any() or not np.isfinite(oof).all():
        raise HarnessError("OOF の点に NaN/inf がある、または予測されなかった行がある")
    return oof, models, info


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def set_files(eval_root: Path, s: str) -> dict:
    d = eval_root / s
    return dict(truth=d / "truth.csv", control=d / "submission_score+prior+frag.csv")


def check_only(table: Path, eval_root: Path) -> dict:
    """列名・行数・fold・型・正解と対照のファイルの有無。正解の SMILES と特徴の値は読まない。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(table)
    rep = dict(table=str(table), mode="check-only", columns=check_columns(pf.schema_arrow.names))
    schema = pf.schema_arrow
    non_numeric = [c for c in FEATURES if not (pd.api.types.is_numeric_dtype(schema.field(c).type.to_pandas_dtype()))]
    if non_numeric:
        raise HarnessError(f"数値でない特徴の列: {non_numeric}")
    rep["non_numeric_features"] = non_numeric
    df = pf.read(columns=list(CHECK_READ_COLS)).to_pandas()
    rep["rows"] = check_rows(df)
    if rep["rows"]["n_rows"] != pf.metadata.num_rows:
        raise HarnessError("行数が parquet のメタデータと一致しない")
    missing_files, id_mismatch = [], []
    for s, sub in df.groupby("set", sort=True):
        f = set_files(eval_root, s)
        for name, p in f.items():
            if not p.is_file():
                missing_files.append(str(p))
        if f["truth"].is_file():
            ids = set(pd.read_csv(f["truth"], usecols=["molecule_id"])["molecule_id"].astype(str))
            if ids != set(sub["molecule_id"].astype(str)):
                id_mismatch.append(s)
    rep["missing_files"] = missing_files
    rep["truth_id_mismatch_sets"] = id_mismatch
    rep["features_sha256"] = features_sha256()
    rep["note"] = "学習なし・点なし。正解の SMILES と特徴の値は読んでいない（truth.csv は molecule_id 列だけ）"
    if missing_files or id_mismatch:
        raise HarnessError(f"ファイルの欠け {missing_files} / 分子の一覧の不一致 {id_mismatch}")
    return rep


def load_truth(df: pd.DataFrame, eval_root: Path) -> dict[str, str]:
    from mrr25 import read_truth
    out = {}
    for s in sorted(df["set"].unique()):
        for mid, t in read_truth(set_files(eval_root, s)["truth"]).items():
            out[f"{s}\t{mid}"] = t
    return out


def run_fit(table: Path, eval_root: Path, out_dir: Path, seed: int, approval: str) -> dict:
    import lightgbm as lgb
    t0 = time.time()
    if out_dir.exists():
        raise HarnessError(f"出力先が既にある（実験は上書きしない）: {out_dir}")
    rep = check_only(table, eval_root)
    df = pd.read_parquet(table)
    check_columns(df.columns)
    truth = load_truth(df, eval_root)
    overlaps = check_truth_disjoint(df, truth)
    y = make_labels(df, truth)
    oof, models, info = fit_predict_oof(df, y, seed=seed)
    out_dir.mkdir(parents=True)
    out = df[["set", "molecule_id", "type", "query_fold", "cand_key", "cand_smiles"]].copy()
    out.insert(0, "row", np.arange(len(df), dtype=np.int64))
    out["score"] = oof
    out.to_parquet(out_dir / "oof.parquet", index=False)
    for k, bst in models.items():
        bst.save_model(str(out_dir / f"model_fold{k}.txt"))
    meta = dict(issue=247, table=str(table), table_sha256=sha256_file(table), approval=approval, seed=seed,
                params=lgb_params(seed), num_boost_round=NUM_BOOST_ROUND, lightgbm=lgb.__version__,
                features=list(FEATURES), features_sha256=features_sha256(), train_types=list(TRAIN_TYPES),
                folds=list(FOLDS), fold_info={str(k): v for k, v in info.items()}, truth_fold_overlaps=overlaps,
                n_rows=int(len(df)), n_pos=int(y.sum()), check=rep, secs=round(time.time() - t0, 1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--table", default=str(ROOT / "outputs/local_eval/features/dev_b8.parquet"))
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"), help="<set>/truth.csv と対照の CSV の場所")
    ap.add_argument("--check-only", action="store_true", help="列名・行数・fold・型・ファイルの点検だけ（学習・点なし）")
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--out-root", default=str(ROOT / "outputs/ranker/lambdarank"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--approval", default=None, help="G1 承認のコメントの URL（無ければ当てはめない）")
    a = ap.parse_args(argv)
    if a.check_only:
        print(json.dumps(check_only(Path(a.table), Path(a.eval_root)), indent=2, ensure_ascii=False))
        return 0
    if not a.approval:
        raise SystemExit("当てはめには --approval <G1 承認のコメントの URL> が要る（承認の前は --check-only だけ）")
    if not a.exp_id:
        raise SystemExit("--exp-id が要る")
    meta = run_fit(Path(a.table), Path(a.eval_root), Path(a.out_root) / a.exp_id, a.seed, a.approval)
    print(json.dumps({k: meta[k] for k in ("table_sha256", "features_sha256", "n_rows", "n_pos", "fold_info", "secs")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
