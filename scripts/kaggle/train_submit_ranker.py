#!/usr/bin/env python3
"""
Issue #274: 提出用の LambdaRank を dev の全行で 1 回だけ学習する（時間の測定と配線の確認用。評価には使わない）。

- 表 = exp003 と同じ `outputs/local_eval/features/dev_b8_cc.parquet`（sha256 を EXPECT_TABLE_SHA で固定）
- 特徴 = exp003 と同じ 64 列（FEATURES − prior_unverified_pop ＋ cc, cc_z。sha256 を EXPECT_FEATURES_SHA で固定）
- 学習 = fold 1〜4 の全部、型 a〜d（lambdarank_oof.TRAIN_TYPES、型 e は exp003 と同じく学習に入れない）
- ハイパラ = lambdarank_oof.lgb_params(seed 0)、num_boost_round=100（既定。早期停止・検証集合なし）
- ラベル = lambdarank_oof.make_labels（metric のキー）。dev の行を全部学習に使うので、このモデルを dev で評価しない

    .venv/bin/python scripts/kaggle/train_submit_ranker.py --out-dir outputs/local_eval/ranker/submit_e2e_274
出力: <out-dir>/ranker_submit.txt、features.json、meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/validate"))

import lambdarank_oof as LR  # noqa: E402

EXCLUDE = ("prior_unverified_pop",)
EXTRA = ("cc", "cc_z")
EXPECT_TABLE_SHA = "304f50c3f763f8da360b050f903ea300b9ad3bc6465d9fff45f99d6f501a9628"   # exp003 の table_sha256
EXPECT_FEATURES_SHA = "e0b73e5cd9e46e0b11c89dea8aa4b0a03c4182efabc3c45daa0f61880c9ed303"   # exp003 の features_sha256
SEED = 0


def submit_features() -> tuple:
    feats = LR.resolve_features(EXCLUDE, EXTRA)
    assert len(feats) == 64, len(feats)
    assert LR.features_sha256(feats) == EXPECT_FEATURES_SHA, "特徴の一覧が exp003 と違う"
    return feats


def train_mask(df: pd.DataFrame) -> np.ndarray:
    """fold 1〜4 の全部のうち、型 a〜d の行（型 e は exp003 と同じく学習に入れない）。"""
    return df["query_fold"].isin(list(LR.FOLDS)).to_numpy() & df["type"].isin(list(LR.TRAIN_TYPES)).to_numpy()


def main(argv=None) -> int:
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "outputs/local_eval/features/dev_b8_cc.parquet"))
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"))
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit(f"出力先が既にある（上書きしない）: {out}")
    table = Path(a.table)
    tsha = LR.sha256_file(table)
    assert tsha == EXPECT_TABLE_SHA, f"表の sha256 が exp003 と違う: {tsha}"
    feats = submit_features()
    df = pd.read_parquet(table)
    LR.check_columns(df.columns, EXTRA)
    LR.check_rows(df)
    truth = LR.load_truth(df, Path(a.eval_root))
    LR.check_truth_disjoint(df, truth)
    tr = train_mask(df)
    d = df[tr].reset_index(drop=True)
    y = LR.make_labels(d, truth)
    g = LR.group_ids(d)
    ds = lgb.Dataset(d[list(feats)].to_numpy(np.float64), label=y, group=LR.group_sizes(g), feature_name=list(feats))
    bst = lgb.train(LR.lgb_params(SEED), ds, num_boost_round=LR.NUM_BOOST_ROUND)
    assert tuple(bst.feature_name()) == feats
    out.mkdir(parents=True)
    bst.save_model(str(out / "ranker_submit.txt"))
    (out / "features.json").write_text(json.dumps(list(feats), indent=1))
    meta = dict(issue=274, purpose="時間の測定と配線の確認だけ（dev の全行で学習したので評価には使わない）",
                table=str(table), table_sha256=tsha, features_sha256=LR.features_sha256(feats), n_features=len(feats),
                params=LR.lgb_params(SEED), num_boost_round=LR.NUM_BOOST_ROUND, lightgbm=lgb.__version__,
                train_types=list(LR.TRAIN_TYPES), folds=list(LR.FOLDS), n_train_rows=int(len(d)),
                n_train_molecules=int(len(pd.unique(g))), n_pos=int(y.sum()),
                model_sha256=LR.sha256_file(out / "ranker_submit.txt"),
                features_json_sha256=LR.sha256_file(out / "features.json"), secs=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
