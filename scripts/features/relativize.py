"""
候補群の中の相対化（Issue #244）。arXiv:2611.28096 §2.3 式 (8) の 5 個を、点の各チャネル v について作る:

    v − 平均、z(v) = (v − 平均) / max(σ_v, 0.02)、v − 最大、exp(v) / Σ exp(v)、σ_v

定数（σ の下限 0.02）は論文のまま。σ は母集団の標準偏差（ddof 0。論文に書いていないのでこちらの決め、値を見る前に固定）。
softmax は最大を引いて計算する（値は同じ、オーバーフローを避けるだけ）。候補 1 個の群は 0, 0, 0, 1, 0。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA_FLOOR = 0.02
SUFFIXES = ("dmean", "z", "dmax", "softmax", "sd")


def relativize(v) -> dict:
    """1 群の値 v（1 次元）→ {接尾辞: 同じ長さの float64 の配列}。v は有限であること（呼び出し側で欠けを埋める）。"""
    v = np.asarray(v, np.float64)
    assert v.ndim == 1 and len(v) > 0, "空の群"
    assert np.isfinite(v).all(), "有限でない値がある（欠けは呼び出し側で埋める）"
    mu, sd, mx = v.mean(), v.std(), v.max()
    e = np.exp(v - mx)
    return dict(dmean=v - mu, z=(v - mu) / max(sd, SIGMA_FLOOR), dmax=v - mx, softmax=e / e.sum(),
                sd=np.full(len(v), sd))


def add_relative(df: pd.DataFrame, group_cols, channels) -> pd.DataFrame:
    """df に <ch>_<接尾辞> の列を足して返す（行の順は変えない）。群 = group_cols の値の組。"""
    out = {f"{c}_{s}": np.empty(len(df), np.float64) for c in channels for s in SUFFIXES}
    for _, idx in df.groupby(list(group_cols), sort=False).indices.items():
        for c in channels:
            r = relativize(df[c].to_numpy()[idx])
            for s in SUFFIXES:
                out[f"{c}_{s}"][idx] = r[s]
    return df.assign(**out)
