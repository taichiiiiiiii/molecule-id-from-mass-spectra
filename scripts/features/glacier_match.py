"""
GLACIER の予測スペクトルと観測スペクトルの一致（Issue #259）。純粋な関数だけ（入出力は add_glacier_match.py）。

- ピークの対応: 両方を m/z 順にして二本の指で歩く 1 対 1 の対応（|Δm/z| ≤ tol で対、小さい方を進める。pv.entropy_sim と同じ）
- cosine = Σ_対 a·b / (‖a‖‖b‖)。強度は変換しない
- spectral entropy の類似度（Li et al. 2021、重み付き）: 和 1 に正規化 → S < 3 なら I^(0.25 + 0.25 S) で和 1 に戻す →
  対の強度を足した和のスペクトル（和 1 に正規化）のエントロピー S_AB で 1 − (2 S_AB − S_A − S_B) / ln 4
- まとめ: 観測スペクトル s ごとに、s の区分のうち予測がある区分での平均 → 分子のスペクトルで平均と最大。値が 1 本も無ければ NaN

定数（tol）は呼び出し側が MetFrag-lite の設定から渡す。定義は値を見る前に #259 に固定した。
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

LN4 = math.log(4.0)
ENT_W_LIMIT = 3.0   # Li et al. 2021: S < 3 のときだけ重み付け


def prep(mz, it):
    """強度 > 0 のピークだけを m/z 順に（float64）。空なら長さ 0。"""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    assert mz.shape == it.shape and mz.ndim == 1
    keep = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[keep], it[keep]
    o = np.argsort(mz, kind="stable")
    return mz[o], it[o]


@njit(cache=False)
def _match(qmz, cmz, tol):
    """二本の指の 1 対 1 の対応。戻り値は (q の位置, c の位置) の配列 2 本（対の数の長さ）。"""
    n, m = len(qmz), len(cmz)
    qi = np.empty(min(n, m), np.int64)
    ci = np.empty(min(n, m), np.int64)
    i = j = k = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            i += 1
        elif d > tol:
            j += 1
        else:
            qi[k] = i
            ci[k] = j
            k += 1
            i += 1
            j += 1
    return qi[:k], ci[:k]


@njit(cache=False)
def _entropy(p):
    s = 0.0
    for x in p:
        if x > 0:
            s -= x * math.log(x)
    return s


@njit(cache=False)
def _ent_weight(it):
    p = it / it.sum()
    s = _entropy(p)
    if s < ENT_W_LIMIT:
        p = p ** (0.25 + 0.25 * s)
        p = p / p.sum()
    return p


@njit(cache=False)
def pair_sims(qmz, qit, cmz, cit, tol):
    """(cosine, entropy の類似度)。どちらかが空なら (0, 0)。入力は prep 済み（m/z 順・強度 > 0）。"""
    if len(qmz) == 0 or len(cmz) == 0:
        return 0.0, 0.0
    qi, ci = _match(qmz, cmz, tol)
    dot = 0.0
    for k in range(len(qi)):
        dot += qit[qi[k]] * cit[ci[k]]
    cos = dot / (math.sqrt((qit * qit).sum()) * math.sqrt((cit * cit).sum()))
    a = _ent_weight(qit)
    b = _ent_weight(cit)
    sa, sb = _entropy(a), _entropy(b)
    # 和のスペクトル: 対は a + b、対でないピークはそのまま（総和 2）
    mixed = np.concatenate((a, b))
    for k in range(len(qi)):
        mixed[qi[k]] = a[qi[k]] + b[ci[k]]
        mixed[len(a) + ci[k]] = 0.0
    mixed = mixed / mixed.sum()
    ent = 1.0 - (2.0 * _entropy(mixed) - sa - sb) / LN4
    return min(max(cos, 0.0), 1.0), min(max(ent, 0.0), 1.0)


def spectrum_value(q, preds, tol):
    """観測スペクトル q（prep 済みの (mz, it)）と、q の区分の予測（prep 済みの (mz, it) の list。予測の無い区分は
    入れない）→ (cos, ent) の区分平均。予測が 1 つも無ければ None。"""
    if not preds or len(q[0]) == 0:
        return None
    v = np.array([pair_sims(q[0], q[1], p[0], p[1], tol) for p in preds], np.float64)
    return float(v[:, 0].mean()), float(v[:, 1].mean())


def aggregate(values):
    """スペクトルごとの (cos, ent) か None の list → dict(gl_cos_mean, gl_cos_max, gl_ent_mean, gl_ent_max, has_gl)。"""
    v = np.array([x for x in values if x is not None], np.float64).reshape(-1, 2)
    if len(v) == 0:
        nan = float("nan")
        return dict(gl_cos_mean=nan, gl_cos_max=nan, gl_ent_mean=nan, gl_ent_max=nan, has_gl=0)
    return dict(gl_cos_mean=float(v[:, 0].mean()), gl_cos_max=float(v[:, 0].max()),
                gl_ent_mean=float(v[:, 1].mean()), gl_ent_max=float(v[:, 1].max()), has_gl=1)


SIM_COLS = ("gl_cos_mean", "gl_cos_max", "gl_ent_mean", "gl_ent_max")
REL_SRC = ("gl_cos_mean", "gl_ent_mean")   # 式 (8) の相対化を付ける 2 列（#259 で固定）


def relativize_masked(df, group_cols, channels=REL_SRC):
    """has_gl = 1 の行だけを群（分子）にして式 (8) の 5 個を付ける。has_gl = 0 の行は NaN。行の順は変えない。"""
    from relativize import SUFFIXES, relativize
    out = {f"{c}_{s}": np.full(len(df), np.nan) for c in channels for s in SUFFIXES}
    has = df["has_gl"].to_numpy() == 1
    for _, idx in df.groupby(list(group_cols), sort=False).indices.items():
        idx = idx[has[idx]]
        if len(idx) == 0:
            continue
        for c in channels:
            r = relativize(df[c].to_numpy()[idx])
            for s in SUFFIXES:
                out[f"{c}_{s}"][idx] = r[s]
    return df.assign(**out)
