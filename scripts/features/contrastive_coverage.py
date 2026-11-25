"""
候補の中で対比した断片の被覆（Issue #253）。arXiv:2611.28096 v1 §2.3 の式 (6) を、定数は論文のまま使う。

    c_j = Σ_p w_p [u_jp − median_k u_kp]_+ / max(W, 1e−8)
    h_jp = 1[候補 j がピーク p を説明する]、w_p = √I_p、W = Σ_p w_p、u_jp = κ h_jp、
    κ = τc log(1 + exp(1/τc))、τc = 0.1、median は候補群（shortlist）全体の下側の median（偶数なら下側）

h_jp は MetFrag-lite（data/ext/frag/frag_rescore.py、DEFAULT_CFG のまま、学習なし）の断片で作る:
0〜2 本の切断の中性の断片の質量に、ピークの m/z − (電荷の担い手 + dh·H) が ±tol で入れば 1（切断数の重みは使わない）。
ピークは frag_rescore._prep_spectrum と同じ（有限、強度 > 0）。I_p はデータの強度（基準ピーク = 1）。
スペクトルが複数なら、ピークのあるスペクトルごとの c_j の平均（metfrag_lite と同じ）。無ければ 0。
SMILES が読めない候補は h = 0 として median に入る（c = 0）。候補 1 個なら c = 0。

本体（`coverage`・`lower_median`・`zscore`）は numpy だけ。frag_rescore は `explained_matrix` などで遅れて読む。
"""
from __future__ import annotations

import math

import numpy as np

TAU_C = 0.1                                          # 論文 §2.3 の τc
KAPPA = TAU_C * math.log1p(math.exp(1.0 / TAU_C))    # κ = τc log(1 + exp(1/τc)) ≈ 1.00000454
W_FLOOR = 1e-8                                       # 分母の下限 max(W, 1e−8)
SIGMA_FLOOR = 0.02                                   # z(v) の σ の下限（論文の within-shortlist standardisation）
COLUMNS = ("cc", "cc_z")


def lower_median(u: np.ndarray) -> np.ndarray:
    """列ごと（候補の軸 = 0）の下側の median: n 個を並べた (n−1)//2 番目（0 始まり）。"""
    u = np.asarray(u, np.float64)
    n = u.shape[0]
    if n == 0:
        return np.zeros(u.shape[1:], np.float64)
    return np.sort(u, axis=0)[(n - 1) // 2]


def coverage(h: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """式 (6)。h: (候補, ピーク) の 0/1、intensity: (ピーク,) の I_p（≥ 0）。戻り値 (候補,)。"""
    h = np.asarray(h, np.float64)
    if h.ndim != 2:
        raise ValueError("h は (候補, ピーク) の 2 次元")
    I = np.asarray(intensity, np.float64).ravel()
    if h.shape[1] != len(I):
        raise ValueError("h の列数とピーク数が違う")
    if np.any(I < 0) or not np.all(np.isfinite(I)):
        raise ValueError("I_p は有限で ≥ 0")
    n = h.shape[0]
    if n == 0:
        return np.zeros(0)
    w = np.sqrt(I)
    W = float(w.sum())
    u = KAPPA * h
    med = lower_median(u)
    return (np.maximum(u - med[None, :], 0.0) @ w) / max(W, W_FLOOR)


def zscore(v: np.ndarray) -> np.ndarray:
    """候補の中の標準化 (v − 平均) / max(σ, 0.02)、σ は母集団（ddof 0）。候補 1 個は 0。"""
    v = np.asarray(v, np.float64)
    if len(v) == 0:
        return v.copy()
    return (v - v.mean()) / max(float(v.std(ddof=0)), SIGMA_FLOOR)


# ---------------------------------------------------------------- MetFrag-lite の断片から h を作る
def prep_peaks(spectrum: dict, frag, cfg: dict):
    """frag_rescore._prep_spectrum と同じピーク（有限、強度 > 0）。(I_p, q (shift×peak の中性質量), n_shift, n_peak) か None。"""
    mz = np.asarray(spectrum.get("mz", ()), np.float64).ravel()
    it = np.asarray(spectrum.get("it", ()), np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) == 0:
        return None
    H = frag.H
    shifts = np.array([car + dh * H for car in frag._carriers(spectrum.get("adduct"), spectrum.get("mode"))
                       for dh in cfg["h_shifts"]])
    q = (mz[None, :] - shifts[:, None]).ravel()
    return it, q, len(shifts), len(mz)


def explained_row(levels, weights, prep, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """1 候補 × 1 スペクトル: (h_p の 0/1, MetFrag-lite のピークごとの credit)。credit は _score_spectrum と同じ。"""
    _it, q, ns, npk = prep
    h = np.zeros(npk, bool)
    best = np.zeros(npk)
    for arr, wt in zip(levels, weights):
        if len(arr) == 0 or wt <= 0:
            continue
        a = np.searchsorted(arr, q - tol, "left")
        b = np.searchsorted(arr, q + tol, "right")
        hit = (b > a).reshape(ns, npk).any(axis=0)
        h |= hit
        best = np.where(hit & (wt > best), wt, best)
    return h, best


def molecule_features(spectra: list, smiles: list, frag, cache: dict | None = None) -> dict:
    """1 分子の候補群の cc・cc_z と、配線の点検用に作り直した metfrag_lite（score_molecule と同じ値になるはず）。"""
    c = frag.resolve_cfg(None)
    cache = {} if cache is None else cache
    weights = [1.0, float(c["w1"]), float(c["w2"]), float(c["w3"])][: int(c["max_breaks"]) + 1]
    tol = float(c["tol"])
    n = len(smiles)
    levels = [frag._levels_cached(s, c, cache) for s in smiles]
    preps = [p for p in (prep_peaks(s, frag, c) for s in (spectra or [])) if p is not None]
    cc = np.zeros(n)
    mfl = np.array([None if lv is None else 0.0 for lv in levels], dtype=object)
    if preps:
        acc_cc, acc_mf = np.zeros(n), np.zeros(n)
        for prep in preps:
            it = prep[0]
            hm = np.zeros((n, prep[3]), bool)
            iw = it / it.sum()   # MetFrag-lite の線形の強度（intensity='linear'、mz_power 0）
            for j, lv in enumerate(levels):
                if lv is None:
                    continue
                hm[j], best = explained_row(lv, weights, prep, tol)
                acc_mf[j] += float(iw @ best)
            acc_cc += coverage(hm, it)
        cc = acc_cc / len(preps)
        mfl = np.array([None if lv is None else acc_mf[j] / len(preps) for j, lv in enumerate(levels)], dtype=object)
    return dict(cc=cc, cc_z=zscore(cc), metfrag_lite_rebuilt=mfl, n_spectra_with_peaks=len(preps))
