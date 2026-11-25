#!/usr/bin/env python3
"""
質量で説明できるか（mass-feasibility）の 26 特徴（Issue #238）。

出典は arXiv:2611.28096 §2.3「Mass-feasibility edges and the role of adduct」と Table 1 の**定義だけ**で、
作者のコードは使っていない（依存は RDKit・numpy のみ）。論文の定数は固定（検証の値に合わせて動かさない）:

  - 断片の提案: 重原子のグラフで「環でない単結合 1 本」または「1 つの環の中の 2 本」を切ってできる
    つながった部分（原子の集合）。最大 64 個（MAX_PROPOSALS）
  - 水素の調整 h ∈ {−2, …, 2}（H_SHIFTS）
  - ピークと提案の辺: 直接のイオン（x_p と d = f + Δ + h·m(H)）か中性損失（m0 − x_p と ℓ = M − f + h·m(H)）の
    どちらかの誤差が 20 ppm 未満（PPM、分母は提案の質量）。経路と h は OR でまとめる
  - ピークの重み w_p = √I_p
  - 13 測定（Table 1）と、その候補群の中の平均からの差で 26 列

論文に書かれていない点（docs/research/mass-feasibility-features.md「自前の決めごと」。値を見る前に固定）:

  - ピークの残し方: 有限・I > 0・m/z ≤ m0 + 0.5 のピークのうち強度の上位 100 本（MAX_PEAKS）
  - 64 個への間引き: 重複を除いた原子集合を（大きさ, 原子の index の並び）で並べ、等間隔に 64 個を取る
    （原子の番号は RDKit の canonical SMILES を読み直した順。同じ構造なら書き方によらず同じ提案になる）
  - 付加イオン: metadata の adduct 文字列から質量を計算する。断片のイオンの担い手は
    正イオンで Na > K > H、負イオンで Cl > 脱プロトン。[M+H]+・[M+Na]+ では論文の式とまったく同じ。
    中性損失は (m0 − Δ_前駆体) − (x_p − Δ_断片) と ℓ を比べる（[M+H]+・[M+Na]+ では論文の m0 − x_p と同じ）。
    2M などの多量体の付加イオンでは中性損失の経路を使わない（直接のイオンだけ）
  - 1 分子に複数のスペクトル: スペクトルごとに 13 測定を出し、使えるスペクトル（付加イオンが読めて
    ピークが 1 本以上）の単純平均を候補の値にする。使えるスペクトルが無い分子は全部 0
  - 候補群 = 呼び出し側が渡した候補の並び（ランカーの群と同じもの。dev では run.py の上位 60）

使い方（関数）:
    cols = feature_names()                          # 26 列の名前
    X = group_features(smiles_list, spectra)         # (候補数, 26) の float64
  spectra は dict(mz=..., it=..., prec=..., adduct=...) の並び（scripts/local_eval/run.py の query_spectra と同じ形）。
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------- 論文の定数（固定）
PPM = 20.0
MAX_PROPOSALS = 64
H_SHIFTS = (-2, -1, 0, 1, 2)
# ---------------------------------------------------------------- 自前の決めごと（論文に無い。固定）
MAX_PEAKS = 100
PREC_MARGIN = 0.5      # m/z ≤ m0 + 0.5 のピークだけ残す

M_H = 1.007825032     # RDKit の GetMostCommonIsotopeMass("H") と同じ値（テストで確かめる）
M_E = 0.000548579909

MEASUREMENTS = (
    "n_matched",            # 1  n_j
    "frac_matched",         # 2  n_j / P
    "log_edges",            # 3  log(1 + Σ_p e_jp)
    "props_per_matched",    # 4  Σ_p e_jp / n_j
    "unambig_frac",         # 5  Σ_p 1[e_jp = 1] / n_j
    "int_cov",              # 6  Σ_p w_p h_jp / W
    "excl_int_cov",         # 7  Σ_p w_p b_jp / W
    "excl_peak_frac",       # 8  Σ_p b_jp / P
    "mean_matched_mz",      # 9  Σ_p x_p h_jp / n_j
    "low_int_cov",          # 10 x/m0 < 1/3
    "mid_int_cov",          # 11 1/3 ≤ x/m0 < 2/3
    "high_int_cov",         # 12 x/m0 ≥ 2/3
    "nearprec_int_cov",     # 13 x/m0 ≥ 0.9
)


def feature_names() -> list[str]:
    return [f"mf_{m}" for m in MEASUREMENTS] + [f"mf_{m}_dmean" for m in MEASUREMENTS]


# ---------------------------------------------------------------- 元素の質量
@lru_cache(maxsize=None)
def _elem_mass(sym: str) -> float:
    from rdkit import Chem
    return float(Chem.GetPeriodicTable().GetMostCommonIsotopeMass(sym))


def formula_mass(f: str) -> float:
    """'CH2O2'、'NH4'、'2H2O' のような式の単一同位体の質量（先頭の数字は個数）。"""
    m = re.fullmatch(r"(\d*)((?:[A-Z][a-z]?\d*)+)", f)
    if not m:
        raise ValueError(f"式が読めない: {f!r}")
    k = int(m.group(1) or 1)
    tot = 0.0
    for sym, n in re.findall(r"([A-Z][a-z]?)(\d*)", m.group(2)):
        tot += _elem_mass(sym) * int(n or 1)
    return k * tot


# ---------------------------------------------------------------- 付加イオン
@lru_cache(maxsize=None)
def parse_adduct(adduct: str):
    """'[M+H]+' → (k, Δ_前駆体, Δ_断片)。前駆体 m/z = k·M + Δ_前駆体、断片のイオン = f + Δ_断片 + h·m(H)。
    1 価だけ扱う。読めなければ None。"""
    m = re.fullmatch(r"\[(\d*)M((?:[+-]\d*[A-Z][A-Za-z0-9]*)*)\](\d*)([+-])", (adduct or "").strip())
    if not m or m.group(3) not in ("", "1"):
        return None
    k = int(m.group(1) or 1)
    charge = 1 if m.group(4) == "+" else -1
    groups = re.findall(r"([+-])(\d*[A-Z][A-Za-z0-9]*)", m.group(2))
    try:
        delta = sum((1 if s == "+" else -1) * formula_mass(g) for s, g in groups) - charge * M_E
    except (ValueError, RuntimeError):   # RDKit は元素でない記号（FA・ACN など）で RuntimeError
        return None
    added = {g for s, g in groups if s == "+"}
    if charge > 0:
        carrier = "Na" if "Na" in added else "K" if "K" in added else "H"
        dfrag = _elem_mass(carrier) - M_E
    else:
        dfrag = (_elem_mass("Cl") + M_E) if "Cl" in added else (M_E - _elem_mass("H"))
    return k, float(delta), float(dfrag)


# ---------------------------------------------------------------- 断片の提案
def _components_after_cut(adj: list[list[int]], cut: set, starts) -> list[frozenset]:
    out = []
    for s in starts:
        seen = {s}
        stack = [s]
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if b not in seen and frozenset((a, b)) not in cut:
                    seen.add(b)
                    stack.append(b)
        out.append(frozenset(seen))
    return out


@lru_cache(maxsize=200_000)
def proposals(smiles: str):
    """候補の SMILES → (M, 断片の中性の質量の配列)。読めなければ None。
    M と断片の質量は原子の単一同位体の質量 + その原子に付く水素で数える（電荷は数えない）。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))   # canonical な原子の順に揃える
    if mol is None:
        return None
    pt = Chem.GetPeriodicTable()
    n = mol.GetNumAtoms()
    am = np.zeros(n)
    for a in mol.GetAtoms():
        iso = a.GetIsotope()
        base = pt.GetMassForIsotope(a.GetSymbol(), iso) if iso else pt.GetMostCommonIsotopeMass(a.GetSymbol())
        am[a.GetIdx()] = base + a.GetTotalNumHs() * M_H
    M = float(am.sum())
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j); adj[j].append(i)
    # つながった部分（塩など複数の部分がある SMILES でも、切った部分の中だけで数える）
    piece = {}
    for s in range(n):
        if s not in piece:
            comp = _components_after_cut(adj, set(), [s])[0]
            for a in comp:
                piece[a] = comp
    masks = set()

    def add(cut, ends):
        for c in _components_after_cut(adj, cut, ends):
            if c != piece[next(iter(c))]:
                masks.add(c)

    for b in mol.GetBonds():
        if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE:
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            add({frozenset((i, j))}, (i, j))
    for ring in mol.GetRingInfo().BondRings():
        bs = [mol.GetBondWithIdx(x) for x in ring]
        for p in range(len(bs)):
            for q in range(p + 1, len(bs)):
                e1 = (bs[p].GetBeginAtomIdx(), bs[p].GetEndAtomIdx())
                e2 = (bs[q].GetBeginAtomIdx(), bs[q].GetEndAtomIdx())
                add({frozenset(e1), frozenset(e2)}, e1 + e2)
    order = sorted(masks, key=lambda c: (len(c), tuple(sorted(c))))
    if len(order) > MAX_PROPOSALS:
        L = len(order)
        order = [order[(i * L) // MAX_PROPOSALS] for i in range(MAX_PROPOSALS)]
    f = np.array([am[list(c)].sum() for c in order], dtype=np.float64)
    return M, f


# ---------------------------------------------------------------- ピーク
def prep_peaks(mz, it, prec: float):
    """残すピーク（自前の決めごと）: 有限・I > 0・m/z ≤ m0 + 0.5 のうち強度の上位 MAX_PEAKS 本。m/z の順で返す。"""
    mz = np.asarray(mz, np.float64).ravel()
    it = np.asarray(it, np.float64).ravel()
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz <= prec + PREC_MARGIN)
    mz, it = mz[ok], it[ok]
    if len(mz) > MAX_PEAKS:
        keep = np.lexsort((mz, -it))[:MAX_PEAKS]   # 強度の降順、同じ強度は m/z の小さい順
        mz, it = mz[keep], it[keep]
    o = np.argsort(mz, kind="stable")
    return mz[o], it[o]


# ---------------------------------------------------------------- 辺の数 e_jp
def edge_counts(props, x: np.ndarray, prec: float, k: int, dprec: float, dfrag: float) -> np.ndarray:
    """props: 候補ごとの (M, f) か None の並び。→ e (候補数, ピーク数) の int64: 候補 j の提案のうち
    ピーク p と辺がある（直接のイオンか中性損失が 20 ppm 未満、h のどれか）ものの数。"""
    nc, P = len(props), len(x)
    e = np.zeros((nc, P), np.int64)
    if P == 0:
        return e
    hs = np.array(H_SHIFTS, np.float64) * M_H
    tol = PPM * 1e-6
    for j, pr in enumerate(props):
        if pr is None or len(pr[1]) == 0:
            continue
        M, f = pr
        d = f[:, None] + dfrag + hs[None, :]                                    # (R, 5)
        ok = d > 0
        hit = (np.abs(x[None, None, :] - d[:, :, None]) < tol * d[:, :, None]) & ok[:, :, None]
        feas = hit.any(axis=1)                                                  # (R, P)
        if k == 1:
            loss_obs = (prec - dprec) - (x - dfrag)                             # [M+H]+ なら m0 − x
            ell = M - f[:, None] + hs[None, :]                                  # (R, 5)
            okl = ell > 0
            hitl = (np.abs(loss_obs[None, None, :] - ell[:, :, None]) < tol * ell[:, :, None]) & okl[:, :, None]
            feas |= hitl.any(axis=1)
        e[j] = feas.sum(axis=0)
    return e


# ---------------------------------------------------------------- 13 測定
def measurements(e: np.ndarray, x: np.ndarray, it: np.ndarray, prec: float) -> np.ndarray:
    """Table 1 の 13 測定。e: (候補数, P)。数の分母は 1、強度の分母は 1e-12 で下を止める。"""
    nc, P = e.shape
    out = np.zeros((nc, len(MEASUREMENTS)), np.float64)
    if P == 0 or nc == 0:
        return out
    h = (e > 0).astype(np.float64)
    w = np.sqrt(np.asarray(it, np.float64))
    W = max(float(w.sum()), 1e-12)
    n = h.sum(axis=1)
    nf = np.maximum(n, 1.0)
    tot_e = e.sum(axis=1).astype(np.float64)
    b = h * (h.sum(axis=0) == 1)[None, :]
    r = x / prec if prec > 0 else np.zeros_like(x)
    out[:, 0] = n
    out[:, 1] = n / max(P, 1)
    out[:, 2] = np.log1p(tot_e)
    out[:, 3] = tot_e / nf
    out[:, 4] = (e == 1).sum(axis=1) / nf
    out[:, 5] = (h * w).sum(axis=1) / W
    out[:, 6] = (b * w).sum(axis=1) / W
    out[:, 7] = b.sum(axis=1) / max(P, 1)
    out[:, 8] = (h * x).sum(axis=1) / nf
    out[:, 9] = (h * w * (r < 1 / 3)).sum(axis=1) / W
    out[:, 10] = (h * w * ((r >= 1 / 3) & (r < 2 / 3))).sum(axis=1) / W
    out[:, 11] = (h * w * (r >= 2 / 3)).sum(axis=1) / W
    out[:, 12] = (h * w * (r >= 0.9)).sum(axis=1) / W
    return out


def spectrum_measurements(props, sp: dict):
    """1 本のスペクトル → (候補数, 13)。使えない（付加イオンが読めない・ピーク 0 本）なら None。"""
    ad = parse_adduct(sp.get("adduct"))
    prec = float(sp.get("prec", float("nan")))
    if ad is None or not math.isfinite(prec) or prec <= 0:
        return None
    x, it = prep_peaks(sp["mz"], sp["it"], prec)
    if len(x) == 0:
        return None
    k, dprec, dfrag = ad
    return measurements(edge_counts(props, x, prec, k, dprec, dfrag), x, it, prec)


def group_features(smiles: list[str], spectra: list[dict], stats: dict | None = None) -> np.ndarray:
    """候補群（1 分子の候補の並び）× その分子のスペクトル → (候補数, 26)。
    前の 13 列 = 使えるスペクトルの 13 測定の単純平均、後ろの 13 列 = それから候補群の平均を引いたもの。"""
    props = [proposals(s) for s in smiles]
    per = [m for m in (spectrum_measurements(props, sp) for sp in spectra) if m is not None]
    if stats is not None:
        stats["n_spectra_used"] = stats.get("n_spectra_used", 0) + len(per)
        stats["n_spectra_skipped"] = stats.get("n_spectra_skipped", 0) + len(spectra) - len(per)
        stats["n_unparsed_smiles"] = stats.get("n_unparsed_smiles", 0) + sum(p is None for p in props)
    nc = len(smiles)
    base = np.mean(per, axis=0) if per else np.zeros((nc, len(MEASUREMENTS)))
    dmean = base - base.mean(axis=0, keepdims=True) if nc else base
    return np.concatenate([base, dmean], axis=1)
