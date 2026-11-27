#!/usr/bin/env python3
"""
群 H3: 断片の構造の統計 pfs_*（MetFrag 2.4.5 の S_Peak の形、学習なし・数え上げ。Issue #296）。
定義は Issue #296 の本文（値を見る前に固定）と docs/research/isomer-evidence-features.md §4.2〜4.3。

  断片   重原子のグラフで 0〜2 本の結合を切った連結成分を原子の集合で列挙（2 本切りは結合 ≤ 100 の分子だけ）。
         名前 f = MolFragmentToSmiles(atomsToUse, canonical, isomericSmiles=False)。切断数は集合を作る最少の数
  照合   |mz − (担い手 + dh·H) − M_f| ≤ 0.005（frag_rescore.DEFAULT_CFG、dh −2〜+3）
  bin    floor(x / 0.01)。問い合わせ = 観測の m/z、学習 = M_f + 担い手 + dh·H（差が最小の組）
  数     train の構造ごとに先頭 16 行。ピークごとに 1 つ（切断の少ない → 名前の文字列順）
  θ      θ_mf = (N + α)/(S_m + α(|F_m| + [f∉F_m]) + β)、θ_m⊥ = β/(S_m + α|F_m| + β)、α = 1e-4、β = 0.0025
  除外   fold k の構造・集合の隠す規則（neighbour_ref と同じ関数）・候補自身（own-out、raw と正準の鍵と別名）

本体は numpy と RDKit。frag_rescore（data/ext/frag、CI に無い）はモジュール `F` として引数で受ける
（使うのは mol_graph・_carriers・_prep_spectrum・DEFAULT_CFG）。値は書き出すだけ（MRR・正解のラベルは作らない）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA = 1e-4             # 出典 1 Table 1 の α（dev で動かさない）
BETA = 0.0025            # 出典 1 Table 1 の β（同上）
BIN_WIDTH = 0.01         # 2·(mzppm + mzabs)、mzppm = 0・mzabs = 0.005
QUERY_CAP = 16           # 構造ごと・分子ごとの先頭の本数（run.query_spectra と同じ）
MAX_BONDS_2 = 100        # frag_rescore.DEFAULT_CFG の max_bonds_2
FID_BITS = 40            # 鍵 = (bin << 40) | 断片の番号
COLUMNS = ("pfs_ll", "pfs_ll_per_peak", "pfs_seen_frac", "pfs_ll_fdmax")


# ---------------------------------------------------------------- 断片の列挙
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _split_masks(n, ea, eb, max_breaks):
    """1〜max_breaks 本の結合を外してグラフが割れたとき、各成分の原子の 0/1 の行と切断数 k を返す。"""
    nb = ea.shape[0]
    n2 = nb * (nb - 1) // 2 if max_breaks >= 2 else 0
    cap = 2 * nb + 3 * n2 + 1
    masks = np.zeros((cap, n), np.uint8)
    ks = np.zeros(cap, np.int8)
    parent = np.empty(n, np.int64)
    removed = np.zeros(nb, np.bool_)
    cnt = 0
    for i in range(nb):
        for j in range(i, nb if max_breaks >= 2 else i + 1):
            removed[i] = True
            removed[j] = True
            k = 1 if j == i else 2
            for a in range(n):
                parent[a] = a
            ncomp = n
            for e in range(nb):
                if removed[e]:
                    continue
                ra = _find(parent, ea[e])
                rb = _find(parent, eb[e])
                if ra != rb:
                    parent[ra] = rb
                    ncomp -= 1
            if ncomp >= 2:
                roots = np.empty(n, np.int64)
                for a in range(n):
                    roots[a] = _find(parent, a)
                for r in range(n):
                    if roots[r] == r:
                        if cnt >= masks.shape[0]:
                            nm = np.zeros((masks.shape[0] * 2, n), np.uint8)
                            nm[:cnt] = masks[:cnt]
                            masks = nm
                            nk = np.zeros(masks.shape[0], np.int8)
                            nk[:cnt] = ks[:cnt]
                            ks = nk
                        for a in range(n):
                            masks[cnt, a] = 1 if roots[a] == r else 0
                        ks[cnt] = k
                        cnt += 1
            removed[i] = False
            removed[j] = False
    return masks[:cnt], ks[:cnt]


try:
    from numba import njit
    _find = njit(cache=True, nogil=True)(_find)
    _split_masks = njit(cache=True, nogil=True)(_split_masks)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


@dataclass
class Frags:
    """1 構造の断片（質量の昇順）。mass: 中性の質量、k: 最少の切断数、name: 名前、rank: 構造の中の名前の文字列順。"""
    mass: np.ndarray
    k: np.ndarray
    name: list
    rank: np.ndarray


def enumerate_fragments(smi, F, max_bonds_2: int = MAX_BONDS_2):
    """SMILES の断片。読めなければ None。原子の質量は F.mol_graph（frag_rescore と同じ値）。"""
    g = F.mol_graph(smi)
    if g is None:
        return None
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    w, ea, eb, npieces = g
    n, nb = len(w), len(ea)
    rows = [np.ones((1, n), np.uint8)]
    ks = [np.zeros(1, np.int8)]
    if npieces > 1:
        for ix in Chem.GetMolFrags(m):
            r = np.zeros((1, n), np.uint8)
            r[0, list(ix)] = 1
            rows.append(r)
            ks.append(np.zeros(1, np.int8))
    if nb > 0:
        mb = 2 if nb <= max_bonds_2 else 1
        sm, sk = _split_masks(n, np.asarray(ea, np.int64), np.asarray(eb, np.int64), mb)
        rows.append(sm)
        ks.append(sk)
    R = np.concatenate(rows)
    K = np.concatenate(ks)
    # 同じ原子の集合は 1 つ（最少の切断数）
    order = np.lexsort((K,) + tuple(R[:, a] for a in range(n - 1, -1, -1)))
    R, K = R[order], K[order]
    first = np.ones(len(R), bool)
    first[1:] = np.any(R[1:] != R[:-1], axis=1)
    R, K = R[first], K[first]
    mass = R.astype(np.float64) @ np.asarray(w, np.float64)
    names = [Chem.MolFragmentToSmiles(m, atomsToUse=[int(a) for a in np.nonzero(r)[0]], canonical=True,
                                      isomericSmiles=False, allHsExplicit=False) for r in R]
    o = np.argsort(mass, kind="stable")
    mass, K = mass[o], K[o]
    names = [names[i] for i in o]
    rank = np.empty(len(names), np.int64)
    rank[np.argsort(np.array(names, dtype=object), kind="stable")] = np.arange(len(names))
    return Frags(mass=mass, k=K, name=names, rank=rank)


# ---------------------------------------------------------------- 照合
def bin_of(x):
    return np.floor(np.asarray(x, np.float64) / BIN_WIDTH).astype(np.int64)


def prep(spec: dict, F):
    """ピーク（frag_rescore._prep_spectrum と同じ選び方）: (mz, √I の重み, shifts)。ピークが無ければ None。"""
    c = F.DEFAULT_CFG
    mz = np.asarray(spec.get("mz", ()), np.float64).ravel()
    it = np.asarray(spec.get("it", ()), np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) == 0:
        return None
    H = F.H
    shifts = np.array([car + dh * H for car in F._carriers(spec.get("adduct"), spec.get("mode")) for dh in c["h_shifts"]])
    return mz, np.sqrt(it), shifts


def expand_matches(mass: np.ndarray, mz: np.ndarray, shifts: np.ndarray, tol: float):
    """全 (ピーク, 断片, shift) の組で |mz − shift − M| ≤ tol のもの。戻り値 (peak, frag, shift, absdiff) の配列。"""
    q = mz[None, :] - shifts[:, None]                 # (ns, npk)
    a = np.searchsorted(mass, q - tol, "left")
    b = np.searchsorted(mass, q + tol, "right")
    ln = (b - a).ravel()
    if ln.sum() == 0:
        e = np.zeros(0, np.int64)
        return e, e, e, np.zeros(0)
    si, pk = np.divmod(np.arange(q.size), q.shape[1])
    rep = np.repeat(np.arange(q.size), ln)
    start = np.repeat(a.ravel(), ln)
    off = np.arange(len(rep)) - np.repeat(np.cumsum(ln) - ln, ln)
    fr = start + off
    peak, shift = pk[rep], si[rep]
    d = np.abs(q.ravel()[rep] - mass[fr])
    keep = d <= tol                                     # searchsorted の端の丸め
    return peak[keep], fr[keep], shift[keep], d[keep]


def count_spectrum(fr: Frags, spec: dict, F):
    """学習の 1 スペクトル: 説明できたピークごとに 1 つの (bin, 断片の番号 i)。切断の少ない → 名前順 → 差の小さい shift。"""
    p = prep(spec, F)
    if p is None or fr is None:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    mz, _w, shifts = p
    peak, fi, si, d = expand_matches(fr.mass, mz, shifts, float(F.DEFAULT_CFG["tol"]))
    if len(peak) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    o = np.lexsort((si, d, fr.rank[fi], fr.k[fi], peak))
    peak, fi, si = peak[o], fi[o], si[o]
    first = np.ones(len(peak), bool)
    first[1:] = peak[1:] != peak[:-1]
    fi, si = fi[first], si[first]
    return bin_of(fr.mass[fi] + shifts[si]), fi


# ---------------------------------------------------------------- 数の表
class CountTable:
    """寄与の記録 (構造, enveda 系の行か, 鍵, 数)。for_set(keep) で使う寄与だけの表を作る。"""

    def __init__(self, struct: np.ndarray, env: np.ndarray, key: np.ndarray, cnt: np.ndarray):
        self.struct = np.asarray(struct, np.int64)
        self.env = np.asarray(env, bool)
        self.key = np.asarray(key, np.int64)
        self.cnt = np.asarray(cnt, np.int64)
        o = np.argsort(self.struct, kind="stable")
        self._by_struct_order = o
        self._s_sorted = self.struct[o]

    @staticmethod
    def from_rows(rows):
        """rows: (構造, env, 鍵) の並び（1 行 = 1 ピーク）。同じ (構造, env, 鍵) を数にまとめる。"""
        if not rows:
            z = np.zeros(0, np.int64)
            return CountTable(z, z.astype(bool), z, z)
        a = np.array(rows, dtype=np.int64)
        u, c = np.unique(a, axis=0, return_counts=True)
        return CountTable(u[:, 0], u[:, 1].astype(bool), u[:, 2], c)

    def records_of(self, structs) -> np.ndarray:
        """構造の番号の集合の記録の位置。"""
        if len(structs) == 0:
            return np.zeros(0, np.int64)
        s = np.asarray(sorted(set(int(x) for x in structs)), np.int64)
        lo = np.searchsorted(self._s_sorted, s, "left")
        hi = np.searchsorted(self._s_sorted, s, "right")
        return np.concatenate([self._by_struct_order[a:b] for a, b in zip(lo, hi)]) if len(s) else np.zeros(0, np.int64)

    def for_set(self, keep: np.ndarray) -> "SetCounts":
        return SetCounts(self, np.asarray(keep, bool))


class SetCounts:
    """1 つの集合で使う数（除外の後）。own(構造) で候補自身の分を引いた値を引ける。"""

    def __init__(self, table: CountTable, keep: np.ndarray):
        self.t, self.keep = table, keep
        k, c = table.key[keep], table.cnt[keep]
        self.ukey, inv = np.unique(k, return_inverse=True)
        self.N = np.bincount(inv, weights=c, minlength=len(self.ukey)).astype(np.int64) if len(k) else np.zeros(0, np.int64)
        b = self.ukey >> FID_BITS
        self.ubin, binv = np.unique(b, return_inverse=True)
        self.S = np.bincount(binv, weights=self.N, minlength=len(self.ubin)).astype(np.int64) if len(b) else np.zeros(0, np.int64)
        self.nF = np.bincount(binv, minlength=len(self.ubin)).astype(np.int64) if len(b) else np.zeros(0, np.int64)

    @staticmethod
    def _look(keys_sorted, vals, q):
        if len(keys_sorted) == 0:
            return np.zeros(len(q), np.int64)
        i = np.clip(np.searchsorted(keys_sorted, q), 0, len(keys_sorted) - 1)
        return np.where(keys_sorted[i] == q, vals[i], 0)

    def own(self, structs):
        """候補自身の構造の、除外の後に残る寄与: (鍵の昇順, 数)。"""
        r = self.t.records_of(structs)
        r = r[self.keep[r]]
        if len(r) == 0:
            z = np.zeros(0, np.int64)
            return z, z
        u, inv = np.unique(self.t.key[r], return_inverse=True)
        return u, np.bincount(inv, weights=self.t.cnt[r]).astype(np.int64)

    def theta_peaks(self, bins: np.ndarray, peak: np.ndarray, keys: np.ndarray, own=None):
        """ピークごとの log θ と「見た」か。bins: (npk,) 問い合わせの bin、peak・keys: 合った (ピーク, 鍵) の組。"""
        npk = len(bins)
        S = self._look(self.ubin, self.S, bins)
        nF = self._look(self.ubin, self.nF, bins)
        Nm = self._look(self.ukey, self.N, keys)
        if own is not None and len(own[0]):
            ok_, oc = own
            on = self._look(ok_, oc, keys)
            Nm = Nm - on
            # bin ごとの自身の分: S から引き、0 になった断片の数を nF から引く
            ob = ok_ >> FID_BITS
            obu, oinv = np.unique(ob, return_inverse=True)
            s_own = np.bincount(oinv, weights=oc).astype(np.int64)
            full = self._look(self.ukey, self.N, ok_)
            z_own = np.bincount(oinv, weights=(full - oc == 0).astype(np.int64)).astype(np.int64)
            S = S - self._look(obu, s_own, bins)
            nF = nF - self._look(obu, z_own, bins)
        if np.any(Nm < 0) or np.any(S < 0) or np.any(nF < 0):
            raise AssertionError("own-out の後に負の数")
        best = np.full(npk, -1, np.int64)                 # −1 = 合う断片なし（⊥）
        if len(peak):
            np.maximum.at(best, peak, Nm)
        matched = best >= 0
        bN = np.maximum(best, 0)
        den = S + ALPHA * nF + BETA
        th = np.where(matched, (bN + ALPHA) / (den + ALPHA * (matched & (bN == 0))), BETA / den)
        return np.log(th), matched & (bN >= 1)


# ---------------------------------------------------------------- 特徴
def candidate_spectrum(fr: Frags | None, fids: np.ndarray | None, p, sc: SetCounts, own, tol: float):
    """1 候補 × 1 スペクトル: (Σ log θ, ピーク数, seen_frac)。fids: 断片ごとの名前の番号（学習に無い名前は −1）。"""
    mz, w, shifts = p
    bins = bin_of(mz)
    if fr is None:
        peak = keys = np.zeros(0, np.int64)
    else:
        peak, fi, _si, _d = expand_matches(fr.mass, mz, shifts, tol)
        f = fids[fi]
        known = f >= 0
        # 学習に無い名前は N = 0（合ったことにはする）: 鍵は存在しない値（−1）にする
        keys = np.where(known, (bins[peak] << FID_BITS) | np.where(known, f, 0), -1)
    lt, seen = sc.theta_peaks(bins, peak, keys, own)
    return float(lt.sum()), len(mz), float((w * seen).sum() / w.sum())


def molecule_features(preps: list, cands: list, sc: SetCounts, owns: list, formulas: list, tol: float):
    """1 分子の候補ごとの 4 列。preps: ピークのある問い合わせスペクトルの prep()、cands: (Frags|None, fids|None)。"""
    n = len(cands)
    out = {c: np.zeros(n) for c in COLUMNS}
    if not preps or n == 0:
        return out
    for i, ((fr, fids), own) in enumerate(zip(cands, owns)):
        r = [candidate_spectrum(fr, fids, p, sc, own, tol) for p in preps]
        out["pfs_ll"][i] = np.mean([a for a, _, _ in r])
        out["pfs_ll_per_peak"][i] = np.mean([a / b for a, b, _ in r])
        out["pfs_seen_frac"][i] = np.mean([s for _, _, s in r])
    out["pfs_ll_fdmax"] = fdmax(out["pfs_ll"], formulas)
    return out


def fdmax(v: np.ndarray, formulas: list) -> np.ndarray:
    """v − 同じ分子式の候補の v の最大（式が None の候補はその候補だけの群）。"""
    v = np.asarray(v, np.float64)
    out = np.zeros(len(v))
    groups: dict = {}
    for i, f in enumerate(formulas):
        groups.setdefault(("__none__", i) if f is None else f, []).append(i)
    for ix in groups.values():
        out[ix] = v[ix] - v[ix].max()
    return out


def formula_of(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    return None if m is None else rdMolDescriptors.CalcMolFormula(m)
