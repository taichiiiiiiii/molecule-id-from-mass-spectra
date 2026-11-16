#!/usr/bin/env python3
"""#120: 断片の説明で異性体を並べ替える学習器（fragnn）の速度測定。学習はしない。v3 のスペクトル・点は読まない。

事前登録: Issue #120（入力・モデル・止め方・採否・外挿の式は Issue のとおり。ここでは変えない）。
段（それぞれ別のプロセスで動かし、`/usr/bin/time -l` で peak RSS を測る）:

  count   : #109 と同じ DROP・内部検証で学習の正解構造と ±10 ppm 窓の負例を作り、和集合の構造数を数える。
            負例は 2 通り: design = train 由来の構造（train_structs のキー）だけ（#120 レビュー 1 回目 major の直し、
            train とプールの非対称の近道を断つ）、allpool = #109 と同じ使えるプール全体（参考、保守的な判定に使う）。
            seed 0 で正解構造から 1,000 構造の切片を選び、切片の窓の負例と一緒に書く（slice.npz）
  frag    : 切片の 1,000 構造（a）と、その窓の負例すべて（b）の断片を、既存の frag_rescore.fragment_levels で
            列挙する（コードは変えない）。秒/構造とバイト/構造（frag_a.npz / frag_b.npz）
  spectra : train.parquet を row group ごとに 1 回読み、切片のスペクトルの強度の上位 32 本を取る（spectra.npz）
  step    : 切片の実際のスペクトル・実際の窓の負例（design と allpool の 2 通り）で、batch 128 × 候補 128 の
            1 ステップ（CPU の特徴作り＋MPS の順・逆伝播。組の数は #109 の padded_layout と同じく PAD_ROWS の倍数に
            詰める）を空回し 5 回のあと 50 回測る。順伝播だけの秒/バッチも測る
  report  : 外挿の式（#120）で合計の時間を出し、64 時間で判定する（bench_report.json）

純粋な関数（CI でテストする）は torch・rdkit を import しない。numba の関数は cache=False（キャッシュを書かない）。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ISSUE = 120
SEED_SLICE = 0
N_SLICE = 1_000
TOP_PEAKS = 32
TOL = 0.005                      # frag_rescore.DEFAULT_CFG の tol
H_SHIFTS = tuple(range(-2, 4))   # frag_rescore.DEFAULT_CFG の h_shifts（-2..+3）
LEVEL_W = (1.0, 1.0, 0.6)        # frag_rescore の weights = [1.0, w1, w2]（k = 0/1/2）
N_LEVELS = 3
F_PEAK = 17
F_PAIR = 1
F_SPEC = 3
K_CAND, BATCH = 128, 128
SEED = 42
LR, WD, WARMUP, CLIP = 1e-3, 1e-5, 2_000, 5.0
EVAL_EVERY = 2_000
STEPS_JUDGE, STEPS_MIN = 28_000, 12_000    # #109 段 b の実績 / 止め方の最短（参考）
N_VAL_SPECTRA = 39_437                     # #109 の内部検証のスペクトル数
LIMIT_HOURS = 64.0
EXPECTED_TRAIN_STRUCTS = 214_772           # #109 train_report.json（n_train_structures）
EXPECTED_VAL_STRUCTS = 4_383
EXPECTED_POOL_ALLOWED = 655_670
EXPECTED_DROP = 55_369
EXPECTED_DROP_KEY = 55_038
EXPECTED_TRAIN_SPECTRA = 1_942_968
PAD_ROWS = 2_048                           # #109 と同じ実装の定数（MPS の形の数を抑える。モデルの定義ではない）
HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
PPM = 10.0
VAL_FRAC, VAL_SEED = 0.02, 0
H = 1.00782503207
E = 0.00054857990
PROTON = H - E

# 特徴の並び（F_PEAK = 17）
#  0 強度の割合（線形） 1 強度の割合（平方根） 2 m/z / prec 3 前駆イオンの印
#  4..7 最も安い説明の段 one-hot（なし・k0・k1・k2） 8 その説明の |Δm| / tol
#  9..14 その説明の水素の移動 one-hot（-2..+3） 15 担い手の番号（0 主 / 1 付加） 16 log1p(許容内の一致の数)


# ---------------------------------------------------------------- 純粋な関数（CI）

def drop_key_space(drop_raw: set, raw_to_key: dict, holdout_keys: dict, split_keys) -> set:
    """#109 align_train.drop_key_space と同じ定義（PR #110 未マージのため写し）。"""
    out = set(map(str, split_keys))
    for ks in holdout_keys.values():
        out |= set(map(str, ks))
    out |= {raw_to_key[k] for k in drop_raw if k in raw_to_key}
    return out


def negative_csr(pos_rows: np.ndarray, mass: np.ndarray, allowed: np.ndarray, ppm: float = PPM):
    """#109 align_train.negative_csr と同じ定義（写し）。各正解の pool 行について ±ppm 窓の allowed の行（自分を除く）。"""
    pos_rows = np.asarray(pos_rows, np.int64)
    tgt = mass[pos_rows]
    tol = tgt * ppm * 1e-6
    a = np.searchsorted(mass, tgt - tol, "left")
    b = np.searchsorted(mass, tgt + tol, "right")
    lens = b - a
    starts = np.repeat(a - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    flat = np.arange(lens.sum(), dtype=np.int64) + starts
    owner = np.repeat(np.arange(len(pos_rows)), lens)
    keep = allowed[flat] & (flat != pos_rows[owner])
    flat, owner = flat[keep], owner[keep]
    offsets = np.zeros(len(pos_rows) + 1, np.int64)
    np.add.at(offsets, owner + 1, 1)
    return np.cumsum(offsets), flat


def split_val_keys(keys, frac: float = VAL_FRAC, seed: int = VAL_SEED) -> set:
    """#109 align_train.split_val_keys と同じ定義（写し）。"""
    ks = sorted(set(keys))
    n = int(round(frac * len(ks)))
    perm = np.random.default_rng(seed).permutation(len(ks))
    return {ks[i] for i in perm[:n]}


def sample_candidates(rng, pos, csr_id, offsets, flat, k: int = K_CAND):
    """#109 align_train.sample_candidates と同じ定義（写し）。候補 0 = 正解、負例は窓から重複なしに k−1 個。"""
    b = len(pos)
    cand = np.zeros((b, k), np.int64)
    mask = np.zeros((b, k), bool)
    cand[:, 0] = pos
    mask[:, 0] = True
    for i in range(b):
        lo, hi = offsets[csr_id[i]], offsets[csr_id[i] + 1]
        n = hi - lo
        if n == 0:
            continue
        m = min(n, k - 1)
        sel = flat[lo + rng.choice(n, size=m, replace=False)] if n > m else flat[lo:hi][rng.permutation(n)]
        cand[i, 1:1 + m] = sel
        mask[i, 1:1 + m] = True
    return cand, mask


def padded_pairs(n: int, pad_to: int = PAD_ROWS) -> int:
    """組の数 n を pad_to の倍数（最低 pad_to）に切り上げる（#109 padded_layout と同じ考え方）。"""
    return max(pad_to, -(-n // pad_to) * pad_to)


def union_count(pos_rows, flat) -> int:
    """正解の pool 行と負例の pool 行の和集合（重複を除いた数）。"""
    return int(len(np.union1d(np.unique(np.asarray(pos_rows, np.int64)), np.unique(np.asarray(flat, np.int64)))))


def select_slice(keys, n: int = N_SLICE, seed: int = SEED_SLICE) -> list:
    """正準キーの集合から seed で n 個（sorted → default_rng(seed).choice、置き換えなし）。"""
    ks = sorted(set(map(str, keys)))
    idx = np.random.default_rng(seed).choice(len(ks), size=min(n, len(ks)), replace=False)
    return [ks[i] for i in sorted(idx)]


def carriers(adduct, mode) -> tuple:
    """frag_rescore._carriers と同じ値（CI で frag_rescore を読めないので写し。ローカルのテストで一致を確かめる）。"""
    na, k, cl, nh4, formate = 22.9897692809, 38.96370668, 34.96885268, 14.0030740048 + 4 * H, 12.0 + 2 * H + 2 * 15.9949146196
    table = {
        '[M+H]+': (PROTON,), '[M-H2O+H]+': (PROTON,), '[M-2H2O+H]+': (PROTON,),
        '[M+NH4]+': (PROTON, nh4 - E), '[M+Na]+': (PROTON, na - E), '[M+K]+': (PROTON, k - E),
        '[M-H]-': (-PROTON,), '[M-H2O-H]-': (-PROTON,), '[M+CH2O2-H]-': (-PROTON, formate - PROTON),
        '[M+Cl]-': (-PROTON, cl + E),
    }
    c = table.get(adduct)
    if c is not None:
        return c
    if isinstance(adduct, str) and adduct.endswith('-'):
        return (-PROTON,)
    if isinstance(adduct, str) and adduct.endswith('+'):
        return (PROTON,)
    return (PROTON,) if (mode or 1) > 0 else (-PROTON,)


def shift_table(adduct, mode):
    """(shift の値, 担い手の番号, dh) の配列。shift = carrier + dh·H。"""
    vals, car, dh = [], [], []
    for ci, c in enumerate(carriers(adduct, mode)):
        for d in H_SHIFTS:
            vals.append(c + d * H); car.append(min(ci, 1)); dh.append(d)
    return np.array(vals, np.float64), np.array(car, np.int8), np.array(dh, np.int8)


def top_peaks(mz, it, k: int = TOP_PEAKS):
    """強度の上位 k 本（同点は m/z の昇順）。有限で強度 > 0 のピークだけ。返り値は m/z の昇順。"""
    mz = np.asarray(mz, np.float64).ravel(); it = np.asarray(it, np.float64).ravel()
    n = min(len(mz), len(it)); mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) > k:
        order = np.lexsort((mz, -it))[:k]
        mz, it = mz[order], it[order]
    o = np.argsort(mz, kind="stable")
    return mz[o], it[o]


def levels_to_csr(levels_list):
    """構造ごとの [k0, k1, k2] の質量の列 → (masses float64, off int64 (n, 4))。None（読めない）は空。"""
    n = len(levels_list)
    off = np.zeros((n, N_LEVELS + 1), np.int64)
    chunks, pos = [], 0
    for j, lv in enumerate(levels_list):
        off[j, 0] = pos
        for k in range(N_LEVELS):
            arr = np.asarray(lv[k], np.float64) if (lv is not None and k < len(lv)) else np.zeros(0)
            chunks.append(arr); pos += len(arr); off[j, k + 1] = pos
    masses = np.concatenate(chunks) if chunks else np.zeros(0)
    return masses, off


def _pair_features_py(masses, off, j, mz, it, npk, prec, sv, sc, sd, ns, tol, out, w):
    """1 組（候補 j × 1 スペクトル）の特徴。out は (P, F_PEAK)、返り値は frag の点（説明された強度の割合）。"""
    tot = 0.0
    tots = 0.0
    for p in range(npk):
        tot += it[p]
        tots += math.sqrt(it[p])
    score = 0.0
    for p in range(npk):
        out[p, 0] = it[p] / tot
        out[p, 1] = math.sqrt(it[p]) / tots
        out[p, 2] = min(mz[p] / prec, 1.5) if prec > 0 else 0.0
        out[p, 3] = 1.0 if abs(mz[p] - prec) <= tol else 0.0
        best_k = -1
        best_d = 1e9
        best_s = 0
        nmatch = 0
        for k in range(3):
            lo, hi = off[j, k], off[j, k + 1]
            if hi <= lo:
                continue
            arr = masses[lo:hi]
            for s in range(ns):
                q = mz[p] - sv[s]
                a = np.searchsorted(arr, q - tol)
                b = np.searchsorted(arr, q + tol, side="right")
                if b > a:
                    nmatch += b - a
                    if best_k == -1 or k == best_k:
                        for t in range(a, b):
                            d = abs(arr[t] - q)
                            if best_k == -1 or d < best_d:
                                best_k, best_d, best_s = k, d, s
        if best_k == -1:
            out[p, 4] = 1.0
        else:
            out[p, 5 + best_k] = 1.0
            out[p, 8] = best_d / tol
            out[p, 9 + (sd[best_s] - H_SHIFTS[0])] = 1.0
            out[p, 15] = float(sc[best_s])
            score += out[p, 0] * w[best_k]
        out[p, 16] = math.log1p(nmatch)
    return score


def _batch_features_py(masses, off, cand_struct, spec_of_pair, pk_mz, pk_it, pk_n, prec, sh_v, sh_c, sh_d, sh_n, tol, w):
    n = len(cand_struct)
    P = pk_mz.shape[1]
    feats = np.zeros((n, P, F_PEAK), np.float32)
    fscore = np.zeros(n, np.float32)
    tmp = np.zeros((P, F_PEAK), np.float64)
    for i in range(n):
        s = spec_of_pair[i]
        tmp[:, :] = 0.0
        fscore[i] = _pair_features_py(masses, off, cand_struct[i], pk_mz[s], pk_it[s], pk_n[s], prec[s],
                                      sh_v[s], sh_c[s], sh_d[s], sh_n[s], tol, tmp, w)
        feats[i] = tmp
    return feats, fscore


try:  # numba は requirements-dev.txt にある。cache=False（キャッシュを書かない）
    from numba import njit, prange

    _pair_features = njit(cache=False, nogil=True)(_pair_features_py)

    @njit(cache=False, parallel=True)
    def _batch_features_nb(masses, off, cand_struct, spec_of_pair, pk_mz, pk_it, pk_n, prec, sh_v, sh_c, sh_d, sh_n,
                           tol, w):
        n = len(cand_struct)
        P = pk_mz.shape[1]
        feats = np.zeros((n, P, F_PEAK), np.float32)
        fscore = np.zeros(n, np.float32)
        for i in prange(n):
            s = spec_of_pair[i]
            tmp = np.zeros((P, F_PEAK), np.float64)
            fscore[i] = _pair_features(masses, off, cand_struct[i], pk_mz[s], pk_it[s], pk_n[s], prec[s],
                                       sh_v[s], sh_c[s], sh_d[s], sh_n[s], tol, tmp, w)
            for p in range(P):
                for f in range(F_PEAK):
                    feats[i, p, f] = tmp[p, f]
        return feats, fscore
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def batch_features(masses, off, cand_struct, spec_of_pair, spec, tol: float = TOL, use_numba: bool = True):
    """組（候補の構造の番号, スペクトルの番号）ごとの特徴 (n, P, 17) と frag の点 (n,)。spec は pack_spectra の dict。"""
    args = (np.asarray(masses, np.float64), np.asarray(off, np.int64), np.asarray(cand_struct, np.int64),
            np.asarray(spec_of_pair, np.int64), spec["mz"], spec["it"], spec["n"], spec["prec"],
            spec["sh_v"], spec["sh_c"], spec["sh_d"], spec["sh_n"], float(tol), np.asarray(LEVEL_W, np.float64))
    if use_numba and HAVE_NUMBA:
        return _batch_features_nb(*args)
    return _batch_features_py(*args)


def pack_spectra(rows: list) -> dict:
    """rows: dict(mz, it, prec, adduct, mode, ce) の並び → 固定長の配列（上位 32 本、shift は最大 12）。"""
    n = len(rows)
    out = dict(mz=np.zeros((n, TOP_PEAKS)), it=np.zeros((n, TOP_PEAKS)), n=np.zeros(n, np.int64), prec=np.zeros(n),
               sh_v=np.zeros((n, 12)), sh_c=np.zeros((n, 12), np.int8), sh_d=np.zeros((n, 12), np.int8),
               sh_n=np.zeros(n, np.int64), spec_feat=np.zeros((n, F_SPEC), np.float32))
    for i, r in enumerate(rows):
        mz, it = top_peaks(r["mz"], r["it"])
        k = len(mz)
        out["mz"][i, :k], out["it"][i, :k], out["n"][i], out["prec"][i] = mz, it, k, float(r["prec"])
        v, c, d = shift_table(r.get("adduct"), r.get("mode"))
        m = len(v)
        out["sh_v"][i, :m], out["sh_c"][i, :m], out["sh_d"][i, :m], out["sh_n"][i] = v, c, d, m
        out["spec_feat"][i] = (1.0 if (r.get("mode") or 1) > 0 else -1.0, math.log1p(k) / math.log(33),
                               float(r.get("ce") or 0.0) / 100.0)
    return out


def extrapolate(n_union: int, sec_per_struct: float, sec_spectra: float, sec_per_step: float, sec_fwd_batch: float,
                steps: int = STEPS_JUDGE, n_val: int = N_VAL_SPECTRA, eval_every: int = EVAL_EVERY,
                limit_h: float = LIMIT_HOURS, sec_pack_per_spectrum: float = 0.0,
                n_spectra: int = EXPECTED_TRAIN_SPECTRA + N_VAL_SPECTRA) -> dict:
    """#120 の外挿の式。前計算 = N_union × 秒/構造、読み込み = 1 回の全体の読み込み ＋ 1 本の詰めの秒 × 全スペクトル数、
    学習 = steps × 秒/ステップ、評価 = (steps / eval_every) × ceil(n_val / BATCH) × 順伝播の秒/バッチ。
    合計 ≤ limit_h → 本番へ。"""
    pre = n_union * sec_per_struct
    spec = sec_spectra + sec_pack_per_spectrum * n_spectra
    tr = steps * sec_per_step
    ev = (steps // eval_every) * math.ceil(n_val / BATCH) * sec_fwd_batch
    tot = pre + spec + tr + ev
    return dict(steps=steps, n_union=n_union, precompute_h=pre / 3600, spectra_h=spec / 3600, train_h=tr / 3600,
                eval_h=ev / 3600, total_h=tot / 3600, limit_h=limit_h,
                verdict="本番へ" if tot / 3600 <= limit_h else "減らす案へ")


# ---------------------------------------------------------------- モデル（torch はここで初めて読む）

def build_model():
    import torch
    from torch import nn

    class FragNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(F_PEAK, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(256 + F_PAIR + F_SPEC, 256), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, feats, valid, pair, spec):
            """feats (N, P, 17)、valid (N, P) bool、pair (N, 1)、spec (N, 3) → (N,) の logit。"""
            h = self.phi(feats)
            w = feats[..., 0:1] * valid[..., None]
            s = (h * w).sum(1)
            m = torch.where(valid[..., None], h, torch.zeros_like(h)).amax(1)
            return self.rho(torch.cat([s, m, pair, spec], -1)).squeeze(-1)

    return FragNN()


# ---------------------------------------------------------------- ローカルの段（CI では動かさない）

def _paths(root: Path) -> dict:
    return dict(pool=root / "data/ext/v4g/casmi26-v2-pool", split=root / "data/ext/foldsafe/split.parquet",
                keys=root / "kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json",
                embed=root / "outputs/align_embed", train=root / "data/train.parquet", frag=root / "data/ext/frag")


def phase_count(root: Path, out: Path) -> dict:
    import pandas as pd
    sys.path.insert(0, str(root / "scripts/train"))
    import sanpier_fold0 as sf
    P = _paths(root)
    t0 = time.time()
    meta = pd.read_parquet(P["pool"] / "pool_meta.parquet", columns=["key", "mass", "formula"])
    assert bool(np.all(np.diff(meta.mass.values) >= 0))
    ts = pd.read_parquet(P["pool"] / "train_structs.parquet", columns=["inchikey14", "key"])
    raw_to_key = dict(zip(ts.inchikey14, ts.key))
    key_to_row = dict(zip(meta.key.values, range(len(meta))))
    split = pd.read_parquet(P["split"], columns=["inchikey14", "key", "fold"])
    hk = json.loads(P["keys"].read_text())
    drop_raw = sf.drop_inchikey14(split.inchikey14, hk)
    assert len(drop_raw) == EXPECTED_DROP, len(drop_raw)
    drop_key = drop_key_space(drop_raw, raw_to_key, hk, split.key)
    assert len(drop_key) == EXPECTED_DROP_KEY, len(drop_key)
    allowed = ~meta.key.isin(drop_key).values
    assert int(allowed.sum()) == EXPECTED_POOL_ALLOWED, int(allowed.sum())
    # #117 で作り直した 4 集合の drop_keys（正準キー＋別名）がすべて DROP ∪ DROP_KEY に入っていること
    h117 = {h: set((root / f"data/holdouts/{h}/drop_keys.txt").read_text().split()) for h in HOLDOUTS}
    h117_missing = {h: len(ks - drop_key - drop_raw) for h, ks in h117.items()}
    assert all(v == 0 for v in h117_missing.values()), h117_missing
    train_key = set(ts.key.values)
    is_trainstruct = meta.key.isin(train_key).values
    erep = json.loads((P["embed"] / "embed_report.json").read_text())
    tr = []
    for k in range(erep["n_shards"]):
        t = pd.read_parquet(P["embed"] / f"idx_{k:03d}.parquet", columns=["row", "kind", "inchikey14", "adduct", "ce",
                                                                          "precursor_mz"])
        tr.append(t[t.kind == 0])
    tr = pd.concat(tr, ignore_index=True)
    tr["key"] = tr.inchikey14.map(raw_to_key)
    keep = tr.key.notna().values & ~tr.key.isin(drop_key).values & ~tr.inchikey14.isin(drop_raw).values
    tr = tr[keep].reset_index(drop=True)
    val_keys = split_val_keys(tr.key.values)
    is_val = tr.key.isin(val_keys).values
    pos_row = np.array([key_to_row[k] for k in tr.key.values], np.int64)
    val_rows = meta.key.isin(val_keys).values
    up_tr = np.unique(pos_row[~is_val]); up_va = np.unique(pos_row[is_val])
    assert len(up_tr) == EXPECTED_TRAIN_STRUCTS and len(up_va) == EXPECTED_VAL_STRUCTS, (len(up_tr), len(up_va))
    assert int((~is_val).sum()) == EXPECTED_TRAIN_SPECTRA and int(is_val.sum()) == N_VAL_SPECTRA, \
        (int((~is_val).sum()), int(is_val.sum()))
    mass = meta.mass.values.astype(np.float64)
    # allpool（#109 と同じ）と design（train 由来の構造だけ）
    off_tr, flat_tr = negative_csr(up_tr, mass, allowed & ~val_rows)
    off_va, flat_va = negative_csr(up_va, mass, allowed)
    doff_tr, dflat_tr = negative_csr(up_tr, mass, allowed & ~val_rows & is_trainstruct)
    doff_va, dflat_va = negative_csr(up_va, mass, allowed & is_trainstruct)
    leak = sf.leak_audit(set(tr.inchikey14), zip(split.inchikey14, split.fold), hk)
    leak_key = sf.leak_audit(set(tr.key), zip(split.key, split.fold), hk)
    leak_all = dict(raw=leak["total"], key=leak_key["total"],
                    neg_in_drop=int(pd.Series(meta.key.values[np.concatenate([flat_tr, flat_va])]).isin(drop_key).sum()),
                    pos_in_drop=int(tr.key.isin(drop_key).sum()), train_neg_in_val=int(val_rows[flat_tr].sum()),
                    design_neg_not_trainstruct=int((~is_trainstruct[np.concatenate([dflat_tr, dflat_va])]).sum()))
    assert all(v == 0 for v in leak_all.values()), leak_all
    n_union = union_count(up_tr, flat_tr)
    n_union_val = union_count(np.concatenate([up_tr, up_va]), np.concatenate([flat_tr, flat_va]))
    dn_union = union_count(up_tr, dflat_tr)
    dn_union_val = union_count(np.concatenate([up_tr, up_va]), np.concatenate([dflat_tr, dflat_va]))
    neg_len = np.diff(off_tr); dneg_len = np.diff(doff_tr)
    form = meta.formula.values

    def same_formula(off, flat):
        return np.array([int((form[flat[off[i]:off[i + 1]]] == form[up_tr[i]]).sum()) for i in range(len(up_tr))])
    same, dsame = same_formula(off_tr, flat_tr), same_formula(doff_tr, dflat_tr)
    # 切片: 学習の正解構造（内部検証を除く）の正準キーから seed 0 で 1,000
    tr_keys = meta.key.values[up_tr]
    sl_keys = select_slice(tr_keys)
    sl_rows = np.array(sorted(key_to_row[k] for k in sl_keys), np.int64)
    ci = np.searchsorted(up_tr, sl_rows)
    assert np.all(up_tr[ci] == sl_rows)
    sl_off = np.concatenate([[0], np.cumsum(neg_len[ci])])
    sl_flat = np.concatenate([flat_tr[off_tr[i]:off_tr[i + 1]] for i in ci])
    dsl_off = np.concatenate([[0], np.cumsum(dneg_len[ci])])
    dsl_flat = np.concatenate([dflat_tr[doff_tr[i]:doff_tr[i + 1]] for i in ci])
    assert np.isin(dsl_flat, sl_flat).all()   # design の負例は allpool の負例の部分集合
    sl_neg_u = np.setdiff1d(np.unique(sl_flat), sl_rows)
    spec_sel = tr[~is_val & np.isin(pos_row, sl_rows)]
    np.savez(out / "slice.npz", sl_rows=sl_rows, sl_off=sl_off, sl_flat=sl_flat, sl_neg_u=sl_neg_u,
             dsl_off=dsl_off, dsl_flat=dsl_flat,
             spec_train_row=spec_sel.row.values.astype(np.int64),
             spec_pool_row=np.array([key_to_row[k] for k in spec_sel.key.values], np.int64),
             spec_ik=spec_sel.inchikey14.values.astype(str))
    rep = dict(n_train_spectra=int((~is_val).sum()), n_val_spectra=int(is_val.sum()), n_train_structs=len(up_tr),
               n_val_structs=len(up_va), drop=len(drop_raw), drop_key=len(drop_key), pool_allowed=int(allowed.sum()),
               h117_missing=h117_missing, pool_trainstruct_rows=int(is_trainstruct.sum()),
               pool_allowed_trainstruct=int((allowed & is_trainstruct).sum()),
               allpool=dict(n_union=n_union, n_union_val=n_union_val, neg_median=float(np.median(neg_len)),
                            neg_zero=int((neg_len == 0).sum()), same_formula_neg_median=float(np.median(same)),
                            same_formula_neg_share=float(same.sum() / max(1, neg_len.sum()))),
               design=dict(n_union=dn_union, n_union_val=dn_union_val, neg_median=float(np.median(dneg_len)),
                           neg_zero=int((dneg_len == 0).sum()), same_formula_neg_median=float(np.median(dsame)),
                           same_formula_neg_share=float(dsame.sum() / max(1, dneg_len.sum()))),
               leak=leak_all,
               slice_structs=len(sl_rows), slice_neg_unique=len(sl_neg_u), slice_spectra=len(spec_sel),
               seconds=round(time.time() - t0, 1))
    return rep


def _load_pool_smiles(root: Path, rows: np.ndarray) -> list:
    import pyarrow.parquet as pq
    t = pq.read_table(_paths(root)["pool"] / "pool_meta.parquet", columns=["smiles"]).column(0)
    return [t[int(i)].as_py() for i in rows]


def phase_frag(root: Path, out: Path, which: str) -> dict:
    sys.path.insert(0, str(_paths(root)["frag"]))
    import frag_rescore as fr
    s = np.load(out / "slice.npz")
    rows = s["sl_rows"] if which == "a" else s["sl_neg_u"]
    smis = _load_pool_smiles(root, rows)
    t0 = time.time(); fr.warmup(); warm = time.time() - t0
    c = fr.resolve_cfg(None)
    t1 = time.time()
    lv = []
    for smi in smis:
        try:
            lv.append(fr.fragment_levels(smi, c["max_breaks"], c["max_bonds_2"], c["max_bonds_3"]))
        except Exception:
            lv.append(None)
    sec = time.time() - t1
    masses, off = levels_to_csr(lv)
    np.savez(out / f"frag_{which}.npz", rows=rows, masses=masses, off=off)
    nfr = np.diff(off[:, [0, 3]], axis=1).ravel()
    return dict(which=which, n_structs=len(rows), seconds=round(sec, 2), warmup_s=round(warm, 2),
                sec_per_struct=sec / max(1, len(rows)), n_none=sum(x is None for x in lv),
                frag_per_struct_mean=float(nfr.mean()), frag_per_struct_median=float(np.median(nfr)),
                frag_per_struct_max=int(nfr.max()), bytes_per_struct=float(masses.nbytes + off.nbytes) / max(1, len(rows)))


def phase_spectra(root: Path, out: Path) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    s = np.load(out / "slice.npz")
    want = np.unique(s["spec_train_row"])
    cols = ["inchikey14", "adduct", "ionization_mode", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
            "collision_energy_ev"]
    t0 = time.time()
    pf = pq.ParquetFile(_paths(root)["train"], buffer_size=1 << 20, pre_buffer=False)
    got, base = {}, 0
    for b in pf.iter_batches(batch_size=8192, columns=cols):
        idx = pa.array(np.arange(base, base + b.num_rows, dtype=np.int64))
        m = pc.is_in(idx, value_set=pa.array(want))
        if pc.any(m).as_py():
            sub = b.filter(m).to_pydict(); ids = idx.filter(m).to_pylist()
            for j, r in enumerate(ids):
                ce = sub["collision_energy_ev"][j]
                got[r] = dict(ik=sub["inchikey14"][j], adduct=sub["adduct"][j], prec=sub["precursor_mz"][j],
                              mode=1 if sub["ionization_mode"][j] == "positive" else -1,
                              mz=sub["ms2_mzs"][j], it=sub["ms2_normalized_intensities"][j],
                              ce=float(np.mean(ce)) if ce else 0.0)
        base += b.num_rows
    sec = time.time() - t0
    order = s["spec_train_row"]
    miss = [int(r) for r in order if int(r) not in got]
    assert not miss, miss[:5]
    bad_ik = sum(got[int(r)]["ik"] != ik for r, ik in zip(order, s["spec_ik"]))
    assert bad_ik == 0, bad_ik
    t1 = time.time()
    pk = pack_spectra([got[int(r)] for r in order])
    sec_pack = time.time() - t1
    np.savez(out / "spectra.npz", **pk)
    return dict(seconds=round(sec, 1), n_rows_scanned=base, n_spectra=len(order), peaks_median=float(np.median(pk["n"])),
                pack_seconds=round(sec_pack, 3), sec_pack_per_spectrum=sec_pack / max(1, len(order)))


def phase_step(root: Path, out: Path, steps: int = 50, warm: int = 5, fwd_batches: int = 20, device: str = "mps",
               negs: str = "design") -> dict:
    import torch
    torch.manual_seed(SEED)
    s = np.load(out / "slice.npz"); sp = dict(np.load(out / "spectra.npz"))
    fa, fb = np.load(out / "frag_a.npz"), np.load(out / "frag_b.npz")
    rows = np.concatenate([fa["rows"], fb["rows"]])
    masses = np.concatenate([fa["masses"], fb["masses"]])
    off = np.concatenate([fa["off"], fb["off"] + len(fa["masses"])])
    struct_of_row = {int(r): i for i, r in enumerate(rows)}
    sl_rows = s["sl_rows"]
    sl_idx = {int(r): i for i, r in enumerate(sl_rows)}
    spec_csr = np.array([sl_idx[int(r)] for r in s["spec_pool_row"]], np.int64)
    pos_struct = np.array([struct_of_row[int(r)] for r in s["spec_pool_row"]], np.int64)
    pre = "dsl_" if negs == "design" else "sl_"
    flat_struct = np.array([struct_of_row[int(r)] for r in s[pre + "flat"]], np.int64)
    sl_off = s[pre + "off"]
    dev = torch.device(device)
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n_spec = len(spec_csr)
    spec_feat = torch.from_numpy(sp["spec_feat"])

    def make_batch(step):
        rng = np.random.default_rng([SEED, 1, step])
        bi = np.random.default_rng([SEED, 0, step]).choice(n_spec, size=BATCH, replace=False)
        cand, mask = sample_candidates(rng, pos_struct[bi], spec_csr[bi], sl_off, flat_struct)
        vi, vk = np.nonzero(mask)
        n = len(vi)
        t = time.time()
        f0, fs0 = batch_features(masses, off, cand[vi, vk], bi[vi], sp)
        tf = time.time() - t
        p = padded_pairs(n)   # 詰め物の組は特徴 0・slot は b*K の外（捨てる）
        feats = np.zeros((p, TOP_PEAKS, F_PEAK), np.float32); feats[:n] = f0
        fs = np.zeros(p, np.float32); fs[:n] = fs0
        vi_p = np.zeros(p, np.int64); vi_p[:n] = vi
        slot = np.full(p, BATCH * K_CAND, np.int64); slot[:n] = vi * K_CAND + vk
        return bi, vi_p, slot, n, feats, fs, tf

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    def run_step(step, train=True):
        t0 = time.time()
        bi, vi, slot_np, n, feats, fs, tf = make_batch(step)
        vmask = np.arange(TOP_PEAKS)[None, :] < sp["n"][bi[vi]][:, None]
        vmask[n:] = False
        valid = torch.from_numpy(vmask).to(dev)
        x = torch.from_numpy(feats).to(dev)
        pair = torch.from_numpy(fs[:, None]).to(dev)
        sf = spec_feat[bi[vi]].to(dev)
        slot = torch.from_numpy(slot_np).to(dev)
        t1 = time.time()
        if train:
            model.train()
            logit = model(x, valid, pair, sf)
            flat = torch.full((BATCH * K_CAND + 1,), float("-inf"), device=dev).index_put((slot,), logit)
            loss = torch.nn.functional.cross_entropy(flat[:BATCH * K_CAND].view(BATCH, K_CAND),
                                                     torch.zeros(BATCH, dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
            lv = float(loss.detach().cpu())
        else:
            model.eval()
            with torch.no_grad():
                logit = model(x, valid, pair, sf)
            lv = float(logit.mean().cpu())
        sync()
        t2 = time.time()
        return dict(total=t2 - t0, feat=tf, prep=t1 - t0 - tf, gpu=t2 - t1, n_pairs=n, n_padded=len(vi), loss=lv)

    for i in range(warm):
        run_step(10_000 + i)
    rec = [run_step(i) for i in range(steps)]
    fwd = [run_step(20_000 + i, train=False) for i in range(fwd_batches)]
    mean = lambda xs, k: float(np.mean([x[k] for x in xs]))  # noqa: E731
    return dict(device=str(dev), negs=negs, steps=steps, warm=warm, sec_per_step=mean(rec, "total"),
                sec_feat=mean(rec, "feat"), sec_prep=mean(rec, "prep"), sec_gpu=mean(rec, "gpu"),
                sec_per_step_sd=float(np.std([x["total"] for x in rec])), pairs_per_step=mean(rec, "n_pairs"),
                padded_shapes=sorted({x["n_padded"] for x in rec}),
                sec_fwd_batch=mean(fwd, "total"), loss_first=rec[0]["loss"], loss_last=rec[-1]["loss"],
                numba_threads=int(os.environ.get("NUMBA_NUM_THREADS", "0")) or None,
                mps_driver_allocated_gb=(round(torch.mps.driver_allocated_memory() / 1e9, 2) if dev.type == "mps" else None))


def phase_report(out: Path) -> dict:
    c = json.loads((out / "count.json").read_text())
    fa, fb = json.loads((out / "frag_a.json").read_text()), json.loads((out / "frag_b.json").read_text())
    sp = json.loads((out / "spectra.json").read_text())
    assert c["n_val_spectra"] == N_VAL_SPECTRA and c["n_train_spectra"] == EXPECTED_TRAIN_SPECTRA
    sps = max(fa["sec_per_struct"], fb["sec_per_struct"])
    bps = max(fa["bytes_per_struct"], fb["bytes_per_struct"])
    res = {}
    for negs in ("design", "allpool"):
        st = json.loads((out / f"step_{negs}.json").read_text())
        for name in ("n_union", "n_union_val"):
            n = c[negs][name]
            res[f"{negs}/{name}"] = {str(k): extrapolate(n, sps, sp["seconds"], st["sec_per_step"], st["sec_fwd_batch"],
                                                         steps=k, sec_pack_per_spectrum=sp["sec_pack_per_spectrum"])
                                     for k in (STEPS_JUDGE, STEPS_MIN)}
    # 判定: 28,000 ステップの 4 通り（design/allpool × union/union_val）の合計の最大で決める（保守的）
    worst = max((v[str(STEPS_JUDGE)] for v in res.values()), key=lambda r: r["total_h"])
    return dict(issue=ISSUE, sec_per_struct_used=sps, bytes_per_struct_used=bps,
                frag_disk_gb_design=c["design"]["n_union_val"] * bps / 1e9,
                frag_disk_gb_allpool=c["allpool"]["n_union_val"] * bps / 1e9,
                extrapolation=res, worst_total_h=worst["total_h"], verdict=worst["verdict"],
                wallclock_cap_h=min(1.5 * worst["total_h"], 96.0))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="#120 fragnn の速度測定（学習しない）")
    ap.add_argument("phase", choices=["count", "frag_a", "frag_b", "spectra", "step_design", "step_allpool", "report"])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.phase.startswith("step"):
        for k, v in (("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6"), ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")):
            if os.environ.get(k) != v:
                raise SystemExit(f"set {k}={v}")
    fn = {"count": lambda: phase_count(root, out), "frag_a": lambda: phase_frag(root, out, "a"),
          "frag_b": lambda: phase_frag(root, out, "b"), "spectra": lambda: phase_spectra(root, out),
          "step_design": lambda: phase_step(root, out, device=a.device, negs="design"),
          "step_allpool": lambda: phase_step(root, out, device=a.device, negs="allpool"),
          "report": lambda: phase_report(out)}[a.phase]
    r = fn()
    (out / f"{a.phase}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
