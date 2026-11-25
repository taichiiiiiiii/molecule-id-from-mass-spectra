#!/usr/bin/env python3
"""
Issue #240（2026-11-25 昼 項目 4）: 自前の「天然物らしさ」分類器を fold ごとに 5 本（k = 0〜4）学習し、dev の候補に予測を付ける。

定義は #240 issuecomment-7152664644（学習の前に固定）のとおり。要点:
  - 正例 = v2-pool の src ∈ {2, 3}（COCONUT）で質量が PubChem 層の範囲のもの
  - 負例 = (A) PubChem 層の構造のうち COCONUT・LOTUS・NPAtlas に無いもの。正例の 1 Da の区間ごとに seed 20261125 で
    1.15 倍を引き、除外の後に正例と同じ数だけ残す（質量を揃える）
  - 入力 = 最大の断片・立体を消した Morgan カウント（r2、2048）＋ RDKit 記述子 20 個（DESC_NAMES、固定）
  - LightGBM `objective=binary`、ほかは既定（100 回、早期停止なし）、seed 20261125。fold k のモデル = bucket ≠ k の行
  - bucket = train の構造に当たればその fold_identity、当たらなければ blake2b(キー) mod 5
  - 学習の直前に 4 鍵（生の InChIKey14・正規化キー・scorer_key・identity_group）で fold k の構造との重なり 0 を assert
  - held-out AUC は k = 1〜4 だけ。fold-0 では予測も評価もしない
  - 予測の表: プールの全行 ＋ B8 の 20 集合の PubChem の上位 60。全行 model_fold = query_fold ∈ {1..4} を assert

段（結果は --out-dir に。途中から再開できる）:
    .venv/bin/python scripts/features/np_likeness.py prepare     # 正例・負例の表（鍵・bucket）
    .venv/bin/python scripts/features/np_likeness.py featurize   # 特徴（疎行列）
    .venv/bin/python scripts/features/np_likeness.py train       # 5 本（fold ごとに log 1 つ、既にあれば止まる）
    .venv/bin/python scripts/features/np_likeness.py predict     # dev の候補の予測の表と report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261125
OVERSAMPLE = 1.15
PPM = 10.0
NBITS = 2048
FOLDS = (0, 1, 2, 3, 4)
EVAL_FOLDS = (1, 2, 3, 4)
MULTI = -2   # 2 つの fold の構造に当たる行: どのモデルの学習にも入れない（bucket = -2）
DESC_NAMES = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount",
              "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings", "FractionCSP3", "HeavyAtomCount",
              "NumHeteroatoms", "n_C", "n_N", "n_O", "n_S", "n_P", "n_halogen", "formal_charge")
LGB_PARAMS = dict(objective="binary", seed=SEED, verbose=-1, num_threads=8)
NUM_BOOST_ROUND = 100   # LightGBM の既定値

POOL = ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet"
POP = ROOT / "data/ext/pop/pool_popularity.csv"
NPK = ROOT / "data/ext/npk/descriptors.parquet"
STRUCT = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
TIER = ROOT / "data/ext/pubchem_tier"
OUT = ROOT / "outputs/local_eval/np_likeness"
B8_SETS = tuple(f"b8_f{k}_{t}_t10" for k in EVAL_FOLDS for t in "abcde")

# ---------------------------------------------------------------- 純粋な関数（テストあり）
# heavy_comp_str・comp_hash・wl_hash は #236 の scripts/bench/candidate_recall.py（PR #237、未マージ）と同じ定義の写し
_TOK = re.compile(r"\[[^\]]*\]|Br|Cl|[BCNOPSFI]|[bcnops]")
_BRA = re.compile(r"\[\d*([A-Z][a-z]?|[a-z][a-z]?)")
ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"}


def heavy_comp_str(smi: str) -> tuple:
    c: Counter = Counter()
    for t in _TOK.findall(smi):
        if t[0] == "[":
            m = _BRA.match(t)
            if not m:
                continue
            e = m.group(1)
            e = e[0].upper() + e[1:]
        else:
            e = t[0].upper() + t[1:]
        if e != "H":
            c[e] += 1
    return tuple(sorted(c.items()))


def comp_hash(comp: tuple) -> int:
    s = ",".join(f"{e}{n}" for e, n in comp).encode()
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little", signed=True)


def wl_hash(smi: str, iters: int = 3) -> int:
    """重原子のグラフの WL の hash（結合次数・電荷・H・立体を見ない）。ORGANIC の外の元素・読めない分子は 0（何とでも両立）。"""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return 0
    heavy = [a for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    if any(a.GetSymbol() not in ORGANIC for a in heavy):
        return 0
    idx = {a.GetIdx(): i for i, a in enumerate(heavy)}
    nb = [[idx[n.GetIdx()] for n in a.GetNeighbors() if n.GetIdx() in idx] for a in heavy]
    lab = [a.GetAtomicNum() for a in heavy]
    for _ in range(iters):
        lab = [hash((lab[i], tuple(sorted(lab[j] for j in nb[i])))) for i in range(len(heavy))]
    h = hash((len(heavy), tuple(sorted(lab))))
    return h if h != 0 else 1


def hash_bucket(key: str, n: int = 5) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % n


def assign_bucket(keys, train_fold) -> np.ndarray:
    """train_fold: 行ごとの train の構造の fold_identity（当たらなければ -1、2 つの fold なら MULTI）。当たれば その fold、
    無ければ hash mod 5、MULTI はそのまま（どのモデルの学習にも入れない）。"""
    tf = np.asarray(train_fold, dtype=np.int64)
    hb = np.fromiter((hash_bucket(str(k)) for k in keys), np.int64, count=len(tf))
    return np.where(tf == MULTI, MULTI, np.where(tf >= 0, tf, hb))


def train_mask(bucket: np.ndarray, k: int) -> np.ndarray:
    """fold k のモデルの学習の行: bucket != k かつ MULTI でない。"""
    b = np.asarray(bucket)
    return (b != k) & (b >= 0)


def fold_keysets(st: pd.DataFrame, k: int) -> dict:
    """fold k の構造の 4 鍵の集合。生・正規化・scorer_key はどれも InChIKey14 なので 1 つの集合でも比べる。"""
    s = st[st.fold_identity.astype(int) == k]
    keys = set(s.inchikey14.astype(str)) | set(s.scorer_key.astype(str))
    return dict(keys=keys, groups=set(s.identity_group.astype(str)))


def assert_no_fold_overlap(rows: pd.DataFrame, fk: dict, k: int) -> dict:
    """学習の行（列 raw_key・canon_key・scorer_key・identity_group・compat）が fold k の構造と 4 鍵で重ならないこと。
    canon_key が空の行は compat = False（組成＋WL が pool・train のどの構造とも両立しない）であることを要する。"""
    raw = rows.raw_key.astype(str).isin(fk["keys"])
    has_c = rows.canon_key.notna() & (rows.canon_key.astype(str) != "")
    can = has_c & rows.canon_key.astype(str).isin(fk["keys"])
    sc = rows.scorer_key.notna() & rows.scorer_key.astype(str).isin(fk["keys"])
    grp = rows.identity_group.notna() & rows.identity_group.astype(str).isin(fk["groups"])
    no_c_compat = (~has_c) & rows.compat.astype(bool)
    counts = dict(raw_inchikey14=int(raw.sum()), canonical_key=int(can.sum()), scorer_key=int(sc.sum()),
                  identity_group=int(grp.sum()), uncanonicalised_but_compatible=int(no_c_compat.sum()))
    assert all(v == 0 for v in counts.values()), f"fold {k} のモデルの学習の行が fold {k} の構造と重なる: {counts}"
    return counts


def check_pred_folds(df: pd.DataFrame) -> None:
    """予測の表: 全行 model_fold = query_fold、query_fold は 1〜4 だけ（fold-0 の問い合わせの行は 0）。"""
    assert len(df) > 0
    assert (df.model_fold.astype(int).values == df.query_fold.astype(int).values).all(), "model_fold と query_fold が違う行がある"
    assert df.query_fold.astype(int).isin(list(EVAL_FOLDS)).all(), "fold-0（または範囲外）の問い合わせの行がある"


def in_window(mass: np.ndarray, centers: np.ndarray, ppm: float = PPM) -> np.ndarray:
    """mass[i] がどれかの中心 t の ±t·ppm·1e-6 に入るか。centers は未整列でよい。"""
    c = np.sort(np.asarray(centers, dtype=np.float64))
    m = np.asarray(mass, dtype=np.float64)
    if len(c) == 0:
        return np.zeros(len(m), bool)
    j = np.searchsorted(c, m)
    out = np.zeros(len(m), bool)
    for jj in (j - 1, j):
        ok = (jj >= 0) & (jj < len(c))
        t = c[np.clip(jj, 0, len(c) - 1)]
        out |= ok & (np.abs(m - t) <= t * ppm * 1e-6)
    return out


def mass_matched_draw(pos_mass: np.ndarray, tier_mass: np.ndarray, seed: int = SEED, over: float = OVERSAMPLE) -> dict:
    """正例の 1 Da の区間ごとに、層（質量で整列済み）のその区間の行から ceil(n_b·over) を非復元で引く。
    返り値: {区間: (引いた層の行番号〈引いた順〉, 正例の数)}。区間は昇順に 1 つの rng で引く。"""
    rng = np.random.default_rng(seed)
    bins = np.floor(pos_mass).astype(np.int64)
    cnt = Counter(bins.tolist())
    out = {}
    for b in sorted(cnt):
        a = int(np.searchsorted(tier_mass, b, "left")); e = int(np.searchsorted(tier_mass, b + 1, "left"))
        n_av = e - a
        m = min(int(math.ceil(cnt[b] * over)), n_av)
        idx = a + rng.choice(n_av, size=m, replace=False) if m > 0 else np.zeros(0, np.int64)
        out[b] = (np.asarray(idx, np.int64), int(cnt[b]))
    return out


def keep_first_per_bin(bin_of: np.ndarray, ok: np.ndarray, need: dict) -> np.ndarray:
    """引いた順の行のうち ok の行を、区間ごとに need[b] 件まで残す mask。"""
    used: Counter = Counter()
    keep = np.zeros(len(ok), bool)
    for i, (b, o) in enumerate(zip(bin_of.tolist(), ok.tolist())):
        if o and used[b] < need[b]:
            keep[i] = True; used[b] += 1
    return keep


def featurize_one(smi: str):
    """→ (Morgan のカウント〈非零の (bit, count) の配列〉, 記述子 20 個) か None。最大の断片、立体を消す。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = _LFC().choose(m)
        Chem.RemoveStereochemistry(m)
        v = _GEN().GetCountFingerprintAsNumPy(m)
    except Exception:
        return None
    nz = np.nonzero(v)[0]
    el = Counter(a.GetSymbol() for a in m.GetAtoms())
    d = [Descriptors.MolWt(m), Descriptors.MolLogP(m), rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumHBD(m),
         rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
         rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumAliphaticRings(m),
         rdMolDescriptors.CalcNumSaturatedRings(m), rdMolDescriptors.CalcFractionCSP3(m), m.GetNumHeavyAtoms(),
         rdMolDescriptors.CalcNumHeteroatoms(m), el["C"], el["N"], el["O"], el["S"], el["P"],
         el["F"] + el["Cl"] + el["Br"] + el["I"], Chem.GetFormalCharge(m)]
    return nz.astype(np.int32), v[nz].astype(np.float32), np.asarray(d, np.float32)


_gen = _lfc = None


def _GEN():
    global _gen
    if _gen is None:
        from rdkit.Chem import rdFingerprintGenerator
        _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
    return _gen


def _LFC():
    global _lfc
    if _lfc is None:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        _lfc = rdMolStandardize.LargestFragmentChooser()
    return _lfc


def featurize(smiles: list[str]):
    """→ (CSR 行列 [Morgan 2048 | 記述子 20]、読めた mask)。読めない行は全 0 の行で ok = False。"""
    from scipy import sparse
    ind, dat, ptr, ok = [], [], [0], np.zeros(len(smiles), bool)
    nd = len(DESC_NAMES)
    for i, s in enumerate(smiles):
        r = featurize_one(s)
        if r is not None:
            nz, val, d = r
            ind.append(nz); dat.append(val)
            ind.append(np.arange(NBITS, NBITS + nd, dtype=np.int32)); dat.append(d)
            ok[i] = True
            ptr.append(ptr[-1] + len(nz) + nd)
        else:
            ptr.append(ptr[-1])
    X = sparse.csr_matrix((np.concatenate(dat) if dat else np.zeros(0, np.float32),
                           np.concatenate(ind) if ind else np.zeros(0, np.int32), np.asarray(ptr, np.int64)),
                          shape=(len(smiles), NBITS + nd), dtype=np.float32)
    return X, ok


def feature_names() -> list[str]:
    return [f"mc{i}" for i in range(NBITS)] + list(DESC_NAMES)


def train_model(X, y: np.ndarray):
    import lightgbm as lgb
    ds = lgb.Dataset(X, label=y.astype(np.float32), feature_name=feature_names(), free_raw_data=False)
    return lgb.train(dict(LGB_PARAMS), ds, num_boost_round=NUM_BOOST_ROUND)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC（同点は平均順位）。"""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- 実データ

def _pmap(fn, items, workers: int, chunk: int = 2000):
    from concurrent.futures import ProcessPoolExecutor
    parts = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, parts))


def _keys_job(smis):
    """SMILES → (生の InChIKey14 か ''、組成の hash、WL の hash)。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        rk = ""
        if m is not None:
            try:
                rk = Chem.MolToInchiKey(m)[:14]
            except Exception:
                rk = ""
        out.append((rk, comp_hash(heavy_comp_str(s)), wl_hash(s)))
    return out


def _hash_job(smis):
    return [(comp_hash(heavy_comp_str(s)), wl_hash(s)) for s in smis]


def _tokey_job(smis):
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    import mrr25
    return [mrr25.to_key(s) or "" for s in smis]


def _feat_job(smis):
    X, ok = featurize(list(smis))
    return X, ok


def tier_smiles(idx: np.ndarray) -> list[str]:
    off = np.load(TIER / "pc_off.npy", mmap_mode="r"); buf = np.load(TIER / "pc_smiles.npy", mmap_mode="r")
    out = []
    for i in idx.tolist():
        a, b = int(off[i]), int(off[i + 1])
        out.append(bytes(buf[a:b]).decode("ascii"))
    return out


def load_struct() -> pd.DataFrame:
    st = pd.read_parquet(STRUCT, columns=["inchikey14", "scorer_key", "identity_group", "fold_identity", "smiles", "mono_mass"])
    st["identity_group"] = st.identity_group.astype(str)
    return st


def struct_map(st: pd.DataFrame) -> dict:
    """InChIKey14（生・scorer_key のどちらでも）→ (fold_identity, scorer_key, identity_group)。1 つの鍵が 2 つの fold の構造に
    当たる場合（ある構造の scorer_key が別の fold の構造の生の鍵と同じ、などの正準化の非冪等）は fold = MULTI とする。"""
    mp: dict = {}
    for r in st.itertuples(index=False):
        for k in (str(r.inchikey14), str(r.scorer_key)):
            v = (int(r.fold_identity), str(r.scorer_key), str(r.identity_group))
            if k in mp and mp[k][0] != v[0]:
                mp[k] = (MULTI, mp[k][1], mp[k][2])
            elif k not in mp:
                mp[k] = v
    return mp


def log_to(path: Path):
    f = open(path, "a")

    def log(*a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True); f.write(s + "\n"); f.flush()
    return log


def stage_prepare(out: Path, workers: int, log) -> None:
    t0 = time.time()
    tmass = np.load(TIER / "pc_mass.npy", mmap_mode="r")
    lo, hi = float(tmass[0]), float(tmass[-1])
    pool = pd.read_parquet(POOL, columns=["key", "smiles", "mass", "src", "pid"])
    pop = pd.read_csv(POP, usecols=["pool_row", "inchikey14", "np_coconut", "np_lotus", "np_npatlas"])
    assert (pop.pool_row.values == np.arange(len(pop))).all() and len(pop) == len(pool)
    assert (pool.pid.values == np.arange(len(pool))).all()
    pool["raw_key"] = pop.inchikey14.astype(str).values
    log(f"pool {len(pool)} 行、raw_key == key の割合 {float((pool.raw_key == pool.key).mean()):.4f}、層の質量 {lo:.3f}〜{hi:.3f}")
    npk = set(pd.read_parquet(NPK, columns=["ik14"]).ik14.astype(str))
    np_pool = pool.src.isin([2, 3]).values
    np_flag = (pop.np_coconut.values == 1) | (pop.np_lotus.values == 1) | (pop.np_npatlas.values == 1)
    np_raw = set(pool.key[np_pool]) | set(pool.raw_key[np_pool]) | set(pop.inchikey14[np_flag].astype(str)) | npk
    np_canon = set(pool.key[np_pool])
    st = load_struct(); smap = struct_map(st)

    # 正例
    P = pool[np_pool & (pool.mass >= lo) & (pool.mass <= hi)].copy()
    P["source"] = "pool"; P["label"] = 1; P["canon_key"] = P.key; P["compat"] = True; P["row"] = P.pid
    log(f"正例 {len(P)}（src2 {int((P.src == 2).sum())}・src3 {int((P.src == 3).sum())}）")

    # 組成＋WL の hash（pool 全行と train の構造）: 負例の正規化キーを計算するかの絞り込み
    ref_smi = pool.smiles.astype(str).tolist() + st.smiles.astype(str).tolist()
    hh = [x for part in _pmap(_hash_job, ref_smi, workers, 5000) for x in part]
    ref_c = {c for c, _ in hh}; ref_cw = {(c, w) for c, w in hh}; ref_c_wl0 = {c for c, w in hh if w == 0}
    log(f"参照の hash {len(hh)} 件 {time.time() - t0:.0f}s")

    # 負例: 質量を揃えて引く
    draws = mass_matched_draw(P.mass.values, tmass)
    idx = np.concatenate([d[0] for d in draws.values()]); binof = np.concatenate([np.full(len(d[0]), b) for b, d in draws.items()])
    need = {b: d[1] for b, d in draws.items()}
    smis = tier_smiles(idx)
    log(f"層から {len(idx)} 行を引いた {time.time() - t0:.0f}s")
    kk = [x for part in _pmap(_keys_job, smis, workers) for x in part]
    raw = np.array([x[0] for x in kk], dtype=object); ch = np.array([x[1] for x in kk]); wh = np.array([x[2] for x in kk])
    compat = np.array([(c in ref_c) and (w == 0 or c in ref_c_wl0 or (c, w) in ref_cw) for c, w in zip(ch, wh)])
    canon = np.array([""] * len(idx), dtype=object)
    ci = np.where(compat & (raw != ""))[0]
    if len(ci):
        ck = [x for part in _pmap(_tokey_job, [smis[i] for i in ci], workers, 500) for x in part]
        canon[ci] = ck
    log(f"負例の鍵 {time.time() - t0:.0f}s、組成＋WL が両立して to_key した行 {len(ci)}")
    is_np = np.array([(r in np_raw) or (c != "" and c in np_canon) for r, c in zip(raw, canon)])
    first = ~pd.Series(raw).duplicated().values
    nokey = compat & (raw != "") & (canon == "")   # 組成＋WL が両立するのに to_key が失敗（正規化キーで点検できない）
    ok = (raw != "") & ~is_np & first & ~nokey
    keep = keep_first_per_bin(binof, ok, need)
    N = pd.DataFrame(dict(row=idx[keep], smiles=np.array(smis, dtype=object)[keep], mass=np.asarray(tmass[idx[keep]]),
                          raw_key=raw[keep], canon_key=canon[keep], compat=compat[keep]))
    N["source"] = "tier"; N["label"] = 0
    short = {int(b): int(need[b] - int(keep[binof == b].sum())) for b in need if need[b] > int(keep[binof == b].sum())}
    log(f"負例 {len(N)}（読めない {int((raw == '').sum())}・天然物 {int(is_np.sum())}・重複 {int((~first).sum())} を除外、"
        f"不足の区間 {len(short)} 個・不足 {sum(short.values())} 行）")

    rows = pd.concat([P[["source", "row", "smiles", "mass", "raw_key", "canon_key", "compat", "label"]], N], ignore_index=True)
    tf, sk, gr = [], [], []
    for r, c in zip(rows.raw_key.astype(str), rows.canon_key.astype(str)):
        h1 = smap.get(r); h2 = smap.get(c) if c else None
        hit = h1 or h2
        f = hit[0] if hit else -1
        if h1 and h2 and h1[0] != h2[0]:
            f = MULTI
        tf.append(f); sk.append(hit[1] if hit else None); gr.append(hit[2] if hit else None)
    rows["train_fold"] = tf; rows["scorer_key"] = sk; rows["identity_group"] = gr
    bkey = np.where(rows.source == "pool", rows.canon_key.astype(str), rows.raw_key.astype(str))
    rows["bucket"] = assign_bucket(bkey, rows.train_fold.values)
    rows["mass_bin"] = np.floor(rows.mass.values).astype(np.int64)
    rows.to_parquet(out / "rows.parquet", index=False)
    summ = dict(n_pos=int((rows.label == 1).sum()), n_neg=int((rows.label == 0).sum()), tier_mass=[lo, hi],
                n_drawn=int(len(idx)), n_unparsable=int((raw == "").sum()), n_np_excluded=int(is_np.sum()),
                n_dup=int((~first).sum()), n_to_key_failed_compatible=int(nokey.sum()), n_to_key=int(len(ci)), short_bins=len(short), short_rows=int(sum(short.values())),
                n_multi_fold_rows=int((rows.bucket == MULTI).sum()),
                train_struct_rows=dict(pos=int(((rows.label == 1) & (rows.train_fold >= 0)).sum()),
                                       neg=int(((rows.label == 0) & (rows.train_fold >= 0)).sum())),
                bucket_counts={str(b): dict(pos=int(((rows.bucket == b) & (rows.label == 1)).sum()),
                                            neg=int(((rows.bucket == b) & (rows.label == 0)).sum())) for b in FOLDS},
                secs=round(time.time() - t0, 1))
    (out / "prepare.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    log("prepare", json.dumps(summ, ensure_ascii=False))


def _feat_parallel(smis: list[str], workers: int):
    from scipy import sparse
    parts = _pmap(_feat_job, smis, workers, 2000)
    return sparse.vstack([p[0] for p in parts]).tocsr(), np.concatenate([p[1] for p in parts])


def stage_featurize(out: Path, workers: int, log) -> None:
    from scipy import sparse
    t0 = time.time()
    rows = pd.read_parquet(out / "rows.parquet")
    X, ok = _feat_parallel(rows.smiles.astype(str).tolist(), workers)
    sparse.save_npz(out / "X_rows.npz", X); np.save(out / "ok_rows.npy", ok)
    log(f"学習の行の特徴 {X.shape} 読めない {int((~ok).sum())} {time.time() - t0:.0f}s")
    pool = pd.read_parquet(POOL, columns=["key", "smiles"])
    Xp, okp = _feat_parallel(pool.smiles.astype(str).tolist(), workers)
    sparse.save_npz(out / "X_pool.npz", Xp); np.save(out / "ok_pool.npy", okp)
    log(f"プール全行の特徴 {Xp.shape} 読めない {int((~okp).sum())} {time.time() - t0:.0f}s")
    pc = b8_pubchem()
    Xc, okc = _feat_parallel(pc.smiles.astype(str).tolist(), workers)
    sparse.save_npz(out / "X_b8pc.npz", Xc); np.save(out / "ok_b8pc.npy", okc); pc.to_parquet(out / "b8pc.parquet", index=False)
    log(f"B8 の PubChem の候補の特徴 {Xc.shape} 読めない {int((~okc).sum())} {time.time() - t0:.0f}s")


def b8_pubchem() -> pd.DataFrame:
    """B8 の 20 集合の PubChem の上位 60（pubchem_fold0.json: 分子 → [[smiles, f·z, score_key], ...]）を fold ごとに重複なしで。"""
    rows = []
    for h in B8_SETS:
        k = int(h[4])
        j = json.loads((ROOT / f"outputs/local_eval/{h}/pubchem_fold0.json").read_text())
        for mol, lst in j.items():
            for s, _fz, key in lst:
                rows.append((k, str(key), str(s)))
    df = pd.DataFrame(rows, columns=["query_fold", "key", "smiles"]).drop_duplicates(["query_fold", "key"])
    return df.reset_index(drop=True)


def dev_centers() -> dict:
    """fold k の問い合わせ（#236 の dev 一覧）の窓の中心 = その構造の mono_mass。"""
    p = ROOT / "outputs/local_eval/np_likeness/dev_list.csv.gz"
    dev = pd.read_csv(p, usecols=["inchikey14", "fold"])
    st = pd.read_parquet(STRUCT, columns=["inchikey14", "mono_mass"]).drop_duplicates("inchikey14")
    d = dev.merge(st, on="inchikey14", how="left")
    assert d.mono_mass.notna().all()
    assert set(d.fold.astype(int)) == set(EVAL_FOLDS)
    return {k: d.mono_mass[d.fold == k].values for k in EVAL_FOLDS}


def stage_train(out: Path, log0) -> None:
    from scipy import sparse
    rows = pd.read_parquet(out / "rows.parquet"); X = sparse.load_npz(out / "X_rows.npz"); ok = np.load(out / "ok_rows.npy")
    st = load_struct()
    for k in FOLDS:
        mp = out / f"model_fold{k}.txt"; lp = out / f"train_fold{k}.log"
        assert not mp.exists() and not lp.exists(), f"fold {k} は学習済み（学び直さない）: {mp}"
        log = log_to(lp)
        t0 = time.time()
        tr = train_mask(rows.bucket.values, k) & ok
        fk = fold_keysets(st, k)
        ov = assert_no_fold_overlap(rows[tr], fk, k)
        log(f"fold {k}: 学習の行 {int(tr.sum())}（正 {int((rows.label.values[tr] == 1).sum())}・負 {int((rows.label.values[tr] == 0).sum())}）"
            f"、fold {k} の構造との重なり（4 鍵） {json.dumps(ov)}、params {json.dumps(LGB_PARAMS)} rounds {NUM_BOOST_ROUND}")
        bst = train_model(X[np.where(tr)[0]], rows.label.values[tr])
        bst.save_model(str(mp))
        log(f"fold {k}: 学習 {time.time() - t0:.1f}s、保存 {mp.name}")
        (out / f"overlap_fold{k}.json").write_text(json.dumps(dict(fold=k, n_train=int(tr.sum()), overlap=ov), indent=1))


def stage_predict(out: Path, log) -> None:
    import lightgbm as lgb
    from scipy import sparse
    rows = pd.read_parquet(out / "rows.parquet"); X = sparse.load_npz(out / "X_rows.npz"); ok = np.load(out / "ok_rows.npy")
    centers = dev_centers()
    rep = dict(issue=240, folds={})
    pool = pd.read_parquet(POOL, columns=["key", "smiles", "mass", "pid"])
    Xp = sparse.load_npz(out / "X_pool.npz"); okp = np.load(out / "ok_pool.npy")
    pc = pd.read_parquet(out / "b8pc.parquet"); Xc = sparse.load_npz(out / "X_b8pc.npz"); okc = np.load(out / "ok_b8pc.npy")
    # 学習の行の鍵（in_train の判定）: pool の行は pid、層の候補は正規化キー・生の鍵
    parts = []
    for k in FOLDS:
        tr = train_mask(rows.bucket.values, k) & ok
        rec = dict(n_train=int(tr.sum()))
        if k in EVAL_FOLDS:
            bst = lgb.Booster(model_file=str(out / f"model_fold{k}.txt"))
            ev = (rows.bucket.values == k) & ok
            s_ev = bst.predict(X[np.where(ev)[0]]); y_ev = rows.label.values[ev]
            istr = rows.train_fold.values[ev] >= 0
            rec["heldout"] = dict(n_pos=int((y_ev == 1).sum()), n_neg=int((y_ev == 0).sum()), auc=auc(y_ev, s_ev),
                                  auc_train_struct_rows=auc(y_ev[istr], s_ev[istr]) if istr.any() else None,
                                  n_train_struct_rows=dict(pos=int((y_ev[istr] == 1).sum()), neg=int((y_ev[istr] == 0).sum())),
                                  auc_other_rows=auc(y_ev[~istr], s_ev[~istr]),
                                  mean_score_pos=float(s_ev[y_ev == 1].mean()), mean_score_neg=float(s_ev[y_ev == 0].mean()))
            # 学習の行での AUC と平均（学習の中と外の差の記録）
            rs = np.random.default_rng(SEED).choice(np.where(tr)[0], size=min(200000, int(tr.sum())), replace=False)
            s_tr = bst.predict(X[np.sort(rs)]); y_tr = rows.label.values[np.sort(rs)]
            rec["train_sample"] = dict(n=int(len(rs)), auc=auc(y_tr, s_tr), mean_score_pos=float(s_tr[y_tr == 1].mean()),
                                       mean_score_neg=float(s_tr[y_tr == 0].mean()))
            # 候補の漏れ: 学習の行のうち fold k の問い合わせの窓に入る割合
            w = in_window(rows.mass.values[tr], centers[k])
            lab = rows.label.values[tr]
            rec["train_rows_in_fold_windows"] = dict(pos=float(w[lab == 1].mean()), neg=float(w[lab == 0].mean()),
                                                     all=float(w.mean()))
            # 予測: プール全行
            in_tr_pid = set(rows.row.values[tr & (rows.source.values == "pool")].tolist())
            ttr = tr & (rows.source.values == "tier")
            tr_tier_keys = set(rows.canon_key[ttr].astype(str)) | set(rows.raw_key[ttr].astype(str))
            tr_tier_keys.discard("")
            sp = np.full(len(pool), np.nan); sp[okp] = bst.predict(Xp[np.where(okp)[0]])
            dfp = pd.DataFrame(dict(source="pool", key=pool.key.values, smiles=pool.smiles.values, pid=pool.pid.values,
                                    query_fold=k, model_fold=k, np_score=sp,
                                    in_train=np.isin(pool.pid.values, list(in_tr_pid)) | pool.key.astype(str).isin(tr_tier_keys).values,
                                    in_window=in_window(pool.mass.values, centers[k])))
            q = pc[pc.query_fold == k]; qi = q.index.values
            sc = np.full(len(q), np.nan); okq = okc[qi]; sc[okq] = bst.predict(Xc[qi[okq]])
            tr_keys = set(rows.canon_key[tr].astype(str)) | set(rows.raw_key[tr].astype(str))
            dfc = pd.DataFrame(dict(source="pubchem_b8", key=q.key.values, smiles=q.smiles.values, pid=-1,
                                    query_fold=k, model_fold=k, np_score=sc,
                                    in_train=q.key.astype(str).isin(tr_keys).values, in_window=True))
            parts += [dfp, dfc]
            rec["pred_rows"] = dict(pool=int(len(dfp)), pool_in_window=float(dfp.in_window.mean()),
                                    pool_in_train=float(dfp.in_train.mean()), pubchem_b8=int(len(dfc)),
                                    pubchem_b8_in_train=float(dfc.in_train.mean()) if len(dfc) else None)
        else:
            rec["note"] = "fold-0: 学習して保存するだけ（予測も評価もしない）"
        ovp = out / f"overlap_fold{k}.json"
        rec["overlap_4keys"] = json.loads(ovp.read_text())["overlap"]
        rep["folds"][str(k)] = rec
        log(f"fold {k}: {json.dumps(rec, ensure_ascii=False)}")
    pred = pd.concat(parts, ignore_index=True)
    check_pred_folds(pred)
    pred.to_parquet(out / "pred_dev.parquet", index=False)
    rep["pred_dev"] = dict(n_rows=int(len(pred)), n_fold0_query_rows=int((pred.query_fold == 0).sum()),
                           model_fold_eq_query_fold=bool((pred.model_fold == pred.query_fold).all()),
                           n_nan=int(pred.np_score.isna().sum()))
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    log("report", json.dumps(rep["pred_dev"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "featurize", "train", "predict"])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    log = log_to(a.out_dir / f"{a.stage}.log")
    import rdkit, lightgbm
    log(f"stage {a.stage} rdkit {rdkit.__version__} lightgbm {lightgbm.__version__}")
    if a.stage == "prepare":
        stage_prepare(a.out_dir, a.workers, log)
    elif a.stage == "featurize":
        stage_featurize(a.out_dir, a.workers, log)
    elif a.stage == "train":
        stage_train(a.out_dir, log)
    else:
        stage_predict(a.out_dir, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
