#!/usr/bin/env python3
"""
Issue #236（playbook §5 11/26 の前倒し）: dev の分子（fold 1〜4）だけで、自前の候補集合が正解を含む割合（recall）を型ごとに数える。

定義は #236 issuecomment-7150781116（数える前に固定）のとおり。要点:
  - 行 = dev の一覧（docs/bench/dev_list.csv.gz、make_lists.py が作る）の 1 構造。型 = B8 の 20 集合の 5 型（used_b8 = 1）＋型なし
  - 正解のキー = mrr25.to_key（metric と同じ）。B8 の行は集合の truth.csv、型なしは train_structs.smiles
  - 候補集合: (i) プールの ±10 ppm 窓（pool remove の型は集合の drop_keys を除く、run.visible_window と同じ）、
    (ii) PubChem 層の ±10 ppm 窓（層そのもの）、(iii) 和集合
  - 窓の中心: B8 の行は集合の pc_logits.npz の __targets__、型なしは train の 10 種の付加イオンのスペクトルの
    chem.neutral_mass の中央値（無ければ全スペクトルの有限値の中央値、それも無ければ中心なし＝recall なし）
  - 照合の絞り込み: SMILES の文字列から数えた重原子の組成（H と電荷を除く）が正解と同じ候補だけ、さらに重原子の
    グラフの WL の hash（結合次数・電荷・H・立体を見ない。金属などを含む・読めない分子は 0 =「どれとも両立」）が
    両立する候補だけ to_key を計算する。to_key が同じなら組成も重原子のグラフも同じなので、絞り込みで正解は落ちない。
    --check-wl で dev の 1 集合（b8_f1_a_t10）の組成だけ／組成＋WL の一致数が同じことを assert する

holdout（fold-0）の正解を読む経路は無い。入力の構造に fold-0 の構造（data/ext/foldsafe/split.parquet）や
fold が 1〜4 でない行、B8 の 20 集合でない集合名が入ってきたら止まる。

    .venv/bin/python scripts/bench/candidate_recall.py --build-comp   # PubChem 層の組成の hash（一度だけ、scratch に）
    .venv/bin/python scripts/bench/candidate_recall.py --check-wl     # b8_f1_a_t10 で WL の絞り込みの確かめ
    .venv/bin/python scripts/bench/candidate_recall.py --rows all     # 全 dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/validate", "scripts/local_eval", "scripts/cv", "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / _p))

PPM = 10.0
DEV_LIST = ROOT / "docs/bench/dev_list.csv.gz"
TIER = ROOT / "data/ext/pubchem_tier"
POOL = ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
TRAIN = ROOT / "data/train.parquet"
SPLIT0 = ROOT / "data/ext/foldsafe/split.parquet"
SCRATCH = Path(os.environ.get("I236_SCRATCH", "/Users/taichi/.hermes/profiles/massspec/cache/scratch/i236"))
B8_RE = re.compile(r"^b8_f[1-4]_[a-e]_t10$")
DEV_FOLDS = {1, 2, 3, 4}
ROW_TYPES = ("a", "b", "c", "d", "e", "none")
CONTROL = "submission_score+prior+frag.csv"
CONTROL_PC = "submission_score+prior+frag+pubchem.csv"
WL_CHECK_SET = "b8_f1_a_t10"
ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"}   # headroom.graph_sig と同じ

# ---------------------------------------------------------------- 純粋な関数（テストあり）
_TOK = re.compile(r"\[[^\]]*\]|Br|Cl|[BCNOPSFI]|[bcnops]")
_BRA = re.compile(r"\[\d*([A-Z][a-z]?|[a-z][a-z]?)")


def heavy_comp_str(smi: str) -> tuple:
    """SMILES の文字列から重原子の組成（H・電荷・同位体の数字を除く元素の数）を数える。RDKit で読まない
    （sanitize=False の RDKit の数え方と一致することを層の標本で assert する）。"""
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
    """組成 → int64（決まった hash。衝突は絞り込みを緩めるだけで、漏れにはならない）。"""
    s = ",".join(f"{e}{n}" for e, n in comp).encode()
    return int.from_bytes(hashlib.blake2b(s, digest_size=8).digest(), "little", signed=True)


def wl_hash(smi: str, iters: int = 3) -> int:
    """重原子のグラフの WL の hash（元素と隣接だけ。結合次数・電荷・H・立体を見ない）。プロセスをまたいで同じ値
    （int の tuple の hash だけを使う）。ORGANIC の外の元素を含む・読めない分子は 0（どの候補とも両立とする）。"""
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


def compatible(c_hash: np.ndarray, c_wl: np.ndarray | None, thashes: set, twls: set | None) -> np.ndarray:
    """候補の mask: 組成の hash が正解のどれかと同じ、かつ（WL を使うなら）WL が両立（どちらかが 0 なら両立）。"""
    m = np.isin(c_hash, list(thashes))
    if c_wl is None or twls is None or 0 in twls:
        return m
    return m & ((c_wl == 0) | np.isin(c_wl, list(twls)))


def window_bounds(mass: np.ndarray, target: float, ppm: float = PPM) -> tuple[int, int]:
    """run.Pool.window・pubchem.Tier.window と同じ ±ppm の [a, b)。"""
    tol = target * ppm * 1e-6
    return int(np.searchsorted(mass, target - tol, "left")), int(np.searchsorted(mass, target + tol, "right"))


def center_mass(nm, adducts, t10: tuple) -> float:
    """型なしの行の窓の中心: 10 種の付加イオンのスペクトルの中性質量の中央値、無ければ全スペクトルの有限値の中央値、
    それも無ければ NaN（中心なし）。"""
    nm = np.asarray(nm, np.float64)
    ok = np.isfinite(nm)
    in10 = np.isin(np.asarray(adducts, dtype=object).astype(str), list(t10)) & ok
    if in10.any():
        return float(np.median(nm[in10]))
    if ok.any():
        return float(np.median(nm[ok]))
    return float("nan")


def summarize(rows: list[dict], n_dev: int) -> dict:
    """行（type・found_i・found_ii・n_i・n_ii・has_center）→ 6 行の表。5 型の和＋型なし＝ n_dev を assert する。"""
    out = {}
    for t in ROW_TYPES:
        sel = [r for r in rows if r["type"] == t]
        n = len(sel)
        if n == 0:
            out[t] = dict(n=0)
            continue
        fi = sum(bool(r["found_i"]) for r in sel)
        fii = sum(bool(r["found_ii"]) for r in sel)
        fiii = sum(bool(r["found_i"] or r["found_ii"]) for r in sel)
        out[t] = dict(n=n, used_b8=int(t != "none"), recall_i=round(fi / n, 4), recall_ii=round(fii / n, 4),
                      recall_iii=round(fiii / n, 4), k_i=fi, k_ii=fii, k_iii=fiii,
                      mean_n_i=round(float(np.mean([r["n_i"] for r in sel])), 1),
                      mean_n_ii=round(float(np.mean([r["n_ii"] for r in sel])), 1),
                      mean_n_i_plus_ii=round(float(np.mean([r["n_i"] + r["n_ii"] for r in sel])), 1),
                      n_no_center=sum(not r["has_center"] for r in sel))
    tot = sum(out[t]["n"] for t in ROW_TYPES)
    assert tot == n_dev, f"5 型＋型なし {tot} ≠ dev {n_dev}"
    return out


def guard_dev_only(dev, fold0_keys: set) -> None:
    """holdout（fold-0）の構造が入っていないことの関門。"""
    assert set(dev.fold.astype(int)) <= DEV_FOLDS, f"fold 1〜4 でない行がある: {set(dev.fold)}"
    bad = (set(dev.inchikey14.astype(str)) | set(dev.key.astype(str)) | set(dev.scorer_key.astype(str))) & fold0_keys
    assert not bad, f"fold-0 の構造が入っている: {len(bad)}"
    sets = set(dev.b8_set.dropna().astype(str))
    assert all(B8_RE.match(s) for s in sets), f"B8 の 20 集合でない集合名: {sets}"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- PubChem 層の組成の hash（一度だけ）
def _tier_slice(a: int, b: int) -> list[str]:
    off = np.load(TIER / "pc_off.npy", mmap_mode="r")
    buf = np.load(TIER / "pc_smiles.npy", mmap_mode="r")
    o = np.asarray(off[a:b + 1]).astype(np.int64)
    raw = bytes(buf[o[0]:o[-1]])
    base = o[0]
    return [raw[o[i] - base:o[i + 1] - base].decode("ascii") for i in range(b - a)]


def _tier_chunk(args):
    a, b, out_path = args
    smis = _tier_slice(a, b)
    h = np.fromiter((comp_hash(heavy_comp_str(s)) for s in smis), np.int64, count=len(smis))
    mm = np.load(out_path, mmap_mode="r+")
    mm[a:b] = h
    mm.flush()
    return b - a


def build_comp(workers: int) -> dict:
    from multiprocessing import get_context
    mass = np.load(TIER / "pc_mass.npy", mmap_mode="r")
    n = len(mass)
    out = SCRATCH / "tier_comp_hash.npy"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.int64, shape=(n,))
    del mm
    step = 1_000_000
    jobs = [(a, min(a + step, n), str(out)) for a in range(0, n, step)]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as p:
        for _ in p.imap_unordered(_tier_chunk, jobs):
            pass
    secs = time.time() - t0
    # 標本で RDKit の数え方と一致することを確かめる
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(n, size=100_000, replace=False))
    hh = np.load(out, mmap_mode="r")
    off = np.load(TIER / "pc_off.npy", mmap_mode="r")
    buf = np.load(TIER / "pc_smiles.npy", mmap_mode="r")
    checked = bad = 0
    for i in idx:
        s = bytes(buf[int(off[i]):int(off[i + 1])]).decode("ascii")
        m = Chem.MolFromSmiles(s, sanitize=False)
        if m is None:
            continue
        r = tuple(sorted(Counter(x.GetSymbol() for x in m.GetAtoms() if x.GetAtomicNum() > 1).items()))
        checked += 1
        if comp_hash(r) != int(hh[i]):
            bad += 1
    assert bad == 0, f"文字列の組成が RDKit と違う: {bad}/{checked}"
    res = dict(n=n, secs=round(secs, 1), sample=len(idx), sample_rdkit_parsed=checked, sample_mismatch=bad,
               file=str(out))
    (SCRATCH / "tier_comp_hash.json").write_text(json.dumps(res, indent=1))
    return res


def _tier_wl_chunk(args):
    a, b, out_path = args
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    smis = _tier_slice(a, b)
    w = np.fromiter((wl_hash(s) for s in smis), np.int64, count=len(smis))
    mm = np.load(out_path, mmap_mode="r+")
    mm[a:b] = w
    mm.flush()
    return b - a


def build_wl(workers: int) -> dict:
    """PubChem 層の全行の WL の hash（scratch の tier_wl_hash.npy）。窓ごとに計算すると組成が同じ候補が窓あたり
    数千あり、22 万分子では重複が多すぎるため、層の全体で 1 回だけ計算する。"""
    from multiprocessing import get_context
    n = len(np.load(TIER / "pc_mass.npy", mmap_mode="r"))
    out = SCRATCH / "tier_wl_hash.npy"
    tmp = SCRATCH / "tier_wl_hash.tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int64, shape=(n,))
    del mm
    step = 500_000
    jobs = [(a, min(a + step, n), str(tmp)) for a in range(0, n, step)]
    t0 = time.time()
    done = 0
    with get_context("spawn").Pool(workers) as p:
        for k in p.imap_unordered(_tier_wl_chunk, jobs):
            done += k
            if done % 10_000_000 < step:
                print(f"  wl {done}/{n} {time.time() - t0:.0f}s", flush=True)
    assert done == n
    os.replace(tmp, out)
    res = dict(n=n, secs=round(time.time() - t0, 1), file=str(out),
               n_zero=int((np.load(out, mmap_mode="r") == 0).sum()))
    (SCRATCH / "tier_wl_hash.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- 行の用意
def load_rows(only_set: str | None = None) -> tuple[list[dict], int]:
    import pandas as pd
    import library as L
    import holdout_expand as HE
    from casmi import chem
    dev = pd.read_csv(DEV_LIST, dtype={"identity_group": str})
    sp = pd.read_parquet(SPLIT0)
    guard_dev_only(dev, set(sp.inchikey14.astype(str)) | set(sp.key.astype(str)))
    n_dev = len(dev)
    rows: list[dict] = []
    b8 = dev[dev.used_b8 == 1]
    for h, g in b8.groupby("b8_set"):
        if only_set and h != only_set:
            continue
        assert B8_RE.match(h), h
        tr = pd.read_csv(ROOT / f"outputs/local_eval/{h}/truth.csv").set_index("molecule_id")
        zf = np.load(ROOT / f"outputs/local_eval/{h}/pc_logits.npz", allow_pickle=True)
        targets = {str(m): float(t) for m, t in zf["__targets__"]}
        rule = L.hide_rule(h)
        hidden = sorted(L.drop_keys(h)) if rule.get("pool") == "remove" else []
        assert set(tr.index.astype(str)) == set(g.b8_molecule_id.astype(str)), f"{h}: truth.csv と一覧の分子が違う"
        for r in g.itertuples():
            mid = str(r.b8_molecule_id)
            assert str(tr.loc[mid, "inchikey14"]) == str(r.b8_truth_key)
            rows.append(dict(inchikey14=r.inchikey14, type=str(r.b8_type), set=h, molecule_id=mid,
                             truth=str(tr.loc[mid, "smiles"]), target=targets.get(mid, float("nan")), hidden=hidden))
    if only_set:
        return rows, len(rows)
    import pyarrow.parquet as pq
    none = dev[dev.used_b8 == 0]
    want = set(none.inchikey14.astype(str))
    ts = pd.read_parquet(TSTRUCT, columns=["inchikey14", "smiles"])
    smi = dict(zip(ts.inchikey14.astype(str), ts.smiles.astype(str)))
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    parts = []
    for b in pf.iter_batches(batch_size=8192, columns=["inchikey14", "precursor_mz", "adduct"]):
        d = b.to_pandas()
        d["inchikey14"] = d.inchikey14.astype(str)
        d = d[d.inchikey14.isin(want)]
        if len(d):
            d["adduct"] = d.adduct.astype(str)
            d["nm"] = chem.neutral_mass(d.precursor_mz.values.astype(np.float64), d.adduct.values)
            parts.append(d[["inchikey14", "adduct", "nm"]])
    meta = pd.concat(parts, ignore_index=True)
    cen = {k: center_mass(g.nm.values, g.adduct.values, HE.TEST10) for k, g in meta.groupby("inchikey14", sort=False)}
    for k in none.inchikey14.astype(str):
        rows.append(dict(inchikey14=k, type="none", set=None, molecule_id=None, truth=smi[k],
                         target=cen.get(k, float("nan")), hidden=[]))
    return rows, n_dev


# ---------------------------------------------------------------- 並列の道具
def _keys_job(items):
    """items = [(row_id, truth, cand_i, cand_ii)] → [(row_id, n_truth_keys, found_i, found_ii, n_tokey)]。"""
    import mrr25
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for rid, truth, ci, cii in items:
        tk = mrr25.truth_keys(truth)
        fi = bool(tk) and any(mrr25.to_key(s) in tk for s in ci)
        fii = bool(tk) and any(mrr25.to_key(s) in tk for s in cii)
        out.append((rid, len(tk), fi, fii, len(ci) + len(cii)))
    return out


def _pmap(fn, jobs, workers):
    from multiprocessing import get_context
    if not jobs:
        return []
    with get_context("spawn").Pool(workers) as p:
        return list(p.imap(fn, jobs))


# ---------------------------------------------------------------- 照合（主プロセスで窓と絞り込み、WL と to_key は並列）
def match_rows(rows: list[dict], workers: int, use_wl: bool, log=print) -> tuple[list[dict], dict]:
    import pandas as pd
    t0 = time.time()
    meta = pd.read_parquet(POOL, columns=["key", "smiles", "mass"])
    order = np.argsort(meta["mass"].values, kind="stable")
    pkey = meta["key"].values[order].astype(str)
    psmi = np.asarray(meta["smiles"].values[order], dtype=object)
    pmass = meta["mass"].values[order].astype(np.float64)
    phash = np.fromiter((comp_hash(heavy_comp_str(s)) for s in psmi), np.int64, count=len(psmi))
    tmass = np.load(TIER / "pc_mass.npy", mmap_mode="r")
    thash = np.load(SCRATCH / "tier_comp_hash.npy", mmap_mode="r")
    assert len(thash) == len(tmass), "PubChem 層の組成の hash の長さが層と違う（--build-comp をやり直す）"
    log(f"  pool/tier loaded {time.time() - t0:.0f}s")
    for r in rows:   # 正解の組成と WL
        ts_ = [s.strip() for s in r["truth"].split(",") if s.strip()]
        r["thashes"] = {comp_hash(heavy_comp_str(s)) for s in ts_}
        r["twls"] = {wl_hash(s) for s in ts_} if use_wl else None
    log(f"  truth comp/wl {time.time() - t0:.0f}s")
    if use_wl:
        twl = np.load(SCRATCH / "tier_wl_hash.npy", mmap_mode="r")
        assert len(twl) == len(tmass), "PubChem 層の WL の hash の長さが層と違う（--build-wl をやり直す）"
        pwl = np.fromiter((wl_hash(s) for s in psmi), np.int64, count=len(psmi))
        log(f"  pool WL {time.time() - t0:.0f}s")
    stats = dict(same_comp_pool=0, same_comp_tier=0)
    for r in rows:   # 窓 → 組成 →（WL）の順に絞る。to_key を計算する候補だけを残す
        r["has_center"] = bool(np.isfinite(r["target"]))
        r["n_i"] = r["n_ii"] = 0
        r["pi"] = np.zeros(0, np.int64)
        r["ti"] = np.zeros(0, np.int64)
        if not r["has_center"]:
            continue
        a, b = window_bounds(pmass, r["target"])
        idx = np.arange(a, b)
        if r["hidden"]:
            idx = idx[~np.isin(pkey[idx], r["hidden"])]
        r["n_i"] = int(len(idx))
        m = compatible(phash[idx], None, r["thashes"], None)
        stats["same_comp_pool"] += int(m.sum())
        if use_wl:
            m = compatible(phash[idx], pwl[idx], r["thashes"], r["twls"])
        r["pi"] = idx[m]
        a, b = window_bounds(tmass, r["target"])
        r["n_ii"] = int(max(0, b - a))
        if b > a:
            hh = np.asarray(thash[a:b])
            m = compatible(hh, None, r["thashes"], None)
            stats["same_comp_tier"] += int(m.sum())
            if use_wl:
                m = compatible(hh, np.asarray(twl[a:b]), r["thashes"], r["twls"])
            r["ti"] = a + np.nonzero(m)[0]
    log(f"  windows {time.time() - t0:.0f}s")
    stats["to_key_pool"] = int(sum(len(r["pi"]) for r in rows))
    stats["to_key_tier"] = int(sum(len(r["ti"]) for r in rows))
    off = np.load(TIER / "pc_off.npy", mmap_mode="r")
    buf = np.load(TIER / "pc_smiles.npy", mmap_mode="r")
    items = []
    for j, r in enumerate(rows):   # to_key を並列で（候補の SMILES を渡す）
        ci = [psmi[i] for i in r["pi"].tolist()]
        cii = [bytes(buf[int(off[i]):int(off[i + 1])]).decode("ascii") for i in r["ti"].tolist()]
        items.append((j, r["truth"], ci, cii))
    step = max(1, min(100, len(items) // (workers * 8) or 1))
    chunks = [items[i:i + step] for i in range(0, len(items), step)]
    res = [x for part in _pmap(_keys_job, chunks, workers) for x in part]
    log(f"  to_key {time.time() - t0:.0f}s")
    out = []
    for rid, ntk, fi, fii, ntok in res:
        r = rows[rid]
        out.append(dict(inchikey14=r["inchikey14"], type=r["type"], set=r["set"], molecule_id=r["molecule_id"],
                        has_center=r["has_center"], target=r["target"], n_i=r["n_i"], n_ii=r["n_ii"],
                        found_i=bool(fi), found_ii=bool(fii), n_truth_keys=ntk, n_tokey=ntok))
    assert len(out) == len(rows)
    return out, stats


def top25_check(res: list[dict]) -> dict:
    """B8 の集合: 対照の上位 25 で見つかった正解は (i) に、+pubchem の上位 25 で見つかった正解は (iii) にあること。"""
    import mrr25 as M
    by: dict = {}
    for r in res:
        if r["set"]:
            by.setdefault(r["set"], {})[r["molecule_id"]] = r
    n1 = n2 = 0
    for h, d in by.items():
        assert B8_RE.match(h), h
        truth = M.read_truth(ROOT / f"outputs/local_eval/{h}/truth.csv")
        for name, need in ((CONTROL, "i"), (CONTROL_PC, "iii")):
            preds = M.read_submission(ROOT / f"outputs/local_eval/{h}/{name}")
            for mid, r in d.items():
                if M.reciprocal_rank(M.truth_keys(truth[mid]), preds.get(mid, []), 25) > 0:
                    ok = r["found_i"] if need == "i" else (r["found_i"] or r["found_ii"])
                    assert ok, f"{h} {mid}: {name} の上位 25 にあるのに ({need}) に無い"
                    if need == "i":
                        n1 += 1
                    else:
                        n2 += 1
    return dict(sets=len(by), control_top25_hits_in_i=n1, pubchem_top25_hits_in_iii=n2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-comp", action="store_true")
    ap.add_argument("--build-wl", action="store_true", help="PubChem 層の全行の WL の hash（--build-comp の後）")
    ap.add_argument("--check-wl", action="store_true", help=f"{WL_CHECK_SET} で組成だけ／組成＋WL の一致数を比べる")
    ap.add_argument("--rows", default=None, help="all = 全 dev、または B8 の集合名 1 つ")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/i236_recall"))
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    if a.build_comp:
        print(json.dumps(build_comp(a.workers)))
        return
    if a.build_wl:
        print(json.dumps(build_wl(a.workers)))
        return
    log = lambda s: print(s, flush=True)
    if a.check_wl:
        rows, _ = load_rows(WL_CHECK_SET)
        r_wl, st_wl = match_rows([dict(r) for r in rows], a.workers, True, log)
        r_no, st_no = match_rows([dict(r) for r in rows], a.workers, False, log)
        k = lambda rs, c: sum(bool(r[c]) for r in rs)
        key = lambda rs: sorted((r["molecule_id"], bool(r["found_i"]), bool(r["found_ii"])) for r in rs)
        res = dict(set=WL_CHECK_SET, n=len(rows), wl=dict(found_i=k(r_wl, "found_i"), found_ii=k(r_wl, "found_ii")),
                   comp_only=dict(found_i=k(r_no, "found_i"), found_ii=k(r_no, "found_ii")),
                   per_molecule_equal=key(r_wl) == key(r_no), stats_wl=st_wl, stats_comp_only=st_no,
                   secs=round(time.time() - t0, 1))
        (out / "wl_check.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res))
        assert res["wl"] == res["comp_only"] and res["per_molecule_equal"], res
        return
    only = None if a.rows in (None, "all") else a.rows
    if only:
        assert B8_RE.match(only), only
    cache = out / ("rows.pkl" if not only else f"rows_{only}.pkl")
    dsha = sha256(DEV_LIST)
    c = pickle.loads(cache.read_bytes()) if cache.is_file() else None
    if c is not None and c["dev_list_sha256"] == dsha:
        rows, n_dev = c["rows"], c["n_dev"]
    else:
        rows, n_dev = load_rows(only)
        cache.write_bytes(pickle.dumps(dict(dev_list_sha256=dsha, rows=rows, n_dev=n_dev)))
    log(f"rows {len(rows)} (dev {n_dev}) prepared in {time.time() - t0:.0f}s")
    res, stats = match_rows(rows, a.workers, True, log)
    import pandas as pd
    pd.DataFrame(res).to_parquet(out / ("per_molecule.parquet" if not only else f"per_molecule_{only}.parquet"), index=False)
    chk = top25_check(res)
    summ = dict(issue=236, definitions="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/236#issuecomment-7150781116",
                ppm=PPM, dev_list_sha256=dsha, table=summarize(res, n_dev) if not only else None,
                top25_check=chk, filter_stats=stats, n_rows=len(res),
                n_truth_without_key=sum(r["n_truth_keys"] == 0 for r in res), secs=round(time.time() - t0, 1))
    if only:
        k = lambda c_: sum(bool(r[c_]) for r in res)
        summ["set_counts"] = dict(n=len(res), found_i=k("found_i"), found_ii=k("found_ii"))
    (out / ("recall_summary.json" if not only else f"recall_{only}.json")).write_text(
        json.dumps(summ, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
