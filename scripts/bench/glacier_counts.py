#!/usr/bin/env python3
"""
Issue #242（2026-11-25 昼 項目 5）: GLACIER の予測を dev の候補に付けるときの本数と、MassSpecGym との重なりを数える。

定義は #242 issuecomment-7153124801（数える前に固定）のとおり。要点:
  - dev = docs/bench/dev_list.csv.gz（#236）。(A) used_b8 = 1 の 3,901 分子、(B) 全 220,443 構造
  - 窓の中心・大きさは #236 の outputs/local_eval/i236_recall/per_molecule.parquet（target・n_i・n_ii）
  - 区分 = gl_runner.py と同じ (付加イオン, 装置の印, CE の 5 eV の区切り)。被覆 (r) [M+H]+・[M+Na]+、(h) [M+H]+ だけ。
    CE が NaN のスペクトルは 20・40・60 eV の 3 予測
  - 候補: L60 版 min(60, n_i)、窓版 n_i + n_ii。予測の本数 = Σ 候補 × 区分。窓版は区分ごとの窓の和集合（重複なし）も
  - 漏れ: MSG（data/ext/msg/MassSpecGym.tsv）の train ∪ val・全体（参考: simulation_challenge の train ∪ val）×
    生 InChIKey14・正規化キー（mrr25.to_key）。dev の正解と、窓版の候補（プール・PubChem 層、正解/不正解別）
  - 測定の入力 2,000 本（--sample）: (A) の窓版の (分子, 候補, 区分(r)) から default_rng(20261125) でプール 1,000・層 1,000

区分の関数（instr_family・spectrum_ce・ce_bucket・被覆）は gl_runner.py の同名の関数と同じ規則を自分で書いたもの
（gl_runner.py はライセンスの記載が無いのでリポジトリに写さない。一致は scratch で gl_runner の関数と
B8 の 20 集合の全スペクトル＋train.parquet の 1/10 の batch で比べ、274,157 スペクトルで不一致 0。#242 に記録）。

    .venv/bin/python scripts/bench/glacier_counts.py --dev-list <dev_list.csv.gz> --stage msg      # MSG と当たる行
    .venv/bin/python scripts/bench/glacier_counts.py --dev-list <dev_list.csv.gz> --stage counts   # 本数と重なり
    .venv/bin/python scripts/bench/glacier_counts.py --dev-list <dev_list.csv.gz> --stage sample   # 測定の入力
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/validate", "scripts/local_eval", "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / _p))

DEV_LIST_SHA = "7a69f073fb94049440bd7fc52a4f3f9a116dd45e76a97498de94976ae2943b79"
PER_MOL = ROOT / "outputs/local_eval/i236_recall/per_molecule.parquet"
POOL = ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
TIER = ROOT / "data/ext/pubchem_tier"
TRAIN = ROOT / "data/train.parquet"
MSG = ROOT / "data/ext/msg/MassSpecGym.tsv"
SCRATCH = Path(os.environ.get("I242_SCRATCH", "/Users/taichi/.hermes/profiles/massspec/cache/scratch/i242"))
OUT = ROOT / "outputs/local_eval/i242_glacier"
PPM = 10.0
TOPN = 60
MASS_TOL_DA = 0.001
SEED = 20261125
N_SAMPLE = 1000                     # 層ごと（pool / tier）
SUM_CE = (20.0, 40.0, 60.0)
COV_R = ("[M+H]+", "[M+Na]+")
COV_H = ("[M+H]+",)
B8_RE = re.compile(r"^b8_f[1-4]_[a-e]_t10$")
TYPES = ("a", "b", "c", "d", "e", "none")


# ---------------------------------------------------------------- 区分（gl_runner.py と同じ規則、テストあり）
def instr_family(s) -> int:
    """0 timsTOF, 1 Orbitrap/FT, 2 TOF, 3 IT, 4 QQQ, 5 other（gl_runner.instr_family と同じ規則）。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 5
    t = str(s).lower()
    if "timstof" in t:
        return 0
    if any(w in t for w in ("orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive", "ft-icr", "fticr")):
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qit" in t or "lit" in t:
        return 3
    if "qq" in t or "quattro" in t or "triple" in t:
        return 4
    return 5


def instr_token(instrument) -> str:
    return "Orbitrap" if instr_family(instrument) == 1 else "QTOF"


def spectrum_ce(units, orig, ev) -> float:
    """units == 'NCE' → collision_energy_orig の数の平均、それ以外 → collision_energy_ev の平均（空なら NaN）。"""
    if units == "NCE":
        v = [float(x) for x in re.findall(r"[0-9]+\.?[0-9]*", str(orig))]
        return float(np.mean(v)) if v else float("nan")
    return float(np.mean(ev)) if ev is not None and len(ev) else float("nan")


def ce_bucket(ce: float) -> float:
    if not np.isfinite(ce):
        return float("nan")
    b = float(np.round(ce / 5.0) * 5.0)
    return 5.0 if b < 5 else b


def is_positive(mode) -> bool:
    return (mode == 1) or (isinstance(mode, str) and mode.lower() in ("positive", "pos", "+", "1"))


def spectrum_jobs(adduct, mode, instrument, units, orig, ev, cov) -> set:
    """1 スペクトル → 予測の区分の集合（被覆の外なら空、CE が NaN なら 20・40・60 の 3 つ）。"""
    if not (is_positive(mode) and adduct in cov):
        return set()
    ins = instr_token(instrument)
    b = ce_bucket(spectrum_ce(units, orig, ev))
    if np.isfinite(b):
        return {(adduct, ins, b)}
    return {(adduct, ins, c) for c in SUM_CE}


# ---------------------------------------------------------------- 組成（candidate_recall.heavy_comp_str〈#237〉と同じ）
_TOK = re.compile(r"\[[^\]]*\]|Br|Cl|[BCNOPSFI]|[bcnops]")
_BRA = re.compile(r"\[\d*([A-Z][a-z]?|[a-z][a-z]?)")


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


def union_length(intervals: list) -> int:
    """[a, b) の区間の和集合の長さ。"""
    if not intervals:
        return 0
    iv = sorted((a, b) for a, b in intervals if b > a)
    tot, ca, cb = 0, None, None
    for a, b in iv:
        if cb is None or a > cb:
            if cb is not None:
                tot += cb - ca
            ca, cb = a, b
        else:
            cb = max(cb, b)
    if cb is not None:
        tot += cb - ca
    return tot


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 入力
def tier_arrays():
    return (np.load(TIER / "pc_mass.npy", mmap_mode="r"), np.load(TIER / "pc_off.npy", mmap_mode="r"),
            np.load(TIER / "pc_smiles.npy", mmap_mode="r"))


def tier_slice(off, buf, a: int, b: int) -> list[str]:
    o = np.asarray(off[a:b + 1]).astype(np.int64)
    raw = bytes(buf[o[0]:o[-1]])
    return [raw[o[i] - o[0]:o[i + 1] - o[0]].decode("ascii") for i in range(b - a)]


def load_pool():
    import pandas as pd
    meta = pd.read_parquet(POOL, columns=["key", "smiles", "mass"])
    order = np.argsort(meta["mass"].values, kind="stable")          # run.Pool と同じ並び
    return (meta["key"].values[order].astype(str), meta["smiles"].values[order].astype(str),
            meta["mass"].values[order].astype(np.float64))


def load_msg():
    """MSG の一意な SMILES ごとに (smiles, ik14, fold の集合, sim の fold の集合)。"""
    import pandas as pd
    d = pd.read_csv(MSG, sep="\t", usecols=["smiles", "inchikey", "fold", "simulation_challenge"], dtype=str)
    d["ik14"] = d.inchikey.str[:14]
    d["sim"] = d.simulation_challenge.astype(str).str.lower().isin(("true", "t", "1"))
    g = d.groupby("smiles", sort=True)
    out = []
    for smi, x in g:
        out.append(dict(smiles=smi, ik14=sorted(set(x.ik14)), folds=sorted(set(x.fold)),
                        sim_folds=sorted(set(x.fold[x.sim]))))
    return out, len(d)


# ---------------------------------------------------------------- 段: msg（MSG の構造と当たる行）
ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"}   # candidate_recall（#237）と同じ


def wl_hash(smi: str, iters: int = 3) -> int:
    """重原子のグラフの WL の hash（candidate_recall.wl_hash〈#237〉と同じ。元素と隣接だけ、0 = どれとも両立）。
    生 InChIKey14 または to_key が同じなら重原子のグラフも同じなので、この絞り込みで当たりは落ちない。"""
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


def _msg_job(items):
    """MSG の SMILES → (smiles, ExactMolWt, 組成, WL, to_key)。"""
    import mrr25
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in items:
        m = Chem.MolFromSmiles(s)
        out.append((s, float(ExactMolWt(m)) if m is not None else float("nan"), heavy_comp_str(s), wl_hash(s),
                    mrr25.to_key(s) or ""))
    return out


_W = {}


def _scan_init():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _W["pool"] = load_pool()[1]
    _W["tier"] = tier_arrays()


def _scan_job(jobs):
    """[(tag, a, b, comp, wls)] → 当たりの候補 [(tag, idx, raw ik14, to_key)]。行 [a, b) のうち組成が同じ、
    WL が両立（どちらかが 0 なら両立）する行だけ RDKit の InChIKey と to_key を計算する。"""
    import mrr25
    from rdkit import Chem
    out, n_comp, n_wl = [], 0, 0
    for tag, a, b, comp, wls in jobs:
        if tag == "pool":
            smis = _W["pool"][a:b]
        else:
            _, off, buf = _W["tier"]
            smis = tier_slice(off, buf, a, b)
        for j, s in enumerate(smis):
            if heavy_comp_str(s) != comp:
                continue
            n_comp += 1
            if 0 not in wls:
                w = wl_hash(s)
                if w != 0 and w not in wls:
                    continue
            n_wl += 1
            m = Chem.MolFromSmiles(s)
            ik = (Chem.MolToInchiKey(m) or "")[:14] if m is not None else ""
            out.append((tag, a + j, ik, mrr25.to_key(s) or ""))
    return [("__n__", n_comp, n_wl, "")] + out


def pmap(fn, jobs, workers, initializer=None):
    from multiprocessing import get_context
    if not jobs:
        return []
    with get_context("spawn").Pool(workers, initializer=initializer) as p:
        return [r for part in p.imap_unordered(fn, jobs) for r in part]


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def merge_intervals(iv: list) -> list:
    """[(a, b)] → 重ならない [a, b) の並び。"""
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b > a]


def stage_msg(workers: int) -> dict:
    t0 = time.time()
    msg, n_rows = load_msg()
    info = pmap(_msg_job, chunks([m["smiles"] for m in msg], 500), workers)
    by = {s: (mw, comp, wl, k) for s, mw, comp, wl, k in info}
    for m in msg:
        m["mw"], m["comp"], m["wl"], m["key"] = by[m["smiles"]]
    sets = dict(raw_tv=set(), raw_all=set(), key_tv=set(), key_all=set(), raw_simtv=set(), key_simtv=set())
    for m in msg:
        tv = bool(set(m["folds"]) & {"train", "val"})
        stv = bool(set(m["sim_folds"]) & {"train", "val"})
        for k in m["ik14"]:
            sets["raw_all"].add(k)
            if tv:
                sets["raw_tv"].add(k)
            if stv:
                sets["raw_simtv"].add(k)
        if m["key"]:
            sets["key_all"].add(m["key"])
            if tv:
                sets["key_tv"].add(m["key"])
            if stv:
                sets["key_simtv"].add(m["key"])
    t1 = time.time()
    # 組成ごとに、MSG の構造の質量 ±0.001 Da の行の区間をまとめる（同じ行を 2 回調べない）
    _, _, pmass = load_pool()
    tm, _, _ = tier_arrays()
    by_comp = defaultdict(list)
    for m in msg:
        if np.isfinite(m["mw"]):
            by_comp[m["comp"]].append(m)
    jobs, n_rows_scan = [], {"pool": 0, "tier": 0}
    for comp, ms in by_comp.items():
        wls = frozenset(m["wl"] for m in ms)
        for tag, mass in (("pool", pmass), ("tier", tm)):
            iv = [(int(np.searchsorted(mass, m["mw"] - MASS_TOL_DA, "left")),
                   int(np.searchsorted(mass, m["mw"] + MASS_TOL_DA, "right"))) for m in ms]
            for a, b in merge_intervals(iv):
                n_rows_scan[tag] += b - a
                for c0 in range(a, b, 20000):
                    jobs.append((tag, c0, min(b, c0 + 20000), comp, wls))
    t2 = time.time()
    batches = chunks(jobs, 8)
    res = pmap(_scan_job, batches, workers, initializer=_scan_init)
    hits = {"pool": [], "tier": []}
    n_comp = n_wl = 0
    for tag, i, ik, k in res:
        if tag == "__n__":
            n_comp += i
            n_wl += ik
            continue
        f = dict(raw_tv=ik in sets["raw_tv"], raw_all=ik in sets["raw_all"], key_tv=k in sets["key_tv"],
                 key_all=k in sets["key_all"], raw_simtv=ik in sets["raw_simtv"], key_simtv=k in sets["key_simtv"])
        if any(f.values()):
            hits[tag].append(dict(idx=int(i), ik14=ik, key=k, **f))
    for t in hits:
        hits[t].sort(key=lambda h: h["idx"])
        idx = [h["idx"] for h in hits[t]]
        assert len(idx) == len(set(idx)), f"{t}: 同じ行が 2 回"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with open(SCRATCH / "msg_hits.json", "w") as f:
        json.dump(dict(sets={k: sorted(v) for k, v in sets.items()}, hits=hits), f)
    summ = dict(msg_rows=n_rows, msg_unique_smiles=len(msg), msg_sha256=sha256(MSG), n_comp_groups=len(by_comp),
                n_struct={k: len(v) for k, v in sets.items()}, n_rows_scanned=n_rows_scan, n_jobs=len(jobs),
                n_comp_match=n_comp, n_rdkit=n_wl, n_hits={t: len(v) for t, v in hits.items()},
                hits_by_flag={t: {f: sum(h[f] for h in v) for f in ("raw_tv", "raw_all", "key_tv", "key_all",
                                                                      "raw_simtv", "key_simtv")}
                              for t, v in hits.items()},
                secs=dict(msg=round(t1 - t0, 1), plan=round(t2 - t1, 1), scan=round(time.time() - t2, 1)))
    json.dump(summ, open(SCRATCH / "msg_summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False))
    return summ


# ---------------------------------------------------------------- 段: counts
def load_dev(dev_list: Path):
    import pandas as pd
    assert sha256(dev_list) == DEV_LIST_SHA, "dev_list.csv.gz の sha256 が #236 のものと違う"
    dev = pd.read_csv(dev_list, dtype={"identity_group": str})
    pm = pd.read_parquet(PER_MOL)
    assert len(pm) == len(dev) == 220443
    b8 = pm[pm.type != "none"].copy()
    nn = pm[pm.type == "none"].copy()
    dv8 = dev[dev.used_b8 == 1].rename(columns={"b8_set": "set", "b8_molecule_id": "molecule_id"})
    b8 = b8.merge(dv8[["set", "molecule_id", "inchikey14", "key", "b8_type"]], on=["set", "molecule_id"],
                  how="left", suffixes=("", "_dev"), validate="one_to_one")
    assert b8.key.notna().all() and (b8.type == b8.b8_type).all() and (b8.inchikey14 == b8.inchikey14_dev).all()
    b8 = b8.drop(columns=["inchikey14_dev", "b8_type"])
    dv0 = dev[dev.used_b8 == 0][["inchikey14", "key"]]
    nn = nn.merge(dv0, on="inchikey14", how="left", validate="one_to_one")
    assert nn.key.notna().all()
    rows = pd.concat([b8, nn], ignore_index=True)
    assert len(rows) == 220443 and (rows.type != "none").sum() == 3901
    for s in rows.set.dropna().unique():
        assert B8_RE.match(s), s
    return rows


def categories(rows):
    """各行の区分の集合 (r)・(h)。(A) は集合の test.parquet、型なしは train.parquet（その構造の全スペクトル）。"""
    import pandas as pd
    import pyarrow.parquet as pq
    cols = ["adduct", "ionization_mode", "instrument_type", "collision_energy_orig", "collision_energy_ev",
            "collision_energy_orig_units"]
    cat_r, cat_h = [set() for _ in range(len(rows))], [set() for _ in range(len(rows))]
    n_spec = np.zeros(len(rows), np.int64)
    pos = {}
    for i, (t, s, mid, ik) in enumerate(zip(rows.type, rows.set, rows.molecule_id, rows.inchikey14)):
        pos[(s, mid) if t != "none" else ("none", ik)] = i

    def add(i, r):
        n_spec[i] += 1
        args = (r[0], r[1], r[2], r[5], r[3], r[4])
        cat_r[i] |= spectrum_jobs(*args, COV_R)
        cat_h[i] |= spectrum_jobs(*args, COV_H)

    for s in sorted(rows.set.dropna().unique()):
        d = pd.read_parquet(ROOT / f"data/holdouts/{s}/test.parquet", columns=["molecule_id"] + cols)
        for r in d.itertuples(index=False):
            add(pos[(s, str(r.molecule_id))], (r.adduct, r.ionization_mode, r.instrument_type,
                                               r.collision_energy_orig, r.collision_energy_ev,
                                               r.collision_energy_orig_units))
    want = set(rows.inchikey14[rows.type == "none"].astype(str))
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    for b in pf.iter_batches(batch_size=8192, columns=["inchikey14"] + cols):
        d = b.to_pandas()
        d = d[d.inchikey14.astype(str).isin(want)]
        for r in d.itertuples(index=False):
            add(pos[("none", str(r.inchikey14))], (r.adduct, r.ionization_mode, r.instrument_type,
                                                   r.collision_energy_orig, r.collision_energy_ev,
                                                   r.collision_energy_orig_units))
    return cat_r, cat_h, n_spec


def stage_counts(dev_list: Path) -> dict:
    import pandas as pd
    import mrr25
    import library as L
    t0 = time.time()
    rows = load_dev(dev_list)
    cat_r, cat_h, n_spec = categories(rows)
    t1 = time.time()
    pk, psmi, pmass = load_pool()
    tm, _, _ = tier_arrays()
    tgt = rows.target.values.astype(np.float64)
    ok = np.isfinite(tgt)
    tol = np.where(ok, tgt, 0) * PPM * 1e-6
    pa = np.where(ok, np.searchsorted(pmass, tgt - tol, "left"), 0)
    pb = np.where(ok, np.searchsorted(pmass, tgt + tol, "right"), 0)
    ta = np.where(ok, np.searchsorted(tm, tgt - tol, "left"), 0)
    tb = np.where(ok, np.searchsorted(tm, tgt + tol, "right"), 0)
    # 窓の大きさが #236 と同じこと（層は全行、プールは hidden の無い型で）
    assert (tb - ta == rows.n_ii.values).all(), "層の窓の大きさが #236 と違う"
    keep = rows.type.isin(["a", "b", "d", "none"]).values
    assert (pb - pa == rows.n_i.values)[keep].all(), "プールの窓の大きさが #236 と違う"
    hidden = {s: (L.drop_keys(s) if L.hide_rule(s).get("pool") == "remove" else set())
              for s in rows.set.dropna().unique()}
    n_i, n_ii = rows.n_i.values.astype(np.int64), rows.n_ii.values.astype(np.int64)
    jr = np.array([len(c) for c in cat_r], np.int64)
    jh = np.array([len(c) for c in cat_h], np.int64)
    l60 = np.minimum(TOPN, n_i)
    win = n_i + n_ii
    is_a = (rows.type != "none").values

    def block(mask):
        out = dict(n=int(mask.sum()), n_spec=int(n_spec[mask].sum()),
                   n_no_cov_r=int((jr[mask] == 0).sum()), n_no_cov_h=int((jh[mask] == 0).sum()),
                   mean_cat_r=round(float(jr[mask].mean()), 3), mean_cat_h=round(float(jh[mask].mean()), 3),
                   cands_l60=int(l60[mask].sum()), cands_win=int(win[mask].sum()),
                   cands_win_pool=int(n_i[mask].sum()), cands_win_tier=int(n_ii[mask].sum()),
                   pred_l60_r=int((l60 * jr)[mask].sum()), pred_l60_h=int((l60 * jh)[mask].sum()),
                   pred_win_r=int((win * jr)[mask].sum()), pred_win_h=int((win * jh)[mask].sum()))
        for cov, cats in (("r", cat_r), ("h", cat_h)):
            ivp, ivt = defaultdict(list), defaultdict(list)
            for i in np.nonzero(mask)[0]:
                for c in cats[i]:
                    ivp[c].append((int(pa[i]), int(pb[i])))
                    ivt[c].append((int(ta[i]), int(tb[i])))
            up = sum(union_length(v) for v in ivp.values())
            ut = sum(union_length(v) for v in ivt.values())
            out[f"pred_win_{cov}_unique"] = int(up + ut)
            out[f"pred_win_{cov}_unique_pool"] = int(up)
            out[f"pred_win_{cov}_unique_tier"] = int(ut)
            out[f"n_categories_{cov}"] = len(ivp)
        return out

    counts = {"A": block(is_a), "B": block(np.ones(len(rows), bool))}
    for t in TYPES:
        counts[f"type_{t}"] = block((rows.type == t).values)
    t2 = time.time()
    # 標本づくり用に (A) の行の区分と窓を残す（漏れの段より先に）
    SCRATCH.mkdir(parents=True, exist_ok=True)
    keep_rows = [dict(i=int(i), set=rows.set.iat[i], molecule_id=rows.molecule_id.iat[i],
                      pa=int(pa[i]), pb=int(pb[i]), ta=int(ta[i]), tb=int(tb[i]), n_i=int(n_i[i]),
                      cats=sorted([list(c) for c in cat_r[i]], key=lambda c: (c[0], c[1], c[2])))
                 for i in np.nonzero(is_a)[0]]
    json.dump(dict(rows=keep_rows), open(SCRATCH / "a_rows.json", "w"))
    OUT.mkdir(parents=True, exist_ok=True)
    if not (SCRATCH / "msg_hits.json").is_file():
        out = dict(issue=242, dev_list_sha256=DEV_LIST_SHA, counts=counts, leak=None,
                   secs=dict(categories=round(t1 - t0, 1), counts=round(t2 - t1, 1)))
        json.dump(out, open(OUT / "glacier_counts_noleak.json", "w"), indent=1, ensure_ascii=False)
        print(json.dumps(dict(A=counts["A"], B=counts["B"], leak="skipped (msg_hits.json 無し)"), ensure_ascii=False))
        return out
    # ---- 漏れ（MSG）
    mh = json.load(open(SCRATCH / "msg_hits.json"))
    sets = {k: set(v) for k, v in mh["sets"].items()}
    leak = {}
    for name, mask in (("A", is_a), ("B", np.ones(len(rows), bool))):
        r = rows[mask]
        leak[name] = {"truth": {s: dict(raw=int(r.inchikey14.isin(sets[f"raw_{s}"]).sum()),
                                        key=int(r.key.isin(sets[f"key_{s}"]).sum())) for s in ("tv", "all", "simtv")},
                      "n": int(mask.sum())}
    leak["A_by_type"] = {t: {s: dict(raw=int(rows[rows.type == t].inchikey14.isin(sets[f"raw_{s}"]).sum()),
                                     key=int(rows[rows.type == t].key.isin(sets[f"key_{s}"]).sum()))
                             for s in ("tv", "all")} for t in TYPES}
    # 候補: 窓に入る当たり行を数え、正解/不正解に分ける
    tk_cache, tr_cache = {}, {}
    tsm = None

    def truth_keys(i):
        nonlocal tsm
        if i not in tk_cache:
            t, s, mid, ik = rows.type.iat[i], rows.set.iat[i], rows.molecule_id.iat[i], rows.inchikey14.iat[i]
            if t != "none":
                if s not in tr_cache:
                    tr_cache[s] = pd.read_csv(ROOT / f"outputs/local_eval/{s}/truth.csv").set_index("molecule_id")
                smi = str(tr_cache[s].loc[mid, "smiles"])
            else:
                if tsm is None:
                    ts = pd.read_parquet(TSTRUCT, columns=["inchikey14", "smiles"])
                    tsm = dict(zip(ts.inchikey14.astype(str), ts.smiles.astype(str)))
                smi = tsm[str(ik)]
            tk_cache[i] = mrr25.truth_keys(smi)
        return tk_cache[i]

    cand = {}
    for src, A, B in (("pool", pa, pb), ("tier", ta, tb)):
        H = sorted(mh["hits"][src], key=lambda h: h["idx"])
        hidx = np.array([h["idx"] for h in H], np.int64)
        res = {nm: {f: dict(correct=0, wrong=0) for f in ("raw_tv", "raw_all", "key_tv", "key_all", "raw_simtv",
                                                          "key_simtv")} for nm in ("A", "B")}
        lo = np.searchsorted(hidx, A, "left")
        hi = np.searchsorted(hidx, B, "left")
        for i in np.nonzero(hi > lo)[0]:
            hid = hidden.get(rows.set.iat[i], set()) if src == "pool" and isinstance(rows.set.iat[i], str) else set()
            tks = None
            for h in H[lo[i]:hi[i]]:
                if src == "pool" and hid and pk[h["idx"]] in hid:
                    continue                               # pool remove の集合で隠した候補は候補でない
                if tks is None:
                    tks = truth_keys(int(i))
                good = h["key"] in tks
                for f in res["B"]:
                    if h[f]:
                        for nm in (("A", "B") if is_a[i] else ("B",)):
                            res[nm][f]["correct" if good else "wrong"] += 1
        cand[src] = res
    leak["candidates"] = cand
    out = dict(issue=242, definitions="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/242#issuecomment-7153124801",
               dev_list_sha256=DEV_LIST_SHA, per_molecule_sha256=sha256(PER_MOL), counts=counts, leak=leak,
               secs=dict(categories=round(t1 - t0, 1), counts=round(t2 - t1, 1), leak=round(time.time() - t2, 1)))
    json.dump(out, open(OUT / "glacier_counts.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(A=counts["A"], B=counts["B"], secs=out["secs"]), ensure_ascii=False))
    return out


# ---------------------------------------------------------------- 段: sample（測定の入力）
def stage_sample(out_csv: Path) -> dict:
    import csv
    import library as L
    from rdkit import Chem
    a = json.load(open(SCRATCH / "a_rows.json"))["rows"]
    pk, psmi, _ = load_pool()
    _, toff, tbuf = tier_arrays()
    hidden = {}
    rng = np.random.default_rng(SEED)
    out = []
    for src in ("pool", "tier"):
        w = np.array([(r["pb"] - r["pa"] if src == "pool" else r["tb"] - r["ta"]) * len(r["cats"]) for r in a],
                     np.float64)
        p = w / w.sum()
        got = 0
        while got < N_SAMPLE:
            r = a[int(rng.choice(len(a), p=p))]
            c = r["cats"][int(rng.integers(len(r["cats"])))]
            if src == "pool":
                i = int(rng.integers(r["pa"], r["pb"]))
                s = r["set"]
                if s not in hidden:
                    hidden[s] = L.drop_keys(s) if L.hide_rule(s).get("pool") == "remove" else set()
                if pk[i] in hidden[s]:
                    continue                                   # 隠した候補は引き直す（候補でないため）
                smi = psmi[i]
            else:
                i = int(rng.integers(r["ta"], r["tb"]))
                smi = tier_slice(toff, tbuf, i, i + 1)[0]
            m = Chem.MolFromSmiles(smi)
            out.append(dict(row=len(out), source=src, set=r["set"], molecule_id=r["molecule_id"], src_idx=i,
                            smiles=smi, adduct=c[0], instr=c[1], ce=c[2],
                            n_heavy=m.GetNumHeavyAtoms() if m is not None else -1))
            got += 1
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    summ = dict(n=len(out), bytes=out_csv.stat().st_size, sha256=sha256(out_csv), seed=SEED,
                by_source=Counter(r["source"] for r in out),
                n_heavy_mean={s: round(float(np.mean([r["n_heavy"] for r in out if r["source"] == s])), 2)
                              for s in ("pool", "tier")},
                n_bad_smiles=sum(r["n_heavy"] < 0 for r in out),
                n_categories=len({(r["adduct"], r["instr"], r["ce"]) for r in out}))
    print(json.dumps(summ, ensure_ascii=False))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("msg", "counts", "sample"), required=True)
    ap.add_argument("--dev-list", type=Path, default=ROOT / "docs/bench/dev_list.csv.gz")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-csv", type=Path, default=ROOT / "kernels/glacier_timing/input/glacier_timing_input.csv")
    a = ap.parse_args()
    if a.stage == "msg":
        stage_msg(a.workers)
    elif a.stage == "counts":
        stage_counts(a.dev_list)
    else:
        stage_sample(a.out_csv)


if __name__ == "__main__":
    main()
