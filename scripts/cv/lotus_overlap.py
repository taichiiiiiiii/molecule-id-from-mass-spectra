#!/usr/bin/env python3
"""
Issue #129: LOTUS 候補（qwertykeypadapple/casmi26-lotus-candidates の candidates_lotus.parquet）が、
今の候補プール（pool_meta）・train・PubChem 層（data/ext/pubchem_tier）に対してどれだけ新しいかを数える。
数え上げだけで、定数の選択や採否には使わない（定義は Issue #129 の事前登録のとおり）。

キー:
  ik14  : RDKit の InChIKey の先頭 14 文字（互変異性の正準化なし。chem.raw_key）
  canon : chem.score_key（指標と同じ互変異性の正準化）

段（出力は --out-dir の下だけ。リポジトリや元のフォルダには書かない）:
  keys    : LOTUS の canon、pool の ik14、正解の canon を計算して keys_*.json に置く（計算済みなら飛ばす）
  counts  : 全数の重なり → counts.json
  pubchem : pool・train に無い LOTUS 分子の標本（seed 0、打ち切りの規則つき）が PubChem 層にあるか → pubchem.json

    .venv/bin/python scripts/cv/lotus_overlap.py --lotus <dir>/candidates_lotus.parquet --out-dir <scratch> keys
    .venv/bin/python scripts/cv/lotus_overlap.py --lotus ... --out-dir <scratch> counts
    .venv/bin/python scripts/cv/lotus_overlap.py --lotus ... --out-dir <scratch> pubchem
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHEM_CODE = "data/ext/v4g/casmi26-v3-models/code"
POOL_REL = "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet"
TS_REL = "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
TRUTH = {"v3_c1": "outputs/holdouts/v3_c1/truth.csv", "v3_c2": "outputs/holdouts/v3_c2/truth.csv",
         "enveda_v1": "outputs/holdouts/enveda_v1/truth.csv", "v3_c2n": "outputs/holdouts/v3_c2n/truth.csv",
         "e180_c2": "data/holdouts/e180_c2/truth.csv"}
PPM = 1.0                       # PubChem 層の窓（同じ構造の中性の単一同位体質量どうしの一致を見る）
SEED, N_MAX, N_MIN, N_PILOT, BUDGET_S, N_CONTROL = 0, 1000, 200, 100, 1800.0, 100


# ---------------------------------------------------------------- 純粋な関数（単体テストの対象）

def frac(k: int, n: int) -> float:
    return round(k / n, 6) if n else float("nan")


def overlap_counts(lotus: list[tuple[str, str | None]], pool_ik: set, pool_canon: set,
                   train_ik: set, train_canon: set) -> tuple[dict, list[int]]:
    """lotus = [(ik14, canon)]。返り値: (数の dict, 「新しい」分子の位置の一覧)。

    新しい = pool にも train にも、ik14 でも canon でも無い。canon が None の分子は ik14 だけで判定する。
    """
    n = len(lotus)
    c = dict(pool_ik=0, pool_canon=0, pool_any=0, train_ik=0, train_canon=0, train_any=0, pool_or_train=0)
    new = []
    for i, (ik, ck) in enumerate(lotus):
        pi, pc = ik in pool_ik, ck is not None and ck in pool_canon
        ti, tc = ik in train_ik, ck is not None and ck in train_canon
        c["pool_ik"] += pi; c["pool_canon"] += pc; c["pool_any"] += pi or pc
        c["train_ik"] += ti; c["train_canon"] += tc; c["train_any"] += ti or tc
        if pi or pc or ti or tc:
            c["pool_or_train"] += 1
        else:
            new.append(i)
    out = dict(n=n)
    for k, v in c.items():
        out[k] = int(v); out[k + "_frac"] = frac(int(v), n)
    out["new"] = len(new); out["new_frac"] = frac(len(new), n)
    return out, new


def truth_containment(truth: list[tuple[str, str | None]], lotus_ik: set, lotus_canon: set,
                      pool_ik: set, pool_canon: set) -> dict:
    """truth = [(ik14, canon)]。LOTUS にある正解の数（ik14 / canon / どちらか）と、そのうち pool に無い数。"""
    r = dict(n=len(truth), lotus_ik=0, lotus_canon=0, lotus_any=0, lotus_any_not_in_pool=0, pool_any=0)
    for ik, ck in truth:
        li, lc = ik in lotus_ik, ck is not None and ck in lotus_canon
        inpool = ik in pool_ik or (ck is not None and ck in pool_canon)
        r["lotus_ik"] += li; r["lotus_canon"] += lc; r["lotus_any"] += li or lc
        r["pool_any"] += inpool
        r["lotus_any_not_in_pool"] += (li or lc) and not inpool
    return {k: int(v) for k, v in r.items()}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def choose_n(t_per_mol: float, n_max: int = N_MAX, n_min: int = N_MIN, budget_s: float = BUDGET_S) -> int:
    """打ち切りの規則（#129 事前登録）: n_max·t ≤ budget なら n_max、超えたら max(n_min, floor(budget / t))。"""
    if t_per_mol <= 0 or n_max * t_per_mol <= budget_s:
        return n_max
    return max(n_min, int(math.floor(budget_s / t_per_mol)))


def sample_order(keys: list[str], seed: int = SEED) -> list[str]:
    """ik14 の昇順に並べてから default_rng(seed).permutation の順。先頭から使う。"""
    s = sorted(keys)
    perm = np.random.default_rng(seed).permutation(len(s))
    return [s[i] for i in perm]


def window_bounds(mass: np.ndarray, target: float, ppm: float = PPM) -> tuple[int, int]:
    tol = target * ppm * 1e-6
    return int(np.searchsorted(mass, target - tol, "left")), int(np.searchsorted(mass, target + tol, "right"))


def window_smiles(mass: np.ndarray, off: np.ndarray, buf: np.ndarray, target: float, ppm: float = PPM) -> list[str]:
    a, b = window_bounds(mass, target, ppm)
    if b <= a:
        return []
    o = np.asarray(off[a:b + 1]).astype(np.int64)
    raw = bytes(buf[o[0]:o[-1]]); base = o[0]
    return [raw[o[i] - base:o[i + 1] - base].decode("ascii") for i in range(b - a)]


def in_tier(mass, off, buf, ik: str, target: float, raw_key, ppm: float = PPM) -> tuple[bool, int, bool]:
    """(層にあるか, 窓の大きさ, 質量が層の範囲内か)。窓の SMILES を raw_key（ik14）にして ik と比べる。"""
    in_range = bool(mass[0] <= target <= mass[-1])
    smis = window_smiles(mass, off, buf, target, ppm)
    hit = False
    for s in smis:
        if raw_key(s) == ik:
            hit = True
            break
    return hit, len(smis), in_range


# ---------------------------------------------------------------- 入出力

def _chem():
    sys.path.insert(0, str(ROOT / CHEM_CODE))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from casmi import chem
    return chem


def _w_canon(smis):
    chem = _chem()
    return [chem.score_key(s) for s in smis]


def _w_raw(smis):
    chem = _chem()
    return [chem.raw_key(s) for s in smis]


def _parallel(fn, smis: list[str], procs: int, chunk: int = 2000) -> list:
    from concurrent.futures import ProcessPoolExecutor
    parts = [smis[i:i + chunk] for i in range(0, len(smis), chunk)]
    out = []
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for j, r in enumerate(ex.map(fn, parts), 1):
            out.extend(r)
            if j % 20 == 0 or j == len(parts):
                print(f"  {fn.__name__} {j}/{len(parts)}", flush=True)
    return out


def _dump(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, p)


def _load_lotus(path: Path):
    import pandas as pd
    return pd.read_parquet(path, columns=["smiles", "inchikey14", "monoisotopic_mass"])


def stage_keys(a) -> None:
    import pandas as pd
    out = Path(a.out_dir)
    lot = _load_lotus(Path(a.lotus))
    p = out / "keys_lotus_canon.json"
    if not p.is_file():
        t = time.time(); ck = _parallel(_w_canon, lot.smiles.tolist(), a.procs)
        _dump(p, dict(ik14=lot.inchikey14.tolist(), canon=ck, seconds=round(time.time() - t, 1)))
    p = out / "keys_pool_ik14.json"
    if not p.is_file():
        pm = pd.read_parquet(ROOT / POOL_REL, columns=["key", "smiles"])
        t = time.time(); ik = _parallel(_w_raw, pm.smiles.tolist(), a.procs)
        _dump(p, dict(key=pm.key.tolist(), ik14=ik, seconds=round(time.time() - t, 1)))
    p = out / "keys_pool_check.json"
    if not p.is_file():
        pm = pd.read_parquet(ROOT / POOL_REL, columns=["key", "smiles"])
        idx = np.random.default_rng(SEED).choice(len(pm), 1000, replace=False)
        ck = _w_canon(pm.smiles.values[idx].tolist())
        agree = int(sum(c == k for c, k in zip(ck, pm.key.values[idx])))
        _dump(p, dict(n=1000, seed=SEED, score_key_equals_key=agree))
    p = out / "keys_truth.json"
    if not p.is_file():
        res = {}
        for s, rel in TRUTH.items():
            tr = pd.read_csv(ROOT / rel, dtype=str)
            res[s] = dict(ik14=tr.inchikey14.tolist(), canon=_w_canon(tr.smiles.fillna("").tolist()), path=rel)
        _dump(p, res)
    print("keys done", flush=True)


def _train_sets():
    import pandas as pd
    import pyarrow.parquet as pq
    f = pq.ParquetFile(ROOT / "data/train.parquet", buffer_size=1 << 20, pre_buffer=False)
    ik = set()
    for b in f.iter_batches(columns=["inchikey14"], batch_size=8192):
        ik.update(str(x) for x in b.column(0).to_pylist() if x is not None)
    ts = pd.read_parquet(ROOT / TS_REL, columns=["inchikey14", "key"])
    ts_ik = set(ts.inchikey14.astype(str))
    canon = set(ts.key[ts.inchikey14.astype(str).isin(ik)].astype(str))
    return ik, canon, dict(train_ik14=len(ik), train_structs_rows=len(ts), train_ik14_not_in_train_structs=len(ik - ts_ik),
                           train_structs_ik14_not_in_train=len(ts_ik - ik), train_canon=len(canon))


def _pool_sets(out: Path):
    pk = json.loads((out / "keys_pool_ik14.json").read_text())
    pool_ik = {k for k in pk["ik14"] if k}
    return pool_ik, set(pk["key"]), dict(pool_rows=len(pk["key"]), pool_ik14=len(pool_ik), pool_canon=len(set(pk["key"])),
                                         pool_ik14_none=sum(k is None for k in pk["ik14"]))


def stage_counts(a) -> None:
    out = Path(a.out_dir)
    lk = json.loads((out / "keys_lotus_canon.json").read_text())
    lotus = list(zip(lk["ik14"], lk["canon"]))
    pool_ik, pool_canon, pinfo = _pool_sets(out)
    train_ik, train_canon, tinfo = _train_sets()
    ov, new = overlap_counts(lotus, pool_ik, pool_canon, train_ik, train_canon)
    lot_ik, lot_canon = set(lk["ik14"]), {c for c in lk["canon"] if c}
    tk = json.loads((out / "keys_truth.json").read_text())
    truth = {s: truth_containment(list(zip(v["ik14"], v["canon"])), lot_ik, lot_canon, pool_ik, pool_canon) for s, v in tk.items()}
    res = dict(issue=129, lotus=dict(rows=len(lotus), ik14_unique=len(lot_ik), canon_unique=len(lot_canon),
                                     canon_none=sum(c is None for c in lk["canon"]),
                                     canon_differs_from_ik14=sum(c is not None and c != i for i, c in lotus),
                                     canon_seconds=lk["seconds"]),
               pool=pinfo, pool_check=json.loads((out / "keys_pool_check.json").read_text()), train=tinfo,
               overlap=ov, truth=truth)
    _dump(out / "counts.json", res)
    _dump(out / "new_index.json", new)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def _tier_run(ids: list[str], ik2mass: dict, chem, tier, log_every: int = 50) -> list[dict]:
    rows = []
    for j, ik in enumerate(ids, 1):
        t = time.perf_counter()
        hit, w, rng = in_tier(tier.mass, tier.off, tier.buf, ik, ik2mass[ik], chem.raw_key)
        rows.append(dict(ik14=ik, hit=hit, window=w, in_range=rng, sec=time.perf_counter() - t))
        if j % log_every == 0:
            print(f"  tier {j}/{len(ids)}", flush=True)
    return rows


def _summ(rows: list[dict]) -> dict:
    n = len(rows); k = sum(r["hit"] for r in rows)
    lo, hi = wilson(k, n)
    return dict(n=n, in_tier=k, frac=frac(k, n), wilson95=[round(lo, 4), round(hi, 4)],
                out_of_mass_range=sum(not r["in_range"] for r in rows),
                window_median=float(np.median([r["window"] for r in rows])) if rows else None,
                sec_per_mol=round(float(np.mean([r["sec"] for r in rows])), 4) if rows else None)


def stage_pubchem(a) -> None:
    out = Path(a.out_dir)
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    from pubchem import Tier
    chem = _chem()
    lot = _load_lotus(Path(a.lotus))
    ik2mass = dict(zip(lot.inchikey14, lot.monoisotopic_mass.astype(float)))
    new = json.loads((out / "new_index.json").read_text())
    order = sample_order([lot.inchikey14.iloc[i] for i in new], SEED)
    tier = Tier(ROOT / "data/ext/pubchem_tier")
    pilot = _tier_run(order[:N_PILOT], ik2mass, chem, tier)
    t_pilot = float(np.mean([r["sec"] for r in pilot]))
    n = min(choose_n(t_pilot), len(order))
    print(f"pilot sec/mol={t_pilot:.4f} -> N={n}", flush=True)
    rest = _tier_run(order[N_PILOT:n], ik2mass, chem, tier)
    rows = pilot + rest
    pool_ik, _, _ = _pool_sets(out)
    ctrl_ids = sample_order([k for k in lot.inchikey14 if k in pool_ik], SEED)[:N_CONTROL]
    ctrl = _tier_run(ctrl_ids, ik2mass, chem, tier)
    res = dict(issue=129, seed=SEED, ppm=PPM, population_new=len(order), pilot_n=N_PILOT, pilot_sec_per_mol=round(t_pilot, 4),
               rule=f"N = {N_MAX} if {N_MAX}*t <= {BUDGET_S}s else max({N_MIN}, floor({BUDGET_S}/t))", chosen_n=n,
               sample=_summ(rows), control_in_pool=_summ(ctrl),
               tier_rows=int(len(tier.mass)), tier_mass_range=[float(tier.mass[0]), float(tier.mass[-1])])
    _dump(out / "pubchem.json", res)
    _dump(out / "pubchem_rows.json", dict(sample=rows, control=ctrl))
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["keys", "counts", "pubchem"])
    ap.add_argument("--lotus", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    od = Path(a.out_dir).resolve()
    if od == ROOT or ROOT in od.parents:
        raise SystemExit("--out-dir はリポジトリの外に置く（元のフォルダへのリンクに書かないため）")
    od.mkdir(parents=True, exist_ok=True)
    dict(keys=stage_keys, counts=stage_counts, pubchem=stage_pubchem)[a.stage](a)


if __name__ == "__main__":
    main()
