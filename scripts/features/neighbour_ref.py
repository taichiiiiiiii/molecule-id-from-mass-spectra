#!/usr/bin/env python3
"""
群 N1: 近傍の参照の照合の特徴（MS2Query の入力 4・5 の形、Issue #261）。定義は docs/research/neighbour-reference-features.md。

候補ごとに:
  nbr_ms_mean   候補の構造に Tanimoto で近い参照の構造（隠す規則の後に参照の残る構造）上位 K_NEIGHBOURS = 10 個について、
                構造ごとに「その構造の残った参照スペクトルそれぞれと問い合わせのエントロピー類似度（問い合わせが複数なら最大）」
                の平均を取り、10 個を平均（MS2Query Methods / 補足 Note 3「構造ごとに平均してから平均」、重みなし = 補足 Table S3）
  nbr_tani_mean その 10 個と候補の Tanimoto の平均（MS2Query の入力 5）
  nbr_n         使えた近傍の数（参照のある構造が 10 未満のときだけ 10 より小さい）
近傍 0 件、または問い合わせのスペクトルが無い分子は nbr_ms_mean = NaN（nbr_tani_mean は近傍 0 件のときだけ NaN）。

構造の単位は train_structs.parquet の key（scorer の正準の鍵）。候補自身の構造も、参照が残っていれば近傍に入る
（MS2Query のコード create_sqlite_database.py: calculate_highest_tanimoto_score(list_of_spectra, list_of_spectra, 10)
は自分自身を除かない）。隠す規則は集合の hide_rule.json と drop_keys.txt を、raw の inchikey14 と正準の鍵の両方で当てる
（互変異性の別名の参照も隠す。library.load_refs より厳しい側）。

    .venv/bin/python scripts/features/neighbour_ref.py --set b8_f1_a_t10 \
        --table outputs/local_eval/features/dev_b8.parquet \
        --out outputs/local_eval/features/parts_nbr_dev/b8_f1_a_t10.parquet

値は書き出すだけ（MRR・正解のラベルは作らない）。記録の json には秒数・行数・NaN の数だけを書く。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
K_NEIGHBOURS = 10          # MS2Query: Methods「10 chemically similar library molecules」（定数は論文のまま、dev で動かさない）
NBITS = 10226              # fp_bits.npy の長さ（既存の指紋）
ENVEDA = {"enveda-180", "enveda-np-examples"}
POOL = ROOT / "data/ext/v4g/casmi26-v2-pool"


# ---------------------------------------------------------------- 隠す規則
def drop_canon(drop: set, raw2canon: dict) -> set:
    """drop_keys（raw の inchikey14 か正準の鍵）を正準の鍵の集合に直す。両方の書き方を受ける。"""
    canon_vals = set(raw2canon.values())
    return {raw2canon[k] for k in drop if k in raw2canon} | {k for k in drop if k in canon_vals}


def keep_rows(row_canon: np.ndarray, row_lib: np.ndarray, hidden_canon: set, rule: str) -> np.ndarray:
    """train の行（参照スペクトル）のうち使ってよい行。rule = all なら隠す構造の行をすべて、
    enveda_libs なら隠す構造の enveda 系の行だけを除く（library.load_refs と同じ規則を正準の鍵で）。"""
    hid = pd.Series(row_canon).isin(hidden_canon).to_numpy().copy()
    if rule == "enveda_libs":
        hid &= pd.Series(row_lib).isin(ENVEDA).to_numpy()
    elif rule != "all":
        raise ValueError(f"知らない hide_rule: {rule}")
    return ~hid


# ---------------------------------------------------------------- 近傍（Tanimoto 上位 k）
def topk_neighbours(cand: np.ndarray, ref_packed: np.ndarray, avail: np.ndarray, k: int = K_NEIGHBOURS,
                    nbits: int = NBITS, ref_chunk: int = 8192, cand_block: int = 1024):
    """cand: (n, nbits) の 0/1、ref_packed: (m, ceil(nbits/8)) の packbits、avail: (m,) bool（参照の残る構造）。
    返り値 idx (n, k) int64（足りない所は -1）、tani (n, k) float64（足りない所は NaN）。
    順は Tanimoto の大きい順、同点は参照の番号の小さい順（厳密。Tanimoto は整数の比を float64 で割った値で比べる）。"""
    n, m = len(cand), len(ref_packed)
    ca = cand.sum(1, dtype=np.float64)      # cand は 0/1 の uint8 のまま持ち、塊ごとに float32 にする（メモリ）
    best_key = np.full((n, k), -np.inf)
    best_idx = np.full((n, k), -1, np.int64)
    for r0 in range(0, m, ref_chunk):
        r1 = min(m, r0 + ref_chunk)
        R = np.unpackbits(ref_packed[r0:r1], axis=1)[:, :nbits].astype(np.float32)
        rb = R.sum(1, dtype=np.float64)
        ok = avail[r0:r1]
        w = r1 - r0
        kk = min(k, w)
        for c0 in range(0, n, cand_block):
            c1 = min(n, c0 + cand_block)
            inter = (cand[c0:c1].astype(np.float32) @ R.T).astype(np.float64)
            uni = ca[c0:c1, None] + rb[None, :] - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(uni > 0, inter / uni, 0.0)   # 整数の比を float64 で 1 回割るだけなので、同じ比は同じ値
            t[:, ~ok] = -np.inf
            # 行ごとに (Tanimoto 降順, 番号 昇順) の上位 kk を厳密に選ぶ: kk 番目の値 v より大きいものすべて + v に等しいもの
            # のうち番号の小さい順に残りの数だけ（列は番号の昇順なので cumsum で数える）
            v = np.partition(t, w - kk, axis=1)[:, w - kk][:, None]
            gt = t > v
            eq = t == v
            need = kk - gt.sum(1, keepdims=True)
            sel = gt | (eq & (np.cumsum(eq, axis=1) <= need))
            _rr, cols = np.nonzero(sel)
            cols = cols.reshape(c1 - c0, kk)
            pk = np.take_along_axis(t, cols, 1)
            allk = np.concatenate([best_key[c0:c1], pk], 1)
            alli = np.concatenate([best_idx[c0:c1], cols + r0], 1)
            # 番号 -1（まだ無い）は -inf と組なので並びの後ろに行く
            o = np.lexsort((np.where(alli < 0, np.iinfo(np.int64).max, alli), -allk), axis=-1)[:, :k]
            best_key[c0:c1] = np.take_along_axis(allk, o, 1)
            best_idx[c0:c1] = np.take_along_axis(alli, o, 1)
    valid = np.isfinite(best_key)
    idx = np.where(valid, best_idx, -1)
    # 返す Tanimoto は選んだ近傍について float64 で作り直す（選ぶときの値と同じ式。点検を兼ねる）
    tani = np.full((n, k), np.nan)
    for i in range(n):
        js = idx[i][valid[i]]
        if len(js):
            R = np.unpackbits(ref_packed[js], axis=1)[:, :nbits].astype(np.float64)
            inter = R @ cand[i].astype(np.float64)
            uni = cand[i].sum() + R.sum(1) - inter
            tani[i, :len(js)] = np.where(uni > 0, inter / np.where(uni > 0, uni, 1), 0.0)
    return idx, tani


# ---------------------------------------------------------------- 特徴
def accumulate(acc: dict, row_struct, row_keep, row_specs, need_map: dict, queries: dict, sim) -> None:
    """参照スペクトルの行を流し読みして、(分子, 構造) ごとに「問い合わせの最大の類似度」の和と本数を足す。
    隠す行（row_keep = False）と、どの分子の近傍にも無い構造の行は使わない。acc[(m, j)] = [和, 本数]。"""
    for j, k, r in zip(row_struct, row_keep, row_specs):
        if not k:
            continue
        for m in need_map.get(int(j), ()):
            qs = queries.get(m)
            if not qs:
                continue
            v = max(sim(q, r) for q in qs)
            a = acc.setdefault((m, int(j)), [0.0, 0])
            a[0] += v
            a[1] += 1


def struct_scores(acc: dict, m) -> dict:
    """分子 m の構造ごとの点（参照スペクトルでの平均）。"""
    return {j: s / n for (mm, j), (s, n) in acc.items() if mm == m and n > 0}


def molecule_features(idx: np.ndarray, tani: np.ndarray, has_query: bool, score_of: dict) -> pd.DataFrame:
    """1 分子の候補（行）ごとの nbr_ms_mean・nbr_tani_mean・nbr_n。score_of: 構造の番号 -> その分子での構造の点。
    idx が -1 の所は近傍なし。問い合わせがあるのに点の無い構造（参照の残らない構造）が近傍に入っていたら止める。"""
    ms = np.full(len(idx), np.nan)
    tm = np.full(len(idx), np.nan)
    nn = np.zeros(len(idx), np.int16)
    for i in range(len(idx)):
        js = [int(j) for j in idx[i] if j >= 0]
        nn[i] = len(js)
        if not js:
            continue
        tm[i] = float(np.mean(tani[i, :len(js)]))
        if not has_query:
            continue
        miss = [j for j in js if j not in score_of]
        if miss:
            raise AssertionError(f"参照の残らない構造 {miss[:3]} が近傍に入った")
        ms[i] = float(np.mean([score_of[j] for j in js]))   # 構造ごとに平均してから平均（重みなし）
    return pd.DataFrame(dict(nbr_ms_mean=ms, nbr_tani_mean=tm, nbr_n=nn))


# ---------------------------------------------------------------- 実データ
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_library(cache_dir: Path):
    """構造（正準の鍵、名前順）・その指紋（pool_fp の行）・train の行の (正準の鍵の番号, enveda か)。cache_dir に保存。"""
    import pyarrow.parquet as pq
    ts = pd.read_parquet(POOL / "train_structs.parquet", columns=["inchikey14", "key"])
    raw2canon = dict(zip(ts.inchikey14, ts.key))
    skeys = np.array(sorted(set(ts.key)))
    sidx = {k: i for i, k in enumerate(skeys)}
    fp_c = cache_dir / "nbr_ref_fp.npy"
    if fp_c.is_file():
        ref_packed = np.load(fp_c)
    else:
        pm = pd.read_parquet(POOL / "pool_meta.parquet", columns=["key"])
        prow = dict(zip(pm.key, range(len(pm))))
        rows = np.array([prow[k] for k in skeys])
        ref_packed = np.asarray(np.load(POOL / "pool_fp.npy", mmap_mode="r")[np.sort(rows)])
        ref_packed = ref_packed[np.argsort(np.argsort(rows))]   # skeys の順に戻す
        np.save(fp_c, ref_packed)
    assert ref_packed.shape == (len(skeys), (NBITS + 7) // 8), "キャッシュの指紋の形が構造の数と合わない"
    rm_c = cache_dir / "nbr_train_rows.parquet"
    if rm_c.is_file():
        rm = pd.read_parquet(rm_c)
    else:
        pf = pq.ParquetFile(ROOT / "data/train.parquet", buffer_size=1 << 20, pre_buffer=False)
        rm = pd.concat([b.to_pandas() for b in pf.iter_batches(batch_size=65536, columns=["inchikey14", "ingest_lib"])],
                       ignore_index=True)
        rm.to_parquet(rm_c, index=False)
    assert len(rm) == pq.ParquetFile(ROOT / "data/train.parquet").metadata.num_rows, "キャッシュの train の行数が違う"
    return skeys, sidx, raw2canon, ref_packed, rm


def run_set(h: str, table: Path, out: Path, cache_dir: Path, mode: str = "hide_rule") -> dict:
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    import library as L
    from casmi import chem
    from casmi import spectra as sp
    from casmi.library import SearchCfg
    T: dict = {}
    t0 = time.time()
    skeys, sidx, raw2canon, ref_packed, rm = load_library(cache_dir)
    T["load_library"] = round(time.time() - t0, 1)
    # 隠す規則（mode = hide_rule: 集合の規則のまま。all_truth: 集合の正解の構造の参照をすべて隠す）
    rule = L.hide_rule(h)["rule"] if mode == "hide_rule" else "all"
    hidden = drop_canon(L.drop_keys(h), raw2canon)
    row_canon = rm.inchikey14.map(raw2canon).to_numpy()
    assert not pd.isna(row_canon).any(), "train の行に train_structs に無い inchikey14 がある"
    keep = keep_rows(row_canon, rm.ingest_lib.to_numpy(), hidden, rule)
    avail = np.zeros(len(skeys), bool)
    avail[pd.Series(row_canon[keep]).map(sidx).to_numpy()] = True
    # 候補
    t1 = time.time()
    tab = pd.read_parquet(table, columns=["set", "molecule_id", "cand_key", "cand_smiles", "is_tier"],
                          filters=[("set", "==", h)])
    assert len(tab), f"{h}: 表に行が無い"
    bits = np.load(POOL / "fp_bits.npy")
    assert len(bits) == NBITS
    pm = pd.read_parquet(POOL / "pool_meta.parquet", columns=["key"])
    prow = dict(zip(pm.key, range(len(pm))))
    pool_fp = np.load(POOL / "pool_fp.npy", mmap_mode="r")
    uniq = tab.drop_duplicates("cand_smiles")[["cand_key", "cand_smiles", "is_tier"]].reset_index(drop=True)
    cand = np.zeros((len(uniq), NBITS), np.uint8)
    for i, (k, s, q) in enumerate(zip(uniq.cand_key, uniq.cand_smiles, uniq.is_tier)):
        if q == 0 and k in prow:
            cand[i] = np.unpackbits(np.asarray(pool_fp[prow[k]]))[:NBITS]
        else:
            fp = chem.raw_fingerprint(s)
            assert fp is not None, f"{h}: 指紋の作れない候補 {s}"
            cand[i] = fp[bits]
    T["cand_fp"] = round(time.time() - t1, 1)
    t2 = time.time()
    idx_u, tani_u = topk_neighbours(cand, ref_packed, avail)
    T["topk"] = round(time.time() - t2, 1)
    assert not ((idx_u >= 0) & ~avail[np.clip(idx_u, 0, None)]).any(), "隠した構造が近傍に入った"
    pos = pd.Series(range(len(uniq)), index=uniq.cand_smiles)
    r = pos.loc[tab.cand_smiles.to_numpy()].to_numpy()
    idx, tani = idx_u[r], tani_u[r]
    # 問い合わせ（run.query_spectra と同じ: 分子ごとに先頭 16 本）
    t3 = time.time()
    te = pd.read_parquet(ROOT / f"data/holdouts/{h}/test.parquet")
    qp = {str(mid): [L._prep(np.asarray(x.ms2_mzs, np.float64), np.asarray(x.ms2_normalized_intensities, np.float64))
                     for x in g.itertuples()][:16] for mid, g in te.groupby("molecule_id", sort=False)}
    c = SearchCfg()

    def sim(q, r_):
        return float(sp.entropy_sim(q[0], q[1], r_[0], r_[1], c.tol, c.ppm_tol))

    # 構造 -> その構造を近傍に持つ分子
    mids = tab.molecule_id.astype(str).to_numpy()
    need_map: dict = {}
    for m, row in zip(mids, idx):
        for j in row[row >= 0]:
            need_map.setdefault(int(j), set()).add(m)
    need = np.array(sorted(need_map))
    need_canon = set(skeys[need])
    raw_need = sorted(r_ for r_, cc in raw2canon.items() if cc in need_canon)
    # 参照スペクトルを train.parquet から行の塊ごとに流し読み（全部を持たない）。隠す規則は keep_rows
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ROOT / "data/train.parquet", buffer_size=1 << 20, pre_buffer=False)
    vs = pa.array(raw_need, type=pf.schema_arrow.field("inchikey14").type)
    acc: dict = {}
    n_used = 0
    for bt in pf.iter_batches(batch_size=8192, columns=L.REF_COLS):
        bt = bt.filter(pc.is_in(bt.column("inchikey14"), value_set=vs))
        if bt.num_rows == 0:
            continue
        d = bt.to_pandas()
        dc = d.inchikey14.map(raw2canon).to_numpy()
        kp = keep_rows(dc, d.ingest_lib.to_numpy(), hidden, rule)
        n_used += int(kp.sum())
        specs = [L._prep(mz, it) if k else None for mz, it, k in zip(d.ms2_mzs, d.ms2_normalized_intensities, kp)]
        accumulate(acc, [sidx[x] for x in dc], kp, specs, need_map, qp, sim)
    T["refs_and_sims"] = round(time.time() - t3, 1)
    t4 = time.time()
    by_m: dict = {}
    for (m, j), (s_, n_) in acc.items():
        by_m.setdefault(m, {})[j] = s_ / n_
    parts = []
    for mid, g in tab.groupby("molecule_id", sort=False):
        pos_ = g.index.to_numpy()
        m = str(mid)
        f = molecule_features(idx[pos_], tani[pos_], bool(qp.get(m)), by_m.get(m, {}))
        f.insert(0, "cand_key", g.cand_key.to_numpy())
        f.insert(0, "molecule_id", mid)
        f.insert(0, "set", h)
        f.index = pos_
        parts.append(f)
    T["features"] = round(time.time() - t4, 1)
    res = pd.concat(parts).loc[tab.index].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    res.to_parquet(tmp, index=False)
    tmp.replace(out)
    rec = dict(set=h, mode=mode, rule=rule, n_rows=int(len(res)), n_molecules=int(res.molecule_id.nunique()),
               n_unique_cand=int(len(uniq)), n_structures=int(len(skeys)), n_avail=int(avail.sum()),
               n_hidden_canon=int(len(hidden)), n_rows_hidden=int((~keep).sum()),
               n_need_structures=int(len(need)), n_ref_spectra_used=n_used, n_mol_struct_pairs=int(len(acc)),
               n_nan=dict(nbr_ms_mean=int(res.nbr_ms_mean.isna().sum()), nbr_tani_mean=int(res.nbr_tani_mean.isna().sum())),
               n_lt_k=int((res.nbr_n < K_NEIGHBOURS).sum()), secs=T, secs_total=round(time.time() - t0, 1),
               out_sha256=sha256_file(out), table_sha256=sha256_file(table), k=K_NEIGHBOURS)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="/Users/taichi/.hermes/profiles/massspec/cache/scratch/nbr_cache")
    ap.add_argument("--mode", choices=["hide_rule", "all_truth"], default="hide_rule")
    a = ap.parse_args()
    cd = Path(a.cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    rec = run_set(a.set, Path(a.table), Path(a.out), cd, a.mode)
    Path(a.out).with_suffix(".json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
