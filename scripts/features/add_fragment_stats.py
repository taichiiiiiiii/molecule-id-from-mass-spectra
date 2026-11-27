#!/usr/bin/env python3
"""
群 H3（pfs_*）の実データの計算（Issue #296）。定義は fragment_stats.py の docstring と Issue #296。

今夜の測定（#296「今夜の測定」）: train の数え上げを構造の部分（--subset N、名前順の inchikey14 から
numpy.random.default_rng(0).choice(N, replace=False)）で作り、dev 1 集合の 4 列をその部分の数で計算する。
--subset 0 で全 train（今夜は回さない）。値は書き出すだけ（MRR・正解のラベルは作らない）。記録の json には
段ごとの秒数・peak RSS・行数・NaN/inf/0 の数だけを書く。

    .venv/bin/python scripts/features/add_fragment_stats.py --set b8_f1_b_t10 --subset 5000 \\
        --table outputs/local_eval/features/dev_b8.parquet \\
        --out outputs/local_eval/features/parts_pfs_dev_sub5000/b8_f1_b_t10.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/features"))
import fragment_stats as P  # noqa: E402
import neighbour_ref as NR  # noqa: E402

POOL = ROOT / "data/ext/v4g/casmi26-v2-pool"
FOLDS = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
TRAIN = ROOT / "data/train.parquet"
COLS = ["inchikey14", "ingest_lib", "adduct", "ionization_mode", "ms2_mzs", "ms2_normalized_intensities"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30   # macOS: bytes


def subset_keys(raw_sorted: np.ndarray, n: int) -> np.ndarray:
    """#296 の部分: 名前順の inchikey14 から default_rng(0) で n 個（n = 0 なら全部）。"""
    if n <= 0 or n >= len(raw_sorted):
        return raw_sorted
    return np.sort(raw_sorted[np.random.default_rng(0).choice(len(raw_sorted), n, replace=False)])


def build_counts(F, ts: pd.DataFrame, use_raw: np.ndarray, T: dict, fid: dict):
    """use_raw の構造の参照（構造ごとに先頭 16 行、隠す前）を数える。戻り値 CountTable と構造ごとの本数。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sid = {k: i for i, k in enumerate(ts.inchikey14)}
    smi = dict(zip(ts.inchikey14, ts.smiles))
    t0 = time.time()
    frags = {}
    for k in use_raw:
        frags[k] = P.enumerate_fragments(smi[k], F)
    T["train_enumerate"] = round(time.time() - t0, 1)
    gid = {}
    for k, fr in frags.items():
        if fr is not None:
            gid[k] = np.array([fid.setdefault(nm, len(fid)) for nm in fr.name], np.int64)
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    vs = pa.array(sorted(use_raw), type=pf.schema_arrow.field("inchikey14").type)
    seen: dict = {}
    S, E, K = [], [], []
    t_io = t_cnt = 0.0
    n_spec = 0
    t1 = time.time()
    for bt in pf.iter_batches(batch_size=8192, columns=COLS):
        bt = bt.filter(pc.is_in(bt.column("inchikey14"), value_set=vs))
        if bt.num_rows == 0:
            continue
        d = bt.to_pandas()
        t2 = time.time()
        for r in d.itertuples(index=False):
            c = seen.get(r.inchikey14, 0)
            if c >= P.QUERY_CAP:
                continue
            seen[r.inchikey14] = c + 1
            n_spec += 1
            fr = frags[r.inchikey14]
            if fr is None:
                continue
            spec = dict(mz=r.ms2_mzs, it=r.ms2_normalized_intensities, adduct=r.adduct,
                        mode=1 if r.ionization_mode == "positive" else -1)
            b, fi = P.count_spectrum(fr, spec, F)
            if len(b):
                S.append(np.full(len(b), sid[r.inchikey14], np.int64))
                E.append(np.full(len(b), r.ingest_lib in NR.ENVEDA, np.int64))
                K.append((b << P.FID_BITS) | gid[r.inchikey14][fi])
        t_cnt += time.time() - t2
    t_io = time.time() - t1 - t_cnt
    T["train_stream_io"] = round(t_io, 1)
    T["train_match_count"] = round(t_cnt, 1)
    if S:
        a = np.stack([np.concatenate(S), np.concatenate(E), np.concatenate(K)], 1)
        u, c = np.unique(a, axis=0, return_counts=True)
        tab = P.CountTable(u[:, 0], u[:, 1].astype(bool), u[:, 2], c)
    else:
        tab = P.CountTable.from_rows([])
    return tab, n_spec, int(sum(1 for v in frags.values() if v is None))


def run_set(F, h: str, table: Path, out: Path, n_subset: int) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import library as L
    T: dict = {}
    t0 = time.time()
    ts = pd.read_parquet(POOL / "train_structs.parquet", columns=["inchikey14", "smiles", "key"])
    raw2canon = dict(zip(ts.inchikey14, ts.key))
    st = pd.read_parquet(FOLDS, columns=["inchikey14", "fold_identity"]).drop_duplicates("inchikey14")
    fold_of = dict(zip(st.inchikey14, st.fold_identity))
    assert all(k in fold_of for k in ts.inchikey14), "structures.parquet に無い train 構造がある"
    use_raw = subset_keys(np.array(sorted(ts.inchikey14)), n_subset)
    T["load"] = round(time.time() - t0, 1)
    F.warmup()
    P.enumerate_fragments("CCOC(=O)c1ccccc1O", F)           # numba の初回の翻訳を測る時間から外す
    fid: dict = {}
    tab, n_spec, n_unparsable_train = build_counts(F, ts, use_raw, T, fid)
    # 集合の除外: fold k の構造、隠す規則（neighbour_ref と同じ関数、raw と正準の鍵の両方）
    t3 = time.time()
    rule = L.hide_rule(h)["rule"]
    hidden = NR.drop_canon(L.drop_keys(h), raw2canon)
    tb = pd.read_parquet(table, columns=["set", "molecule_id", "cand_key", "cand_smiles", "query_fold"],
                         filters=[("set", "==", h)])
    assert len(tb), f"{h}: 表に行が無い"
    folds = tb.query_fold.unique()
    assert len(folds) == 1, "集合の query_fold が 1 つでない"
    k_fold = int(folds[0])
    rec_raw = ts.inchikey14.to_numpy()[tab.struct]
    rec_canon = ts.key.to_numpy()[tab.struct]
    rec_lib = np.where(tab.env, "enveda-180", "other")
    keep = NR.keep_rows(rec_canon, rec_lib, hidden, rule)
    keep &= np.array([fold_of[k] != k_fold for k in rec_raw], bool)
    sc = tab.for_set(keep)
    T["set_counts"] = round(time.time() - t3, 1)
    # 候補の断片と own-out の構造
    t4 = time.time()
    uniq = tb.cand_smiles.drop_duplicates().tolist()
    cfr = {}
    for s in uniq:
        fr = P.enumerate_fragments(s, F)
        cfr[s] = (fr, None if fr is None else np.array([fid.get(nm, -1) for nm in fr.name], np.int64))
    formula = {s: P.formula_of(s) for s in uniq}
    T["cand_enumerate"] = round(time.time() - t4, 1)
    by_canon: dict = {}
    for i, (r_, c_) in enumerate(zip(ts.inchikey14, ts.key)):
        by_canon.setdefault(c_, []).append(i)
    sid = {k: i for i, k in enumerate(ts.inchikey14)}

    def own_structs(ck):
        s = set(by_canon.get(ck, ()))
        if ck in sid:
            s.add(sid[ck])
        if ck in raw2canon:
            s |= set(by_canon.get(raw2canon[ck], ()))
        return s

    # 問い合わせ（run.query_spectra と同じ: 分子ごとに先頭 16 本）
    t5 = time.time()
    te = pd.read_parquet(ROOT / f"data/holdouts/{h}/test.parquet")
    qp = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        sp = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                   adduct=r.adduct, mode=1 if r.ionization_mode == "positive" else -1) for r in g.itertuples()][:P.QUERY_CAP]
        qp[str(mid)] = [p for p in (P.prep(s, F) for s in sp) if p is not None]
    tol = float(F.DEFAULT_CFG["tol"])
    parts = []
    n_own = 0
    for mid, g in tb.groupby("molecule_id", sort=False):
        owns = []
        for ck in g.cand_key:
            os_ = own_structs(ck)
            o = sc.own(os_) if os_ else None
            n_own += int(o is not None and len(o[0]) > 0)
            owns.append(o)
        feats = P.molecule_features(qp.get(str(mid), []), [cfr[s] for s in g.cand_smiles], sc, owns,
                                    [formula[s] for s in g.cand_smiles], tol)
        f = pd.DataFrame(feats, index=g.index)
        f.insert(0, "cand_key", g.cand_key.to_numpy())
        f.insert(0, "molecule_id", g.molecule_id.to_numpy())
        f.insert(0, "set", h)
        parts.append(f)
    T["features"] = round(time.time() - t5, 1)
    res = pd.concat(parts).loc[tb.index].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    res.to_parquet(tmp, index=False)
    tmp.replace(out)
    cnt = {c: dict(nan=int(res[c].isna().sum()), inf=int(np.isinf(res[c]).sum()), zero=int((res[c] == 0).sum()))
           for c in P.COLUMNS}
    return dict(set=h, rule=rule, query_fold=k_fold, subset=int(len(use_raw)), n_train_spectra_counted=int(n_spec),
                n_unparsable_train=n_unparsable_train, n_records=int(len(tab.key)), n_records_kept=int(keep.sum()),
                n_keys_kept=int(len(sc.ukey)), n_bins_kept=int(len(sc.ubin)), n_fragment_names=int(len(fid)),
                n_rows=int(len(res)), n_molecules=int(res.molecule_id.nunique()), n_unique_cand=int(len(uniq)),
                n_unparsable_cand=int(sum(1 for v in cfr.values() if v[0] is None)),
                n_mol_without_peaks=int(sum(1 for m in tb.molecule_id.astype(str).unique() if not qp.get(m))),
                n_rows_with_own=n_own, counts=cnt, secs=T, secs_total=round(time.time() - t0, 1),
                peak_rss_gb=round(peak_rss_gb(), 3), out_sha256=sha256_file(out), table_sha256=sha256_file(table),
                alpha=P.ALPHA, beta=P.BETA, bin_width=P.BIN_WIDTH, note="数は部分の数（全 train では違う）" if n_subset else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=5000)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as F
    rec = run_set(F, a.set, Path(a.table), Path(a.out), a.subset)
    Path(a.out).with_suffix(".json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
