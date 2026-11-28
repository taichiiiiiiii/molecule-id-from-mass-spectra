#!/usr/bin/env python3
"""
群 H3（pfs_*）の実データの計算（Issue #296）。定義は fragment_stats.py の docstring と Issue #296。

train の数え上げ: 構造（--subset N なら名前順の inchikey14 から numpy.random.default_rng(0).choice(N, replace=False)、
--subset 0 で全 train）を塊（--chunk 構造）ごとに 列挙 → train.parquet の流し読み（塊の構造だけ）→ 塊の中で集約。
断片の名前は 64 bit の hash にして文字列は捨てる（#296 の直し。値は名前の辞書の版と同じ）。数の表は 1 回だけ作り、
--counts-cache があれば npz に書く（あれば読む）。そのあと集合を順に回し、集合ごとに fold k の構造・隠す規則・
own-out を当てて 4 列を書く。候補の断片は SMILES ごとに 1 回だけ列挙し、後の集合でも使う SMILES は最後に使う
集合まで持つ。出力が既にある集合は飛ばす（再開）。値は書き出すだけ（MRR・正解のラベルは作らない）。記録の json
には段ごとの秒数・peak RSS・行数・NaN/inf/0 の数だけを書く。

    # 1 集合（#296 の夜の測定と同じ形）
    .venv/bin/python scripts/features/add_fragment_stats.py --set b8_f1_b_t10 --subset 5000 \\
        --table outputs/local_eval/features/dev_b8.parquet \\
        --out outputs/local_eval/features/parts_pfs_dev_sub5000/b8_f1_b_t10.parquet
    # 全 train・dev 20 集合
    .venv/bin/python scripts/features/add_fragment_stats.py --all-sets --subset 0 \\
        --table outputs/local_eval/features/dev_b8_cc_gl2_n1.parquet \\
        --counts-cache outputs/local_eval/features/pfs_counts_full.npz \\
        --out-dir outputs/local_eval/features/parts_pfs_dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
CHUNK = 10_000           # 塊の構造数（#296 の直し。値には効かない）
TABLE_COLS = ["set", "molecule_id", "cand_key", "cand_smiles", "query_fold"]


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


def build_counts(F, ts: pd.DataFrame, use_raw: np.ndarray, T: dict, chunk: int = CHUNK):
    """use_raw の構造の参照（構造ごとに先頭 16 行、隠す前）を数える。戻り値 (CountTable, vocab, 本数, 読めない数)。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sid = {k: i for i, k in enumerate(ts.inchikey14)}
    smi = dict(zip(ts.inchikey14, ts.smiles))
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    ktype = pf.schema_arrow.field("inchikey14").type
    seen: dict = {}
    parts = []
    n_spec = n_unparsable = 0
    t_enum = t_io = t_cnt = t_agg = 0.0
    for c0 in range(0, len(use_raw), chunk):
        keys = use_raw[c0:c0 + chunk]
        t0 = time.time()
        frags = {k: P.enumerate_fragments(smi[k], F, keep_names=False) for k in keys}
        n_unparsable += sum(1 for v in frags.values() if v is None)
        t_enum += time.time() - t0
        vs = pa.array(sorted(keys), type=ktype)
        S, E, B, H = [], [], [], []
        t1 = time.time()
        c_cnt = 0.0
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
                    B.append(b)
                    H.append(fr.h[fi])
            c_cnt += time.time() - t2
        t_io += time.time() - t1 - c_cnt
        t_cnt += c_cnt
        del frags
        t3 = time.time()
        if S:
            parts.append(P.aggregate_records(np.concatenate(S), np.concatenate(E), np.concatenate(B), np.concatenate(H)))
        del S, E, B, H
        t_agg += time.time() - t3
        print(f"chunk {c0 // chunk + 1}/{-(-len(use_raw) // chunk)} structs={c0 + len(keys)} spectra={n_spec} "
              f"records={sum(len(p[0]) for p in parts)} rss_peak={peak_rss_gb():.2f}GB {time.strftime('%T')}", flush=True)
    T["train_enumerate"] = round(t_enum, 1)
    T["train_stream_io"] = round(t_io, 1)
    T["train_match_count"] = round(t_cnt, 1)
    t4 = time.time()
    if parts:
        cat = [np.concatenate([p[i] for p in parts]) for i in range(5)]
        del parts
        tab, vocab = P.CountTable.from_hashed(*cat)
        del cat
    else:
        z = np.zeros(0, np.int64)
        tab, vocab = P.CountTable(z, z.astype(bool), z, z), z
    T["train_aggregate"] = round(t_agg + time.time() - t4, 1)
    return tab, vocab, n_spec, n_unparsable


def save_counts(path: Path, tab, vocab, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez(f, struct=tab.struct, env=tab.env, key=tab.key, cnt=tab.cnt, vocab=vocab,
                 info=np.array(json.dumps(info, ensure_ascii=False)))
    os.replace(tmp, path)


def load_counts(path: Path):
    z = np.load(path, allow_pickle=False)
    info = json.loads(str(z["info"]))
    tab = P.CountTable(z["struct"], z["env"], z["key"], z["cnt"])
    return tab, z["vocab"], info


class Context:
    """集合をまたいで 1 回だけ作るもの: train の構造の表、fold、正準の鍵の別名、数の表、候補の断片のキャッシュ。"""

    def __init__(self, F, ts: pd.DataFrame, fold_arr: np.ndarray, tab, vocab):
        self.F, self.ts, self.fold_arr, self.tab, self.vocab = F, ts, fold_arr, tab, vocab
        self.raw2canon = dict(zip(ts.inchikey14, ts.key))
        self.sid = {k: i for i, k in enumerate(ts.inchikey14)}
        self.by_canon: dict = {}
        for i, c_ in enumerate(ts.key):
            self.by_canon.setdefault(c_, []).append(i)
        self.canon_arr = ts.key.to_numpy()
        self.cfr: dict = {}        # SMILES → (Frags（質量だけ）| None, fids | None)
        self.formula: dict = {}

    def own_structs(self, ck):
        s = set(self.by_canon.get(ck, ()))
        if ck in self.sid:
            s.add(self.sid[ck])
        if ck in self.raw2canon:
            s |= set(self.by_canon.get(self.raw2canon[ck], ()))
        return s

    def cand(self, s):
        if s not in self.cfr:
            fr = P.enumerate_fragments(s, self.F, keep_names=False)
            if fr is None:
                self.cfr[s] = (None, None)
            else:
                self.cfr[s] = (P.Frags(mass=fr.mass, k=fr.k, name=None, rank=None, h=None), P.vocab_ids(self.vocab, fr.h))
            self.formula[s] = P.formula_of(s)
        return self.cfr[s]

    def drop(self, smiles) -> None:
        for s in smiles:
            self.cfr.pop(s, None)
            self.formula.pop(s, None)


def run_set(ctx: Context, h: str, tb: pd.DataFrame, out: Path, last_use: dict | None = None, set_index: int = 0) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import library as L
    F, ts, tab = ctx.F, ctx.ts, ctx.tab
    T: dict = {}
    t0 = time.time()
    # 集合の除外: fold k の構造、隠す規則（neighbour_ref と同じ関数、raw と正準の鍵の両方）。構造ごと（env 2 通り）に
    # 求めて記録に引く（keep_rows は行ごとの関数なので、記録ごとに呼ぶのと同じ値）
    rule = L.hide_rule(h)["rule"]
    hidden = NR.drop_canon(L.drop_keys(h), ctx.raw2canon)
    assert len(tb), f"{h}: 表に行が無い"
    folds = tb.query_fold.unique()
    assert len(folds) == 1, "集合の query_fold が 1 つでない"
    k_fold = int(folds[0])
    n = len(ts)
    k_env = NR.keep_rows(ctx.canon_arr, np.full(n, "enveda-180", dtype=object), hidden, rule)
    k_oth = NR.keep_rows(ctx.canon_arr, np.full(n, "other", dtype=object), hidden, rule)
    s_ok_env = k_env & (ctx.fold_arr != k_fold)
    s_ok_oth = k_oth & (ctx.fold_arr != k_fold)
    keep = np.where(tab.env, s_ok_env[tab.struct], s_ok_oth[tab.struct])
    sc = tab.for_set(keep)
    T["set_counts"] = round(time.time() - t0, 1)
    # 候補の断片（キャッシュ）
    t4 = time.time()
    uniq = tb.cand_smiles.drop_duplicates().tolist()
    n_new = sum(1 for s in uniq if s not in ctx.cfr)
    for s in uniq:
        ctx.cand(s)
    T["cand_enumerate"] = round(time.time() - t4, 1)
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
            os_ = ctx.own_structs(ck)
            o = sc.own(os_) if os_ else None
            n_own += int(o is not None and len(o[0]) > 0)
            owns.append(o)
        feats = P.molecule_features(qp.get(str(mid), []), [ctx.cfr[s] for s in g.cand_smiles], sc, owns,
                                    [ctx.formula[s] for s in g.cand_smiles], tol)
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
    n_unparsable_cand = int(sum(1 for s in uniq if ctx.cfr[s][0] is None))
    if last_use is not None:
        ctx.drop([s for s in uniq if last_use[s] == set_index])
    cnt = {c: dict(nan=int(res[c].isna().sum()), inf=int(np.isinf(res[c]).sum()), zero=int((res[c] == 0).sum()))
           for c in P.COLUMNS}
    return dict(set=h, rule=rule, query_fold=k_fold, n_records_kept=int(keep.sum()),
                n_keys_kept=int(len(sc.ukey)), n_bins_kept=int(len(sc.ubin)),
                n_rows=int(len(res)), n_molecules=int(res.molecule_id.nunique()), n_unique_cand=int(len(uniq)),
                n_cand_enumerated_here=int(n_new), n_cand_cached_after=int(len(ctx.cfr)),
                n_unparsable_cand=n_unparsable_cand,
                n_mol_without_peaks=int(sum(1 for m in tb.molecule_id.astype(str).unique() if not qp.get(m))),
                n_rows_with_own=n_own, counts=cnt, secs=T, secs_set=round(time.time() - t0, 1),
                peak_rss_gb=round(peak_rss_gb(), 3), out_sha256=sha256_file(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", help="1 集合（--out に書く）")
    g.add_argument("--sets", help="カンマ区切りの集合（--out-dir/<set>.parquet）")
    g.add_argument("--all-sets", action="store_true", help="表の set の全集合（表の順、--out-dir/<set>.parquet）")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", help="--set のときの出力")
    ap.add_argument("--out-dir", help="--sets / --all-sets のときの出力の dir")
    ap.add_argument("--subset", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--counts-cache", help="数の表の npz（あれば読み、無ければ作って書く）")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as F
    t0 = time.time()
    T: dict = {}
    table = Path(a.table)
    tb_all = pd.read_parquet(table, columns=TABLE_COLS)
    if a.set:
        if not a.out:
            ap.error("--set には --out が要る")
        sets = [a.set]
        outs = {a.set: Path(a.out)}
    else:
        if not a.out_dir:
            ap.error("--sets / --all-sets には --out-dir が要る")
        sets = list(dict.fromkeys(tb_all["set"])) if a.all_sets else a.sets.split(",")
        outs = {h: Path(a.out_dir) / f"{h}.parquet" for h in sets}
    missing = [h for h in sets if not (tb_all["set"] == h).any()]
    assert not missing, f"表に無い集合: {missing}"
    todo = [h for h in sets if not outs[h].exists()]
    print(f"sets={len(sets)} todo={len(todo)} (skip {len(sets) - len(todo)} existing)", flush=True)
    ts = pd.read_parquet(POOL / "train_structs.parquet", columns=["inchikey14", "smiles", "key"])
    st = pd.read_parquet(FOLDS, columns=["inchikey14", "fold_identity"]).drop_duplicates("inchikey14")
    fold_of = dict(zip(st.inchikey14, st.fold_identity))
    assert all(k in fold_of for k in ts.inchikey14), "structures.parquet に無い train 構造がある"
    fold_arr = np.array([fold_of[k] for k in ts.inchikey14], np.int64)
    use_raw = subset_keys(np.array(sorted(ts.inchikey14)), a.subset)
    use_sha = hashlib.sha256("\n".join(use_raw).encode()).hexdigest()
    T["load"] = round(time.time() - t0, 1)
    F.warmup()
    P.enumerate_fragments("CCOC(=O)c1ccccc1O", F)           # numba の初回の翻訳を測る時間から外す
    cache = Path(a.counts_cache) if a.counts_cache else None
    if cache is not None and cache.exists():
        t1 = time.time()
        tab, vocab, info = load_counts(cache)
        assert info["use_raw_sha256"] == use_sha and info["subset"] == int(len(use_raw)), "数の表の構造の組が違う"
        T["counts_load"] = round(time.time() - t1, 1)
        info = dict(info, loaded_from_cache=True)
    else:
        tab, vocab, n_spec, n_unp = build_counts(F, ts, use_raw, T, a.chunk)
        info = dict(subset=int(len(use_raw)), use_raw_sha256=use_sha, n_train_spectra_counted=int(n_spec),
                    n_unparsable_train=int(n_unp), n_records=int(len(tab.key)), n_record_names=int(len(vocab)),
                    chunk=int(a.chunk), secs_build=dict(T), peak_rss_gb_after_build=round(peak_rss_gb(), 3))
        if cache is not None:
            save_counts(cache, tab, vocab, info)
            info["cache_sha256"] = sha256_file(cache)
        info["loaded_from_cache"] = False
    print(json.dumps(dict(counts=info), ensure_ascii=False), flush=True)
    ctx = Context(F, ts, fold_arr, tab, vocab)
    last_use: dict = {}
    for i, h in enumerate(todo):
        for s in tb_all.loc[tb_all["set"] == h, "cand_smiles"].drop_duplicates():
            last_use[s] = i
    recs = []
    for i, h in enumerate(todo):
        tb = tb_all[tb_all["set"] == h]
        rec = run_set(ctx, h, tb, outs[h], last_use, i)
        rec.update(subset=int(len(use_raw)), train_counts=info, table_sha256=sha256_file(table), alpha=P.ALPHA,
                   beta=P.BETA, bin_width=P.BIN_WIDTH, note="数は部分の数（全 train では違う）" if a.subset else "",
                   secs_total=round(time.time() - t0, 1))
        outs[h].with_suffix(".json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(json.dumps(dict(set=h, secs=rec["secs"], peak_rss_gb=rec["peak_rss_gb"], cached=rec["n_cand_cached_after"]),
                         ensure_ascii=False), flush=True)
        recs.append(rec)
    print(json.dumps(dict(done=len(recs), secs_total=round(time.time() - t0, 1), peak_rss_gb=round(peak_rss_gb(), 3)),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
