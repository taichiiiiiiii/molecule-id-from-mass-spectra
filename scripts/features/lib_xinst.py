#!/usr/bin/env python3
"""
H2（Issue #287）: lib_max_sim を「問い合わせと違う系統の装置の参照だけ」で作り直した 3 列を、dev の集合ごとに parts に書く。
定義は #287 の本文（当てはめの前に固定）と scripts/local_eval/library.lib_scores_xinst。

  lib_max_sim_xi     問い合わせの系統と参照の系統がどちらもあり、違う組 (q, r) の entropy 類似度の最大（無ければ 0.0）
  has_ref_xi         その組が 1 つ以上あれば 1
  lib_max_sim_xi_sa  そのうち参照の付加イオンが問い合わせと同じ組だけの最大（無ければ 0.0）

問い合わせ = run.query_spectra と同じ（分子ごとに test.parquet の先頭 16 本）＋ その行の instrument_type。
参照 = train.parquet の候補の鍵の行（REF_META_COLS）に、load_refs と同じ隠す規則（library.hide_rows）を当てた後。
L と Q を同じ規則で扱う（dump_features の L = load_refs・Q = q_refs はどちらもこの規則）。

作り直しの点検: 同じ参照・同じ問い合わせで系統の規則を外した値（今の lib_max_sim・has_ref の定義）が、入力の表の
lib_max_sim（|差| ≤ 1e-12）・has_ref（完全一致）と全行で一致しなければ止める（part を書かない）。

    .venv/bin/python scripts/features/lib_xinst.py \
        --table outputs/local_eval/features/dev_b8_cc_gl2_n1.parquet \
        --parts outputs/local_eval/features/parts_h2_dev [--sets b8_f1_a_t10]

途中から再開できる（<set>.parquet と <set>.json があれば飛ばす）。正解のラベル・MRR は使わない。
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
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

DEV_SETS = tuple(f"b8_f{k}_{t}_t10" for k in (1, 2, 3, 4) for t in "abcde")
KEY = ["set", "molecule_id", "cand_key"]
OUT_COLS = ("lib_max_sim_xi", "has_ref_xi", "lib_max_sim_xi_sa")
N_QUERY = 16   # run.query_spectra と同じ


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def query_spectra_meta(te: pd.DataFrame) -> dict:
    """run.query_spectra と同じ問い合わせ（分子ごとに先頭 16 本）に instrument_type を足した dict のリスト。"""
    out = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        out[str(mid)] = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                              adduct=r.adduct, inst=r.instrument_type) for r in g.itertuples()][:N_QUERY]
    return out


def set_frame(g: pd.DataFrame, queries: dict, refs_meta: dict) -> tuple[pd.DataFrame, dict]:
    """1 集合の行 g（表の順、列 set・molecule_id・cand_key・lib_max_sim・has_ref）の 3 列と作り直しの点検の数。"""
    import library as L
    n = len(g)
    full, has, xi, hxi, sa = (np.zeros(n), np.zeros(n, np.int8), np.zeros(n), np.zeros(n, np.int8), np.zeros(n))
    mids = g.molecule_id.astype(str).to_numpy()
    keys = g.cand_key.to_numpy()
    for mid, idx in pd.Series(range(n)).groupby(mids, sort=False).indices.items():
        res = L.lib_scores_xinst(queries.get(mid, []), list(keys[idx]), refs_meta)
        for j in idx:
            f_, h_, x_, hx_, s_ = res[keys[j]]
            full[j], has[j], xi[j], hxi[j], sa[j] = f_, h_, x_, hx_, s_
    d_full = np.abs(full - g.lib_max_sim.to_numpy(np.float64))
    chk = dict(n_rows=int(n), max_abs_diff_lib_max_sim=float(d_full.max()) if n else 0.0,
               n_mismatch_lib_max_sim=int((d_full > 1e-12).sum()),
               n_mismatch_has_ref=int((has != g.has_ref.to_numpy().astype(np.int8)).sum()),
               n_has_ref=int(has.sum()), n_has_ref_xi=int(hxi.sum()), n_sa_pos=int((sa > 0).sum()),
               n_xi_lt_full=int((xi < full).sum()))
    out = pd.DataFrame({"set": g["set"].to_numpy(), "molecule_id": g.molecule_id.to_numpy(), "cand_key": keys,
                        "lib_max_sim_xi": xi, "has_ref_xi": hxi, "lib_max_sim_xi_sa": sa})
    return out, chk


def run_set(h: str, table: Path, pdir: Path, table_sha: str) -> dict:
    import library as L
    from instrument import family
    t0 = time.time()
    T: dict = {}
    g = pd.read_parquet(table, columns=KEY + ["lib_max_sim", "has_ref"], filters=[("set", "==", h)])
    assert len(g), f"{h}: 表に行が無い"
    te = pd.read_parquet(ROOT / f"data/holdouts/{h}/test.parquet")
    queries = query_spectra_meta(te)
    rule, drop = L.hide_rule(h), L.drop_keys(h)
    t1 = time.time()
    raw = L._read_rows(ROOT / "data/train.parquet", set(g.cand_key), cols=L.REF_META_COLS)
    T["read_train"] = round(time.time() - t1, 1)
    kept = L.hide_rows(raw, rule, drop)
    refs_meta = L.refs_with_meta(kept)
    t2 = time.time()
    out, chk = set_frame(g, queries, refs_meta)
    T["sims"] = round(time.time() - t2, 1)
    if chk["n_mismatch_lib_max_sim"] or chk["n_mismatch_has_ref"]:
        raise AssertionError(f"{h}: 作り直しの点検に通らない {chk}")
    z = out.has_ref_xi == 0
    if (z & ((out.lib_max_sim_xi != 0) | (out.lib_max_sim_xi_sa != 0))).any():
        raise AssertionError(f"{h}: has_ref_xi = 0 の行に値がある")
    vals = out[list(OUT_COLS)].to_numpy(np.float64)
    if not np.isfinite(vals).all():
        raise AssertionError(f"{h}: NaN か inf がある")
    pp = pdir / f"{h}.parquet"
    tmp = pp.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(pp)
    qf = pd.Series([family(q["inst"]) for v in queries.values() for q in v], dtype=object)
    rf = pd.Series([f for v in refs_meta.values() for _p, f, _a in v], dtype=object)
    rec = dict(set=h, rule=rule["rule"], n_drop_keys=len(drop), n_train_rows_read=int(len(raw)),
               n_train_rows_kept=int(len(kept)), n_molecules=int(out.molecule_id.nunique()),
               n_molecules_with_query=int(sum(1 for m in out.molecule_id.astype(str).unique() if queries.get(m))),
               query_family_counts={str(k): int(v) for k, v in qf.fillna("<none>").value_counts().items()},
               ref_family_counts={str(k): int(v) for k, v in rf.fillna("<none>").value_counts().items()},
               check=chk, secs=T, secs_total=round(time.time() - t0, 1), table_sha256=table_sha,
               out_sha256=sha256_file(pp))
    (pdir / f"{h}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--sets", default=None, help="カンマ区切り（既定は dev の 20 集合）")
    a = ap.parse_args(argv)
    table, pdir = Path(a.table), Path(a.parts)
    pdir.mkdir(parents=True, exist_ok=True)
    sets = a.sets.split(",") if a.sets else list(DEV_SETS)
    table_sha = sha256_file(table)
    for h in sets:
        if (pdir / f"{h}.parquet").is_file() and (pdir / f"{h}.json").is_file():
            meta = json.loads((pdir / f"{h}.json").read_text())
            if meta["table_sha256"] != table_sha:
                raise RuntimeError(f"{h}: 既にある part は別の表から作られた")
            print(f"skip {h}", flush=True)
            continue
        rec = run_set(h, table, pdir, table_sha)
        print(json.dumps({k: rec[k] for k in ("set", "n_train_rows_read", "n_train_rows_kept", "check", "secs_total")},
                         ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
