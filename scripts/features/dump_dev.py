#!/usr/bin/env python3
"""
候補ごとの特徴表を dev（B8 の 20 集合）か holdout（v3 の 4 集合）でまとめて書き出す（Issue #244）。
集合ごとに `run.run(..., dump_features=...)`（= run.py --dump-features）を呼び、つないで漏れの点検をする。
正解のラベル・MRR は作らない。holdout は値を開かない（sha256・行数・NaN/inf 数だけを記録に書く）。

    .venv/bin/python scripts/features/dump_dev.py --which dev \
        --out outputs/local_eval/features/dev_b8.parquet
    .venv/bin/python scripts/features/dump_dev.py --which holdout \
        --out /Users/taichi/.hermes/profiles/massspec/cache/scratch/features_holdout/holdout_v3.parquet

途中から再開できる（集合ごとの parquet と記録の json が <out の親>/parts_<which>/ にあれば飛ばす）。
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
for p in ("scripts/features", "scripts/local_eval", "scripts/validate", "scripts/bench"):
    sys.path.insert(0, str(ROOT / p))

DEV_SETS = [f"b8_f{k}_{t}_t10" for k in (1, 2, 3, 4) for t in "abcde"]
HOLDOUT_SETS = ["v3_c1u_t10", "enveda_v1u_t10", "v3_c2_t10", "v3_c2nu_t10"]
STAGES = ["score", "prior", "frag"]
N_DEV, N_HOLD = 3901, 1358


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def q_rows(sets: list[str], out: Path) -> Path:
    """全集合の Q の鍵の train の行（参照）を train.parquet の 1 回の読みで集める（隠す規則は集合ごとに後で当てる）。"""
    import library as L
    if out.is_file():
        return out
    keys = set()
    for h in sets:
        pc = json.loads((ROOT / f"outputs/local_eval/{h}/pubchem_fold0.json").read_text())
        keys |= {k for v in pc.values() for _s, _f, k in (v or [])}
    df = L._read_rows(ROOT / "data/train.parquet", keys)
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    print(f"qrows: {len(keys)} keys -> {len(df)} train rows", flush=True)
    return out


def nan_inf(tab: pd.DataFrame) -> dict:
    num = tab.select_dtypes(include=[np.number])
    a = num.to_numpy(np.float64)
    return dict(nan={c: int(v) for c, v in zip(num.columns, np.isnan(a).sum(0)) if v},
                inf={c: int(v) for c, v in zip(num.columns, np.isinf(a).sum(0)) if v},
                n_nan=int(np.isnan(a).sum()), n_inf=int(np.isinf(a).sum()))


def leak_checks(tab: pd.DataFrame) -> dict:
    """#244 の漏れの点検 4 つ（dev の表）。"""
    import dump_features as DF
    import make_lists as ML
    import mrr25
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    dev = pd.read_csv(ROOT / "docs/bench/dev_list.csv.gz", dtype=str)
    dev = dev[dev.used_b8 == "1"]
    mols = tab[["set", "molecule_id"]].drop_duplicates()
    m = mols.merge(dev, left_on=["set", "molecule_id"], right_on=["b8_set", "b8_molecule_id"], how="left")
    found = m.inchikey14.notna()
    # (1) fold-0 の問い合わせが 0
    c1 = dict(query_fold_values=sorted(int(v) for v in tab.query_fold.unique()),
              n_molecules=int(len(mols)), n_found_in_dev_list=int(found.sum()),
              n_truth_fold0=int((m.fold[found].astype(int) == 0).sum()),
              n_fold_mismatch=int((m.fold[found].astype(int).values != m.set[found].str[4].astype(int).values).sum()))
    c1["ok"] = c1["query_fold_values"] and min(c1["query_fold_values"]) >= 1 and c1["n_truth_fold0"] == 0 \
        and c1["n_found_in_dev_list"] == c1["n_molecules"] and c1["n_fold_mismatch"] == 0
    # (2) 入れない列が無い
    bad = [c for c in tab.columns if c in DF.EXCLUDED or c.startswith(("src", "train_sid", "np_coconut", "b8_p"))]
    c2 = dict(excluded_present=bad, ok=not bad)
    # (3) model_fold == query_fold
    c3 = dict(n_model_fold_mismatch=int((tab.model_fold != tab.query_fold).sum()),
              n_np_model_fold_mismatch=int((tab.np_model_fold != tab.query_fold).sum()))
    c3["ok"] = c3["n_model_fold_mismatch"] == 0 and c3["n_np_model_fold_mismatch"] == 0
    # (4) dev ∩ holdout が 4 鍵で 0（make_lists と同じ鍵の作り方）
    st = pd.read_parquet(ML.FOLDS, columns=["inchikey14", "scorer_key", "identity_group", "fold_identity"])
    ts = pd.read_parquet(ML.TSTRUCT, columns=["inchikey14", "key"])
    ts_key = dict(zip(ts.inchikey14, ts.key))
    frames = []
    for h in ML.HOLDOUT_SETS + ML.ORIG + ML.A3A:
        tr = ML.holdout_truth(h)
        dp = ROOT / f"data/holdouts/{h}/drop_keys.txt"
        drop = set(dp.read_text().split()) if dp.is_file() else set()
        frames.append(ML.holdout_key_rows(tr, drop, st, ts_key, mrr25.to_key))
    hk = ML.key_sets(pd.concat(frames, ignore_index=True))
    ov = ML.overlap4(m[found].reset_index(drop=True), hk)
    c4 = dict(counts=ov["counts"], n_any=int(ov["any"].sum()), holdout_sets=ML.HOLDOUT_SETS + ML.ORIG + ML.A3A)
    c4["ok"] = c4["n_any"] == 0
    return {"1_no_fold0_queries": c1, "2_no_excluded_columns": c2, "3_model_fold_eq_query_fold": c3,
            "4_dev_holdout_overlap_4keys": c4, "all_ok": bool(c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"])}


def main() -> None:
    import run as R
    import dump_features as DF
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dev", "holdout"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h2", action="store_true", help="#287: lib 系の列を装置の系統の規則で作り直した列に置き換える")
    ap.add_argument("--sets", default=None, help="カンマ区切りで一部の集合だけ（試し用。つなぎと点検はしない）")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out)
    sets = DEV_SETS if a.which == "dev" else HOLDOUT_SETS
    if a.sets:
        sets = a.sets.split(",")
    parts = out.parent / (f"parts_{a.which}" + ("_h2dump" if a.h2 else ""))
    parts.mkdir(parents=True, exist_ok=True)
    qp = q_rows(DEV_SETS if a.which == "dev" else HOLDOUT_SETS, parts / "qrows.parquet")
    scorers: dict = {}
    recs = []
    for h in sets:
        pq_, rj = parts / f"{h}.parquet", parts / f"{h}.json"
        if pq_.is_file() and rj.is_file():
            recs.append(json.loads(rj.read_text()))
            continue
        _t, fold = DF.set_meta(h)
        if fold not in scorers:
            scorers[fold] = DF.NPScorer(fold)
        d = ROOT / f"outputs/local_eval/{h}"
        opts = dict(qrows=str(qp), np_scorer=scorers[fold], h2=a.h2)
        if a.which == "holdout":
            opts["refs_cache"] = str(parts / f"{h}_refs_top60.parquet")
        rec = R.run(h, STAGES, logits_dir=d, truth_dir=d, dump_features=pq_, dump_opts=opts)
        assert rec["control_csv_identical"] is True, f"{h}: 対照の submission とバイト一致しない"
        rj.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        recs.append(rec)
        print(json.dumps({k: rec[k] for k in ("holdout", "n_molecules", "n_rows", "n_nan", "n_inf", "secs")}), flush=True)
    if a.sets:
        return
    tab = pd.concat([pd.read_parquet(parts / f"{h}.parquet") for h in sets], ignore_index=True)
    tmp = out.with_suffix(".tmp.parquet")
    tab.to_parquet(tmp, index=False)
    tmp.replace(out)
    rep = dict(issue=244, which=a.which, out=str(out), out_sha256=sha256_file(out), n_molecules=int(len(tab[["set", "molecule_id"]].drop_duplicates())),
               n_rows=int(len(tab)), n_columns=int(tab.shape[1]), columns=list(tab.columns), sets={r["holdout"]: r for r in recs})
    if a.which == "dev":
        b8 = 0
        for h in sets:
            j = json.loads((ROOT / f"outputs/local_eval/{h}/b8_features.json").read_text())
            b8 += sum(int(m["n_L"]) + int(m["n_Q"]) for m in j["molecules"] if not m.get("missing_logits"))
        rep["b8_sum_nL_nQ"] = b8
        rep["rows_match_b8"] = b8 == len(tab)
        rep["n_molecules_ok"] = rep["n_molecules"] == N_DEV
        rep["nan_inf"] = nan_inf(tab)
        rep["leak"] = leak_checks(tab)
    else:
        rep["n_molecules_ok"] = rep["n_molecules"] == N_HOLD
        ni = nan_inf(tab)
        rep["nan_inf"] = dict(n_nan=ni["n_nan"], n_inf=ni["n_inf"], nan_columns=sorted(ni["nan"]), inf_columns=sorted(ni["inf"]))
        q = tab[["query_fold", "model_fold", "np_model_fold"]]
        rep["folds"] = dict(query_fold=sorted(int(v) for v in q.query_fold.unique()),
                            all_model_eq_query=bool(((q.model_fold == q.query_fold) & (q.np_model_fold == q.query_fold)).all()))
        rep.pop("sets")   # holdout の集合ごとの記録は parts の json に置く（値は無いが、まとめには数だけ）
        rep["sets_n"] = {r["holdout"]: dict(n_molecules=r["n_molecules"], n_rows=r["n_rows"],
                                            control_csv_identical=r["control_csv_identical"]) for r in recs}
    rep["secs"] = round(time.time() - t0, 1)
    rp = out.with_name(out.stem + "_report.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("columns", "sets")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
