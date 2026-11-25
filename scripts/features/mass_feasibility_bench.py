#!/usr/bin/env python3
"""
mass_feasibility.py の速さの測定（Issue #238 手順 4）。**特徴の値は読まない・表示しない**（G1 承認前）。

dev の 1 集合（既定 b8_f1_a_t10）について、scripts/local_eval/run.py と同じ関数（読むだけで変えない）で
score,prior の上位 TOPN（60）の候補の並びを作り、その集合のスペクトル（run.py の query_spectra と同じ形・同じ上限）
で 26 列を計算する。記録するのは時間・件数・NaN/inf の数だけ。truth.csv は読まない。
特徴表は --out（scratch）に書く。記録の後に消す。

    /usr/bin/time -l .venv/bin/python scripts/features/mass_feasibility_bench.py \\
        --holdout b8_f1_a_t10 --out <scratch>/mf_b8_f1_a_t10.parquet --report <scratch>/mf_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/features"))


def candidate_lists(holdout: str, logits: Path):
    """run.py の score → prior → dedup（上位 TOPN）と同じ並び。分子の一覧は pc_logits.npz の __targets__ と
    test.parquet の分子の共通部分（truth.csv は読まない）。"""
    import run as R
    import library as L
    zf = np.load(logits, allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()
    spectra = R.query_spectra(holdout)
    lists = {}
    for mid in spectra:
        if mid not in targets:
            continue
        idx = R.visible_window(pool, targets[mid], hidden)
        rows = R.score_window(idx, pool, R.stage_z(zf[mid], None), fpc)
        rows = R.rerank_with_prior(rows, pool)
        lists[mid] = [pool.smiles[i] for _, i in R.dedup(rows, pool, R.TOPN)]
    return lists, spectra, dict(rule=rule, n_hidden=len(hidden))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="b8_f1_a_t10")
    ap.add_argument("--logits", default=None, help="既定 outputs/local_eval/<holdout>/pc_logits.npz")
    ap.add_argument("--out", required=True, help="特徴表の parquet（scratch に置く）")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    if "v3" in a.holdout:
        raise SystemExit("v3（holdout）には当てない（#238）")
    import mass_feasibility as mf
    t0 = time.time()
    lists, spectra, info = candidate_lists(a.holdout, Path(a.logits) if a.logits else
                                           ROOT / f"outputs/local_eval/{a.holdout}/pc_logits.npz")
    t_cand = time.time() - t0
    st, parts, per_mol = {}, [], []
    t1 = time.time()
    for mid, smis in lists.items():
        tm = time.time()
        X = mf.group_features(smis, spectra.get(mid, []), st)
        per_mol.append(time.time() - tm)
        df = pd.DataFrame(X, columns=mf.feature_names())
        df.insert(0, "molecule_id", mid); df.insert(1, "cand_rank", np.arange(len(smis)))
        parts.append(df)
    t_feat = time.time() - t1
    tab = pd.concat(parts, ignore_index=True)
    vals = tab[mf.feature_names()].to_numpy()
    n_nan, n_inf = int(np.isnan(vals).sum()), int(np.isinf(vals).sum())
    tab.to_parquet(a.out, index=False)
    n_spec = sum(len(spectra.get(m, [])) for m in lists)
    rep = dict(holdout=a.holdout, n_molecules=len(lists), n_candidates=int(len(tab)), n_spectra=n_spec,
               n_columns=len(mf.feature_names()), n_nan=n_nan, n_inf=n_inf,
               secs_candidates=round(t_cand, 1), secs_features=round(t_feat, 2),
               s_per_molecule=round(t_feat / max(len(lists), 1), 4),
               s_per_molecule_median=round(float(np.median(per_mol)), 4),
               s_per_molecule_max=round(float(np.max(per_mol)), 4),
               s_per_spectrum_candidate=round(t_feat / max(sum(len(spectra.get(m, [])) * len(s) for m, s in lists.items()), 1), 7),
               proposal_cache=mf.proposals.cache_info()._asdict(), **st, **info)
    Path(a.report).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    assert n_nan == 0 and n_inf == 0, "NaN/inf がある"


if __name__ == "__main__":
    main()
