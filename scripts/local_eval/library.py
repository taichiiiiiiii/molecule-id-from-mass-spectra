"""
ローカル評価環境のライブラリ照合（Issue #25）。0.479 版と同じエントロピー類似度（casmi.spectra.entropy_sim、
SearchCfg の既定値）で、問い合わせスペクトルと候補の参照スペクトル（学習データ、隠した行を除く）を比べる。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
ENVEDA = {"enveda-180", "enveda-np-examples"}


def hide_rule(holdout: str) -> dict:
    p = ROOT / f"data/holdouts/{holdout}/hide_rule.json"
    return json.loads(p.read_text()) if p.is_file() else dict(rule="all", pool="keep")


def drop_keys(holdout: str) -> set:
    p = ROOT / f"data/holdouts/{holdout}/drop_keys.txt"
    return set(p.read_text().split()) if p.is_file() else set()


def _prep(mz, it):
    from casmi import spectra as sp
    from casmi.library import SearchCfg
    c = SearchCfg()
    return sp.prep_query(np.asarray(mz, np.float64), np.asarray(it, np.float64), c.floor, c.topk, c.power, c.ent_weight)


def load_refs(holdout: str, keys: set, cache: Path) -> dict:
    """keys の構造の参照スペクトル（隠す規則を適用済み）。key -> [(mz, p), ...]。cache に保存して再利用。"""
    if cache.is_file():
        df = pd.read_parquet(cache)
    else:
        rule, drop = hide_rule(holdout), drop_keys(holdout)
        t = pq.read_table(ROOT / "data/train.parquet", columns=["inchikey14", "ingest_lib", "ms2_mzs", "ms2_normalized_intensities"],
                          filters=pc.field("inchikey14").isin(sorted(keys))).to_pandas()
        hid = t.inchikey14.isin(drop)
        if rule["rule"] == "enveda_libs":
            hid &= t.ingest_lib.isin(ENVEDA)
        df = t[~hid].reset_index(drop=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    out: dict = {}
    for k, mz, it in zip(df.inchikey14, df.ms2_mzs, df.ms2_normalized_intensities):
        out.setdefault(k, []).append(_prep(mz, it))
    return out


def lib_scores(queries, cand_keys, refs) -> dict:
    """候補ごとに、問い合わせ（複数）× 参照（複数）のエントロピー類似度の最大値。参照の無い候補は含めない。"""
    from casmi import spectra as sp
    from casmi.library import SearchCfg
    c = SearchCfg()
    qs = [_prep(q["mz"], q["it"]) for q in queries]
    out = {}
    for k in cand_keys:
        rs = refs.get(k)
        if not rs:
            continue
        out[k] = max(float(sp.entropy_sim(qm, qp, rm, rp, c.tol, c.ppm_tol)) for qm, qp in qs for rm, rp in rs)
    return out
