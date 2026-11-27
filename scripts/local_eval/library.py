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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # instrument.py（#287）
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


REF_COLS = ["inchikey14", "ingest_lib", "ms2_mzs", "ms2_normalized_intensities"]
# #287（H2）: 参照の装置と付加イオンも読む版。既定の REF_COLS は変えない（neighbour_ref.py・対照でも使う）
REF_META_COLS = REF_COLS + ["instrument_type", "adduct"]


def _read_rows(path: Path, keys: set, batch_size: int = 8192, cols=None) -> pd.DataFrame:
    """train.parquet の inchikey14 ∈ keys の行（cols、既定 REF_COLS、ファイルの順）。読み込みバッファを小さくして、バッチごとに読んで絞る（#116:
    pq.read_table(filters=...) は最大常駐 12.5 GB だった。buffer_size=1 MiB・pre_buffer=False で約 0.8 GB）。
    中身と順は read_table(filters=isin) と同じ（outputs の比較で確かめた）。"""
    cols = list(REF_COLS if cols is None else cols)
    pf = pq.ParquetFile(path, buffer_size=1 << 20, pre_buffer=False)
    schema = pa.schema([pf.schema_arrow.field(c) for c in cols])
    vs = pa.array(sorted(keys), type=schema.field("inchikey14").type)
    parts = [b.filter(pc.is_in(b.column("inchikey14"), value_set=vs))
             for b in pf.iter_batches(batch_size=batch_size, columns=cols)]
    return pa.Table.from_batches(parts, schema=schema).to_pandas()


def load_refs(holdout: str, keys: set, cache: Path) -> dict:
    """keys の構造の参照スペクトル（隠す規則を適用済み）。key -> [(mz, p), ...]。cache に保存して再利用。"""
    if cache.is_file():
        df = pd.read_parquet(cache)
    else:
        rule, drop = hide_rule(holdout), drop_keys(holdout)
        t = _read_rows(ROOT / "data/train.parquet", keys)
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


# ---------------------------------------------------------------- #287 H2: 装置の系統の規則
def hide_rows(t: pd.DataFrame, rule: dict, drop: set) -> pd.DataFrame:
    """load_refs と同じ隠す規則（drop の inchikey14 の行。rule = enveda_libs なら enveda 系の行だけ）を当てた後の行。"""
    hid = t.inchikey14.isin(drop)
    if rule["rule"] == "enveda_libs":
        hid &= t.ingest_lib.isin(ENVEDA)
    elif rule["rule"] != "all":
        raise ValueError(f"知らない hide_rule: {rule['rule']}")
    return t[~hid].reset_index(drop=True)


def refs_with_meta(t: pd.DataFrame) -> dict:
    """隠す規則を当てた後の行（REF_META_COLS）→ key -> [(prep, 系統, 付加イオン), ...]。系統の欠けは None。"""
    from instrument import family, inst_value
    out: dict = {}
    for k, mz, it, ins, ad in zip(t.inchikey14, t.ms2_mzs, t.ms2_normalized_intensities, t.instrument_type, t.adduct):
        out.setdefault(k, []).append((_prep(mz, it), family(ins), inst_value(ad)))
    return out


def lib_scores_xinst(queries, cand_keys, refs_meta) -> dict:
    """候補ごとに (full, has_ref, xi, has_ref_xi, xi_sa)（#287 の定義）。queries = [dict(mz, it, inst, adduct)]。
    full・has_ref は今の lib_max_sim・has_ref の定義（系統の規則なし、作り直しの点検用）。
    xi = 問い合わせの系統と参照の系統がどちらもあり、違う組だけの最大。has_ref_xi = その組が 1 つ以上ある。
    xi_sa = そのうち参照の付加イオンが問い合わせと同じ組だけの最大。組が無ければ 0.0。"""
    from casmi import spectra as sp
    from casmi.library import SearchCfg
    from instrument import family, inst_value
    c = SearchCfg()
    qs = [(_prep(q["mz"], q["it"]), family(q.get("inst")), inst_value(q.get("adduct"))) for q in queries]
    out = {}
    for k in cand_keys:
        rs = refs_meta.get(k) or []
        full, xi, sa, has_xi = 0.0, 0.0, 0.0, 0
        first = True
        for (qm, qp), qf, qa in qs:
            for (rm, rp), rf, ra in rs:
                v = float(sp.entropy_sim(qm, qp, rm, rp, c.tol, c.ppm_tol))
                full = v if first else max(full, v)
                first = False
                if qf is None or rf is None or qf == rf:
                    continue
                xi = v if not has_xi else max(xi, v)
                has_xi = 1
                if qa is not None and ra is not None and qa == ra:
                    sa = max(sa, v)
        out[k] = (full, int(bool(rs)), xi, has_xi, sa)
    return out
