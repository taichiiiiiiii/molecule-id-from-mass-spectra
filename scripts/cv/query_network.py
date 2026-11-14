#!/usr/bin/env python3
"""
Issue #101: 評価集合の中で、問い合わせ同士のスペクトルのつながり（相互最近傍）を数える。測定だけで採否は決めない。

定義（Issue #101 の本文に事前登録したもの。結果を見て替えない）:
  - 問い合わせのスペクトル: scripts/local_eval/run.py の query_spectra と同じ（分子ごとに test.parquet の行順で先頭 16 本）
  - s_ij = 同じイオン化モードのスペクトルの組だけについて、エントロピー類似度（library.lib_scores と同じ
    casmi.spectra.entropy_sim、SearchCfg の tol・ppm_tol、_prep 済み）の最大値。組 (a, b) の値は
    max(sim(a,b), sim(b,a))。前駆体の質量のずれは補正しない（modified cosine ではない）
  - 共通のイオン化モードのスペクトルが無い組は s 未定義（NaN）で、つながりの候補から外す
  - 最近傍 = s 最大の相手、同点は molecule_id の文字列順で小さいほう。相互最近傍 = 互いに最近傍。しきい値なし
  - 正解の Tanimoto は casmi.chem.raw_fingerprint(smiles)[fp_bits]（run.py の候補の指紋と同じ選択ビット）
  - 分子式は RDKit の CalcMolFormula
  - visible の正解は train.parquet との完全一致（adduct, precursor_mz, ms2_mzs, ms2_normalized_intensities の値の sha1）

    .venv/bin/python scripts/cv/query_network.py --holdout v3_c1
    .venv/bin/python scripts/cv/query_network.py --visible
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QS = [0, 25, 50, 75, 100]
MAX_SPECTRA = 16


# ---------------------------------------------------------------- 純粋な関数（CI でテストする）
def pair_sim(specs_a: list, specs_b: list, sim: Callable) -> float:
    """同じ mode のスペクトルの組だけで max(sim(a,b), sim(b,a)) の最大値。比べる組が無ければ NaN。"""
    best = np.nan
    for a in specs_a:
        for b in specs_b:
            if a["mode"] != b["mode"]:
                continue
            v = max(float(sim(a, b)), float(sim(b, a)))
            if np.isnan(best) or v > best:
                best = v
    return best


def sim_matrix(ids: list, specs: dict, sim: Callable) -> np.ndarray:
    n = len(ids)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = pair_sim(specs[ids[i]], specs[ids[j]], sim)
    return S


def nearest(ids: list, S: np.ndarray) -> list:
    """各分子の最近傍の添字（無ければ None）。同点は molecule_id の文字列順で小さいほう。"""
    order = sorted(range(len(ids)), key=lambda k: ids[k])
    out = []
    for i in range(len(ids)):
        best, bj = None, None
        for j in order:                 # id 順に見て、厳密に大きいときだけ更新 → 同点は id の小さいほう
            if j == i or np.isnan(S[i, j]):
                continue
            if best is None or S[i, j] > best:
                best, bj = S[i, j], j
        out.append(bj)
    return out


def mutual_pairs(ids: list, S: np.ndarray) -> list:
    """相互最近傍の組 (i, j)（i < j の添字）。"""
    nn = nearest(ids, S)
    return [(i, j) for i, j in enumerate(nn) if j is not None and i < j and nn[j] == i]


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def quantiles(x) -> dict | None:
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if x.size == 0:
        return None
    return {f"q{q}": float(np.percentile(x, q)) for q in QS} | {"n": int(x.size)}


def spectrum_key(adduct, prec, mzs, its) -> str:
    s = repr(adduct) + "|" + repr(float(prec)) + "|" + ",".join(repr(float(v)) for v in mzs) + "|" + ",".join(repr(float(v)) for v in its)
    return hashlib.sha1(s.encode()).hexdigest()


# ---------------------------------------------------------------- 入力
def load_queries(path: Path) -> dict:
    """run.py の query_spectra と同じ並びと切り詰め（先頭 16 本）。"""
    import pandas as pd
    te = pd.read_parquet(path)
    out = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        out[str(mid)] = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                              adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == "positive" else -1)
                         for r in g.itertuples()][:MAX_SPECTRA]
    return out


def visible_truth(test_path: Path, train_path: Path) -> tuple[dict, dict]:
    """見えているテストの正解を train.parquet との完全一致で戻す。(mid -> smiles, 診断)。"""
    import pandas as pd
    import pyarrow.parquet as pq
    te = pd.read_parquet(test_path, columns=["molecule_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"])
    tkey = {spectrum_key(r.adduct, r.precursor_mz, r.ms2_mzs, r.ms2_normalized_intensities): str(r.molecule_id) for r in te.itertuples()}
    precs = set(float(v) for v in te.precursor_mz)
    hits: dict = {}   # mid -> {ik14: smiles}
    matched_spectra = 0
    pf = pq.ParquetFile(train_path)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["precursor_mz"]).column(0).to_numpy()
        rows = np.where(np.isin(t, list(precs)))[0]
        if rows.size == 0:
            continue
        sub = pf.read_row_group(g, columns=["adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
                                            "inchikey14", "normalized_smiles"]).take(rows).to_pandas()
        for r in sub.itertuples():
            mid = tkey.get(spectrum_key(r.adduct, r.precursor_mz, r.ms2_mzs, r.ms2_normalized_intensities))
            if mid is None:
                continue
            matched_spectra += 1
            hits.setdefault(mid, {}).setdefault(r.inchikey14, r.normalized_smiles)
    truth, n_none, n_amb = {}, 0, 0
    for mid in te.molecule_id.astype(str).unique():
        h = hits.get(mid, {})
        if len(h) == 1:
            truth[mid] = next(iter(h.values()))
        elif not h:
            n_none += 1
        else:
            n_amb += 1
    return truth, dict(n_test_spectra=len(te), n_matched_train_rows=matched_spectra, n_truth=len(truth),
                       n_no_match=n_none, n_ambiguous=n_amb)


# ---------------------------------------------------------------- 本体
def measure(name: str, specs: dict, truth: dict, sim: Callable, fp_of: Callable, formula_of: Callable, ik_of: Callable) -> dict:
    ids = list(specs)
    S = sim_matrix(ids, specs, sim)
    nn = nearest(ids, S)
    pairs = mutual_pairs(ids, S)
    iu = np.triu_indices(len(ids), 1)
    all_s = S[iu]
    fps = {m: fp_of(truth[m]) for m in ids if m in truth}
    fps = {m: f for m, f in fps.items() if f is not None}

    def tani(i, j):
        a, b = fps.get(ids[i]), fps.get(ids[j])
        return None if a is None or b is None else tanimoto(a, b)

    pt = [tani(i, j) for i, j in pairs]
    pt_ok = [v for v in pt if v is not None]
    base = [tani(i, j) for i, j in zip(*iu) if not np.isnan(S[i, j])]
    base_ok = [v for v in base if v is not None]
    same_f = same_k = n_both = 0
    for i, j in pairs:
        a, b = truth.get(ids[i]), truth.get(ids[j])
        if a is None or b is None:
            continue
        n_both += 1
        fa, fb = formula_of(a), formula_of(b)
        same_f += int(fa is not None and fa == fb)
        ka, kb = ik_of(a), ik_of(b)
        same_k += int(ka is not None and ka == kb)
    return dict(
        set=name, n_molecules=len(ids), n_spectra_used=int(sum(len(v) for v in specs.values())),
        n_with_nearest=int(sum(j is not None for j in nn)),
        n_mutual_pairs=len(pairs), n_connected_molecules=2 * len(pairs), frac_connected=2 * len(pairs) / len(ids) if ids else 0.0,
        s_mutual_pairs=quantiles([S[i, j] for i, j in pairs]),
        s_nearest=quantiles([S[i, j] for i, j in enumerate(nn) if j is not None]),
        s_all_defined_pairs=quantiles(all_s),
        n_defined_pairs=int((~np.isnan(all_s)).sum()), n_pairs=int(all_s.size),
        n_pairs_with_both_truth=n_both,
        tanimoto_mutual_pairs=quantiles(pt_ok),
        n_mutual_tanimoto_ge_0_5=int(sum(v >= 0.5 for v in pt_ok)), n_mutual_tanimoto_ge_0_7=int(sum(v >= 0.7 for v in pt_ok)),
        tanimoto_all_defined_pairs_baseline=quantiles(base_ok),
        frac_all_pairs_tanimoto_ge_0_5=float(np.mean([v >= 0.5 for v in base_ok])) if base_ok else None,
        frac_all_pairs_tanimoto_ge_0_7=float(np.mean([v >= 0.7 for v in base_ok])) if base_ok else None,
        n_mutual_same_formula=same_f, n_mutual_same_inchikey14=same_k,
        pairs=[dict(a=ids[i], b=ids[j], s=float(S[i, j]), tanimoto=t) for (i, j), t in zip(pairs, pt)],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--holdout")
    g.add_argument("--visible", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    import pandas as pd
    from casmi import chem
    from casmi import spectra as sp
    from casmi.library import SearchCfg
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    c = SearchCfg()
    name = "visible" if a.visible else a.holdout
    extra = {}
    if a.visible:
        specs = load_queries(ROOT / "data/test.parquet")
        truth, extra = visible_truth(ROOT / "data/test.parquet", ROOT / "data/train.parquet")
    else:
        specs = load_queries(ROOT / f"data/holdouts/{a.holdout}/test.parquet")
        sys.path.insert(0, str(ROOT / "scripts/local_eval"))
        import run as RUN                      # run.py の query_spectra と同じであることを確かめる
        ref = RUN.query_spectra(a.holdout)
        assert list(ref) == list(specs) and all(
            len(ref[m]) == len(specs[m]) and all(np.array_equal(x["mz"], y["mz"]) and x["mode"] == y["mode"]
                                                 for x, y in zip(ref[m], specs[m])) for m in ref)
        tr = pd.read_csv(ROOT / f"outputs/holdouts/{a.holdout}/truth.csv")
        truth = dict(zip(tr.molecule_id.astype(str), tr.smiles))
    for v in specs.values():   # library._prep と同じ前処理
        for q in v:
            q["pm"], q["pp"] = sp.prep_query(q["mz"], q["it"], c.floor, c.topk, c.power, c.ent_weight)

    def sim(x, y):
        return sp.entropy_sim(x["pm"], x["pp"], y["pm"], y["pp"], c.tol, c.ppm_tol)

    bits = np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy")

    def fp_of(smi):
        f = chem.raw_fingerprint(smi)
        return None if f is None else f[bits]

    def mol(smi):
        return Chem.MolFromSmiles(smi) if isinstance(smi, str) else None

    def formula_of(smi):
        m = mol(smi)
        return None if m is None else rdMolDescriptors.CalcMolFormula(m)

    def ik_of(smi):
        m = mol(smi)
        return None if m is None else Chem.MolToInchiKey(m)[:14]

    res = measure(name, specs, truth, sim, fp_of, formula_of, ik_of)
    res["truth_recovery"] = extra or dict(source=f"outputs/holdouts/{a.holdout}/truth.csv", n_truth=len(truth))
    res["seconds"] = round(time.time() - t0, 1)
    res["timed_out"] = res["seconds"] > 1800
    out = Path(a.out or ROOT / f"outputs/local_eval/query_network_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
