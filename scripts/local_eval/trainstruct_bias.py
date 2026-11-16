#!/usr/bin/env python3
"""
Issue #120 の記録用の診断 (ii): fragnn の点が「fragnn の学習に使った構造」の候補を低く（高く）出すか（採否には使わない）。

定義は #120 の事前登録のコメント（2026-11-16T14:50:50Z、独立レビュー PASS）のとおり。要点:
- 集合は見えているテスト（holdout 名 sub_test）と v3_c1 を別々に。読むのは sub_test の fragnn の値だけ
- 学習の構造の集合 T = prep.npz の学習の正解（is_val が偽の pos_struct）∪ 学習の負例（flat_tr）を struct_rows で
  pool_meta の元の行に戻した、その行の pool_meta の key の集合（内部検証は入れない）。キーの集合として比べる
- 候補は run.py の score 段と同じ窓を pool の key でまとめ（点は最大）、fragnn の点 n が有限のものを分子の中で z（ddof=0）
- d = mean(z | key ∈ T) − mean(z | key ∉ T)。両方の群がある分子だけで、平均・95% 区間（bootstrap 2,000 回、seed 0）・n
- 読み方: sub_test の fragnn で区間の上限 < 0 → "lower_train"（freeze 型を次の回で検討）、それ以外 → "record_only"
- fragnnblind でも同じ数を出して並べる（記録だけ）

    python scripts/local_eval/trainstruct_bias.py --holdouts sub_test,v3_c1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))

READ_HOLDOUT = "sub_test"
PREP = ROOT / "outputs/fragnn_train/prep.npz"
EXPECTED_T = 214_772   # #120 の学習の正解構造の数（train 由来の負例は正解構造の中に入る）


def train_struct_rows(struct_rows: np.ndarray, pos_struct: np.ndarray, is_val: np.ndarray, flat_tr: np.ndarray) -> np.ndarray:
    """学習の正解と学習の負例の和集合を pool_meta の元の行番号で返す（内部検証の pos_struct・flat_va は入れない）。"""
    idx = np.unique(np.concatenate([np.asarray(pos_struct)[~np.asarray(is_val, bool)], np.asarray(flat_tr)]))
    return np.asarray(struct_rows)[idx]


def train_struct_keys(prep_path: Path = PREP, meta_path: Path | None = None) -> tuple[set, dict]:
    import pandas as pd
    z = np.load(prep_path, allow_pickle=False)
    rows = train_struct_rows(z["struct_rows"], z["pos_struct"], z["is_val"], z["flat_tr"])
    meta = pd.read_parquet(meta_path or ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key", "smiles"])
    keys = meta["key"].values[rows]
    return set(keys.tolist()), dict(n_rows=int(len(rows)), n_keys=int(len(set(keys.tolist()))),
                                    smiles=meta["smiles"].values[rows], keys=keys)


def key_mismatch(smiles, keys, key_fn) -> int:
    """key_fn(smiles)（chem.score_key）と pool の key の食い違いの数（記録だけ）。"""
    return int(sum(1 for s, k in zip(smiles, keys) if key_fn(s) != k))


def molecule_bias(cands: dict[str, float], T: set) -> tuple[float | None, dict]:
    """cands = {key: 点}。有限の点だけを z（ddof=0）にし、d = mean(z∈T) − mean(z∉T)。
    外すとき (None, {"excluded": 理由})。"""
    ks = [k for k, v in cands.items() if v is not None and math.isfinite(v)]
    if len(ks) < 2:
        return None, dict(excluded="few_scored", n_scored=len(ks))
    v = np.array([cands[k] for k in ks], np.float64)
    sd = v.std()
    if sd == 0:
        return None, dict(excluded="sd0", n_scored=len(ks))
    zz = (v - v.mean()) / sd
    inn = np.array([k in T for k in ks])
    if inn.all() or not inn.any():
        return None, dict(excluded="one_group", n_scored=len(ks), n_in=int(inn.sum()))
    return float(zz[inn].mean() - zz[~inn].mean()), dict(n_scored=len(ks), n_in=int(inn.sum()), n_out=int((~inn).sum()))


def summarize(d: dict[str, float], infos: dict[str, dict]) -> dict:
    from train_eval import bootstrap_diff
    ex = {}
    for i in infos.values():
        if "excluded" in i:
            ex[i["excluded"]] = ex.get(i["excluded"], 0) + 1
    out = dict(n=len(d), excluded=ex)
    if d:
        b = bootstrap_diff(d, {m: 0.0 for m in d})
        out.update(mean_d=b["diff"], lo=b["lo"], hi=b["hi"],
                   n_in_median=float(np.median([infos[m]["n_in"] for m in d])),
                   n_out_median=float(np.median([infos[m]["n_out"] for m in d])))
    return out


def reading(summary: dict) -> str:
    """事前登録の読み方: 区間の上限 < 0 なら lower_train、それ以外は record_only。"""
    return "lower_train" if summary.get("n", 0) > 0 and summary["hi"] < 0 else "record_only"


def run_holdout(holdout: str, T: set) -> dict:
    import iso_rank_diag as IR
    t0 = time.time()
    cands, _ = IR.window_candidates(holdout)
    res = dict(holdout=holdout, read=holdout == READ_HOLDOUT, n_molecules=len(cands))
    for c, name in (("n", "fragnn"), ("nb", "fragnnblind")):
        d, infos = {}, {}
        for mid, cd in cands.items():
            v, info = molecule_bias({k: a[c] for k, a in cd.items()}, T)
            infos[mid] = info
            if v is not None:
                d[mid] = v
        res[name] = summarize(d, infos)
    if holdout == READ_HOLDOUT:
        res["reading"] = reading(res["fragnn"])
    res["secs"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", default="sub_test,v3_c1")
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/trainstruct_bias.json"))
    a = ap.parse_args()
    from casmi import chem
    T, tinfo = train_struct_keys()
    assert tinfo["n_rows"] == EXPECTED_T, tinfo["n_rows"]
    t1 = time.time()
    mism = key_mismatch(tinfo["smiles"], tinfo["keys"], chem.score_key)
    res = dict(T=dict(n_rows=tinfo["n_rows"], n_keys=tinfo["n_keys"], score_key_mismatch=mism,
                      score_key_secs=round(time.time() - t1, 1)))
    res.update({h: run_holdout(h, T) for h in a.holdouts.split(",")})
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
