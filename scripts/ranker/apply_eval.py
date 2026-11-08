#!/usr/bin/env python3
"""
ランカーの追加点検（Issue #29、採否の条件には入れない）: 保存済みモデル outputs/ranker/model.txt を別のホールドアウトに当て、
run.py の段の出力（submission_<tag>.csv）と MRR@25・差の 95% 区間を比べる。

    python scripts/ranker/apply_eval.py --holdout enveda_v1 --baselines score+prior,score+prior+lib+pubchem
出力: outputs/ranker/apply_<holdout>.json、outputs/local_eval/<holdout>/submission_ranker.csv

Issue #36 案 A（PubChem 入場門、11/9 事前登録）:
    python scripts/ranker/apply_eval.py --holdout v3_c1 --model outputs/ranker/model_c2p.txt --gate-lib --swap-check \\
        --tag gate --baselines score+prior+lib+pubchem,ranker_c2p
  --model      モデルファイル（ROOT からの相対。既定は #29 の outputs/ranker/model.txt）
  --tag        出力名の接尾辞（apply_<holdout>_<tag>.json / submission_ranker_<tag>.csv）。既定は空で従来の名前
  --gate-lib   lib_max ≥ GATE_TAU（run.py の PC_LIB_TAU = 0.479 系統の値 0.9。動かさない）の分子では PubChem 候補を除く
  --swap-check 差し替え試験 (ii): プール由来の正解行だけ lib_sim = SWAP_SIM（0.95、#36 本文の値）、その分子の全行の
               lib_max = max(lib_max, 0.95) にし、同じ構成（門の有無も同じ）で正解 1 位率の前後を出す。参考として門なしも出す
  ベースライン名は submission_<名前>.csv を読む（ranker_c2p = model_c2p の評価出力 submission_ranker_c2p.csv）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
import train_eval as T  # noqa: E402

GATE_TAU = 0.9    # run.py の PC_LIB_TAU（0.479 系統の LIB_TAU）と同じ値。tests で一致を確かめる
SWAP_SIM = 0.95   # #36 本文の差し替え値


def gate_pubchem(df: pd.DataFrame, tau: float = GATE_TAU) -> pd.DataFrame:
    """lib_max ≥ tau の分子から PubChem 候補（src_pubchem = 1）の行を除く。他の行と並びはそのまま。"""
    drop = (df["src_pubchem"] == 1) & (df["lib_max"] >= tau)
    return df[~drop.to_numpy()]


def swap_truth(df: pd.DataFrame, sim: float = SWAP_SIM) -> tuple[pd.DataFrame, int, int]:
    """差し替え (ii): プール由来の正解行の lib_sim = sim、その分子の全行の lib_max = max(lib_max, sim)。
    戻り値: (差し替え後の表, 差し替えた分子数, 対象外の PubChem 由来の正解がある分子数)。"""
    out = df.copy()
    hit = (out["label"] == 1) & (out["src_pubchem"] == 0)
    mids = set(out.loc[hit, "molecule_id"])
    out.loc[hit, "lib_sim"] = sim
    rows = out["molecule_id"].isin(mids)
    out.loc[rows, "lib_max"] = np.maximum(out.loc[rows, "lib_max"].to_numpy(), sim)
    n_pc = int(out[(out["label"] == 1) & (out["src_pubchem"] == 1)].molecule_id.nunique())
    return out, len(mids), n_pc


def top1_rate(df: pd.DataFrame, score_col: str, n_mol: int) -> float:
    """正解が score_col 降順の 1 位（同点は元の並び）の分子の割合。分母は全分子 n_mol。"""
    hits = 0
    for _, g in df.groupby("molecule_id", sort=False):
        g = g.iloc[np.argsort(-g[score_col].to_numpy(), kind="stable")]
        hits += int(g["label"].iloc[0] == 1)
    return round(hits / n_mol, 4)


def main() -> None:
    import lightgbm as lgb
    from mrr25 import read_truth
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--baselines", default="score+prior,score+prior+lib+pubchem,score+joint")
    ap.add_argument("--model", default="outputs/ranker/model.txt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gate-lib", action="store_true")
    ap.add_argument("--swap-check", action="store_true")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    model = lgb.Booster(model_file=str(ROOT / args.model))
    ev0 = T.load([args.holdout])
    ev0["ranker"] = model.predict(ev0[T.FEATURES])
    ev = gate_pubchem(ev0) if args.gate_lib else ev0
    truth = read_truth(ROOT / f"outputs/holdouts/{args.holdout}/truth.csv")
    preds = T.ranked_lists(ev, "ranker")
    with open(ROOT / f"outputs/local_eval/{args.holdout}/submission_ranker{sfx}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["molecule_id", "smiles"])
        for m in truth:
            w.writerow([m, ";".join(preds.get(m, []))])
    rr = T.per_molecule_rr(truth, preds)
    top1_src = ev.sort_values("ranker", ascending=False).groupby("molecule_id").src_pubchem.first()
    res = dict(holdout=args.holdout, n=len(truth), ranker=round(float(np.mean(list(rr.values()))), 4),
               top1_from_pubchem=int(top1_src.sum()), truth_in_pool=int(ev0[(ev0.label == 1) & (ev0.src_pubchem == 0)].molecule_id.nunique()),
               truth_in_pubchem=int(ev0[(ev0.label == 1) & (ev0.src_pubchem == 1)].molecule_id.nunique()))
    if args.tag or args.model != "outputs/ranker/model.txt":
        res["model"] = args.model
    if args.gate_lib:
        mol_lm = ev0.groupby("molecule_id").lib_max.first()
        pc_truth = ev0[(ev0.label == 1) & (ev0.src_pubchem == 1)]
        res.update(gate_tau=GATE_TAU, gate_closed_molecules=int((mol_lm >= GATE_TAU).sum()),
                   pubchem_truth_removed_by_gate=int(pc_truth[pc_truth.lib_max >= GATE_TAU].molecule_id.nunique()))
    if args.swap_check:
        sw, n_sw, n_pc = swap_truth(ev0)
        sw["ranker"] = model.predict(sw[T.FEATURES])
        n = len(truth)
        sc = dict(swap_sim=SWAP_SIM, swapped_molecules=n_sw, pubchem_truth_not_swapped=n_pc)
        for name, gated in (("same_config", args.gate_lib), ("no_gate_reference", False)):
            a = gate_pubchem(ev0) if gated else ev0
            b = gate_pubchem(sw) if gated else sw
            sc[name] = dict(gate=gated, top1_before=top1_rate(a, "ranker", n), top1_after=top1_rate(b, "ranker", n))
        sc["pass"] = bool(sc["same_config"]["top1_after"] >= sc["same_config"]["top1_before"])
        res["swap_check"] = sc
    for tag in args.baselines.split(","):
        p = ROOT / f"outputs/local_eval/{args.holdout}/submission_{tag}.csv"
        if not p.is_file():
            res[tag] = None
            continue
        rb = T.per_molecule_rr(truth, T.read_sub(p))
        res[tag] = round(float(np.mean(list(rb.values()))), 4)
        res[f"ranker_minus_{tag}"] = T.bootstrap_diff(rr, rb)
    (ROOT / f"outputs/ranker/apply_{args.holdout}{sfx}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
