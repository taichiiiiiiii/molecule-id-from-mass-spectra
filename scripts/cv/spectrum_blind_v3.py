#!/usr/bin/env python3
"""
Issue #92: スペクトルを見ない対照（spectrum-blind、pop だけ）を検証 v3 の 4 集合で測る（記録のみ、採否なし）。

定義（#92 の本文で測る前に固定）:
  blind = 質量窓 pool.window(target)（target は outputs/holdout_eval/<h>_fold0/pc_logits.npz の __targets__、
          run.py と同じ ±PPM。hide_rule の pool=remove の集合は drop_keys を窓から除く）の全候補を
          pop（log1p 物質数 + log1p PubMed）の降順・同点は窓の中の元の順に並べ、run.dedup で key の重複を
          除いた上位 25。FP ロジット・frag は使わない。指紋が作れない候補も落とさない。
比べる既存の値: score（run.run(h, ["score"]) で計算）、score+prior・score+prior+frag（既存 result_*.json）。
v3_c2 は正解がプールに無いのに mrr25 > 0 なので、rr > 0 の分子とその理由（pool key と指標キーの違い）を列挙する。

    .venv/bin/python scripts/cv/spectrum_blind_v3.py [--out outputs/local_eval/spectrum_blind_v3.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
K = 25
EXPECT_SPF = {"v3_c1": 0.8535, "v3_c2": 0.0117, "enveda_v1": 0.8376, "v3_c2n": 0.6284}


def blind_rank(idx: np.ndarray, pop: np.ndarray, keys: np.ndarray, hidden: set, k: int = K) -> list[int]:
    """窓 idx（pool 行、元の順）から hidden のキーを除き、pop 降順（同点は元の順）で key の重複を除いた上位 k の行。"""
    import run as R
    idx = np.asarray(idx)
    if hidden:
        idx = idx[~np.isin(keys[idx], list(hidden))]
    order = np.argsort(-pop[idx], kind="stable")
    rows = [(float(pop[idx[o]]), int(idx[o])) for o in order]

    class _P:   # run.dedup は pool.key だけを読む
        key = keys
    return [i for _, i in R.dedup(rows, _P, k)]


def metrics(truth: dict[str, str], preds: dict[str, list[str]]) -> tuple[dict, dict[str, int]]:
    """mrr25（公式実装）・top1・hit@25 と、分子ごとの正解の順位（無ければ 0）。"""
    import mrr25 as M
    ranks = {}
    for mid, t in truth.items():
        tk, r = M.to_key(t), 0
        for j, c in enumerate(preds.get(mid, [])[:K], 1):
            if tk is not None and M.to_key(c) == tk:
                r = j
                break
        ranks[mid] = r
    n = len(truth)
    out = dict(mrr25=round(M.calculate_mrr25(truth, preds), 4),
               top1=round(sum(r == 1 for r in ranks.values()) / n, 4),
               hit25=round(sum(r > 0 for r in ranks.values()) / n, 4), n=n)
    return out, ranks


def read_sub(path: Path) -> dict[str, list[str]]:
    import mrr25 as M
    with open(path, newline="") as f:
        return {r["molecule_id"]: M.parse_candidates(r["smiles"]) for r in csv.DictReader(f)}


def explain_hits(h: str, ranks: dict[str, int], preds: dict[str, list[str]], truth_rows: dict[str, dict],
                 smi2key: dict[str, str], hidden: set) -> list[dict]:
    """rr > 0 の分子: 当たった候補の pool key と、正解の inchikey14・指標キー（tautomer 正準化後）を並べる。"""
    import mrr25 as M
    out = []
    for mid, r in ranks.items():
        if r <= 0:
            continue
        hit = preds[mid][r - 1]
        tr = truth_rows[mid]
        pk = smi2key.get(hit)
        out.append(dict(
            molecule_id=mid, rank=r, rr=round(1 / r, 4),
            hit_smiles=hit, hit_pool_key=pk, hit_metric_key=M.to_key(hit),
            truth_smiles=tr["smiles"], truth_inchikey14=tr["inchikey14"], truth_metric_key=M.to_key(tr["smiles"]),
            hit_pool_key_in_hidden=pk in hidden if pk is not None else None,
            truth_ik14_eq_metric_key=tr["inchikey14"] == M.to_key(tr["smiles"]),
            hit_pool_key_eq_truth_ik14=pk == tr["inchikey14"],
        ))
    return out


def raw_ik14(smi: str) -> str | None:
    """tautomer 正準化なしの InChIKey14（理由の切り分け用）。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m)[:14] if m is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/spectrum_blind_v3.json"))
    args = ap.parse_args()
    import run as R
    import library as L
    import mrr25 as M
    t0 = time.time()
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    res = dict(issue=92, definition="blind = window(target, ±%g ppm) minus hidden, pop desc (stable), dedup by pool key, top %d"
               % (R.PPM, K), holdouts={})
    for h in HOLDOUTS:
        zf = np.load(ROOT / f"outputs/holdout_eval/{h}_fold0/pc_logits.npz", allow_pickle=True)
        targets = {m: float(t) for m, t in zf["__targets__"]}
        truth = M.read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
        with open(ROOT / f"outputs/holdouts/{h}/truth.csv", newline="") as f:
            truth_rows = {r["molecule_id"]: r for r in csv.DictReader(f)}
        rule = L.hide_rule(h)
        hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
        preds, n_win, n_nofp = {}, [], 0
        for mid in truth:
            if mid not in targets:
                preds[mid] = []
                continue
            idx = pool.window(targets[mid])
            if hidden:
                n_win.append(int((~np.isin(pool.key[idx], list(hidden))).sum()))
            else:
                n_win.append(len(idx))
            rows = blind_rank(idx, pool.pop, pool.key, hidden)
            n_nofp += sum(fpc(pool.smiles[i]) is None for i in rows)
            preds[mid] = [pool.smiles[i] for i in rows]
        bm, branks = metrics(truth, preds)
        bm.update(n_no_target=sum(m not in targets for m in truth), window_median=float(np.median(n_win)),
                  window_le25=round(float(np.mean(np.array(n_win) <= K)), 4),
                  top25_without_fp=n_nofp)
        # 既存 stage
        ex = {}
        for tag in ("score+prior", "score+prior+frag"):
            ex[tag] = json.loads((ROOT / f"outputs/local_eval/{h}/result_{tag}.json").read_text())["mrr25"]
        rs = R.run(h, ["score"])
        ex["score"] = rs["mrr25"]
        # 既存 stage の top1・hit25 と v3_c2 の内訳（提出 csv から）
        stage_full, hits = {}, {}
        for tag in ("score", "score+prior", "score+prior+frag"):
            sp = read_sub(ROOT / f"outputs/local_eval/{h}/submission_{tag}.csv")
            sm, sr = metrics(truth, sp)
            assert abs(sm["mrr25"] - ex[tag]) < 1e-4, (h, tag, sm["mrr25"], ex[tag])
            stage_full[tag] = sm
            if h == "v3_c2":
                hits[tag] = (sr, sp)
        assert abs(ex["score+prior+frag"] - EXPECT_SPF[h]) < 1e-9, (h, ex["score+prior+frag"])
        entry = dict(hide_rule=rule, n_hidden=len(hidden), blind=bm, stages_mrr25=ex, stages=stage_full)
        if h == "v3_c2":
            smi_key = dict(zip(pool.smiles, pool.key))
            det = {}
            for tag, (sr, sp) in hits.items():
                det[tag] = explain_hits(h, sr, sp, truth_rows, smi_key, hidden)
            det["blind"] = explain_hits(h, branks, preds, truth_rows, smi_key, hidden)
            for lst in det.values():
                for d in lst:
                    d["hit_raw_ik14"] = raw_ik14(d["hit_smiles"])
                    d["truth_raw_ik14"] = raw_ik14(d["truth_smiles"])
            entry["rr_pos"] = det
        res["holdouts"][h] = entry
        print(h, json.dumps(dict(blind=bm, stages=ex)), flush=True)
    res["secs"] = round(time.time() - t0, 1)
    res["created"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
