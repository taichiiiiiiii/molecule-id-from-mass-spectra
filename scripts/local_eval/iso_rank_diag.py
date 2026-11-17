#!/usr/bin/env python3
"""
Issue #120 の記録用の診断 (i): 同じ分子式の中の順位（採否には使わない）。

定義は #120 の事前登録のコメント（2026-11-16T14:50:50Z、独立レビュー PASS）のとおり。要点:
- 候補は run.py の score 段と同じ ±10 ppm 窓（visible_window → score_window）。pool の key でまとめ、
  点 s・f・n・n_blind は同じキーの全行の最大。代表の SMILES は Pool の並びで最初の行
- G = 正解と同じ分子式（代表の SMILES の CalcMolFormula）の候補。正解の式も同じ規則
- 外す分子: 正解が窓に無い・正解の式が作れない・|G| < 2（数を記録）
- 点 (A) s、(B) MetFrag-lite の f、(C) fragnn の n、(D) fragnnblind の n。同点は悲観的、点の無い候補は下
- 主の指標: G の中の逆数順位の平均。(C−A)・(C−B)・(C−D) を bootstrap_diff（2,000 回、seed 0）
- 読むのは v3_c1 だけ。ほかの集合は記録だけ
- 追加の数え上げ: 対照（score+prior+frag）の (c) のうち新（score+fragnn+prior+frag）で 1 位になった数、
  対照で 1 位だった分子のうち新で 1 位でなくなった数（#118 の miss_classify と同じ定義）

fragnn の点は sanpier（#60）と同じ形の npz（outputs/local_eval/<h>/fragnn_scores.npz・fragnnblind_scores.npz）
から読む。その npz を作る fragnn 段は別の PR（この PR の範囲外）。

    python scripts/local_eval/iso_rank_diag.py --holdouts v3_c1,v3_c2,enveda_v1,v3_c2n
"""
from __future__ import annotations

import argparse
import csv
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

N_BOOT, SEED = 2000, 0
SCORES = ("s", "f", "n", "nb")          # (A) FPNet (B) MetFrag-lite (C) fragnn (D) fragnnblind
NAMES = dict(s="A_fpnet", f="B_metfrag", n="C_fragnn", nb="D_fragnnblind")
DIFFS = (("n", "s"), ("n", "f"), ("n", "nb"))   # (C−A)・(C−B)・(C−D)
READ_HOLDOUT = "v3_c1"
CONTROL_TAG, NEW_TAG = "score+prior+frag", "score+fragnn+prior+frag"


def _num(v) -> float:
    """None・NaN・無限大は点なし（NaN）にする。"""
    if v is None:
        return math.nan
    v = float(v)
    return v if math.isfinite(v) else math.nan


def _nanmax(a: float, b: float) -> float:
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)


def aggregate_by_key(rows: list[dict]) -> dict[str, dict]:
    """rows = [{key, row（Pool の添字）, smiles, s, n, nb}] → {key: {rep_row, smiles, rows, s, n, nb}}。
    点はキーの中の最大（点なしは除く）。代表は Pool の並びで最初の行（row が最小）。"""
    out: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: r["row"]):
        k = r["key"]
        if k not in out:
            out[k] = dict(rep_row=r["row"], smiles=r["smiles"], rows=[], smiles_all=[],
                          **{c: math.nan for c in SCORES})
        a = out[k]
        a["rows"].append(r["row"]); a["smiles_all"].append(r["smiles"])
        for c in ("s", "n", "nb"):
            a[c] = _nanmax(a[c], _num(r.get(c)))
    return out


def build_group(cands: dict[str, dict], truth_key: str | None, formula_fn) -> tuple[list[str] | None, dict]:
    """G（キーの並び、正解を含む）と記録。外す分子は (None, {"excluded": 理由})。
    formula_fn(smiles) → 式か None。同じキーの行で式が食い違う数も数える。"""
    info = dict(n_window_keys=len(cands), n_formula_none=0, n_formula_conflict=0)
    if truth_key is None or truth_key not in cands:
        return None, dict(info, excluded="truth_absent")
    form = {}
    for k, c in cands.items():
        f = formula_fn(c["smiles"])
        if f is None:
            info["n_formula_none"] += 1
        else:
            form[k] = f
        if len({formula_fn(s) for s in c["smiles_all"]}) > 1:
            info["n_formula_conflict"] += 1
    tf = form.get(truth_key)
    if tf is None:
        return None, dict(info, excluded="truth_formula_none")
    G = [k for k in cands if form.get(k) == tf]
    if len(G) < 2:
        return None, dict(info, excluded="g_lt2", g_size=len(G))
    return G, dict(info, g_size=len(G), truth_formula=tf)


def pessimistic_rank(scores: list[float], ti: int) -> int:
    """scores の中の ti 番目（正解）の順位。同点は正解を最後に、点なし（NaN）は点のある候補の下。
    正解に点が無ければ最下位（len(scores)）。"""
    t = scores[ti]
    if math.isnan(t):
        return len(scores)
    return 1 + sum(1 for j, v in enumerate(scores) if j != ti and not math.isnan(v) and v >= t)


def group_ranks(cands: dict[str, dict], G: list[str], truth_key: str) -> dict[str, int]:
    ti = G.index(truth_key)
    return {c: pessimistic_rank([cands[k][c] for k in G], ti) for c in SCORES}


def bootstrap(a: dict, b: dict, n: int = N_BOOT, seed: int = SEED) -> dict:
    """train_eval.bootstrap_diff と同じ（分子の再抽出、2.5・97.5 パーセンタイル）。"""
    from train_eval import bootstrap_diff
    return bootstrap_diff(a, b, n=n, seed=seed)


def summarize(per: dict[str, dict], excluded: dict[str, str]) -> dict:
    """per = {mid: {s: 順位, f:…, n:…, nb:…, g_size}} → 主の指標と差の区間。"""
    rr = {c: {m: 1.0 / p[c] for m, p in per.items()} for c in SCORES}
    out = dict(n=len(per), excluded={r: sum(1 for v in excluded.values() if v == r)
                                     for r in ("truth_absent", "truth_formula_none", "g_lt2")})
    if not per:
        return out
    out["mean_rr"] = {NAMES[c]: round(float(np.mean(list(rr[c].values()))), 4) for c in SCORES}
    out["top1_rate"] = {NAMES[c]: round(float(np.mean([p[c] == 1 for p in per.values()])), 4) for c in SCORES}
    out["g_size_median"] = float(np.median([p["g_size"] for p in per.values()]))
    out["diff"] = {f"{NAMES[a]}-{NAMES[b]}": bootstrap(rr[a], rr[b]) for a, b in DIFFS}
    return out


def read_rules(summary: dict) -> dict:
    """事前登録の読み方（結果の前に固定）。(C−D) は下限 > 0 か否か、(C−B)・(C−A) は 3 つに分ける。"""
    d = summary.get("diff")
    if not d:
        return dict(status="no_molecules")

    def three(x):
        return "C_better" if x["lo"] > 0 else ("C_worse" if x["hi"] < 0 else "inconclusive")
    cd = d[f"{NAMES['n']}-{NAMES['nb']}"]
    return dict(C_minus_D="spectrum_contributes" if cd["lo"] > 0 else "not_shown_spectral",
                C_minus_B=three(d[f"{NAMES['n']}-{NAMES['f']}"]), C_minus_A=three(d[f"{NAMES['n']}-{NAMES['s']}"]))


# ------------------------------------------------------------- 追加の数え上げ（#118 の miss_classify と同じ定義）
def classify_top1(truth_key: str | None, cand_keys: list, truth_formula: str | None, cand_formulas: list) -> str:
    """hit・a（上位 25 に無い）・b（1 位の式が違う）・c（1 位が同じ式の異性体）。"""
    c = cand_keys[:25]
    rank = next((i + 1 for i, k in enumerate(c) if truth_key is not None and k == truth_key), None)
    if rank == 1:
        return "hit"
    if rank is None:
        return "a"
    return "b" if cand_formulas[0] != truth_formula else "c"


def flip_counts(cls_control: dict[str, str], cls_new: dict[str, str]) -> dict:
    """対照の (c) のうち新で 1 位になった数、対照で 1 位だった分子のうち新で 1 位でなくなった数。"""
    c_ids = [m for m, v in cls_control.items() if v == "c"]
    h_ids = [m for m, v in cls_control.items() if v == "hit"]
    return dict(control_c=len(c_ids), control_hit=len(h_ids),
                c_to_hit=sum(1 for m in c_ids if cls_new.get(m) == "hit"),
                hit_lost=sum(1 for m in h_ids if cls_new.get(m) != "hit"))


# ------------------------------------------------------------- 実データ（採点の回だけ）
def formula_of_smiles(smi: str) -> str | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi) if smi else None
    return CalcMolFormula(m) if m is not None else None


def window_candidates(holdout: str, need: tuple = ("n", "nb"), logits_run: str | None = None) -> tuple[dict, dict]:
    """{mid: aggregate_by_key の結果}（s と、npz があれば n・nb）と、正解の辞書を返す。run.py の score 段と同じ窓。"""
    import run as R
    import library as L
    import sanpier as SP
    from mrr25 import read_truth
    zf = np.load(ROOT / f"outputs/holdout_eval/{logits_run or holdout + '_fold0'}/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = read_truth(ROOT / f"outputs/holdouts/{holdout}/truth.csv")
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()
    maps = {}
    for c, name in (("n", "fragnn"), ("nb", "fragnnblind")):
        if c in need:
            p = ROOT / f"outputs/local_eval/{holdout}/{name}_scores.npz"
            if not p.exists():
                raise SystemExit(f"{p} が無い（fragnn 段の PR で作る。#120）")
            maps[c] = SP.load_scores(p)
    out = {}
    for mid in truth:
        if mid not in targets:
            continue
        rows = R.score_window(R.visible_window(pool, targets[mid], hidden), pool, R.stage_z(zf[mid], None), fpc)
        recs = [dict(key=pool.key[i], row=i, smiles=pool.smiles[i], s=s,
                     **{c: maps[c].get(mid, {}).get(i) for c in maps}) for s, i in rows]
        out[mid] = aggregate_by_key(recs)
    return out, truth


def read_preds(path: Path) -> dict[str, list[str]]:
    with open(path, newline="") as f:
        return {r["molecule_id"]: [s for s in r["smiles"].split(";") if s] for r in csv.DictReader(f)}


def run_holdout(holdout: str) -> dict:
    import run as R
    from mrr25 import to_key
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as frag
    frag.warmup()
    t0 = time.time()
    cands, truth = window_candidates(holdout)
    spectra = R.query_spectra(holdout)
    per, excl, info_all, key_mismatch, fcache = {}, {}, {}, 0, {}
    for mid, cd in cands.items():
        tk = to_key(truth[mid]) if truth[mid] else None
        G, info = build_group(cd, tk, formula_of_smiles)
        info_all[mid] = info
        if G is None:
            excl[mid] = info["excluded"]
            continue
        for k in G:   # (B) f は G のキーの全行で計算して最大
            sc = frag.score_molecule(spectra.get(mid, []), cd[k]["smiles_all"], cache=fcache)
            for v in sc:
                cd[k]["f"] = _nanmax(cd[k]["f"], _num(v))
            key_mismatch += sum(1 for s in cd[k]["smiles_all"] if to_key(s) != k)   # 行単位で数える（記録だけ）
        per[mid] = dict(group_ranks(cd, G, tk), g_size=len(G))
    summ = summarize(per, excl)
    summ.update(holdout=holdout, read=holdout == READ_HOLDOUT, n_truth=len(truth), n_scored_molecules=len(cands),
                key_vs_to_key_mismatch_rows_in_G=key_mismatch,
                n_formula_none=sum(i["n_formula_none"] for i in info_all.values()),
                n_formula_conflict=sum(i["n_formula_conflict"] for i in info_all.values()),
                secs=round(time.time() - t0, 1))
    if holdout == READ_HOLDOUT:
        summ["reading"] = read_rules(summ)
    # 追加の数え上げ（同じ回に採り直した対照と新の最終リスト）
    d = ROOT / f"outputs/local_eval/{holdout}"
    pc, pn = d / f"submission_{CONTROL_TAG}.csv", d / f"submission_{NEW_TAG}.csv"
    if pc.exists() and pn.exists():
        import csv as _csv
        trows = {r["molecule_id"]: r for r in _csv.DictReader(open(ROOT / f"outputs/holdouts/{holdout}/truth.csv"))}
        cls = {}
        for tag, p in (("control", pc), ("new", pn)):
            preds = read_preds(p)
            cls[tag] = {}
            for mid, t in truth.items():
                c = preds.get(mid, [])[:25]
                tsmi = trows[mid].get("smiles") or ""
                cls[tag][mid] = classify_top1(to_key(t), [to_key(s) for s in c],
                                              formula_of_smiles(tsmi) if tsmi else None, [formula_of_smiles(s) for s in c[:1]])
        summ["flip"] = flip_counts(cls["control"], cls["new"])
    else:
        summ["flip"] = "submissions_missing"
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", default="v3_c1,v3_c2,enveda_v1,v3_c2n")
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/iso_rank_diag.json"))
    a = ap.parse_args()
    res = {h: run_holdout(h) for h in a.holdouts.split(",")}
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
