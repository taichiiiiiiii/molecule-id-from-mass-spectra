#!/usr/bin/env python3
"""
Issue #234: 重い枠の候補「PubChem 層の中の自前の並べ替え」の仕組みを、保存済みの出力だけで確かめる（学習しない、
PubChem の提案の作り直しもしない）。定義と読みは #234 の本文（2026-11-24T16:49:05Z、数える前）のとおり。

- 対象: outputs/local_eval/v3_c2/pubchem_fold0.json（200 分子、上位 60、fold-0 のロジット）だけ
- 照合: scripts/validate/mrr25.py の to_key / truth_keys（headroom.py の pubchem_channel_aux と同じ）
- (A) 正解が上位 60 にある分子の割合と順位の分布（1・2〜5・6〜25・26〜60）
- (B) 記述のみ: 上位 60 の中の上限 Σ(1 − rr)/n（n = 全分子）
- (C) 記述のみ: 正解が 2〜60 位の分子ごとに、正解より上の候補のうち正解と同じ分子式（RDKit CalcMolFormula）の割合、その中央値
- (D) 判定: 正解が 2〜60 位の分子で、MetFrag-lite（frag_rescore.score_molecule、run.py の frag 段と同じ呼び方）の点が
  「正解 ＞ 1 位」の割合（同点は数えない、None の分子は分母から除く）と、分子の bootstrap 2000 回（seed 0）の 95% CI
- 読み: (D) の 95% 下限 > 0.5 → 「候補にする」、それ以外 → 「作らない」

    PYTHONPYCACHEPREFIX=<scratch> NUMBA_CACHE_DIR=<scratch> <venv>/bin/python scripts/research/pubchem_rerank_mechanism.py \
        [--out outputs/local_eval/pubchem_rerank_mechanism/result.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = "v3_c2"
CACHE = ROOT / f"outputs/local_eval/{HOLDOUT}/pubchem_fold0.json"
LOGITS = ROOT / f"outputs/holdout_eval/{HOLDOUT}_fold0/pc_logits.npz"
TRUTH = ROOT / f"outputs/holdouts/{HOLDOUT}/truth.csv"
RANK_BINS = ((1, 1), (2, 5), (6, 25), (26, 60))
N_BOOT, SEED, THRESH = 2000, 0, 0.5


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def truth_rank(props: list, tk: set, to_key) -> int | None:
    """提案（SMILES の列）の中で to_key が正解のキーに一致する最初の位置（1 始まり）。無ければ None。"""
    if not tk:
        return None
    for i, s in enumerate(props):
        if to_key(s) in tk:
            return i + 1
    return None


def rank_distribution(ranks: list) -> dict:
    out = {}
    for lo, hi in RANK_BINS:
        out[f"{lo}-{hi}"] = sum(1 for r in ranks if r is not None and lo <= r <= hi)
    return out


def headroom_top60(ranks: list, n: int) -> float:
    """(B) 上位 60 にある分子を全部 1 位にできたときの増え Σ(1 − 1/r)/n。"""
    return sum(1.0 - 1.0 / r for r in ranks if r is not None) / n


def same_formula_share(above_formulas: list, truth_formula: str | None) -> float | None:
    """(C) 正解より上の候補のうち、分子式が正解と同じものの割合。式が読めない候補は「違う」に数える。"""
    if not above_formulas or truth_formula is None:
        return None
    return sum(1 for f in above_formulas if f == truth_formula) / len(above_formulas)


def frag_win(s_truth, s_top1) -> int | None:
    """(D) 1 = 正解の点 ＞ 1 位の点、0 = それ以外（同点を含む）、None = どちらかの点が None（分母から除く）。"""
    if s_truth is None or s_top1 is None:
        return None
    return 1 if float(s_truth) > float(s_top1) else 0


def bootstrap_ci(x, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """分子を復元抽出した割合の 2.5・97.5 パーセンタイル。"""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def reading(lo: float) -> str:
    return "候補にする" if lo > THRESH else "作らない"


def formula_of(smi: str):
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    m = Chem.MolFromSmiles(smi) if smi else None
    return CalcMolFormula(m) if m is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/pubchem_rerank_mechanism/result.json"))
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import mrr25 as M
    import rdkit
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    cache = json.loads(CACHE.read_text())
    truth = M.read_truth(TRUTH)
    n = len(truth)
    per = []
    for mid, t in truth.items():
        props = [x[0] for x in cache.get(mid) or []]
        tk = M.truth_keys(t)
        r = truth_rank(props, tk, M.to_key)
        per.append(dict(molecule_id=mid, n_props=len(props), rank=r))
    ranks = [p["rank"] for p in per]
    n_in = sum(r is not None for r in ranks)
    n_top1 = sum(r == 1 for r in ranks)
    mrr25 = sum(1.0 / r for r in ranks if r is not None and r <= 25) / n

    # (C)
    tsmi = {}
    import csv
    with open(TRUTH, newline="") as f:
        for row in csv.DictReader(f):
            tsmi[row["molecule_id"]] = row["smiles"].split(",")[0]
    shares = []
    for p in per:
        r = p["rank"]
        if r is None or r < 2:
            continue
        props = [x[0] for x in cache[p["molecule_id"]]]
        tf = formula_of(tsmi[p["molecule_id"]])
        sh = same_formula_share([formula_of(s) for s in props[: r - 1]], tf)
        p["truth_formula"] = tf
        p["same_formula_above_share"] = sh
        if sh is not None:
            shares.append(sh)

    # (D): run.py の frag 段と同じ呼び方
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as frag
    frag.warmup()
    import run as R
    spectra = R.query_spectra(HOLDOUT)
    fcache: dict = {}
    wins, n_fail, n_tie = [], 0, 0
    for p in per:
        r = p["rank"]
        if r is None or r < 2:
            continue
        props = [x[0] for x in cache[p["molecule_id"]]]
        sp = spectra.get(p["molecule_id"], [])
        s_top1, s_truth = frag.score_molecule(sp, [props[0], props[r - 1]], cache=fcache)
        w = frag_win(s_truth, s_top1)
        p.update(frag_top1=s_top1, frag_truth=s_truth, n_spectra=len(sp), frag_win=w)
        if w is None:
            n_fail += 1
            continue
        if float(s_truth) == float(s_top1):
            n_tie += 1
        wins.append(w)
    lo, hi = bootstrap_ci(wins)
    share = float(np.mean(wins))

    res = dict(
        issue=234, holdout=HOLDOUT, n=n,
        inputs=dict(cache=str(CACHE.relative_to(ROOT)), cache_sha256=sha256_file(CACHE),
                    cache_mtime=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(CACHE))),
                    logits=str(LOGITS.relative_to(ROOT)), logits_sha256=sha256_file(LOGITS),
                    logits_mtime=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(LOGITS))),
                    truth=str(TRUTH.relative_to(ROOT)), truth_sha256=sha256_file(TRUTH),
                    frag_rescore_sha256=sha256_file(ROOT / "data/ext/frag/frag_rescore.py")),
        rdkit=rdkit.__version__, n_cached=sum(1 for m in truth if m in cache),
        A=dict(n_in_top60=n_in, share_in_top60=round(n_in / n, 4), rank_bins=rank_distribution(ranks)),
        check_headroom_2_2=dict(n_top1=n_top1, n_in_top25=sum(r is not None and r <= 25 for r in ranks),
                                mrr25_channel_alone=round(mrr25, 4)),
        B_descriptive=dict(headroom_top60=round(headroom_top60(ranks, n), 4)),
        C_descriptive=dict(n_molecules=len(shares),
                           median_same_formula_share_above=(round(float(np.median(shares)), 4) if shares else None)),
        D=dict(n_rank2_60=sum(1 for r in ranks if r is not None and r >= 2), n_frag_fail=n_fail,
               n_used=len(wins), n_truth_gt_top1=int(sum(wins)), n_tie=n_tie,
               share_truth_gt_top1=round(share, 4), ci95=[round(lo, 4), round(hi, 4)],
               n_boot=N_BOOT, seed=SEED, threshold=THRESH, reading=reading(lo)),
        seconds=round(time.time() - t0, 1),
    )
    # headroom.md §2.2 の既知の値と違えば止める（同じキャッシュ・同じ照合のはず）
    known = dict(n_in_top60=145, n_top1=65, n_in_top25=138, mrr25=0.4183)
    got = dict(n_in_top60=n_in, n_top1=n_top1, n_in_top25=res["check_headroom_2_2"]["n_in_top25"], mrr25=round(mrr25, 4))
    res["check_headroom_2_2"]["matches_known"] = got == known
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    import pandas as pd
    pd.DataFrame(per).to_csv(out.with_name("per_molecule.csv"), index=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if got != known:
        print(f"headroom.md §2.2 と不一致: {got} != {known}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
