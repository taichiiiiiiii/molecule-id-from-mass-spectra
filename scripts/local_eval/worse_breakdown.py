#!/usr/bin/env python3
"""
Issue #120 の記録用の測定: fragnn・整列で「悪くなった分子」の新しい 1 位の内訳（採否には使わない）。

定義は #120 の事前登録のコメント（2026-11-17T15:55:20Z、2026-11-18 夜 項目 4）のとおり。要点:
- 既存の出力の per-molecule の rr だけを使う（新しい採点はしない）。悪くなった分子 = rr(新) < rr(対照)
  A fragnn・enveda_v1 / B fragnn・v3_c2n / C 整列・v3_c2n。対照は score+prior+frag
- 悪くなった分子の数が A 19・B 26・C 41 と一致しなければ表を作らずに止める（exit 2）
- 悪くなった分子ごとに、新しいリストの 1 位を
  (a) 正解と同じ分子式（RDKit CalcMolFormula）/ (b) 別の分子式、
  (x) fragnn の学習構造 T に入る / 入らない、(y) 競技の train の構造に入る / 入らない で数える。1 位が無い・読めない → 「なし」
- キーは 生（chem.raw_key）と 正準化後（chem.score_key）の 2 種類。表の「入る」は どちらかで入る。生だけ・正準化後だけの表も併記

    python scripts/local_eval/worse_breakdown.py [--out outputs/local_eval/worse_breakdown.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/cv"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))

CTRL_TAG = "score+prior+frag"
# (名前, holdout, 新のタグ, 期待する悪化の数〈fragnn_v3_check.json・align_v3_check.json の worse〉)
TARGETS = (
    ("fragnn_enveda_v1", "enveda_v1", "score+fragnn+prior+frag", 19),
    ("fragnn_v3_c2n", "v3_c2n", "score+fragnn+prior+frag", 26),
    ("align_v3_c2n", "v3_c2n", "score+align+prior+frag", 41),
)
KEY_MODES = ("any", "raw", "canon")
NONE_CELL = "none"


def worse_molecules(rr_new: Mapping[str, float], rr_ctrl: Mapping[str, float]) -> list[str]:
    """rr(新) < rr(対照) の分子（対照の分子の並び順）。新に無い分子は rr 0。"""
    return [m for m in rr_ctrl if rr_new.get(m, 0.0) < rr_ctrl[m]]


def rank_of(truth_key: str | None, cands: Sequence[str], key_fn: Callable[[str], str | None], top_k: int = 25) -> int:
    """正解が最初に出る順位（1 始まり）。top_k 以内に無ければ 0（圏外）。"""
    if truth_key is None:
        return 0
    for r, c in enumerate(cands[:top_k], start=1):
        if key_fn(c) == truth_key:
            return r
    return 0


def membership(smiles: str, raw_fn, canon_fn, raw_set: set, canon_set: set) -> dict:
    """生のキー・正準化後のキーそれぞれで集合に入るか。any = どちらかで入る。"""
    rk, ck = raw_fn(smiles), canon_fn(smiles)
    raw_in = rk is not None and rk in raw_set
    canon_in = ck is not None and ck in canon_set
    return dict(raw_key=rk, canon_key=ck, raw=bool(raw_in), canon=bool(canon_in), any=bool(raw_in or canon_in))


def classify(truth_smiles: str, top1_smiles: str | None, *, formula_fn, raw_fn, canon_fn,
             T_raw: set, T_canon: set, tr_raw: set, tr_canon: set) -> dict:
    """1 分子分。formula は (a) same / (b) diff / none（1 位が無い・読めない）。"""
    rec: dict = dict(top1=top1_smiles)
    tf = formula_fn(truth_smiles)
    rec["truth_formula"] = tf
    rec["truth_in_T"] = membership(truth_smiles, raw_fn, canon_fn, T_raw, T_canon)
    rec["truth_in_train"] = membership(truth_smiles, raw_fn, canon_fn, tr_raw, tr_canon)
    f1 = formula_fn(top1_smiles) if top1_smiles else None
    rec["top1_formula"] = f1
    if f1 is None or tf is None:
        rec["formula"] = NONE_CELL
        return rec
    rec["formula"] = "same" if f1 == tf else "diff"
    rec["in_T"] = membership(top1_smiles, raw_fn, canon_fn, T_raw, T_canon)
    rec["in_train"] = membership(top1_smiles, raw_fn, canon_fn, tr_raw, tr_canon)
    return rec


def table(records: Sequence[dict], mode: str = "any") -> dict:
    """(a)/(b) × T × train の 8 マス ＋ なし。キーは "<same|diff>|T<in|out>|train<in|out>"。"""
    cells = {f"{f}|T{t}|train{y}": 0 for f in ("same", "diff") for t in ("in", "out") for y in ("in", "out")}
    none = 0
    for r in records:
        if r["formula"] == NONE_CELL:
            none += 1
            continue
        t = "in" if r["in_T"][mode] else "out"
        y = "in" if r["in_train"][mode] else "out"
        cells[f"{r['formula']}|T{t}|train{y}"] += 1
    return dict(mode=mode, cells=cells, none=none, n=len(records),
                same=sum(v for k, v in cells.items() if k.startswith("same")),
                diff=sum(v for k, v in cells.items() if k.startswith("diff")))


def formula_rdkit(smiles: str | None) -> str | None:
    if not smiles:
        return None
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(m) if m is not None else None


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_target(name, holdout, new_tag, expected, sets, fns) -> dict:
    import mrr25
    from frag_v3_check import per_molecule_rr
    truth_p = ROOT / f"outputs/holdouts/{holdout}/truth.csv"
    new_p = ROOT / f"outputs/local_eval/{holdout}/submission_{new_tag}.csv"
    ctl_p = ROOT / f"outputs/local_eval/{holdout}/submission_{CTRL_TAG}.csv"
    truth = mrr25.read_truth(truth_p)
    new, ctl = mrr25.read_submission(new_p), mrr25.read_submission(ctl_p)
    rr_new, rr_ctl = per_molecule_rr(truth, new), per_molecule_rr(truth, ctl)
    worse = worse_molecules(rr_new, rr_ctl)
    out = dict(name=name, holdout=holdout, new=new_tag, ctrl=CTRL_TAG, n=len(truth),
               files={str(p.relative_to(ROOT)): sha256(p) for p in (truth_p, new_p, ctl_p)},
               n_worse=len(worse), expected_worse=expected, match=len(worse) == expected)
    if not out["match"]:
        return out
    recs = []
    for m in worse:
        nl, cl = new.get(m, []), ctl.get(m, [])
        tk = mrr25.to_key(truth[m])
        r = classify(truth[m], nl[0] if nl else None, formula_fn=fns["formula"], raw_fn=fns["raw"],
                     canon_fn=fns["canon"], **sets)
        r.update(molecule_id=m, truth=truth[m], rr_ctrl=rr_ctl[m], rr_new=rr_new.get(m, 0.0),
                 rank_ctrl=rank_of(tk, cl, mrr25.to_key), rank_new=rank_of(tk, nl, mrr25.to_key),
                 top1_changed=(fns["canon"](nl[0]) if nl else None) != (fns["canon"](cl[0]) if cl else None),
                 top1_canon_eq_to_key=(fns["canon"](nl[0]) == mrr25.to_key(nl[0])) if nl else None)
        recs.append(r)
    out["tables"] = {mode: table(recs, mode) for mode in KEY_MODES}
    out["aux"] = dict(
        top1_changed=sum(bool(r["top1_changed"]) for r in recs),
        ctrl_rank1=sum(r["rank_ctrl"] == 1 for r in recs),
        new_out_of_25=sum(r["rank_new"] == 0 for r in recs),
        truth_in_T_any=sum(r["truth_in_T"]["any"] for r in recs),
        truth_in_train_any=sum(r["truth_in_train"]["any"] for r in recs),
        canon_vs_to_key_mismatch=sum(r["top1_canon_eq_to_key"] is False for r in recs))
    out["molecules"] = recs
    return out


def main() -> None:
    import pandas as pd
    from casmi import chem
    from trainstruct_bias import EXPECTED_T, train_struct_keys

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/worse_breakdown.json"))
    a = ap.parse_args()
    t0 = time.time()
    T_canon, tinfo = train_struct_keys()
    t1 = time.time()
    raw_keys = [chem.raw_key(s) for s in tinfo["smiles"]]
    T_raw = {k for k in raw_keys if k is not None}
    T_raw_none = sum(k is None for k in raw_keys)
    raw_secs = round(time.time() - t1, 1)
    ts = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet", columns=["inchikey14", "key"])
    tr_raw, tr_canon = set(ts["inchikey14"].dropna()), set(ts["key"].dropna())
    sets = dict(T_raw=T_raw, T_canon=T_canon, tr_raw=tr_raw, tr_canon=tr_canon)
    fns = dict(formula=formula_rdkit, raw=chem.raw_key, canon=chem.score_key)
    res = dict(issue=120, definition="#120 2026-11-17T15:55:20Z",
               sets=dict(T_rows=tinfo["n_rows"], T_expected=EXPECTED_T, T_canon=len(T_canon), T_raw=len(T_raw),
                         T_raw_none=int(T_raw_none),
                         T_raw_secs=raw_secs, train_raw=len(tr_raw), train_canon=len(tr_canon),
                         train_union=len(tr_raw | tr_canon)))
    ok = True
    for name, h, tag, exp in TARGETS:
        r = run_target(name, h, tag, exp, sets, fns)
        res[name] = r
        print(name, "worse", r["n_worse"], "expected", exp, "match", r["match"])
        ok &= r["match"]
    res["all_match"] = ok
    res["secs"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for name, *_ in TARGETS:
        if "tables" in res[name]:
            print(name, json.dumps(res[name]["tables"]["any"], ensure_ascii=False), json.dumps(res[name]["aux"]))
    if not ok:
        print("[stop] 悪くなった分子の数が事前登録の値と一致しない。表は作っていない")
        sys.exit(2)


if __name__ == "__main__":
    main()
