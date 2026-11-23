#!/usr/bin/env python3
"""
Issue #210（#195 A4・A5）: 伸びしろの分解。対照（score+prior+frag）の上位 25 と候補の表から、分子ごとに失点の型を分け、
型ごとに「その区分の分子を全部 1 位にできたときの MRR の上限」を出す（採否には使わない。次の対象を決める材料）。

区分（定義は #210 の本文で計算の前に固定。ここで替えない）:
- i_pool_offwin : 正解はプール（隠した候補を除く）にあるが、±10 ppm の窓の外
- i_pubchem     : プールの窓に無く、PubChem 層の ±10 ppm の窓にある
- i_none        : プールの窓にも PubChem 層の窓にも無い（プールの窓の外にあれば i_pool_offwin が先）
- ii_below25    : プールの窓にあるが、対照の上位 25 に無い
- iii_rank2_25  : 対照の 2〜25 位
- iv_top1       : 対照の 1 位
上限 = Σ_{区分の分子} (1 − rr) / n。全区分の上限の和 + 対照の MRR = 1。

照合: scripts/validate/mrr25.py の to_key / truth_keys（metric と同じ）。窓の中の正解の有無は、正解と重原子の組成
（H と電荷を除く元素の数）が同じ候補だけ to_key を計算して調べる（InChIKey14 が同じなら重原子の組成も同じ）。
プールにあるか（窓の外も含む）は、プールの key と正解の score_key（casmi.chem、scorer のキー）の一致。

A5（見える test 400 分子）: outputs/test_library_match.parquet（scripts/eda/library_coverage.py の出力）の
best_cos ≥ 0.99 を「train の写しらしい」とし、Wilson の 95% 区間を付ける。プールの窓の候補の数と、最良一致の構造
（best_key、train の raw InChIKey14）がプールの窓にあるか（候補の raw_key と比べる）も数える。
見える test はデータのページで training dataset の例とされているので、この割合は隠れテストの比率の推定に使えない。

    .venv/bin/python scripts/research/headroom.py --out outputs/local_eval/headroom_1123/headroom.json

出力 json は gitignore の outputs の下（数値は docs/research/headroom.md に写し、写しを docs/research/headroom-1123/ に置く）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PPM = 10.0
K = 25
CONTROL = "submission_score+prior+frag.csv"
A5_COS = 0.99
CATS = ("i_pool_offwin", "i_pubchem", "i_none", "ii_below25", "iii_rank2_25", "iv_top1")
EXPECTED_MRR = {"v3_c1": 0.8535, "v3_c2": 0.0, "enveda_v1": 0.8376, "v3_c2n": 0.6284,
                "v3_c1u_t10": 0.8878, "v3_c2_t10": 0.0, "enveda_v1u_t10": 0.8513, "v3_c2nu_t10": 0.6357}
ORIG = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
UNION = ("v3_c1u_t10", "v3_c2_t10", "enveda_v1u_t10", "v3_c2nu_t10")
MISMATCH_EXIT = 2

_ELEM = re.compile(r"([A-Z][a-z]?)(\d*)")


# ---------------------------------------------------------------- 純粋な関数（テストあり）
def heavy_comp_from_formula(formula: str | None) -> tuple | None:
    """'C10H12N2O+' → (('C',10),('N',2),('O',1))。H と電荷・同位体の印は捨てる。読めなければ None。"""
    if not formula:
        return None
    f = re.sub(r"[+-]\d*$", "", formula.strip())
    if not f or not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", f):
        return None
    c: Counter = Counter()
    for el, n in _ELEM.findall(f):
        if el in ("H", "D", "T"):
            continue
        c[el] += int(n) if n else 1
    return tuple(sorted(c.items()))


def heavy_comp_from_smiles(smi: str) -> tuple | None:
    """SMILES の重原子の組成（sanitize なしで数える。明示の [H] 原子は数えない）。読めなければ None。"""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    c: Counter = Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() > 1)
    return tuple(sorted(c.items()))


def rank_of(rr: float) -> int | None:
    """rr（1/r か 0）→ 順位 r（無ければ None）。"""
    return None if rr <= 0 else int(round(1.0 / rr))


def classify(r: int | None, in_window: bool, in_pool: bool, in_pubchem: bool) -> str:
    """1 分子の区分。r は対照の順位（上位 25 に無ければ None）。窓の外の判定は r が None のときだけ使う。"""
    if r == 1:
        return "iv_top1"
    if r is not None:
        assert 2 <= r <= K, r
        return "iii_rank2_25"
    if in_window:
        return "ii_below25"
    if in_pool:
        return "i_pool_offwin"
    if in_pubchem:
        return "i_pubchem"
    return "i_none"


def summarize(recs: list[dict]) -> dict:
    """recs = [{cat, rr}]。区分ごとの分子数・割合・上限 Σ(1−rr)/n と、全体の MRR・1 位の割合。"""
    n = len(recs)
    assert n > 0
    mrr = sum(r["rr"] for r in recs) / n
    out: dict = dict(n=n, mrr=round(mrr, 4), mrr_raw=mrr, top1_share=round(sum(r["cat"] == "iv_top1" for r in recs) / n, 4),
               cats={})
    tot = 0.0
    for c in CATS:
        sel = [r for r in recs if r["cat"] == c]
        gain = sum(1.0 - r["rr"] for r in sel) / n
        tot += gain
        out["cats"][c] = dict(n=len(sel), share=round(len(sel) / n, 4), bound=round(gain, 4), bound_raw=gain)
    out["sum_bounds_plus_mrr"] = round(tot + mrr, 10)
    assert sum(v["n"] for v in out["cats"].values()) == n
    assert abs(tot + mrr - 1.0) < 1e-9, tot + mrr
    assert out["top1_share"] <= out["mrr"] + 1e-12
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson のスコア区間（95%）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 実データ
def set_paths(h: str) -> dict:
    if h in ORIG:
        return dict(sub=ROOT / f"outputs/local_eval/{h}/{CONTROL}", truth=ROOT / f"outputs/holdouts/{h}/truth.csv",
                    logits=ROOT / f"outputs/holdout_eval/{h}_fold0/pc_logits.npz")
    return dict(sub=ROOT / f"outputs/local_eval/{h}/{CONTROL}", truth=ROOT / f"outputs/local_eval/{h}/truth.csv",
                logits=ROOT / f"outputs/local_eval/{h}/pc_logits.npz")


class PoolT:
    """run.Pool と同じ並び（mass の安定ソート）に formula の重原子の組成を足したもの。"""

    def __init__(self):
        import pandas as pd
        meta = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet",
                               columns=["key", "smiles", "formula", "mass"])
        order = np.argsort(meta["mass"].values, kind="stable")
        self.key = meta["key"].values[order]
        self.smiles = meta["smiles"].values[order]
        self.formula = meta["formula"].values[order]
        self.mass = meta["mass"].values[order].astype(np.float64)
        self._hc: dict = {}
        self.keyset = set(self.key)

    def window(self, target: float, ppm: float = PPM) -> np.ndarray:   # run.Pool.window と同じ式
        tol = target * ppm * 1e-6
        a = np.searchsorted(self.mass, target - tol, "left")
        b = np.searchsorted(self.mass, target + tol, "right")
        return np.arange(a, b)

    def hc(self, i: int):
        f = self.formula[i]
        if f not in self._hc:
            self._hc[f] = heavy_comp_from_formula(f)
        return self._hc[f]


def in_candidates(smis, hcs, tkeys: set, thc, to_key) -> bool:
    """候補の SMILES（と重原子の組成）に正解のキーがあるか。組成が同じか不明な候補だけ to_key を計算する。"""
    for s, h in zip(smis, hcs):
        if h is not None and thc is not None and h != thc:
            continue
        if to_key(s) in tkeys:
            return True
    return False


def run_set(h: str, pool: PoolT, tier, L, M, chem) -> tuple[dict, list[dict]]:
    p = set_paths(h)
    truth = M.read_truth(p["truth"])
    preds = M.read_submission(p["sub"])
    zf = np.load(p["logits"], allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    rule = L.hide_rule(h)
    hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
    visible_keys = pool.keyset - hidden
    recs = []
    for mid, t in truth.items():
        tk = M.truth_keys(t)
        rr = M.reciprocal_rank(tk, preds.get(mid, []), K)
        r = rank_of(rr)
        t_smis = [s.strip() for s in t.split(",") if s.strip()]
        thc = heavy_comp_from_smiles(t_smis[0]) if t_smis and not M._INCHIKEY.match(t_smis[0]) else None
        rec = dict(set=h, molecule_id=mid, rr=rr, rank=r, n_window=None, in_window=None, in_pool=None, in_pubchem=None)
        if mid not in targets:
            raise SystemExit(f"{h} {mid}: 窓の中心（__targets__）が無い")
        idx = pool.window(targets[mid])
        if hidden:
            idx = idx[~np.isin(pool.key[idx], list(hidden))]
        rec["n_window"] = int(len(idx))
        in_win = in_candidates(pool.smiles[idx], [pool.hc(i) for i in idx], tk, thc, M.to_key)
        rec["in_window"] = bool(in_win)
        if r is not None and not in_win:
            raise SystemExit(f"{h} {mid}: 上位 25 にあるのに窓に無い（定義の食い違い）")
        if r is None and not in_win:
            sk = {chem.score_key(s) for s in t_smis} - {None}
            rec["in_pool"] = bool(sk & visible_keys)
            if not rec["in_pool"]:
                smis = tier.window(targets[mid])
                rec["n_pubchem_window"] = len(smis)
                rec["in_pubchem"] = bool(in_candidates(smis, [heavy_comp_from_smiles(s) for s in smis], tk, thc, M.to_key))
        rec["cat"] = classify(r, bool(in_win), bool(rec["in_pool"]), bool(rec["in_pubchem"]))
        recs.append(rec)
    s = summarize(recs)
    s.update(inputs=dict(submission_sha256=sha256_file(p["sub"]), truth_sha256=sha256_file(p["truth"]),
                         logits_sha256=sha256_file(p["logits"])), pool_rule=rule.get("pool"), n_hidden=len(hidden),
             n_window_median=float(np.median([r["n_window"] for r in recs])))
    return s, recs


def run_a5(pool: PoolT, chem) -> tuple[dict, list[dict]]:
    import pandas as pd
    lm_p = ROOT / "outputs/test_library_match.parquet"
    lm = pd.read_parquet(lm_p)
    zf = np.load(ROOT / "outputs/holdout_eval/sub_test_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    test_ids = set(pd.read_parquet(ROOT / "data/test.parquet", columns=["molecule_id"]).molecule_id.astype(str))
    assert set(lm.molecule_id.astype(str)) == test_ids, "test_library_match の分子が見える test と違う"
    assert set(targets) == test_ids, "sub_test_fold0 の __targets__ の分子が見える test と違う"
    rows = []
    for r in lm.itertuples(index=False):
        mid = str(r.molecule_id)
        idx = pool.window(targets[mid])
        copy_like = bool(r.best_cos >= A5_COS)
        raw = {chem.raw_key(s) for s in pool.smiles[idx]} if copy_like and r.best_key is not None else set()
        rows.append(dict(molecule_id=mid, best_cos=float(r.best_cos), best_key=r.best_key, n_window=int(len(idx)),
                         copy_like=bool(r.best_cos >= A5_COS),
                         best_in_pool_window=bool(r.best_key is not None and r.best_key in raw)))
    n = len(rows)
    k_copy = sum(x["copy_like"] for x in rows)
    k_best_pool = sum(x["copy_like"] and x["best_in_pool_window"] for x in rows)
    k_nonempty = sum(x["n_window"] > 0 for x in rows)
    nw = np.array([x["n_window"] for x in rows])
    out = dict(n=n, cos_threshold=A5_COS, input_sha256=sha256_file(lm_p),
               copy_like=dict(k=k_copy, share=round(k_copy / n, 4), wilson95=[round(v, 4) for v in wilson(k_copy, n)]),
               copy_like_and_best_in_pool_window=dict(k=k_best_pool, share=round(k_best_pool / n, 4),
                                                      wilson95=[round(v, 4) for v in wilson(k_best_pool, n)]),
               pool_window_nonempty=dict(k=k_nonempty, share=round(k_nonempty / n, 4),
                                         wilson95=[round(v, 4) for v in wilson(k_nonempty, n)]),
               pool_window_size=dict(median=float(np.median(nw)), p10=float(np.percentile(nw, 10)),
                                     p90=float(np.percentile(nw, 90)), min=int(nw.min()), max=int(nw.max())))
    return out, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/headroom_1123/headroom.json"))
    ap.add_argument("--sets", default=",".join(ORIG + UNION))
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import library as L
    import mrr25 as M
    import pubchem as PC
    from casmi import chem
    import rdkit
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    pool = PoolT()
    tier = PC.Tier()
    res: dict = dict(issue=210, control=CONTROL, ppm=PPM, rdkit=rdkit.__version__, sets={})
    per_mol: list[dict] = []
    bad = []
    for h in a.sets.split(","):
        s, recs = run_set(h, pool, tier, L, M, chem)
        res["sets"][h] = s
        per_mol += recs
        print(f"{h}: n {s['n']} mrr {s['mrr']:.4f} top1 {s['top1_share']:.4f} "
              + " ".join(f"{c}={v['n']}" for c, v in s["cats"].items()), flush=True)
        if h in EXPECTED_MRR and s["mrr"] != EXPECTED_MRR[h]:
            bad.append((h, s["mrr"], EXPECTED_MRR[h]))
    if bad:
        print(f"MRR が既存の値と食い違う: {bad}", file=sys.stderr)
        sys.exit(MISMATCH_EXIT)
    a5, a5_rows = run_a5(pool, chem)
    res["a5_visible_test"] = a5
    res["secs"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    import pandas as pd
    pd.DataFrame(per_mol).to_csv(out.with_name("headroom_per_molecule.csv"), index=False)
    pd.DataFrame(a5_rows).to_csv(out.with_name("a5_visible_test_per_molecule.csv"), index=False)
    print(json.dumps(a5, ensure_ascii=False))
    print(f"secs {res['secs']}")


if __name__ == "__main__":
    main()
