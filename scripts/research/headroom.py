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
    sig = graph_sig(smi)
    return None if sig is None else sig[0]


ORGANIC = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"}
_WL_IDS: dict = {}


def _wl_id(x) -> int:
    """WL の印を、このプロセスの中で一意な整数にする（分子をまたいで同じ印は同じ整数）。"""
    if x not in _WL_IDS:
        _WL_IDS[x] = len(_WL_IDS)
    return _WL_IDS[x]


def graph_sig(smi: str, iters: int = 3) -> tuple | None:
    """(重原子の組成, 重原子のグラフの WL の印)。結合次数・電荷・H・立体を見ないので、立体違い・互変異性体・
    保護の違いでは変わらない（to_key が同じなら印も同じ。to_key を計算する候補を絞るための必要条件）。
    金属などを含む分子は InChI が結合を切るので、WL の印を None にして組成だけで絞る。読めなければ None。"""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None
    heavy = [a for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    comp = tuple(sorted(Counter(a.GetSymbol() for a in heavy).items()))
    if any(el not in ORGANIC for el, _ in comp):
        return (comp, None)
    idx = {a.GetIdx(): i for i, a in enumerate(heavy)}
    nb = [[idx[n.GetIdx()] for n in a.GetNeighbors() if n.GetIdx() in idx] for a in heavy]
    lab = [_wl_id(("e", a.GetSymbol())) for a in heavy]
    for _ in range(iters):
        lab = [_wl_id((lab[i], tuple(sorted(lab[j] for j in nb[i])))) for i in range(len(heavy))]
    return (comp, tuple(sorted(lab)))


def sig_compatible(a: tuple | None, b: tuple | None) -> bool:
    """候補の印 a と正解の印 b が「同じ to_key でありうる」か。どちらかが読めなければ True（調べる側）。"""
    if a is None or b is None:
        return True
    if a[0] != b[0]:
        return False
    if a[1] is None or b[1] is None:
        return True
    return a[1] == b[1]


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


def in_candidates(smis, comps, tkeys: set, tsig, to_key, sig_of=None) -> bool:
    """候補の SMILES に正解のキーがあるか。まず重原子の組成（comps、None は不明）で、次に sig_of（SMILES → 印）が
    あれば WL の印で絞り、両立する候補だけ to_key を計算する（to_key が同じなら組成も印も同じなので漏れない）。"""
    tcomp = None if tsig is None else tsig[0]
    for s, h in zip(smis, comps):
        if h is not None and tcomp is not None and h != tcomp:
            continue
        if sig_of is not None and not sig_compatible(sig_of(s), tsig):
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
        tsig = graph_sig(t_smis[0]) if len(t_smis) == 1 and not M._INCHIKEY.match(t_smis[0]) else None
        rec = dict(set=h, molecule_id=mid, rr=rr, rank=r, n_window=None, in_window=None, in_pool=None, in_pubchem=None)
        if mid not in targets:
            raise SystemExit(f"{h} {mid}: 窓の中心（__targets__）が無い")
        idx = pool.window(targets[mid])
        if hidden:
            idx = idx[~np.isin(pool.key[idx], list(hidden))]
        rec["n_window"] = int(len(idx))
        in_win = in_candidates(pool.smiles[idx], [pool.hc(i) for i in idx], tk, tsig, M.to_key, graph_sig)
        rec["in_window"] = bool(in_win)
        if r is not None and not in_win:
            raise SystemExit(f"{h} {mid}: 上位 25 にあるのに窓に無い（定義の食い違い）")
        if r is None and not in_win:
            sk = {chem.score_key(s) for s in t_smis} - {None}
            rec["in_pool"] = bool(sk & visible_keys)
            if not rec["in_pool"]:
                smis = tier.window(targets[mid])
                rec["n_pubchem_window"] = len(smis)
                sigs = {x: graph_sig(x) for x in smis}
                rec["in_pubchem"] = bool(in_candidates(smis, [None if sigs[x] is None else sigs[x][0] for x in smis], tk,
                                                       tsig, M.to_key, sigs.get))
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


RANK_BINS = ((2, 2), (3, 5), (6, 10), (11, 25))


def rank_bins(ranks: list) -> dict:
    """補助: 2〜25 位の分子の順位の内訳（分子数と、その分子を 1 位にしたときの上限 Σ(1 − 1/r) の n で割る前の和）。"""
    out = {}
    for lo, hi in RANK_BINS:
        sel = [r for r in ranks if r is not None and lo <= r <= hi]
        out[f"{lo}-{hi}"] = dict(n=len(sel), gain_sum=round(sum(1.0 - 1.0 / r for r in sel), 4))
    return out


def pubchem_channel_aux(h: str, M) -> dict | None:
    """補助（区分の定義の外。計算の後に足した記述で、順位付けの見込みの根拠にだけ使う）: 既存の PubChem チャネルの
    キャッシュ outputs/local_eval/<h>/pubchem_fold0.json（run.py の pubchem・joint 段、fold-0 のロジット、上位 60）
    で、正解が提案の何位にあるか。チャネルだけで並べたときの MRR@25 と、上位 60 に入る分子の数。"""
    cp = ROOT / f"outputs/local_eval/{h}/pubchem_fold0.json"
    if not cp.is_file():
        return None
    cache = json.loads(cp.read_text())
    p = set_paths(h)
    truth = M.read_truth(p["truth"])
    ranks = []
    for mid, t in truth.items():
        tk = M.truth_keys(t)
        props = [x[0] for x in cache.get(mid) or []]
        r = next((i + 1 for i, s in enumerate(props) if M.to_key(s) in tk), None)
        ranks.append(r)
    n = len(truth)
    return dict(n=n, n_cached=sum(1 for m in truth if m in cache), cache_sha256=sha256_file(cp),
                n_in_top60=sum(r is not None for r in ranks), n_top1=sum(r == 1 for r in ranks),
                n_in_top25=sum(r is not None and r <= K for r in ranks),
                mrr25_channel_alone=round(sum(1.0 / r for r in ranks if r is not None and r <= K) / n, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/headroom_1123/headroom.json"))
    ap.add_argument("--sets", default=",".join(ORIG + UNION))
    ap.add_argument("--aux-only", action="store_true", help="既存の --out の json に PubChem チャネルの補助の数だけ足す")
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
    if a.aux_only:
        out = Path(a.out)
        res = json.loads(out.read_text())
        res["aux_pubchem_channel"] = {h: pubchem_channel_aux(h, M) for h in res["sets"]}
        import pandas as pd
        pm = pd.read_csv(out.with_name("headroom_per_molecule.csv"))
        res["aux_rank_bins"] = {h: rank_bins([int(r) for r in g["rank"].dropna()]) for h, g in pm.groupby("set", sort=False)}
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(res["aux_pubchem_channel"], ensure_ascii=False, indent=1))
        return
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
