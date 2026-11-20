#!/usr/bin/env python3
"""
Issue #171 §5・補足 6・7: rep の点が train の構造（U）のおとりを、train に無い構造（O）のおとりより高く出すか
（d_dom）を、MetFrag-lite と比べて測る（採否には使わない。#146 の dseen.py の型）。

定義（#171 §5 と補足 6・7 のとおり。ここで替えない）:
- 候補 = run.py の score 段の窓（±10 ppm、visible_window → score_window、e180_c2k は hide_rule keep）を pool の key で
  まとめたもの。rep の点・MetFrag-lite の点はキーの全行の最大（iso_rank_diag._nanmax と同じ規則。点なしの行は除く）
- rep の点 = run_rep_once.sh が書いた outputs/local_eval/<h>/rep_scores.npz（z をかける前の生の rep(c|q)、
  leave-own-structure-out 込み、窓の全候補）。npz の行の集合と窓の行の集合が分子ごとに一致しなければ止める
- 正解の除き方 E は dseen.truth_exclusion、群 S・U・O は dseen.group_of（dseen.Ctx の T・TR）
- 有限を求める点は rep と MetFrag-lite の 2 つだけ。rep の点なし（キーの全行で NaN）の候補は群から外す
- z は分子ごとに残った候補で ddof 0。sd 0 の判定も rep と MetFrag-lite の 2 つ
- 外す理由（この順）: no_logits → no_candidates → no_U → no_O → sd0
- d_dom = mean(z|U) − mean(z|O)、Δ_dom = d_dom(rep) − d_dom(MetFrag-lite)。記録だけ: d_seen = mean(z|S) − mean(z|U)
  （S と U の両方がある分子だけ）、Δ_seen = d_seen(rep) − d_seen(MetFrag-lite)
- 区間は dseen.bootstrap_ci（丸める前の lo・hi、2,000 回・seed 0、molecule_id の昇順）。帯: lo > 0 → higher_train、
  hi < 0 → lower_train、それ以外 → none
- 読む集合 v3_c1・e180_c2k。記録だけ enveda_v1・v3_c2n・v3_c2。読む 2 集合のどちらかが higher_train なら
  class2_risk = True（採用時の提出の事前登録に「class 2 型の危険」として書く）

    .venv/bin/python scripts/local_eval/rep_dseen.py [--sets v3_c1,e180_c2k,enveda_v1,v3_c2n,v3_c2]

出力: outputs/local_eval/rep_dseen.json、rep_dseen_per_molecule.csv。純粋な関数（CI）は rdkit を import しない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/local_eval", "scripts/validate", "scripts/ranker", "data/ext/v4g/casmi26-v3-models/code"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

READ_SETS = ("v3_c1", "e180_c2k")
RECORD_SETS = ("enveda_v1", "v3_c2n", "v3_c2")
ALL_SETS = READ_SETS + RECORD_SETS
REASONS = ("no_logits", "no_candidates", "no_U", "no_O", "sd0")
POINTS = ("rep", "f")   # rep・MetFrag-lite
WINDOW_MISMATCH_EXIT = 3


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 純粋な関数（CI）
def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))


def nanmax_all(vals) -> float:
    """iso_rank_diag._nanmax を順に当てたもの（NaN・None は除く。全部無ければ NaN）。"""
    v = math.nan
    for x in vals:
        x = math.nan if x is None else float(x)
        if math.isnan(v):
            v = x
        elif not math.isnan(x):
            v = max(v, x)
    return v


def zscore(v) -> np.ndarray | None:
    v = np.asarray(v, np.float64)
    if len(v) == 0:
        return None
    sd = v.std()
    if not sd > 0:
        return None
    return (v - v.mean()) / sd


def molecule_d(cands: list[dict]) -> tuple[dict | None, str | None]:
    """cands = [{group, rep, f}]（正解は除いた後）。rep と f が有限の候補だけ。(rec, None) か (None, 理由)。"""
    c = [x for x in cands if _finite(x.get("rep")) and _finite(x.get("f"))]
    if not c:
        return None, "no_candidates"
    g = np.array([x["group"] for x in c])
    S, U, O = g == "S", g == "U", g == "O"
    if not U.any():
        return None, "no_U"
    if not O.any():
        return None, "no_O"
    z = {k: zscore([x[k] for x in c]) for k in POINTS}
    if any(z[k] is None for k in POINTS):
        return None, "sd0"
    rec = dict(n_cands=len(c), n_S=int(S.sum()), n_U=int(U.sum()), n_O=int(O.sum()),
               d_dom={k: float(z[k][U].mean() - z[k][O].mean()) for k in POINTS},
               d_seen={k: (float(z[k][S].mean() - z[k][U].mean()) if S.any() else None) for k in POINTS})
    return rec, None


def band(lo: float, hi: float) -> str:
    """dseen.band と同じしきい値（丸める前の lo > 0 / hi < 0）。名前だけ #171 §5 のもの。"""
    if lo > 0:
        return "higher_train"
    if hi < 0:
        return "lower_train"
    return "none"


def summarize_set(recs: dict, excluded: dict) -> dict:
    from dseen import bootstrap_ci
    out: dict = dict(n=len(recs), excluded={r: sum(1 for v in excluded.values() if v == r) for r in REASONS})
    if not recs:
        out.update(delta_dom=None, band="none", delta_seen=None)
        return out
    x = {m: r["d_dom"]["rep"] - r["d_dom"]["f"] for m, r in recs.items()}
    b = bootstrap_ci(x)
    out["delta_dom"] = dict(mean=round(b["mean"], 4), lo=round(b["lo"], 4), hi=round(b["hi"], 4), n=b["n"],
                            lo_raw=b["lo"], hi_raw=b["hi"],
                            better=int(sum(v > 0 for v in x.values())), worse=int(sum(v < 0 for v in x.values())))
    out["band"] = band(b["lo"], b["hi"])
    xs = {m: r["d_seen"]["rep"] - r["d_seen"]["f"] for m, r in recs.items() if r["d_seen"]["rep"] is not None}
    out["delta_seen"] = None
    if xs:
        bs = bootstrap_ci(xs)
        out["delta_seen"] = dict(mean=round(bs["mean"], 4), lo=round(bs["lo"], 4), hi=round(bs["hi"], 4), n=bs["n"],
                                 band=band(bs["lo"], bs["hi"]))
    for kind in ("d_dom", "d_seen"):
        out[kind] = {}
        for k, name in (("rep", "rep"), ("f", "metfrag")):
            v = [r[kind][k] for r in recs.values() if r[kind][k] is not None]
            out[kind][name] = dict(mean=round(float(np.mean(v)), 4), n=len(v)) if v else None
    for g in ("n_S", "n_U", "n_O", "n_cands"):
        out[f"{g}_median"] = float(np.median([r[g] for r in recs.values()]))
    return out


def class2_risk(bands: dict) -> bool:
    return any(bands.get(s) == "higher_train" for s in READ_SETS)


# ---------------------------------------------------------------- 実データ
def run_set(h: str, ctx, per_rows: list) -> dict:
    import iso_rank_diag as IR
    import library as L
    import sanpier as SP
    from dseen import group_of, is_truth, logits_path, metfrag_max, truth_exclusion, truth_path
    from mrr25 import read_truth
    t0 = time.time()
    R, pool = ctx.R, ctx.pool
    done = json.loads((ROOT / f"outputs/local_eval/{h}/rep_once.done").read_text())
    npz = ROOT / f"outputs/local_eval/{h}/rep_scores.npz"
    rj = json.loads((ROOT / f"outputs/local_eval/{h}/rep_scores.json").read_text())
    npz_sha = sha256_file(npz)
    assert npz_sha == rj["rep_scores_sha256"], f"{h}: rep_scores.npz の sha256 が rep_scores.json と違う"
    reps = SP.load_scores(npz)
    zf = np.load(logits_path(h, ROOT), allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    tp = truth_path(h, ROOT)
    truth = read_truth(tp)
    with open(tp, newline="") as f:
        trows = {r["molecule_id"]: r for r in csv.DictReader(f)}
    rule = L.hide_rule(h)
    hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
    spectra = R.query_spectra(h)
    recs, excl, n_rows, n_truth_removed, mismatch = {}, {}, 0, 0, []
    fcache = {}
    for mid in truth:
        if mid not in targets:
            excl[mid] = "no_logits"
            continue
        rows = R.score_window(R.visible_window(pool, targets[mid], hidden), pool, R.stage_z(zf[mid], None), ctx.fpc)
        rs = reps.get(mid)
        if rs is None or set(rs) != {int(i) for _, i in rows}:
            mismatch.append(mid)
            continue
        n_rows += len(rows)
        cd = IR.aggregate_by_key([dict(key=pool.key[i], row=i, smiles=pool.smiles[i], s=s) for s, i in rows])
        tr = trows.get(mid, {})
        E = truth_exclusion(tr.get("inchikey14") or None, tr.get("smiles") or None, ctx.fns, ctx.aidx)
        cands = []
        for k, a in cd.items():
            raws = [ctx.raw_key(s) for s in a["smiles_all"]]
            if is_truth(k, raws, E):
                n_truth_removed += 1
                continue
            rv = nanmax_all(rs[int(r)] for r in a["rows"])
            if not math.isfinite(rv):   # rep の点なしは群から外す（MetFrag-lite は求めない）
                continue
            fv = metfrag_max(ctx, spectra.get(mid, []), a["smiles_all"], fcache)
            cands.append(dict(key=k, group=group_of(k, raws, ctx.T_canon, ctx.T_raw, ctx.TR), rep=rv, f=fv))
        rec, why = molecule_d(cands)
        if rec is None:
            excl[mid] = why
        else:
            recs[mid] = rec
            per_rows.append(dict(holdout=h, molecule_id=mid, delta_dom=rec["d_dom"]["rep"] - rec["d_dom"]["f"],
                                 d_dom_rep=rec["d_dom"]["rep"], d_dom_metfrag=rec["d_dom"]["f"],
                                 d_seen_rep=rec["d_seen"]["rep"], d_seen_metfrag=rec["d_seen"]["f"],
                                 n_S=rec["n_S"], n_U=rec["n_U"], n_O=rec["n_O"], n_cands=rec["n_cands"]))
    if mismatch:
        print(f"[stop] {h}: rep_scores.npz の行と窓の行が {len(mismatch)} 分子で違う: {mismatch[:10]}")
        sys.exit(WINDOW_MISMATCH_EXIT)
    out = summarize_set(recs, excl)
    out.update(holdout=h, read=h in READ_SETS, n_truth=len(truth), hide_rule=rule, hidden_keys=len(hidden),
               window_rows=n_rows, truth_removed_total=n_truth_removed, rep_once_commit=done["commit"],
               files_sha256={"rep_scores.npz": npz_sha, "logits": sha256_file(logits_path(h, ROOT)),
                             "truth": sha256_file(tp)},
               secs=round(time.time() - t0, 1))
    return out


def run_all(sets: list[str]) -> dict:
    from dseen import Ctx, ram_peak_gb
    t0 = time.time()
    commits = {}
    for h in sets:
        p = ROOT / f"outputs/local_eval/{h}/rep_once.done"
        assert p.exists(), f"{h}: rep_once.done が無い（run_rep_once.sh が終わっていない）"
        commits[h] = json.loads(p.read_text())["commit"]
    assert len(set(commits.values())) == 1, f"rep_once.done の commit sha が違う: {commits}"
    ctx = Ctx(ROOT)
    per_rows: list = []
    res: dict = dict(issue=171, definition="#171 §5 + 補足 6・7", sets=sets, inputs=ctx.info,
                     script_sha256=sha256_file(Path(__file__)), rep_once_commit=next(iter(commits.values())))
    for h in sets:
        res[h] = run_set(h, ctx, per_rows)
        print(h, json.dumps({k: res[h][k] for k in ("n", "excluded", "delta_dom", "band")}, ensure_ascii=False), flush=True)
    bands = {s: res[s]["band"] for s in READ_SETS if s in res}
    res["read_bands"] = bands
    res["class2_risk"] = class2_risk(bands)
    res["secs"] = round(time.time() - t0, 1)
    res["ram_peak_gb"] = ram_peak_gb()
    (ROOT / "outputs/local_eval/rep_dseen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    with open(ROOT / "outputs/local_eval/rep_dseen_per_molecule.csv", "w", newline="") as f:
        if per_rows:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0]))
            w.writeheader()
            w.writerows(per_rows)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="#171 §5 Δ_dom")
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    a = ap.parse_args()
    sets = a.sets.split(",")
    assert set(READ_SETS) <= set(sets), "読む 2 集合は必ず含める"
    assert set(sets) <= set(ALL_SETS), sets
    res = run_all(sets)
    print(json.dumps({k: res.get(k) for k in ("read_bands", "class2_risk", "secs", "ram_peak_gb")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
