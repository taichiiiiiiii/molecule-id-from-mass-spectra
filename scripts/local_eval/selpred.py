#!/usr/bin/env python3
"""
選択的予測の測定（Issue #216）: 対照 score,prior,frag の「1 位の確信度」と「1 位が正解か」の関係を記録する。

定義は #216 の事前登録のとおり（数字を見てから変えない。しきい値は選ばない）:
  L        = frag 後の重複除去した上位 60（run.py の lists[mid]）。各候補の点 f は prior 段の z(s) + 0.15·pop
  κ_gap    = f(L[0]) − f(L[1])（frag の入れ替えで負になりうる）
  κ_conf   = softmax(f over L, T=1)[0]（T は固定）
  κ_lib    = lib_max（L の候補の参照とのエントロピー類似度の最大、参照が無ければ 0。run.py の lib 段の diag と同じ）
  |L| = 1  → κ_gap = +inf・κ_conf = 1、|L| = 0 → 3 種とも −inf
  正解     = mrr25.reciprocal_rank == 1
  AURC     = 損失 1 − 正解、確信度の降順、同点は中の順がランダムとしたときの期待値（厳密）
  AUC      = P(κ_A > κ_B) + 0.5·P(同点)（A = 正解がプールにある型、B = v3_c2_t10）
  CI       = 分子の bootstrap 2000 回（seed 0）のパーセンタイル 95%

run.py は変えない。最初に、計算し直した submission が outputs/local_eval/<h>/submission_score+prior+frag.csv と
バイト一致することを確かめ、一致しなければ測らずに止める（rc 2）。

    .venv/bin/python scripts/local_eval/selpred.py                     # 4 集合
    .venv/bin/python scripts/local_eval/selpred.py --holdouts v3_c1u_t10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOLDOUTS = ("v3_c1u_t10", "v3_c2_t10", "enveda_v1u_t10", "v3_c2nu_t10")
C2 = "v3_c2_t10"                      # 正解がプールに無い型
AUC_MAIN = "v3_c1u_t10"               # 主の比較相手（C1 型）
STAGES = ["score", "prior", "frag"]
N_BOOT, SEED = 2000, 0
KAPPAS = ("gap", "conf", "lib")


# ---------------------------------------------------------------- 純粋な関数（単体テストの対象）
def confidences(fs) -> tuple[float, float]:
    """L の点 f（最終の並び順）から (κ_gap, κ_conf)。|L|=1 → (+inf, 1)、|L|=0 → (−inf, −inf)。"""
    f = np.asarray(fs, np.float64)
    if f.size == 0:
        return -math.inf, -math.inf
    if f.size == 1:
        return math.inf, 1.0
    e = np.exp(f - f.max())
    return float(f[0] - f[1]), float(e[0] / e.sum())


def aurc(kappa, loss) -> float:
    """確信度 kappa の降順で受け入れたときのリスク・カバレッジ曲線の面積（位置 k=1..n のリスクの平均）。
    同点の組の中は順がランダムとしたときの期待値: 組の j 番目のリスク = (前までの誤り + j·e/m) / (前までの数 + j)。"""
    k = np.asarray(kappa, np.float64)
    l = np.asarray(loss, np.float64)
    n = len(k)
    if n == 0:
        return math.nan
    order = np.argsort(-k, kind="stable")
    ks, ls = k[order], l[order]
    total, before_n, before_e, i = 0.0, 0, 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and ks[j + 1] == ks[i]:
            j += 1
        m = j - i + 1
        e = float(ls[i:j + 1].sum())
        pos = np.arange(1, m + 1)
        total += float(((before_e + pos * e / m) / (before_n + pos)).sum())
        before_n += m; before_e += e; i = j + 1
    return total / n


def aurc_summary(kappa, loss) -> dict:
    """AURC・random（誤り率）・oracle（誤りを先に棄てる）・relAURC。誤りが 0 か全部なら relAURC は NaN。"""
    l = np.asarray(loss, np.float64)
    a = aurc(kappa, l)
    rnd = float(l.mean()) if len(l) else math.nan
    orc = aurc(-l, l)
    rel = (a - orc) / (rnd - orc) if len(l) and rnd - orc > 1e-12 else math.nan
    return dict(aurc=a, random=rnd, oracle=orc, rel_aurc=rel)


def auc(a, b) -> float:
    """P(a > b) + 0.5·P(a == b)（a・b は 2 集合の確信度。±inf も比べられる）。"""
    a = np.asarray(a, np.float64)[:, None]
    b = np.asarray(b, np.float64)[None, :]
    if a.size == 0 or b.size == 0:
        return math.nan
    return float((a > b).mean() + 0.5 * (a == b).mean())


def _pct(vals) -> list:
    v = np.asarray([x for x in vals if np.isfinite(x)], np.float64)
    if v.size == 0:
        return [None, None]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def boot_aurc(kappa, loss, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    k, l = np.asarray(kappa, np.float64), np.asarray(loss, np.float64)
    rng = np.random.default_rng(seed)
    out = {"aurc": [], "rel_aurc": []}
    for _ in range(n_boot):
        ix = rng.integers(0, len(k), len(k))
        s = aurc_summary(k[ix], l[ix])
        out["aurc"].append(s["aurc"]); out["rel_aurc"].append(s["rel_aurc"])
    return {f"{m}_ci95": _pct(v) for m, v in out.items()}


def boot_auc(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> list:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    rng = np.random.default_rng(seed)
    return _pct([auc(a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]) for _ in range(n_boot)])


def submission_bytes(truth_ids, preds: dict) -> bytes:
    """run.py と同じ書き方（csv.writer、newline=''、truth の順、候補は ';' つなぎ）の submission のバイト列。"""
    buf = io.StringIO(newline="")
    w = csv.writer(buf); w.writerow(["molecule_id", "smiles"])
    for mid in truth_ids:
        w.writerow([mid, ";".join(preds[mid])])
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------- 対照の計算し直し
def _run_module():
    spec = importlib.util.spec_from_file_location("local_eval_run", ROOT / "scripts/local_eval/run.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["local_eval_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def recompute(holdout: str) -> dict:
    """run.py の run(holdout, score,prior,frag, logits_dir=truth_dir=outputs/local_eval/<h>) と同じ手順で lists を作り、
    分子ごとの確信度と正解を返す。submission のバイト一致を確かめ、不一致なら SystemExit(2)。"""
    import pandas as pd
    R = _run_module()
    from mrr25 import read_truth, reciprocal_rank, truth_keys
    import library as L
    t0 = time.time()
    d = ROOT / f"outputs/local_eval/{holdout}"
    R.check_stages(STAGES)
    zf = np.load(R.logits_path(f"{holdout}_fold0", d), allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = read_truth(R.truth_path(holdout, d))
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as frag
    frag.warmup()
    fuse = R._load("gl_fuse", ROOT / "data/ext/v4g/casmi26-glacier/gl_fuse.py")
    meta = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["mass", "formula"])
    formula_of = meta["formula"].values[np.argsort(meta["mass"].values, kind="stable")]
    spectra, fcache = R.query_spectra(holdout), {}
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()
    lists = {}
    for mid in truth:
        if mid not in targets:
            continue
        idx = R.visible_window(pool, targets[mid], hidden)
        rows = R.score_window(idx, pool, R.stage_z(zf[mid], None), fpc)
        rows = R.rerank_with_prior(rows, pool)
        rows = R.frag_rerank(rows, pool, spectra.get(mid, []), frag, fuse, fcache, formula_of)
        lists[mid] = R.dedup(rows, pool, R.TOPN)
    preds = {mid: [pool.smiles[i] for _, i in lists[mid][:R.K]] if mid in lists else [] for mid in truth}
    got = submission_bytes(list(truth), preds)
    ref_path = d / "submission_score+prior+frag.csv"
    ref = ref_path.read_bytes()
    match = dict(ref=str(ref_path.relative_to(ROOT)), ref_sha256=hashlib.sha256(ref).hexdigest(),
                 got_sha256=hashlib.sha256(got).hexdigest(), byte_identical=got == ref)
    if not match["byte_identical"]:
        print(json.dumps(dict(holdout=holdout, **match), ensure_ascii=False), flush=True)
        raise SystemExit(2)
    # κ_lib: run.py の lib 段の diag と同じ（L の候補の参照、隠す規則を適用、参照が無ければ 0）
    keys = {pool.key[i] for rows in lists.values() for _, i in rows}
    refs = L.load_refs(holdout, keys, d / f"selpred_refs_top{R.TOPN}.parquet")
    mol = []
    for mid in truth:
        rows = lists.get(mid, [])
        g, c = confidences([r[0] for r in rows])
        if rows:
            sim = L.lib_scores(spectra.get(mid, []), [pool.key[i] for _, i in rows], refs)
            lib = max(sim.values(), default=0.0)
        else:
            lib = -math.inf
        rr = reciprocal_rank(truth_keys(truth[mid]), preds[mid])
        mol.append(dict(molecule_id=mid, n_list=len(rows), gap=g, conf=c, lib=float(lib), rr=rr, top1=int(rr == 1.0)))
    for m in mol:
        assert all(not math.isnan(m[k]) for k in KAPPAS), m
    return dict(holdout=holdout, match=match, molecules=mol, secs=round(time.time() - t0, 1),
                n_list_lt2=sum(m["n_list"] < 2 for m in mol), n_list0=sum(m["n_list"] == 0 for m in mol))


def summarise(per: dict) -> dict:
    """集合ごとの AURC と CI、C2 型対ほかの型の AUC と CI（主は AUC_MAIN）。"""
    out = {"definitions": __doc__.split("run.py は変えない")[0].strip(), "n_boot": N_BOOT, "seed": SEED, "sets": {}, "auc_vs_c2": {}}
    for h, r in per.items():
        mol = r["molecules"]
        loss = np.array([1 - m["top1"] for m in mol], np.float64)
        s = dict(n=len(mol), top1=round(float(1 - loss.mean()), 4), mrr25=round(float(np.mean([m["rr"] for m in mol])), 4),
                 n_list_lt2=r["n_list_lt2"], n_list0=r["n_list0"], match=r["match"], secs=r["secs"], kappa={})
        for k in KAPPAS:
            kap = np.array([m[k] for m in mol], np.float64)
            fin = kap[np.isfinite(kap)]
            summ = {m_: (round(v, 4) if np.isfinite(v) else None) for m_, v in aurc_summary(kap, loss).items()}
            summ.update(boot_aurc(kap, loss))
            summ["median"] = round(float(np.median(fin)), 4) if fin.size else None
            s["kappa"][k] = summ
        out["sets"][h] = s
    if C2 in per:
        b = {k: [m[k] for m in per[C2]["molecules"]] for k in KAPPAS}
        for h in per:
            if h == C2:
                continue
            a = {k: [m[k] for m in per[h]["molecules"]] for k in KAPPAS}
            out["auc_vs_c2"][h] = {"role": "main" if h == AUC_MAIN else "secondary",
                                   **{k: dict(auc=round(auc(a[k], b[k]), 4), ci95=boot_auc(a[k], b[k])) for k in KAPPAS}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", default=",".join(HOLDOUTS))
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/selpred.json"))
    args = ap.parse_args()
    per = {}
    for h in args.holdouts.split(","):
        per[h] = recompute(h)
        print(json.dumps(dict(holdout=h, byte_identical=per[h]["match"]["byte_identical"], secs=per[h]["secs"])), flush=True)
        with open(ROOT / f"outputs/local_eval/{h}/selpred_per_molecule.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per[h]["molecules"][0])); w.writeheader(); w.writerows(per[h]["molecules"])
    res = summarise(per)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({h: {k: v["aurc"] for k, v in s["kappa"].items()} for h, s in res["sets"].items()}, indent=1))


if __name__ == "__main__":
    main()
