#!/usr/bin/env python3
"""
B8 振り分け器（Issue #216）: 1 集合ぶんの特徴と候補の列を作る（当てはめはしない）。

事前登録: 下書き issuecomment-7121558819 §2 ＋ 確定 issuecomment-7134581003「実装で決める細部」。
  L   = 対照 score,prior,frag の重複除去した上位 60（run.py の lists[mid]、selpred.recompute と同じ手順）。
        作り直した submission が outputs/local_eval/<h>/submission_score+prior+frag.csv とバイト一致しなければ止める（rc 2）
  Q   = pubchem.run_holdout（run.py の pubchem 段と同じ呼び方、プールのキーを除く、f·z 順、最大 60）
  y   = 正解（mrr25.to_key）が visible_window（隠したキーを除いた ±10 ppm）の候補のどれかと一致 → 1
  特徴（生の値。欠け・無限の置き換えは当てはめ側 b8_fit_eval.py で学習側の値を使って行う）:
        conf_L, gap_L（selpred.confidences）、conf_Q, gap_Q（同じ定義を Q の f·z に）、
        d = best_pool − Q[0] の f·z（best_pool は run.py の pubchem 段と同じ: 生の z、隠したキーを除く窓の最大）、
        lib = lib_max（run.py の lib 段の diag と同じ）、log1p(|visible_window|)、log1p(|PubChem 層の窓|)
  pubchem_fold0.json の隣に pc_logits.npz の sha256 を書き（pubchem_fold0.logits_sha256）、食い違えば止める（rc 3）。

    .venv/bin/python scripts/research/b8_features.py --holdout b8_f1_c_t10 \
        --dir outputs/local_eval/b8_f1_c_t10      # pc_logits.npz・truth.csv のある実のディレクトリ
出力: <dir>/b8_features.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/local_eval", "scripts/validate", "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / p))

FEATURES = ("conf_L", "gap_L", "conf_Q", "gap_Q", "d", "lib", "log_nwin_pool", "log_nwin_pc")
SIGNS = dict(conf_L=+1, gap_L=+1, conf_Q=-1, gap_Q=-1, d=+1, lib=+1, log_nwin_pool=0, log_nwin_pc=0)
STAGES = ["score", "prior", "frag"]
SHA_NAME = "pubchem_fold0.logits_sha256"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_cache_sha(d: Path, logits: Path) -> str:
    """pubchem のキャッシュがこの集合のロジットで作られたことの印。印が無いのにキャッシュがあれば止める（作った物が不明）。
    印があって食い違えば止める。無ければ書く。返り値は sha256。"""
    got = sha256_file(logits)
    mark, cache = d / SHA_NAME, d / "pubchem_fold0.json"
    if mark.is_file():
        want = mark.read_text().strip()
        if want != got:
            raise SystemExit(f"rc3: {mark} の sha256 {want[:12]}… と {logits} の {got[:12]}… が違う")
    else:
        if cache.is_file():
            raise SystemExit(f"rc3: {cache} があるのに {SHA_NAME} が無い（どのロジットで作ったか不明）")
        mark.write_text(got + "\n")
    return got


def q_conf(fz) -> tuple[float, float]:
    """Q の f·z（降順）に selpred.confidences と同じ定義。|Q|=0 は (nan, nan)（当てはめ側で 0 に置く）。"""
    import selpred
    if len(fz) == 0:
        return math.nan, math.nan
    return selpred.confidences(fz)


def main() -> None:
    import pandas as pd
    import selpred
    import pubchem as PC
    import library as L
    from mrr25 import read_truth, to_key, truth_keys
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--dir", required=True, help="pc_logits.npz・truth.csv・submission_score+prior+frag.csv のある実のディレクトリ")
    args = ap.parse_args()
    t0 = time.time()
    h, d = args.holdout, Path(args.dir)
    assert d.resolve() == (ROOT / f"outputs/local_eval/{h}").resolve(), "run.py・pubchem.py が読み書きする場所と違う"
    R = selpred._run_module()
    R.check_stages(STAGES)
    logits = R.logits_path(f"{h}_fold0", d)
    sha = check_cache_sha(d, logits)
    print(f"logits {logits} sha256 {sha}", flush=True)
    zf = np.load(logits, allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = read_truth(R.truth_path(h, d))
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as frag
    frag.warmup()
    fuse = R._load("gl_fuse", ROOT / "data/ext/v4g/casmi26-glacier/gl_fuse.py")
    meta = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["mass", "formula"])
    formula_of = meta["formula"].values[np.argsort(meta["mass"].values, kind="stable")]
    spectra, fcache = R.query_spectra(h), {}
    rule = L.hide_rule(h)
    hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
    lists, wins = {}, {}
    for mid in truth:
        if mid not in targets:
            continue
        idx = R.visible_window(pool, targets[mid], hidden)
        wins[mid] = idx
        rows = R.score_window(idx, pool, R.stage_z(zf[mid], None), fpc)
        rows = R.rerank_with_prior(rows, pool)
        rows = R.frag_rerank(rows, pool, spectra.get(mid, []), frag, fuse, fcache, formula_of)
        lists[mid] = R.dedup(rows, pool, R.TOPN)
    preds = {mid: [pool.smiles[i] for _, i in lists[mid][:R.K]] if mid in lists else [] for mid in truth}
    got = selpred.submission_bytes(list(truth), preds)
    ref = (d / "submission_score+prior+frag.csv").read_bytes()
    if got != ref:
        print(json.dumps(dict(holdout=h, byte_identical=False)), flush=True)
        raise SystemExit(2)
    t_l = time.time() - t0
    keys = {pool.key[i] for rows in lists.values() for _, i in rows}
    refs = L.load_refs(h, keys, d / f"refs_top{R.TOPN}.parquet")
    pool_keys = set(pool.key) - hidden
    t1 = time.time()
    pcres = PC.run_holdout(h, {m: targets[m] for m in lists}, zf, fpc.bits, pool_keys)   # run.py の pubchem 段と同じ
    t_pc = time.time() - t1
    tier = PC.Tier()
    mols = []
    for mid in truth:
        rows = lists.get(mid, [])
        tk = truth_keys(truth[mid])
        if mid not in targets:
            mols.append(dict(molecule_id=mid, missing_logits=True))
            continue
        g, c = selpred.confidences([r[0] for r in rows])
        lib = max(L.lib_scores(spectra.get(mid, []), [pool.key[i] for _, i in rows], refs).values(), default=0.0) if rows else -math.inf
        q = pcres.get(mid) or []
        gq, cq = q_conf([r[1] for r in q])
        bp = [r[0] for r in R.score_window(pool.window(targets[mid]), pool, zf[mid].astype(np.float32), fpc)
              if pool.key[r[1]] not in hidden]
        best_pool = max(bp) if bp else None
        if not q:
            dd = math.nan            # |Q| = 0（当てはめ側で学習側の d の最大に置く）
        elif best_pool is None:
            dd = -math.inf           # プールの窓が空（当てはめ側で学習側の有限の最小に置く）
        else:
            dd = float(best_pool - q[0][1])
        tol = targets[mid] * PC.PPM * 1e-6
        n_pc = int(np.searchsorted(tier.mass, targets[mid] + tol, "right") - np.searchsorted(tier.mass, targets[mid] - tol, "left"))
        y = int(any(to_key(pool.smiles[i]) in tk for i in wins[mid])) if tk else 0
        mols.append(dict(molecule_id=mid, y=y, conf_L=c, gap_L=g, conf_Q=cq, gap_Q=gq, d=dd, lib=float(lib),
                         log_nwin_pool=math.log1p(len(wins[mid])), log_nwin_pc=math.log1p(max(n_pc, 0)),
                         n_L=len(rows), n_Q=len(q),
                         L=[[pool.key[i], pool.smiles[i]] for _, i in rows],
                         Q=[[k, s] for s, _fz, k in q]))
    enc = lambda v: (None if isinstance(v, float) and math.isnan(v) else ("inf" if v == math.inf else ("-inf" if v == -math.inf else v)))
    out = dict(holdout=h, logits=str(logits), logits_sha256=sha, n=len(mols), hide_rule=rule,
               control_submission_sha256=hashlib.sha256(ref).hexdigest(), byte_identical=True,
               secs_total=round(time.time() - t0, 1), secs_lists=round(t_l, 1), secs_pubchem=round(t_pc, 1),
               molecules=[{k: enc(v) for k, v in m.items()} for m in mols])
    (d / "b8_features.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "molecules"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
