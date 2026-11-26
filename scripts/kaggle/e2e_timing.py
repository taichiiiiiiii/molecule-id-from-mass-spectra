#!/usr/bin/env python3
"""
Issue #274: 自前の最小版（プール＋PubChem 層 → exp003 の 64 列 → LambdaRank）を、test を「何も隠さないホールドアウト」として
dev の表と同じコードの経路に通し、段ごとの秒を測る（Kaggle の CPU カーネルの中で実行する。提出はしない）。

段（順に実行。各段の終わりに `TAG {json}` を 1 行出す）:
  prepare   test を data/holdouts/<name>/ に置く（hide_rule = all・drop_keys 空・pool keep、truth はダミー）
  logits    scripts/cv/local_fold0_logits.py（fpnet_fold0.pt、holdout と同じ 1 本）
  pubchem   pubchem.run_holdout（PubChem 層の ±10 ppm 窓、ECFP4 で上位 PC_N1 → 全ビットの f·z、K_KEEP 60）
  qrows     層の候補（Q）の鍵の train の行（library._read_rows、train.parquet の流し読み）
  features  run.run(..., dump_features=...)（プールの窓の採点・prior・断片の並べ替え = 対照の L、L の参照、特徴の 62 列）
  cc        add_contrastive_coverage.run_set（cc・cc_z）
  rank      LambdaRank（dev の全行で学習した 1 本、64 列）→ cand_key で重複を除いた上位 25 → submission.csv
features の中の部品の秒（重なりを含み、足し算にならない）は `within` に書く。

外挿（事前に固定）: 1,200 分子（見える test の 3 倍）の秒 = 段の秒 × 1,200 / 分子数（固定費も比例させる上側の見積もり）。
上限 = 9 h の 50% = 16,200 s。1 段でも、または全段の合計が超えれば over に名前と超過分を書く。

    python scripts/kaggle/e2e_timing.py --test data/test.parquet --sample data/sample_submission.csv \
        --ranker <ranker_submit.txt> --features <features.json> --out-dir <dir> [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/local_eval", "scripts/features", "scripts/ranker", "scripts/validate"):
    sys.path.insert(0, str(ROOT / _p))

NAME = "sub_test"
TARGET_N = 1200
LIMIT_SECS = 0.5 * 9 * 3600      # 9 h の 50%
FALLBACK = "CCO"
K = 25
ENVEDA = ["enveda-180", "enveda-np-examples"]
EXPECT_FEATURES_SHA = "e0b73e5cd9e46e0b11c89dea8aa4b0a03c4182efabc3c45daa0f61880c9ed303"   # exp003
STAGE_NAMES = ("prepare", "logits", "pubchem", "qrows", "features", "cc", "rank")


def tag(stage: str, **kw) -> None:
    print("TAG " + json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extrapolate(stage_secs: dict, n: int, target_n: int = TARGET_N, limit: float = LIMIT_SECS) -> dict:
    """段の秒 × target_n / n（固定費も比例させる上側の見積もり）と、上限を超える段・合計。"""
    assert n > 0
    f = target_n / n
    ext = {k: round(v * f, 1) for k, v in stage_secs.items()}
    total = round(sum(stage_secs.values()) * f, 1)
    over = [dict(stage=k, secs=v, over_by=round(v - limit, 1)) for k, v in ext.items() if v > limit]
    if total > limit:
        over.append(dict(stage="total", secs=total, over_by=round(total - limit, 1)))
    return dict(target_n=target_n, factor=round(f, 4), limit_secs=limit, stages=ext, total=total, over=over)


class Within:
    """関数の呼び出しの秒と回数を数える（モジュールの属性を差し替える）。"""

    def __init__(self):
        self.secs: dict = {}
        self.calls: dict = {}

    def wrap(self, mod, attr: str, name: str | None = None) -> None:
        fn = getattr(mod, attr)
        name = name or f"{mod.__name__}.{attr}"

        def w(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                self.secs[name] = self.secs.get(name, 0.0) + time.perf_counter() - t
                self.calls[name] = self.calls.get(name, 0) + 1
        setattr(mod, attr, w)

    def report(self) -> dict:
        return {k: dict(secs=round(v, 2), calls=self.calls[k]) for k, v in sorted(self.secs.items())}


def prepare(test: Path, sample: Path, limit: int | None) -> list[str]:
    import pandas as pd
    hd, ed = ROOT / f"data/holdouts/{NAME}", ROOT / f"outputs/local_eval/{NAME}"
    for d in (hd, ed):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    ids = [str(x) for x in pd.read_csv(sample).molecule_id]
    te = pd.read_parquet(test)
    if limit:
        keep = list(dict.fromkeys(te.molecule_id.astype(str)))[:limit]
        te = te[te.molecule_id.astype(str).isin(keep)]
        ids = [m for m in ids if m in set(keep)]
    te.to_parquet(hd / "test.parquet", index=False)
    pd.DataFrame(dict(molecule_id=ids)).assign(smiles=FALLBACK).to_csv(hd / "sample_submission.csv", index=False)
    (hd / "drop_keys.txt").write_text("")
    (hd / "hide_rule.json").write_text(json.dumps(dict(rule="all", enveda_libs=ENVEDA, pool="keep",
                                                       note="#274: 何も隠さない"), indent=2))
    pd.DataFrame(dict(molecule_id=ids, inchikey14="", smiles=FALLBACK)).to_csv(ed / "truth.csv", index=False)
    return ids


def write_submission(out: Path, ids: list[str], lists: dict) -> dict:
    n_rank = n_fb = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule_id", "smiles"])
        for m in ids:
            lst = (lists.get(m) or [])[:K]
            if lst:
                n_rank += 1
            else:
                lst, n_fb = [FALLBACK], n_fb + 1
            w.writerow([m, ";".join(lst)])
    return dict(ranker=n_rank, fallback_cco=n_fb)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--ranker", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="手元の試し用: 先頭 N 分子だけ")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    secs: dict = {}
    within = Within()
    t_all = time.time()

    def done(stage, t0, **kw):
        secs[stage] = round(time.time() - t0, 2)
        tag(stage, secs=secs[stage], elapsed=round(time.time() - t_all, 1), **kw)

    t0 = time.time()
    ids = prepare(a.test, a.sample, a.limit)
    done("prepare", t0, n_ids=len(ids))

    t0 = time.time()
    ed = ROOT / f"outputs/local_eval/{NAME}"
    subprocess.run([sys.executable, "scripts/cv/local_fold0_logits.py", "--holdout", NAME, "--out-dir", str(ed)],
                   cwd=ROOT, check=True)
    import numpy as np
    zf = np.load(ed / "pc_logits.npz", allow_pickle=True)
    targets = {str(m): float(t) for m, t in zf["__targets__"]}
    done("logits", t0, n_logits=len(targets))

    t0 = time.time()
    import pubchem as PC
    import run as R
    pool = R.load_pool()
    bits = np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy")
    t_pool = round(time.time() - t0, 2)
    pcres = PC.run_holdout(NAME, targets, zf, bits, set(pool.key))
    (ed / "pubchem_fold0.logits_sha256").write_text(sha256_file(ed / "pc_logits.npz"))
    done("pubchem", t0, n=len(pcres), load_pool_secs=t_pool, n_q=int(sum(len(v or []) for v in pcres.values())))
    del pool

    t0 = time.time()
    import library as L
    qkeys = {k for v in pcres.values() for _s, _f, k in (v or [])}
    qr = L._read_rows(ROOT / "data/train.parquet", qkeys)
    qp = ed / "qrows.parquet"
    qr.to_parquet(qp, index=False)
    done("qrows", t0, n_keys=len(qkeys), n_rows=int(len(qr)))
    del qr

    t0 = time.time()
    import dump_features as DF
    import mass_feasibility as MF
    DF.set_meta = lambda h: ("test", 0)   # test は型も fold も無い。fold 0 = holdout と同じモデル（FPNet・NP）
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as frag
    for mod, attr in ((R, "score_window"), (R, "rerank_with_prior"), (R, "frag_rerank"), (R, "load_pool"),
                      (L, "load_refs"), (L, "lib_scores"), (DF, "build"), (MF, "group_features"),
                      (frag, "score_molecule")):
        within.wrap(mod, attr)
    within.wrap(DF.NPScorer, "__call__", "NPScorer.__call__")
    tab_p = ed / "features.parquet"
    rec = R.run(NAME, ["score", "prior", "frag"], logits_dir=ed, truth_dir=ed, dump_features=tab_p,
                dump_opts=dict(qrows=str(qp), np_scorer=DF.NPScorer(0)))
    done("features", t0, n_molecules=rec["n_molecules"], n_rows=rec["n_rows"], n_L=rec["n_L"], n_Q=rec["n_Q"],
         n_nan=rec["n_nan"], n_inf=rec["n_inf"])

    t0 = time.time()
    import pandas as pd
    import add_contrastive_coverage as ACC
    key = pd.read_parquet(tab_p, columns=ACC.KEY_COLS)
    parts = ed / "parts_cc"
    parts.mkdir(exist_ok=True)
    sp_in = parts / f"{NAME}.in.parquet"
    key.to_parquet(sp_in, index=False)
    crec = json.loads(Path(ACC.run_set((NAME, sp_in, parts))).read_text())
    done("cc", t0, n_rows=crec["n_rows"], metfrag_rebuild_ok=crec["metfrag_rebuild_ok"])

    t0 = time.time()
    import lightgbm as lgb
    import lambdarank_oof as LR
    from evaluate import ranked_lists
    from mrr25 import validate_submission
    feats = tuple(json.loads(a.features.read_text()))
    assert LR.features_sha256(feats) == EXPECT_FEATURES_SHA, "特徴の一覧が exp003 と違う"
    tab = pd.concat([pd.read_parquet(tab_p), pd.read_parquet(parts / f"{NAME}.parquet")], axis=1)
    bst = lgb.Booster(model_file=str(a.ranker))
    assert tuple(bst.feature_name()) == feats, "モデルの特徴の一覧が features.json と違う"
    X = tab[list(feats)].to_numpy(np.float64)
    assert np.isfinite(X).all(), "特徴に NaN/inf"
    sc = bst.predict(X)
    assert np.isfinite(sc).all()
    oof = tab[["set", "molecule_id", "cand_key", "cand_smiles"]].copy()
    oof.insert(0, "row", np.arange(len(tab)))
    oof["score"] = sc
    lists = {mid: lst for (_s, mid), lst in ranked_lists(oof, K).items()}
    sub = a.out_dir / "submission.csv"
    counts = write_submission(sub, ids, lists)
    with open(sub, newline="") as f:
        got = validate_submission(list(csv.reader(f)), expected_ids=set(ids))
    n_cands = [len(v) for v in got.values()]
    done("rank", t0, **counts, n_cand_min=min(n_cands), n_cand_max=max(n_cands))

    n = len(ids)
    rep = dict(issue=274, n_molecules=n, n_logits=len(targets), stages=secs, total=round(sum(secs.values()), 1),
               wall=round(time.time() - t_all, 1), within_features=within.report(),
               extrapolation=extrapolate(secs, n), counts=counts, n_cand_min=min(n_cands), n_cand_max=max(n_cands),
               ranker_sha256=sha256_file(a.ranker), features_sha256=LR.features_sha256(feats),
               submission_sha256=sha256_file(sub), table_rows=int(len(tab)), cpu_count=os.cpu_count(),
               python=sys.version.split()[0], lightgbm=lgb.__version__)
    (a.out_dir / "timing.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    tag("end", **{k: rep[k] for k in ("n_molecules", "total", "wall")}, over=rep["extrapolation"]["over"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
