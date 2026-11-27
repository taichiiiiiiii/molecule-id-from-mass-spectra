#!/usr/bin/env python3
"""
Issue #294: 版 A の Kaggle 時間測定 版 3。版 2（#278）に無い 2 段 GLACIER と N1 だけを Kaggle の CPU で測る（提出しない）。

段（#294 に push の前に固定。順に、それぞれ別のプロセス。1 段が落ちても、依存しない段は続ける）:
  base     scripts/kaggle/e2e_timing.py --limit N（見える test の先頭 N 分子）で候補の表 features.parquet を作る。
           秒は参考（判定の土台は版 2 の 12,448.5 s）
  n1       neighbour_ref.run_set（3 段すべて: cand_fp・topk・refs_and_sims、前後の load_library・features）
  glcount  glacier_l60.select_rows・build_jobs と同じ規則で、base の表の本数/分子（プールの行 ＋ 層の行）
  glpred   dev の jobs.parquet の先頭の固定の 4,000 本（docs/bench/gl_dev_jobs_head4000.csv.gz）を
           glacier_l60.make_predict_fn と同じ読み込みで予測する。threads = os.cpu_count()

外挿と判定（#294 に固定）:
  GLACIER = ms/本 × max(312.25, 見える test の本数/分子) × 1,200 + 固定費（wheel の導入・重みの読み込み）
  N1      = （N1 の段の全体の秒 / N）× 1,200
  合計    = 12,448.5 + GLACIER + N1。≤ 16,200 関門内 / ≤ 32,400 上限内・関門外 / それより上 上限外

    python scripts/kaggle/e2e_timing_v3.py all --test <test.parquet> --sample <sample_submission.csv> \\
        --ranker <ranker_submit.txt> --features <features.json> --out-dir <dir> [--n-mol 40] [--gl-jobs 4000]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/bench", "scripts/features", "scripts/local_eval", "scripts/validate"):
    sys.path.insert(0, str(ROOT / _p))

NAME = "sub_test"                 # e2e_timing.py の NAME（何も隠さない holdout）
TARGET_N = 1200
V2_TOTAL = 12_448.5               # 版 2 の 1,200 分子への外挿（e2e_timing_278_v2.json）
GATE = 16_200.0                   # 9 h の 50%
LIMIT = 32_400.0                  # 9 h
DEV_JOBS = 544_886 + 673_214      # dev（B8 の 3,901 分子）の GLACIER の本数: プール（#242 L60 版）＋ 層（#268）
DEV_MOLS = 3_901
DEV_JOBS_PER_MOL = DEV_JOBS / DEV_MOLS      # 312.25
GL_JOBS_FILE = "docs/bench/gl_dev_jobs_head4000.csv.gz"
GL_JOBS_SRC_SHA = "7417992e1cc5100113002743f66799568eeb0b4ffa56b8386e3810b94465171b"   # glacier_l60/jobs.parquet
GL_RUNNER_SHA = "5c4f74313ac54263d05d987c39d012d81ec57c5f35e1095ecd49a5bcb6440fca"
GL_CKPT_SHA = "9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12"
GL_N = 4000


def tag(stage: str, **kw) -> None:
    print("TAG " + json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 外挿と判定（純粋な関数、テストあり）
def verdict_of(total: float, gate: float = GATE, limit: float = LIMIT) -> str:
    if total <= gate:
        return "関門内"
    if total <= limit:
        return "上限内・関門外"
    return "上限外"


def extrapolate(gl_pred_secs: float, gl_n: int, gl_fixed_secs: float, jobs_per_mol_test: float | None,
                n1_secs: float, n_mol: int, target_n: int = TARGET_N, v2_total: float = V2_TOTAL) -> dict:
    """#294 に固定した式。jobs_per_mol_test が None（数えられなかった）なら dev の値だけで判定する。"""
    assert gl_n > 0 and n_mol > 0
    ms = gl_pred_secs / gl_n * 1000.0
    j = max(DEV_JOBS_PER_MOL, jobs_per_mol_test if jobs_per_mol_test is not None else 0.0)
    gl = ms / 1000.0 * j * target_n + gl_fixed_secs
    s1 = n1_secs / n_mol
    n1 = s1 * target_n
    total = v2_total + gl + n1
    over = total - GATE
    main = "glacier" if gl >= n1 else "n1"
    return dict(gl_ms_per_job=round(ms, 3), jobs_per_mol_dev=round(DEV_JOBS_PER_MOL, 2),
                jobs_per_mol_test=None if jobs_per_mol_test is None else round(jobs_per_mol_test, 2),
                jobs_per_mol_used=round(j, 2), gl_fixed_secs=round(gl_fixed_secs, 1), gl_1200=round(gl, 1),
                n1_secs_per_mol=round(s1, 3), n1_1200=round(n1, 1), v2_total=v2_total, total=round(total, 1),
                gate=GATE, limit=LIMIT, margin_to_gate=round(GATE - total, 1), verdict=verdict_of(total),
                main_cause=main if over > 0 else None, target_n=target_n)


# ---------------------------------------------------------------- 段: glpred
def find_glacier_pkg(pattern: str = "/kaggle/input/**/gl_runner.py") -> Path:
    """gl_runner.py の sha256 と ckpt/glacier.pt の sha256 が両方合う場所（realpath で重ねる）。無ければ止める。"""
    hits = sorted({os.path.realpath(h) for h in glob.glob(pattern, recursive=True)})
    ok = []
    for h in hits:
        d = Path(h).parent
        if sha256_file(h) == GL_RUNNER_SHA and (d / "ckpt/glacier.pt").is_file():
            ok.append(d)
    ok = [d for d in ok if sha256_file(d / "ckpt/glacier.pt") == GL_CKPT_SHA]
    if not ok:
        raise FileNotFoundError(f"sha256 の合う gl_runner.py ＋ glacier.pt が無い（見つけた gl_runner.py: {hits}）")
    return ok[0]


def read_gl_jobs(p: Path, n: int):
    import pandas as pd
    j = pd.read_csv(p, dtype={"smiles": str, "adduct": str, "instr": str})
    assert list(j.columns) == ["row", "smiles", "adduct", "instr", "ce"], list(j.columns)
    assert (j.row.to_numpy() == range(len(j))).all() and len(j) >= n
    return j.iloc[:n]


def stage_glpred(a) -> dict:
    import numpy as np

    def log(*x):
        print(time.strftime("%H:%M:%S"), *x, flush=True)
    t0 = time.time()
    pkg = Path(a.gl_pkg) if a.gl_pkg else find_glacier_pkg()
    site = Path(a.gl_site)
    sys.path.insert(0, str(pkg))
    import gl_runner as R
    if a.gl_install:
        R.install_site(R.find_wheels(str(pkg), a.gl_wheels), str(site), log)
    t_install = time.time() - t0
    import glacier_l60 as GL
    t1 = time.time()
    threads = os.cpu_count() or 1
    fn = GL.make_predict_fn(pkg, site, threads, log)
    t_load = time.time() - t1
    jobs = read_gl_jobs(ROOT / GL_JOBS_FILE, a.gl_jobs)
    keys = list(zip(jobs.smiles.astype(str), jobs.adduct.astype(str), jobs.instr.astype(str), jobs.ce.astype(float)))
    t2 = time.time()
    preds, fail, stats = fn(keys)
    t_pred = time.time() - t2
    fail = np.asarray(fail)
    ok = sum(p is not None and len(p[0]) > 0 for p in preds)
    rec = dict(n_jobs=len(keys), n_unique=len(set(keys)), n_ok=int(ok), n_fail_smiles=int((fail == GL.FAIL_SMILES).sum()),
               install_secs=round(t_install, 2), load_secs=round(t_load, 2), pred_secs=round(t_pred, 2),
               ms_per_job=round(t_pred / len(keys) * 1000, 3), threads_requested=threads, pkg=str(pkg),
               jobs_file_sha256=sha256_file(ROOT / GL_JOBS_FILE), **{k: v for k, v in stats.items()})
    return rec


# ---------------------------------------------------------------- 段: glcount
def stage_glcount(a) -> dict:
    import numpy as np
    import pandas as pd
    import glacier_l60 as GL
    tab = pd.read_parquet(a.table, columns=["set", "molecule_id", "cand_key", "cand_smiles", "is_tier", "ctrl_rank"])
    mols = tab[["set", "molecule_id"]].drop_duplicates().copy()
    mols["molecule_id"] = mols.molecule_id.astype(str)
    out = {}
    for part in ("pool", "tier"):
        L = GL.select_rows(tab, part)
        nL = L.groupby(["set", "molecule_id"]).size().rename("n_L").reset_index()
        m = mols.merge(nL, on=["set", "molecule_id"], how="left").fillna({"n_L": 0})
        m["n_L"] = m.n_L.astype(np.int64)
        jobs, m2, n_spec = GL.build_jobs(ROOT, m, L)
        out[part] = dict(n_L=int(len(L)), n_jobs=int(len(jobs)),
                         n_unique=int(jobs[["smiles", "adduct", "instr", "ce"]].drop_duplicates().shape[0]),
                         n_mols_no_cov=int((m2.n_cat_r == 0).sum()), n_query_spectra=int(n_spec))
    n_mol = int(a.n_mol_expected or len(mols))
    total = out["pool"]["n_jobs"] + out["tier"]["n_jobs"]
    return dict(n_mol_table=int(len(mols)), n_mol=n_mol, n_jobs=total, jobs_per_mol=round(total / n_mol, 3), parts=out)


# ---------------------------------------------------------------- 段: n1
def stage_n1(a) -> dict:
    import neighbour_ref as NR
    cache = Path(a.out_dir) / "nbr_cache"
    if cache.exists():
        import shutil
        shutil.rmtree(cache)          # キャッシュなしの初回を測る
    cache.mkdir(parents=True)
    out = Path(a.out_dir) / "n1_features.parquet"
    rec = NR.run_set(NAME, Path(a.table), out, cache, "hide_rule")
    return rec


# ---------------------------------------------------------------- 全体
def run_child(stage: str, a, extra: list[str], log_path: Path) -> tuple[int, dict | None, float]:
    """自分を別プロセスで 1 段だけ回し、最後の RESULT 行の json を返す。"""
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), stage, "--out-dir", str(a.out_dir)] + extra
    t0 = time.time()
    res = None
    with open(log_path, "w") as lg:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            lg.write(line)
            if line.startswith("RESULT "):
                res = json.loads(line[len("RESULT "):])
        rc = p.wait()
    return rc, res, round(time.time() - t0, 2)


def stage_all(a) -> dict:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    rep = dict(issue=294, n_mol=a.n_mol, gl_jobs=a.gl_jobs, cpu_count=os.cpu_count(), python=sys.version.split()[0],
               code_commit=os.environ.get("CASMI_CODE_COMMIT"), stages={}, errors={})
    # 1) base
    t0 = time.time()
    cmd = [sys.executable, "-u", "scripts/kaggle/e2e_timing.py", "--test", a.test, "--sample", a.sample,
           "--ranker", a.ranker, "--features", a.features, "--out-dir", str(out / "base"), "--limit", str(a.n_mol),
           "--pubchem-workers", "auto"]
    with open(out / "base.log", "w") as lg:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=dict(os.environ))
        for line in p.stdout:
            print(line, end="", flush=True)
            lg.write(line)
        rc = p.wait()
    base = None
    if (out / "base/timing.json").is_file():
        base = json.loads((out / "base/timing.json").read_text())
    rep["stages"]["base"] = dict(rc=rc, wall=round(time.time() - t0, 2),
                                 stages=(base or {}).get("stages"), n_molecules=(base or {}).get("n_molecules"),
                                 table_rows=(base or {}).get("table_rows"))
    tag("base", **rep["stages"]["base"])
    table = ROOT / f"outputs/local_eval/{NAME}/features.parquet"
    if rc != 0 or not table.is_file():
        rep["errors"]["base"] = f"rc {rc}, features.parquet {'あり' if table.is_file() else 'なし'}（base.log を見る）"
    n_mol = int((base or {}).get("n_molecules") or a.n_mol)
    # 2) n1, 3) glcount（base の表が要る）
    n1 = glc = None
    if table.is_file():
        rc, n1, w = run_child("n1", a, ["--table", str(table)], out / "n1.log")
        rep["stages"]["n1"] = dict(rc=rc, wall=w, rec=n1)
        tag("n1", rc=rc, wall=w, secs=(n1 or {}).get("secs"), secs_total=(n1 or {}).get("secs_total"))
        if rc != 0 or n1 is None:
            rep["errors"]["n1"] = f"rc {rc}（n1.log を見る）"
        rc, glc, w = run_child("glcount", a, ["--table", str(table), "--n-mol-expected", str(n_mol)], out / "glcount.log")
        rep["stages"]["glcount"] = dict(rc=rc, wall=w, rec=glc)
        tag("glcount", rc=rc, wall=w, jobs_per_mol=(glc or {}).get("jobs_per_mol"))
        if rc != 0 or glc is None:
            rep["errors"]["glcount"] = f"rc {rc}（glcount.log を見る）"
    # 4) glpred（base に依存しない）
    extra = ["--gl-jobs", str(a.gl_jobs), "--gl-site", a.gl_site] + (["--gl-pkg", a.gl_pkg] if a.gl_pkg else []) \
        + (["--gl-wheels", a.gl_wheels] if a.gl_wheels else []) + ([] if a.gl_install else ["--no-gl-install"])
    rc, glp, w = run_child("glpred", a, extra, out / "glpred.log")
    rep["stages"]["glpred"] = dict(rc=rc, wall=w, rec=glp)
    tag("glpred", rc=rc, wall=w, **{k: (glp or {}).get(k) for k in ("ms_per_job", "pred_secs", "load_secs",
                                                                     "install_secs", "threads", "n_ok")})
    if rc != 0 or glp is None:
        rep["errors"]["glpred"] = f"rc {rc}（glpred.log を見る）"
    if n1 is not None and glp is not None:
        rep["extrapolation"] = extrapolate(glp["pred_secs"], glp["n_jobs"], glp["install_secs"] + glp["load_secs"],
                                           (glc or {}).get("jobs_per_mol"), n1["secs_total"], n_mol)
        rep["extrapolation"]["n_mol_measured"] = n_mol
        rep["extrapolation"]["n1_secs_measured"] = n1["secs_total"]
        tag("verdict", **rep["extrapolation"])
    rep["wall"] = round(time.time() - t_all, 1)
    (out / "timing_v3.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    tag("end", wall=rep["wall"], errors=rep["errors"])
    return rep


def build_gl_jobs_file(src: Path, dst: Path, n: int = GL_N) -> dict:
    """dev の glacier_l60/jobs.parquet（sha256 固定）の先頭 n 本を gzip の CSV に（mtime 0、同じ入力から同じバイト）。"""
    import pandas as pd
    assert sha256_file(src) == GL_JOBS_SRC_SHA, "jobs.parquet の sha256 が #242 の manifest と違う"
    j = pd.read_parquet(src, columns=["row", "smiles", "adduct", "instr", "ce"]).iloc[:n]
    raw = j.to_csv(index=False, lineterminator="\n").encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as g:
            g.write(raw)
    return dict(n=len(j), sha256=sha256_file(dst), csv_sha256=hashlib.sha256(raw).hexdigest(), src_sha256=GL_JOBS_SRC_SHA)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("all", "n1", "glcount", "glpred", "make-jobs"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test")
    ap.add_argument("--sample")
    ap.add_argument("--ranker")
    ap.add_argument("--features")
    ap.add_argument("--table")
    ap.add_argument("--n-mol", type=int, default=40)
    ap.add_argument("--n-mol-expected", type=int, default=None)
    ap.add_argument("--gl-jobs", type=int, default=GL_N)
    ap.add_argument("--gl-pkg", default=None, help="既定: /kaggle/input の gl_runner.py を sha256 で探す")
    ap.add_argument("--gl-site", default="/kaggle/working/gl_site")
    ap.add_argument("--gl-wheels", default=None)
    ap.add_argument("--no-gl-install", dest="gl_install", action="store_false", help="手元の rehearsal: 入れ済みの site")
    a = ap.parse_args(argv)
    if a.stage == "make-jobs":
        print(json.dumps(build_gl_jobs_file(ROOT / "outputs/local_eval/glacier_l60/jobs.parquet", ROOT / GL_JOBS_FILE)))
        return 0
    if a.stage == "all":
        rep = stage_all(a)
        return 0 if not rep["errors"] else 1
    t0 = time.time()
    rec = dict(n1=stage_n1, glcount=stage_glcount, glpred=stage_glpred)[a.stage](a)
    rec["stage_wall"] = round(time.time() - t0, 2)
    print("RESULT " + json.dumps(rec, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
