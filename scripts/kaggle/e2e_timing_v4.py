#!/usr/bin/env python3
"""
Issue #302: 版 A の時間測定 版 4。GLACIER の予測の段だけを Kaggle の GPU（T4）で測る（提出しない）。

#302 に push の前に固定したこと:
  - 測る段: #294 と同じ dev の先頭の固定の 4,000 本（docs/bench/gl_dev_jobs_head4000.csv.gz）
  - 重み・コード: gl_runner.py・ckpt/glacier.pt は sha256 で探す（e2e_timing_v3.find_glacier_pkg と同じ）
  - device = cuda。torch.cuda.is_available() が偽なら止める（CPU に戻さない）。glacier_l60.make_predict_fn の
    device で R.load_model と R.Predictor の両方に渡す。batch の組み方はランナーの既定のまま（32 と gl_runner の既定）
  - ms/本 = fn(keys) の wall / 本数、固定費 = wheel の導入と探索 ＋ make_predict_fn の wall（重みの読み込み・CUDA への移し込み）
  - 外挿（B5 の直しの案、#195）: 12,448.5 / 3 × 1.24 ＋ (ms/本 / 1000 × 312.25 × 400 ＋ 固定費) × 1.24 ＋ 8.67 × 400 × 1.24
  - 判定: ≤ 16,200 関門内 / ≤ 32,400 上限内・関門外 / それより上 上限外
  - 上限: --deadline（time.time() の値）を過ぎたら予測を止め、途中の ms/本（予測した canonical の本数あたり）を出す。判定しない

    python scripts/kaggle/e2e_timing_v4.py --out-dir <dir> [--device cuda] [--gl-jobs 4000] [--deadline <epoch>] \\
        [--gl-pkg <dir>] [--gl-site <dir>] [--no-gl-install]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/bench", "scripts/kaggle"):
    sys.path.insert(0, str(ROOT / _p))

import e2e_timing_v3 as V3  # noqa: E402

TARGET_MOL = 400                  # 公式: 隠れテスト ~400 分子・~1,500 スペクトル
SPEC_RATIO = 1.24                 # 1,500 / 1,213（見える test のスペクトル）、全段に掛ける
V2_TOTAL_1200 = V3.V2_TOTAL       # 12,448.5（版 2 の 1,200 分子への外挿）
N1_SECS_PER_MOL = 8.67            # #294 の Kaggle CPU の実測
JOBS_PER_MOL = V3.DEV_JOBS_PER_MOL
GATE = V3.GATE
LIMIT = V3.LIMIT


def extrapolate_v4(gl_ms_per_job: float, gl_fixed_secs: float) -> dict:
    """#302 に固定した式。"""
    v2 = V2_TOTAL_1200 / 3.0 * SPEC_RATIO
    gl = (gl_ms_per_job / 1000.0 * JOBS_PER_MOL * TARGET_MOL + gl_fixed_secs) * SPEC_RATIO
    n1 = N1_SECS_PER_MOL * TARGET_MOL * SPEC_RATIO
    total = v2 + gl + n1
    return dict(gl_ms_per_job=round(gl_ms_per_job, 3), gl_fixed_secs=round(gl_fixed_secs, 2),
                jobs_per_mol=round(JOBS_PER_MOL, 2), target_mol=TARGET_MOL, spec_ratio=SPEC_RATIO,
                v2_part=round(v2, 1), glacier=round(gl, 1), n1=round(n1, 1), total=round(total, 1),
                gate=GATE, limit=LIMIT, margin_to_gate=round(GATE - total, 1), verdict=V3.verdict_of(total))


def require_device(device: str):
    """cuda を頼んで使えなければ止める（CPU に戻さない）。使う device の名前を返す。"""
    import torch
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device cuda を頼んだが torch.cuda.is_available() が偽（CPU に戻さず止める）")
        return torch.cuda.get_device_name(0)
    return "cpu"


def run(a) -> dict:
    import numpy as np

    def log(*x):
        print(time.strftime("%H:%M:%S"), *x, flush=True)
    rep = dict(issue=302, device=a.device, gl_jobs=a.gl_jobs, cpu_count=os.cpu_count(), python=sys.version.split()[0],
               code_commit=os.environ.get("CASMI_CODE_COMMIT"), deadline=a.deadline)
    rep["gpu"] = require_device(a.device)
    t0 = time.time()
    pkg = Path(a.gl_pkg) if a.gl_pkg else V3.find_glacier_pkg()
    site = Path(a.gl_site)
    sys.path.insert(0, str(pkg))
    import gl_runner as R
    if a.gl_install:
        R.install_site(R.find_wheels(str(pkg), a.gl_wheels), str(site), log)
    t_install = time.time() - t0
    import glacier_l60 as GL
    t1 = time.time()
    fn = GL.make_predict_fn(pkg, site, os.cpu_count() or 1, log, device=a.device,
                            deadline=a.deadline if a.deadline else float("inf"))
    t_load = time.time() - t1
    jobs = V3.read_gl_jobs(ROOT / V3.GL_JOBS_FILE, a.gl_jobs)
    keys = list(zip(jobs.smiles.astype(str), jobs.adduct.astype(str), jobs.instr.astype(str), jobs.ce.astype(float)))
    t2 = time.time()
    preds, fail, stats = fn(keys)
    t_pred = time.time() - t2
    fail = np.asarray(fail)
    finished = bool(stats.get("finished", True))
    ok = sum(p is not None and len(p[0]) > 0 for p in preds)
    rep.update(n_jobs=len(keys), n_unique=len(set(keys)), n_ok=int(ok), n_fail_smiles=int((fail == GL.FAIL_SMILES).sum()),
               install_secs=round(t_install, 2), load_secs=round(t_load, 2), pred_secs=round(t_pred, 2),
               finished=finished, pkg=str(pkg), jobs_file_sha256=V3.sha256_file(ROOT / V3.GL_JOBS_FILE),
               stats=stats)
    if a.device.startswith("cuda"):
        import torch
        rep["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    if finished:
        rep["ms_per_job"] = round(t_pred / len(keys) * 1000, 3)
        rep["extrapolation"] = extrapolate_v4(t_pred / len(keys) * 1000, t_install + t_load)
    else:   # 上限で止めた: 途中の値だけ（判定しない）
        n = int(stats.get("n_model_calls") or 0)
        rep["partial_ms_per_model_call"] = round(stats["t_pred"] / n * 1000, 3) if n else None
        rep["extrapolation"] = None
    print("TAG " + json.dumps(dict(stage="glpred", **{k: rep.get(k) for k in (
        "gpu", "finished", "ms_per_job", "pred_secs", "load_secs", "install_secs", "n_ok", "max_mem_gb",
        "partial_ms_per_model_call")}), ensure_ascii=False), flush=True)
    if rep["extrapolation"]:
        print("TAG " + json.dumps(dict(stage="verdict", **rep["extrapolation"]), ensure_ascii=False), flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gl-jobs", type=int, default=V3.GL_N)
    ap.add_argument("--deadline", type=float, default=0.0, help="time.time() の値。0 = 上限なし")
    ap.add_argument("--gl-pkg", default=None, help="既定: /kaggle/input の gl_runner.py を sha256 で探す")
    ap.add_argument("--gl-site", default="/kaggle/working/gl_site")
    ap.add_argument("--gl-wheels", default=None)
    ap.add_argument("--no-gl-install", dest="gl_install", action="store_false", help="手元の rehearsal: 入れ済みの site")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        rep = run(a)
        rc = 0
    except Exception as e:      # 原因の行を timing_v4.json に残してから落とす
        rep = dict(issue=302, device=a.device, error=repr(e)[:2000])
        rc = 1
    rep["wall"] = round(time.time() - t0, 1)
    (out / "timing_v4.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print("TAG " + json.dumps(dict(stage="end", rc=rc, wall=rep["wall"], error=rep.get("error")), ensure_ascii=False),
          flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
