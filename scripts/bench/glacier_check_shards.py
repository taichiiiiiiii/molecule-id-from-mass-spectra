#!/usr/bin/env python3
"""Issue #268: GLACIER の予測の out-dir（glacier_l60.py の list → run）がすべて完了しているかを確かめる。

    .venv/bin/python scripts/bench/glacier_check_shards.py --out-dir outputs/local_eval/glacier_tier

点検: jobs.parquet の sha256 が manifest と一致、shard がすべて shard_done（npz の sha256・区間・jobs の sha256）、
shard の n_rows の合計 = manifest の n_jobs = glacier_l60.EXPECTED[part]["n_jobs"]（holdout の part は manifest の値）。
予測の値は見ない（数えるのは本数・失敗の数・秒・peak だけ）。すべて通れば exit 0、どれかが駄目なら exit 1。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bench"))

import glacier_l60 as G  # noqa: E402


def check(out_dir: Path) -> dict:
    man = json.loads((out_dir / "manifest.json").read_text())
    part = man.get("part", "pool")
    jsha = G.sha256(out_dir / "jobs.parquet")
    bounds = G.shard_bounds(man["n_jobs"], man["shard_size"])
    done = [G.shard_done(out_dir, i, a, b, jsha) for i, (a, b) in enumerate(bounds)]
    metas = []
    for i in range(len(bounds)):
        js = G.shard_paths(out_dir, i)[1]
        metas.append(json.loads(js.read_text()) if js.exists() else {})
    n_rows = sum(m.get("n_rows", 0) for m, d in zip(metas, done) if d)
    secs = [m["seconds"] for m in metas if "seconds" in m]
    rep = dict(out_dir=str(out_dir), part=part, jobs_sha_ok=jsha == man["outputs"]["jobs.parquet"],
               n_shards=len(bounds), n_done=int(sum(done)), n_rows_done=int(n_rows), n_jobs=man["n_jobs"],
               expected_n_jobs=G.expected_n_jobs(man),   # #290: holdout の part は manifest の本数
               n_ok=int(sum(m.get("n_ok", 0) for m in metas)),
               n_fail_pred=int(sum(m.get("n_fail_pred", 0) for m in metas)),
               n_fail_smiles=int(sum(m.get("n_fail_smiles", 0) for m in metas)),
               seconds_sum=round(sum(secs), 1),
               maxrss_bytes_max=max((m.get("maxrss_bytes", 0) for m in metas), default=0))
    rep["ok"] = bool(rep["jobs_sha_ok"] and rep["n_done"] == rep["n_shards"]
                     and rep["n_rows_done"] == rep["n_jobs"] == rep["expected_n_jobs"])
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = check(a.out_dir.resolve())
    print(json.dumps(rep, ensure_ascii=False))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
