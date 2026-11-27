#!/usr/bin/env python3
"""
Issue #278: 手元の実データで pubchem.run_holdout の 1 worker のメモリ（共有 = mmap の層の配列、専用 = それ以外）を測り、
workers=N の結果が workers=1 と完全に一致するかを確かめる（評価の数値には使わない）。

- 入力: 既存の fold-safe のロジット outputs/local_eval/<holdout>/pc_logits.npz の先頭 --n 分子、プールの鍵（run.load_pool）
- キャッシュは --out-dir（scratch）の下に書くので、holdout の pubchem_fold0.json には触らない
- 5 秒ごとに自分と子プロセスの `vmmap --summary`（mapped file の resident = 共有）と `footprint`（phys_footprint = 専用）を
  取り、最大値を JSON に書く。peak RSS は外側の `/usr/bin/time -l` で取る

    /usr/bin/time -l python scripts/bench/pubchem_parallel_probe.py --holdout v3_c1u_t10 --n 20 --workers 1 --out-dir <scratch>/w1
    python scripts/bench/pubchem_parallel_probe.py --holdout v3_c1u_t10 --n 20 --workers 2 --out-dir <scratch>/w2 \\
        --compare <scratch>/w1/pubchem_fold0.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

UNIT = {"B": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def _bytes(tok: str) -> int:
    m = re.fullmatch(r"([0-9.]+)([BKMGT])?", tok)
    return int(float(m.group(1)) * UNIT[m.group(2) or "B"]) if m else 0


def vmmap_mapped_file_resident(pid: int) -> int:
    """`vmmap --summary` の mapped file の行の RESIDENT（層の npy の mmap はここに入る）。"""
    out = subprocess.run(["vmmap", "--summary", str(pid)], capture_output=True, text=True, timeout=60).stdout
    for line in out.splitlines():
        if line.startswith("mapped file"):
            toks = line.split()
            return _bytes(toks[3])   # mapped file  VIRTUAL RESIDENT DIRTY ...
    return 0


def phys_footprint(pid: int) -> int:
    out = subprocess.run(["footprint", "-f", "bytes", "-p", str(pid)], capture_output=True, text=True, timeout=60).stdout
    m = re.search(r"Footprint:\s*([0-9]+)\s*B", out) or re.search(r"phys_footprint:\s*([0-9]+)", out)
    return int(m.group(1)) if m else 0


def children(pid: int) -> list[int]:
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


class Sampler(threading.Thread):
    def __init__(self, every: float = 5.0):
        super().__init__(daemon=True)
        self.every, self.stop, self.peak = every, threading.Event(), {}

    def run(self):
        while not self.stop.is_set():
            for pid in [os.getpid()] + children(os.getpid()):
                try:
                    sh, pr = vmmap_mapped_file_resident(pid), phys_footprint(pid)
                except Exception:  # noqa: BLE001  終わりかけのプロセス
                    continue
                key = "main" if pid == os.getpid() else "child"
                p = self.peak.setdefault(key, dict(shared_max=0, private_max=0, n=0))
                p["shared_max"], p["private_max"] = max(p["shared_max"], sh), max(p["private_max"], pr)
                p["n"] += 1
            self.stop.wait(self.every)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--no-sample", action="store_true", help="vmmap/footprint の見張りを止める")
    ap.add_argument("--compare", type=Path, default=None, help="workers=1 の pubchem_fold0.json（完全一致を確かめる）")
    a = ap.parse_args(argv)
    import numpy as np
    import pubchem as PC
    import run as R
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cache = a.out_dir / "pubchem_fold0.json"
    assert not cache.exists(), f"{cache} が既にある（測り直しは別のディレクトリで）"
    zf = np.load(ROOT / f"outputs/local_eval/{a.holdout}/pc_logits.npz", allow_pickle=True)
    targets = {str(m): float(t) for m, t in zf["__targets__"]}
    targets = dict(list(targets.items())[:a.n])
    bits = np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy")
    pool_keys = set(R.load_pool().key)
    smp = Sampler()
    if not a.no_sample:
        smp.start()
    t0 = time.time()
    res = PC.run_holdout(a.holdout, targets, zf, bits, pool_keys, workers=a.workers, cache=cache)
    secs = round(time.time() - t0, 2)
    smp.stop.set()
    if smp.is_alive():
        smp.join()
    rep = dict(issue=278, holdout=a.holdout, n=len(targets), workers=a.workers, secs=secs,
               n_q=int(sum(len(v) for v in res.values())), peak=smp.peak, cache=str(cache))
    if a.compare:
        ref = json.loads(a.compare.read_text())
        mine = json.loads(cache.read_text())
        same_keys = list(ref) == list(mine)
        same_lists = all([r[2] for r in ref[m]] == [r[2] for r in mine[m]] and [r[0] for r in ref[m]] == [r[0] for r in mine[m]]
                         for m in ref) if same_keys else False
        same_fz = all([r[1] for r in ref[m]] == [r[1] for r in mine[m]] for m in ref) if same_lists else False
        rep["compare"] = dict(ref=str(a.compare), same_molecules=same_keys, same_keys_and_order=same_lists,
                              same_fz_exact=same_fz, bytes_identical=a.compare.read_bytes() == cache.read_bytes())
    (a.out_dir / "probe.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if a.compare and not rep["compare"]["bytes_identical"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
