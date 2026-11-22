#!/usr/bin/env python3
"""#196: fragnn v2 の学習の起動の関門（計画 2026-11-23 夜 項目 2 の手順 5、#196 の 16:20Z の追記 3）。

1 回だけ測って判定し、JSON を 1 行出す。終了コード 0 = 関門を満たす（起動してよい）、1 = 満たさない（待つ）。
  (i)   memory_pressure の空き ≥ 25%
  (ii)  vm_stat の free + inactive + speculative ≥ 10.9 GB（= max(10, footprint の見込み 6.9 + 4)）
  (iii) swap の空き ≥ 0.5 GB（計画の値。#196 の 300 MB より厳しい方）
  (iv)  このリポジトリの重いジョブ（fragnn_train.py などの python）が 0 本、かつ
        ほかの python / MLX のプロセスで RSS ≥ 3 GiB のものが 0 本（skill の記録: 他の GPU ジョブと同時だと 1/3 の速さ）
測るだけで、どのプロセスにも触らない。純粋な関数（judge・parse_*）は CI でテストする。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

MIN_MP_FREE_PCT = 25
MIN_AVAIL_GB = 10.9
MIN_SWAP_FREE_GB = 0.5
BIG_RSS_GIB = 3.0
HEAVY_SCRIPTS = ("fragnn_train.py", "fragnn_bench.py", "fragnn_v2_bench.py", "align_local_embed.py", "align_train.py",
                 "local_fold0_logits.py", "fragnn_score.py")


def parse_memory_pressure(txt: str):
    m = re.search(r"free percentage:\s*(\d+)%", txt)
    return int(m.group(1)) if m else None


def parse_swap_free_gb(txt: str):
    m = re.search(r"free\s*=\s*([\d.]+)([MG])", txt)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1024.0 if m.group(2) == "M" else v


def parse_vm_stat_avail_gb(txt: str):
    m = re.search(r"page size of (\d+) bytes", txt)
    if not m:
        return None
    ps = int(m.group(1))

    def get(k):
        g = re.search(rf"{k}:\s+(\d+)", txt)
        return int(g.group(1)) * ps / 1e9 if g else 0.0
    return round(get("Pages free") + get("Pages speculative") + get("Pages inactive"), 2)


def _first_token(cmd: str) -> str:
    return cmd.strip().split(" ")[0]


def is_python_or_mlx(cmd: str) -> bool:
    """python（パスに空白があっても /bin/python を含めば）か MLX（コマンドに mlx を含む）か。シェル・time の行は数えない。"""
    first = os.path.basename(_first_token(cmd))
    if first in ("bash", "sh", "zsh", "time") or _first_token(cmd) == "/usr/bin/time":
        return False
    low = cmd.lower()
    return first.startswith("python") or "/bin/python" in cmd or "mlx" in low


def classify(ps_lines, me: int, big_gib: float = BIG_RSS_GIB) -> dict:
    """ps -Ao pid=,rss=,command= の行 → (このリポジトリの重いジョブ, ほかの大きな python/MLX)。rss は KiB。"""
    own, big = [], []
    for line in ps_lines:
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid, rss_kib, cmd = int(parts[0]), int(parts[1]), parts[2]
        if pid == me or not is_python_or_mlx(cmd):
            continue
        if any(s in cmd for s in HEAVY_SCRIPTS):
            own.append(f"{pid} {cmd[:120]}")
        elif rss_kib / (1024 ** 2) >= big_gib:
            big.append(f"{pid} rss={rss_kib / 1024 ** 2:.1f}GiB {cmd[:120]}")
    return dict(own_heavy=own, big_other=big)


def judge(mp_free_pct, avail_gb, swap_free_gb, own_heavy, big_other, min_mp: int = MIN_MP_FREE_PCT,
          min_avail: float = MIN_AVAIL_GB, min_swap: float = MIN_SWAP_FREE_GB) -> dict:
    """測れなかった値（None）は満たさないとする（保守的）。"""
    checks = {"mp_free_pct": mp_free_pct is not None and mp_free_pct >= min_mp,
              "avail_gb": avail_gb is not None and avail_gb >= min_avail,
              "swap_free_gb": swap_free_gb is not None and swap_free_gb >= min_swap,
              "own_heavy_0": len(own_heavy) == 0, "big_other_0": len(big_other) == 0}
    return dict(ok=all(checks.values()), checks=checks, mp_free_pct=mp_free_pct, avail_gb=avail_gb,
                swap_free_gb=swap_free_gb, own_heavy=own_heavy, big_other=big_other,
                thresholds=dict(min_mp_free_pct=min_mp, min_avail_gb=min_avail, min_swap_free_gb=min_swap,
                                big_rss_gib=BIG_RSS_GIB))


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def measure() -> dict:
    mp = parse_memory_pressure(_run(["memory_pressure"]))
    av = parse_vm_stat_avail_gb(_run(["vm_stat"]))
    sw = parse_swap_free_gb(_run(["sysctl", "vm.swapusage"]))
    me = {os.getpid(), os.getppid()}
    lines = _run(["ps", "-Ao", "pid=,rss=,command="]).splitlines()
    c = classify([ln for ln in lines if not (ln.strip().split(None, 1) or ["0"])[0] in {str(p) for p in me}], -1)
    return judge(mp, av, sw, c["own_heavy"], c["big_other"])


def main() -> None:
    import time
    r = measure()
    r["at"] = time.strftime("%F %T")
    print("GATE " + json.dumps(r, ensure_ascii=False), flush=True)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
