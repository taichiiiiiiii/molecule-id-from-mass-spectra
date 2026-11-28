#!/usr/bin/env python3
"""H3 の数の表の大きさに比例する部分の peak の測定（Issue #296、全 train の前の関門の外挿用）。

全 train と同じ件数の合成の記録（構造・env・bin・名前の hash・数、aggregate_records の出力と同じ並び）を作り、
add_fragment_stats.build_counts の最後の段（塊をつなぐ → CountTable.from_hashed）と、集合ごとの段
（記録ごとの keep → for_set → own）を同じ順に通して、各段の後の maximum RSS を書く。値（θ・列）は作らない。

    python scripts/research/h3_counts_mem_bench.py --records 18460000 --names 3000000 --structs 275810
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/features"))
import fragment_stats as P  # noqa: E402


def rss() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, required=True)
    ap.add_argument("--names", type=int, required=True)
    ap.add_argument("--structs", type=int, required=True)
    ap.add_argument("--chunks", type=int, default=28)
    a = ap.parse_args()
    out = dict(args=vars(a), rss_gb={})
    out["rss_gb"]["start"] = rss()
    rng = np.random.default_rng(0)
    names = rng.integers(-2 ** 63, 2 ** 63 - 1, size=a.names, dtype=np.int64)
    per = a.records // a.chunks
    parts = []
    for c in range(a.chunks):
        n = per if c < a.chunks - 1 else a.records - per * (a.chunks - 1)
        lo = c * a.structs // a.chunks
        hi = (c + 1) * a.structs // a.chunks
        s = np.sort(rng.integers(lo, hi, size=n)).astype(np.int64)
        parts.append((s, rng.integers(0, 2, size=n).astype(np.int64), rng.integers(0, 200_000, size=n).astype(np.int64),
                      names[rng.integers(0, a.names, size=n)], rng.integers(1, 4, size=n).astype(np.int64)))
    del names
    out["rss_gb"]["parts"] = rss()
    t0 = time.time()
    cat = [np.concatenate([p[i] for p in parts]) for i in range(5)]
    del parts
    out["rss_gb"]["concat"] = rss()
    tab, vocab = P.CountTable.from_hashed(*cat)
    del cat
    out["rss_gb"]["from_hashed"] = rss()
    out["secs_from_hashed"] = round(time.time() - t0, 1)
    t1 = time.time()
    s_ok = rng.random(a.structs) < 0.75
    keep = np.where(tab.env, s_ok[tab.struct], s_ok[tab.struct])
    sc = tab.for_set(keep)
    out["rss_gb"]["for_set"] = rss()
    for i in range(200):
        sc.own({int(i * 997 % a.structs)})
    out["rss_gb"]["own"] = rss()
    out["secs_set"] = round(time.time() - t1, 1)
    out["n_records"] = int(len(tab.key))
    out["n_vocab"] = int(len(vocab))
    out["table_bytes"] = int(tab.struct.nbytes + tab.env.nbytes + tab.key.nbytes + tab.cnt.nbytes
                             + tab._by_struct_order.nbytes + tab._s_sorted.nbytes)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
