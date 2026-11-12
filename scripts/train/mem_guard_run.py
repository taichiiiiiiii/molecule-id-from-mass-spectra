"""共用の Mac で重いかもしれないコマンドを、メモリと時間の上限つきで 1 本だけ走らせる包み（#88）。

    python scripts/train/mem_guard_run.py --log logs/x.log --report logs/x.json \
        --max-rss-gb 8 --max-minutes 30 --min-free-pct 10 -- python some_stage.py ...

- macOS では `/usr/bin/time -l` で包み、その「maximum resident set size」を peak_rss_time_bytes に記録する
  （これは直接の子プロセスだけの値。multiprocessing の子は含まない）
- 数秒おきに (a) `memory_pressure` の「System-wide memory free percentage」、(b) プロセス木全体の RSS 合計、
  (c) 経過時間を見て、どれかが上限を越えたらプロセス木ごと止める。memory_pressure の行が読めないときも
  「読めない＝安全と言えない」として止める
- 結果（終了コード、止めた理由、時間、RSS のピーク）を --report に JSON で書く

標準ライブラリだけを使う。CI（Linux）では free_pct_fn を差し替えてテストする。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Callable, List, Optional

FREE_RE = re.compile(r'System-wide memory free percentage:\s*(\d+)%')
TIME_RSS_RE = re.compile(r'^\s*(\d+)\s+maximum resident set size', re.M)


def free_pct_macos() -> Optional[int]:
    """memory_pressure の空き率（%）。行が無い・コマンドが無いときは None。"""
    try:
        out = subprocess.run(['memory_pressure'], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = FREE_RE.search(out)
    return int(m.group(1)) if m else None


def tree_rss_bytes(root_pid: int) -> int:
    """root_pid とその子孫の RSS の合計（ps の rss は KiB）。"""
    try:
        out = subprocess.run(['ps', '-Ao', 'pid=,ppid=,rss='], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    kids, rss = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, r = (int(x) for x in parts)
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = r
    tot, stack, seen = 0, [root_pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        tot += rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return tot * 1024


def run_guarded(cmd: List[str], log_path: str, max_rss_bytes: int, max_seconds: float, min_free_pct: int,
                poll_s: float = 5.0, free_pct_fn: Callable[[], Optional[int]] = free_pct_macos,
                use_time: Optional[bool] = None, env: Optional[dict] = None, cwd: Optional[str] = None) -> dict:
    use_time = (sys.platform == 'darwin' and os.path.exists('/usr/bin/time')) if use_time is None else use_time
    full = (['/usr/bin/time', '-l'] + list(cmd)) if use_time else list(cmd)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    t0 = time.time()
    reason, peak_tree, min_free_seen, polls = None, 0, None, 0
    with open(log_path, 'ab') as log:
        log.write(f'[mem_guard] start {time.strftime("%F %T")} cmd={full!r}\n'.encode())
        log.flush()
        p = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        while True:
            try:
                p.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            polls += 1
            fp = free_pct_fn()
            rss = tree_rss_bytes(p.pid)
            peak_tree = max(peak_tree, rss)
            if fp is not None:
                min_free_seen = fp if min_free_seen is None else min(min_free_seen, fp)
            if fp is None:
                reason = 'free_pct_unreadable'
            elif fp < min_free_pct:
                reason = f'free_pct {fp} < {min_free_pct}'
            elif rss > max_rss_bytes:
                reason = f'tree_rss {rss} > {max_rss_bytes}'
            elif time.time() - t0 > max_seconds:
                reason = f'wall > {max_seconds:.0f}s'
            if reason:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                except ProcessLookupError:
                    p.wait()
                break
        wall = time.time() - t0
        log.write(f'[mem_guard] end rc={p.returncode} wall={wall:.1f}s reason={reason}\n'.encode())
    peak_time = None
    if use_time:
        with open(log_path, 'rb') as fh:
            txt = fh.read().decode('utf-8', 'replace')
        found = TIME_RSS_RE.findall(txt)
        peak_time = int(found[-1]) if found else None
    return dict(cmd=list(cmd), returncode=p.returncode, killed_reason=reason, wall_s=round(wall, 1),
                peak_rss_time_bytes=peak_time, peak_tree_rss_poll_bytes=peak_tree, min_free_pct_seen=min_free_seen,
                polls=polls, log=log_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--max-rss-gb', type=float, default=8.0)
    ap.add_argument('--max-minutes', type=float, default=30.0)
    ap.add_argument('--min-free-pct', type=int, default=10)
    ap.add_argument('--poll-s', type=float, default=5.0)
    ap.add_argument('--cwd', default=None)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    if not cmd:
        ap.error('command missing (put it after --)')
    r = run_guarded(cmd, a.log, int(a.max_rss_gb * 1024 ** 3), a.max_minutes * 60, a.min_free_pct, a.poll_s,
                    free_pct_fn=free_pct_macos, cwd=a.cwd)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, 'w') as fh:
        json.dump(r, fh, indent=1, ensure_ascii=False)
    print(json.dumps(r, ensure_ascii=False))
    return 0 if r['returncode'] == 0 and r['killed_reason'] is None else 1


if __name__ == '__main__':
    sys.exit(main())
