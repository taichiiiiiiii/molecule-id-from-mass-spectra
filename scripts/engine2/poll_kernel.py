"""Kaggle カーネルの見張り（#165）。終わるときに --issue の Issue へ終状態を自動でコメントする。

    python poll_kernel.py --kernel taichiiiii/casmi26-engine2-build-b-run1 --issue 165 --log <path> \
        [--interval 300] [--fetch-dir <dir>] [--repo taichiiiiiiii/molecule-id-from-mass-spectra]

- 5 分おきに `kaggle kernels status <kernel>` を読み、--log に 1 行ずつ書く。
- 終わる条件: COMPLETE / ERROR / CANCEL、または「読めない」（空・認証切れ・想定外の文言・rc≠0）が 3 回続いた。
- 終わるとき: --fetch-dir があれば小さい出力だけ取り出し（ログ・*.json。npz は取らない）、本文をファイルに書いて
  `gh issue comment <--issue> -R <repo> -F <file>`（失敗したら 3 回まで再試行し、結果を --log に書く）。
  ERROR / CANCEL / 読めない ではカーネルのログの最後 30 行を本文に入れる。
- Issue の番号は必ず引数で渡す（既定値なし）。クローンの外の固定パスから動かす前提で、リポジトリのファイルは読まない。
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

TERMINAL = ("COMPLETE", "ERROR", "CANCEL")
MAX_UNREADABLE = 3
TAIL_LINES = 30
SMALL_FILES = r"(.*\.log|.*\.json)$"


def classify(text):
    """`kaggle kernels status` の出力 → COMPLETE / ERROR / CANCEL / RUNNING / QUEUED / UNREADABLE。"""
    if not text or not text.strip():
        return "UNREADABLE"
    low = text.lower()
    if any(w in low for w in ("authentication", "unauthorized", "401", "403", "forbidden", "credentials", "token")):
        return "UNREADABLE"
    m = re.search(r'has status "(?:KernelWorkerStatus\.)?([A-Za-z_]+)"', text)
    if not m:
        return "UNREADABLE"
    s = m.group(1).upper()
    if s == "COMPLETE":
        return "COMPLETE"
    if s in ("ERROR", "FAILED"):
        return "ERROR"
    if s.startswith("CANCEL"):
        return "CANCEL"
    if s == "RUNNING":
        return "RUNNING"
    if s in ("QUEUED", "NEW", "PENDING"):
        return "QUEUED"
    return "UNREADABLE"


def log_tail(text, n=TAIL_LINES):
    """Kaggle のログ（stream の JSON 配列）なら data を繋いでから、そうでなければ生の行で、最後の n 行。"""
    try:
        arr = json.loads(text)
        body = "".join(str(x.get("data", "")) for x in arr if isinstance(x, dict))
    except Exception:
        body = text
    lines = body.rstrip("\n").split("\n")
    return "\n".join(lines[-n:])


def comment_body(kernel, state, last_raw, when, history, tail=None, files=None, check=None, issue=None):
    """終状態のコメントの本文（文字列）。gh は呼ばない。"""
    out = [f"### 見張りの自動記録: `{kernel}` は **{state}**（{when}）",
           f"- 最後の `kaggle kernels status` の出力: `{(last_raw or '').strip()[:300]}`",
           f"- 見張りの経過: 最初の記録 {history[0] if history else '-'}、最後 {history[-1] if history else '-'}、"
           f"読んだ回数 {len(history)}"]
    if state == "UNREADABLE":
        out.append(f"- 状態が {MAX_UNREADABLE} 回続けて読めなかったので見張りを止めた（カーネル自体の状態は未確認。"
                   "認証切れなら `kaggle kernels status` を手で確かめる）")
    if files is not None:
        out.append("- 取り出した小さい出力（ログ・json。npz は取っていない）: "
                   + (", ".join(f"`{f}`" for f in files) if files else "なし"))
    if check is not None:
        out.append("- `check.json`（先頭 3,000 文字）:\n```\n" + check[:3000] + "\n```")
    if tail is not None:
        out.append(f"- カーネルのログの最後 {TAIL_LINES} 行:\n```\n" + tail.replace("```", "'''") + "\n```")
    if state == "COMPLETE":
        out.append("- 次: 表 2 本の取り出しと検証（sha256・恒等式・グループ数・統計）は次の回が行う。この記録は数値の判定をしていない")
    out.append(f"\n（`poll_kernel.py` が自動で書いた。Issue #{issue}）")
    return "\n".join(out) + "\n"


def write_body(path, body):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(body)
    return path


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--issue", required=True, type=int)
    ap.add_argument("--log", required=True)
    ap.add_argument("--repo", default="taichiiiiiiii/molecule-id-from-mass-spectra")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--fetch-dir")
    ap.add_argument("--kaggle", default=shutil.which("kaggle") or "kaggle")
    ap.add_argument("--gh", default=shutil.which("gh") or "gh")
    a = ap.parse_args(argv)

    def L(msg):
        with open(a.log, "a") as fh:
            fh.write(f"{now()} {msg}\n")

    L(f"poller start pid {os.getpid()} ppid {os.getppid()} kernel {a.kernel} issue {a.issue} "
      f"script {os.path.abspath(__file__)}")
    history, bad, last_raw, state = [], 0, "", None
    while True:
        try:
            r = subprocess.run([a.kaggle, "kernels", "status", a.kernel], capture_output=True, text=True, timeout=120)
            last_raw = (r.stdout + r.stderr).strip()
            st = classify(last_raw) if r.returncode == 0 else "UNREADABLE"
        except Exception as e:  # noqa: BLE001
            last_raw, st = f"exception {e!r}", "UNREADABLE"
        history.append(f"{now()} {st}")
        L(f"{st} | {last_raw[:200]}")
        if st in TERMINAL:
            state = st
            break
        bad = bad + 1 if st == "UNREADABLE" else 0
        if bad >= MAX_UNREADABLE:
            state = "UNREADABLE"
            break
        time.sleep(a.interval)

    files, check, tail = None, None, None
    if a.fetch_dir and state != "UNREADABLE":
        os.makedirs(a.fetch_dir, exist_ok=True)
        try:
            subprocess.run([a.kaggle, "kernels", "output", a.kernel, "-p", a.fetch_dir, "--file-pattern", SMALL_FILES],
                           capture_output=True, text=True, timeout=600)
        except Exception as e:  # noqa: BLE001
            L(f"fetch failed {e!r}")
        files = sorted(os.path.relpath(p, a.fetch_dir) for p in glob.glob(os.path.join(a.fetch_dir, "**", "*"), recursive=True)
                       if os.path.isfile(p))
        cj = [p for p in glob.glob(os.path.join(a.fetch_dir, "**", "check.json"), recursive=True)]
        if cj:
            check = open(cj[0]).read()
        logs = sorted(glob.glob(os.path.join(a.fetch_dir, "*.log")))
        if state in ("ERROR", "CANCEL"):
            tail = log_tail(open(logs[0]).read()) if logs else "（ログを取り出せなかった）"
    elif state in ("ERROR", "CANCEL"):
        tail = "（--fetch-dir が無いのでログを取り出していない）"
    body = comment_body(a.kernel, state, last_raw, now(), history, tail=tail, files=files, check=check, issue=a.issue)
    bp = write_body(os.path.join(os.path.dirname(os.path.abspath(a.log)), f"final_comment_{a.issue}.md"), body)
    L(f"DONE {state}; body {bp}")
    for attempt in range(3):
        r = subprocess.run([a.gh, "issue", "comment", str(a.issue), "-R", a.repo, "-F", bp], capture_output=True, text=True)
        L(f"gh issue comment rc {r.returncode} {(r.stdout + r.stderr).strip()[:300]}")
        if r.returncode == 0:
            return 0
        time.sleep(60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
