"""Kaggle カーネルの見張り（#165）。終わるときに --issue の Issue へ終状態を自動でコメントする。

    python poll_kernel.py --kernel taichiiiii/casmi26-engine2-build-b-run1 --issue 165 --log <path> \
        [--interval 300] [--fetch-dir <dir>] [--repo taichiiiiiiii/molecule-id-from-mass-spectra]

- 5 分おきに `kaggle kernels status <kernel>` を読み、--log に 1 行ずつ書く。
- 終わる条件: COMPLETE / ERROR / CANCEL、または「読めない」（空・認証切れ・想定外の文言・rc≠0）が 3 回続いた。
- 終わるとき: --fetch-dir があれば小さい出力だけ取り出し（ログ・*.json。npz は取らない）、本文をファイルに書いて
  `gh issue comment <--issue> -R <repo> -F <file>`（失敗したら 3 回まで再試行し、結果を --log に書く）。
  ERROR / CANCEL / 読めない ではカーネルのログの最後 30 行を本文に入れる。
- Issue の番号は必ず引数で渡す（既定値なし）。クローンの外の固定パスから動かす前提で、リポジトリのファイルは読まない。

提出の採点結果の見張り（#164）:

    python poll_kernel.py submission --ref <submission ref> --issue 164 --line 0.490 --log <path> [--interval 900]

- 15 分おきに `kaggle competitions submissions -c <slug> --csv` を読み、ref の行を探す。見つからない・読めない回は
  error として --log に書き、4 回続いたら止める。COMPLETE / ERROR、または --max-hours（既定 14 h）で終わる。
- 終わるとき、Public の値と事前登録の帯の機械的な読み（≥ line → 床として採用、< line → 再現しない・原因を調べる・
  追加の提出なし）を本文ファイルに書いて `gh issue comment` する。
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


# ---- 提出の採点結果の見張り（#164）: `poll_kernel.py submission --ref <ref> --issue <n> ...` ----
SUB_MAX_ERRORS = 4


def parse_submission(csv_text, ref):
    """`kaggle competitions submissions -c <slug> --csv` の出力から ref の行を探す。

    返り値: (state, public, row)。state は COMPLETE / ERROR / PENDING / NOTFOUND / UNREADABLE。
    public は COMPLETE のときだけ float（読めなければ state を UNREADABLE にする）。
    """
    import csv
    import io
    if not csv_text or not csv_text.strip():
        return "UNREADABLE", None, None
    lines = csv_text.replace("\r\n", "\n").strip().split("\n")
    i0 = next((i for i, l in enumerate(lines) if l.startswith("ref,")), None)
    if i0 is None:
        return "UNREADABLE", None, None
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[i0:]))))
    if not rows or "status" not in rows[0] or "publicScore" not in rows[0]:
        return "UNREADABLE", None, None
    row = next((r for r in rows if str(r.get("ref", "")).strip() == str(ref)), None)
    if row is None:
        return "NOTFOUND", None, None
    s = str(row.get("status", "")).split(".")[-1].strip().upper()
    if s == "COMPLETE":
        try:
            return "COMPLETE", float(row.get("publicScore")), row
        except (TypeError, ValueError):
            return "UNREADABLE", None, row
    if s in ("ERROR", "FAILED"):
        return "ERROR", None, row
    if s in ("PENDING", "RUNNING", "QUEUED", "NEW"):
        return "PENDING", None, row
    return "UNREADABLE", None, row


def band_reading(public, line):
    """事前登録の帯の機械的な読み（#164 §5）。public が None なら None。"""
    if public is None:
        return None
    if public >= line:
        return f"Public {public:.3f} ≥ {line:.3f} → 床として採用"
    return f"Public {public:.3f} < {line:.3f} → 再現しない・原因を調べる・追加の提出なし"


def submission_body(ref, state, public, line, row, when, history, issue):
    """採点結果のコメントの本文。gh は呼ばない。"""
    out = [f"### 見張りの自動記録: 提出 `{ref}` は **{state}**（{when}）"]
    if row is not None:
        out.append(f"- 行: date `{row.get('date', '')}`、description `{row.get('description', '')}`、"
                   f"status `{row.get('status', '')}`、publicScore `{row.get('publicScore', '')}`")
    if state == "COMPLETE":
        out.append(f"- 帯（事前登録、機械的な読み）: {band_reading(public, line)}")
        out.append("- この読みで定数・セルを選び直さない（事前登録どおり）")
    elif state == "ERROR":
        out.append("- 採点の再実行が ERROR。帯は読めない（スコアなし）。追加の提出はしない、原因は次の回で調べる")
    elif state == "TIMEOUT":
        out.append("- 見張りの上限時間を過ぎた（提出の状態は未確定。`kaggle competitions submissions` を手で確かめる）")
    else:
        out.append(f"- 提出の状態が {SUB_MAX_ERRORS} 回続けて読めない／見つからなかったので見張りを止めた"
                   "（認証切れなら手で確かめる）")
    out.append(f"- 見張りの経過: 最初 {history[0] if history else '-'}、最後 {history[-1] if history else '-'}、"
               f"読んだ回数 {len(history)}")
    out.append(f"\n（`poll_kernel.py submission` が自動で書いた。Issue #{issue}）")
    return "\n".join(out) + "\n"


def submission_main(argv):
    ap = argparse.ArgumentParser(prog="poll_kernel.py submission")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--issue", required=True, type=int)
    ap.add_argument("--log", required=True)
    ap.add_argument("--line", required=True, type=float, help="帯の線（例 0.490）")
    ap.add_argument("--competition", default="enveda-CASMI26-molecule-id-mass-spectra")
    ap.add_argument("--repo", default="taichiiiiiiii/molecule-id-from-mass-spectra")
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--max-hours", type=float, default=14.0)
    ap.add_argument("--kaggle", default=shutil.which("kaggle") or "kaggle")
    ap.add_argument("--gh", default=shutil.which("gh") or "gh")
    a = ap.parse_args(argv)

    def L(msg):
        with open(a.log, "a") as fh:
            fh.write(f"{now()} {msg}\n")

    L(f"submission poller start pid {os.getpid()} ppid {os.getppid()} ref {a.ref} issue {a.issue} "
      f"line {a.line} script {os.path.abspath(__file__)}")
    t0, history, bad = time.time(), [], 0
    state, public, row = None, None, None
    while True:
        try:
            r = subprocess.run([a.kaggle, "competitions", "submissions", "-c", a.competition, "--csv"],
                               capture_output=True, text=True, timeout=120)
            raw = r.stdout + r.stderr
            st, pub, rw = parse_submission(raw, a.ref) if r.returncode == 0 else ("UNREADABLE", None, None)
        except Exception as e:  # noqa: BLE001
            raw, st, pub, rw = f"exception {e!r}", "UNREADABLE", None, None
        history.append(f"{now()} {st}")
        L(f"{st} | public {pub} | {('error ' if st in ('UNREADABLE', 'NOTFOUND') else '')}"
          f"{(rw or {}).get('status', raw.strip()[:150])}")
        if st in ("COMPLETE", "ERROR"):
            state, public, row = st, pub, rw
            break
        bad = bad + 1 if st in ("UNREADABLE", "NOTFOUND") else 0
        if bad >= SUB_MAX_ERRORS:
            state, row = "UNREADABLE", rw
            break
        if time.time() - t0 > a.max_hours * 3600:
            state, row = "TIMEOUT", rw
            break
        time.sleep(a.interval)
    body = submission_body(a.ref, state, public, a.line, row, now(), history, a.issue)
    bp = write_body(os.path.join(os.path.dirname(os.path.abspath(a.log)), f"submission_comment_{a.issue}_{a.ref}.md"), body)
    L(f"DONE {state} public {public}; body {bp}")
    for attempt in range(3):
        r = subprocess.run([a.gh, "issue", "comment", str(a.issue), "-R", a.repo, "-F", bp], capture_output=True, text=True)
        L(f"gh issue comment rc {r.returncode} {(r.stdout + r.stderr).strip()[:300]}")
        if r.returncode == 0:
            return 0
        time.sleep(60)
    return 1


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "submission":
        return submission_main(argv[1:])
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
        # カーネル自体のログ（<slug>.log）を先に、無ければ他の *.log（build.log など）
        own = os.path.join(a.fetch_dir, a.kernel.split("/")[-1] + ".log")
        logs = ([own] if os.path.isfile(own) else []) + sorted(p for p in glob.glob(os.path.join(a.fetch_dir, "*.log")) if p != own)
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
