#!/bin/bash
# #120 fragnn 本番の学習（scripts/train/fragnn_train.py）の再開ループ。#109 段 b の再開ループ（logs/align_b_loop.sh）と同じ判定の型。
# 使い方: bash fragnn_loop.sh <logbase> <train_report.json> -- <cmd...>
#   各回: MIN_FREE_PCT=${GUARD_PCT:-15} memguard_run.sh <logbase>.<n> <cmd...>   （MEMGUARD で memguard のパスを替えられる、テスト用）
#   判定（終了コード＋report の status と train.stop_reason＋その回のログ）:
#     99                                        → WAIT_S 待って再開（memguard の停止）
#     0 かつ status paused                      → WAIT_S 待って再開（学習中に avail < 2 GB）
#     0 かつ status ok・stop_reason converged   → 終了 0（止め方で終了 = 収束）
#     0 かつ status ok・stop_reason max_steps   → 終了 0（完了 = 上限 300,000 ステップ）
#     0 かつ status ok・stop_reason wallclock_cap → 終了 0（経過時間の上限。続けるかはユーザーが判断）
#     0 かつ status ok・それ以外の stop_reason  → 終了 2（記録）
#     0 かつ status error                       → 終了 2（エラー）
#     0 かつ report 無し・running・その他       → 終了 2（記録）
#     1 かつ ログに ^vm_stat avail( の行 かつ 最後の行が memguard の exit=1 → WAIT_S 待って再開（起動時の空き不足）
#     1 のそれ以外（set PYTORCH_MPS_ / another heavy job / status ok の再開拒否 / meta の不一致 / 例外）→ 終了 2（エラー）
#     その他の終了コード（143 など）            → 終了 2（エラー）
#     注: 子が SIGKILL（jetsam など）で死ぬと /usr/bin/time -l は 1 を返す（実測、#126 起動前レビュー）ので rc1_other で止まる
#   再開は最大 MAX_RESTARTS 回（既定 12、超えたら終了 3）。判定は 1 回ごとに <logbase>.loop に 1 行。
#   LOCK_FILE を渡すと、走っている間だけ pid を書く（生きた pid が持っていれば起動せず終了 4）。
# 実行中のこのファイルを書き換えない（直すなら別名で作る）。
set -u
LOGBASE="$1"; REPORT="$2"; shift 2
[ "${1:-}" = "--" ] && shift
GUARD=${MEMGUARD:-/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh}
MAX_RESTARTS=${MAX_RESTARTS:-12}
WAIT_S=${WAIT_S:-600}
LOCK=${LOCK_FILE:-}
PY=${LOOP_PY:-/usr/bin/python3}
LOOPLOG="$LOGBASE.loop"
if [ -n "$LOCK" ]; then
  if [ -f "$LOCK" ]; then
    op=$(sed -n 's/^pid=//p' "$LOCK" | head -1)
    if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then
      echo "$(date '+%F %T') lock $LOCK is held by live pid $op -> not starting" >> "$LOOPLOG"; exit 4
    fi
  fi
  printf 'pid=%s\nowner=massspec #120 fragnn 学習の再開ループ\nstarted=%s\nlog=%s.<n>\nloop_log=%s\n' \
    "$$" "$(date '+%F %T')" "$LOGBASE" "$LOOPLOG" > "$LOCK"
  trap 'rm -f "$LOCK"' EXIT
fi

status_of() {   # report の "status stop_reason"（無ければ none）
  [ -f "$REPORT" ] || { echo "none -"; return; }
  "$PY" -c 'import json,sys
try:
    r = json.load(open(sys.argv[1]))
    print(r.get("status", "none"), (r.get("train") or {}).get("stop_reason") or "-")
except Exception:
    print("unreadable -")' "$REPORT"
}

decide() {   # $1 = 終了コード, $2 = その回のログ → "resume|done|stop <理由>"
  local rc="$1" log="$2" st sr
  case "$rc" in
    99) echo "resume memguard_kill"; return;;
    0)
      read -r st sr <<< "$(status_of)"
      case "$st" in
        paused) echo "resume paused_low_memory";;
        ok)
          case "$sr" in
            converged|max_steps|wallclock_cap) echo "done $sr";;
            *) echo "stop status_ok_unknown_stop_reason_$sr";;
          esac;;
        error) echo "stop status_error";;
        *) echo "stop rc0_status_$st";;
      esac
      return;;
    1)
      if grep -q '^vm_stat avail(' "$log" 2>/dev/null && tail -n 1 "$log" | grep -q ' exit=1$'; then
        echo "resume start_low_avail"
      else
        echo "stop rc1_other"
      fi
      return;;
    *) echo "stop rc_$rc"; return;;
  esac
}

n=0; restarts=0
while :; do
  n=$((n + 1))
  log="$LOGBASE.$n"
  echo "$(date '+%F %T') round $n start" >> "$LOOPLOG"
  MIN_FREE_PCT=${GUARD_PCT:-15} bash "$GUARD" "$log" "$@"
  rc=$?
  d=$(decide "$rc" "$log")
  echo "$(date '+%F %T') round $n rc=$rc decision=$d log=$log" >> "$LOOPLOG"
  case "$d" in
    resume*)
      if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "$(date '+%F %T') stop max_restarts=$MAX_RESTARTS" >> "$LOOPLOG"; exit 3
      fi
      restarts=$((restarts + 1))
      sleep "$WAIT_S";;
    done*) echo "$(date '+%F %T') finished ($d)" >> "$LOOPLOG"; exit 0;;
    *) echo "$(date '+%F %T') stopped ($d)" >> "$LOOPLOG"; exit 2;;
  esac
done
