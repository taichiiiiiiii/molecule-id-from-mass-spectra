#!/bin/bash
# #196: fragnn v2 の学習の関門つきの待ちループ（計画 2026-11-23 夜 項目 2 の手順 5）。
# 使い方: bash fragnn_gate_wait.sh <logbase> <train_report.json> -- <cmd...>
#   GATE_EVERY_S（既定 300）ごとに `$GATE_PY $GATE_SCRIPT` を 1 回呼ぶ（終了 0 = 関門を満たす）。
#   満たしたら fragnn_loop.sh（同じディレクトリ、LOOP で替えられる）を STOP_ON_MEMGUARD=1 で exec する
#   （memguard の rc 99 では次の round に進まない）。GATE_MAX_S（既定 86400 = 24 h）を超えたら起動せずに終了 5。
#   判定は 1 回ごとに <logbase>.gate に 1 行（GATE の JSON）、起動・時間切れは <logbase>.loop にも 1 行。
# 実行中のこのファイルを書き換えない（直すなら別名で作る）。
set -u
LOGBASE="$1"; REPORT="$2"; shift 2
[ "${1:-}" = "--" ] && shift
HERE="$(cd "$(dirname "$0")" && pwd)"
GATE_PY=${GATE_PY:-/usr/bin/python3}
GATE_SCRIPT=${GATE_SCRIPT:-$HERE/fragnn_gate.py}
LOOP=${LOOP:-$HERE/fragnn_loop.sh}
EVERY=${GATE_EVERY_S:-300}
MAX=${GATE_MAX_S:-86400}
GATELOG="$LOGBASE.gate"
LOOPLOG="$LOGBASE.loop"
start=$(date +%s)
echo "$(date '+%F %T') gate_wait start pid=$$ every=${EVERY}s max=${MAX}s" >> "$LOOPLOG"
while :; do
  "$GATE_PY" "$GATE_SCRIPT" >> "$GATELOG" 2>&1
  rc=$?
  if [ "$rc" = "0" ]; then
    echo "$(date '+%F %T') gate ok -> start fragnn_loop.sh" >> "$LOOPLOG"
    STOP_ON_MEMGUARD=1 exec bash "$LOOP" "$LOGBASE" "$REPORT" -- "$@"
  fi
  now=$(date +%s)
  if [ $((now - start + EVERY)) -gt "$MAX" ]; then
    echo "$(date '+%F %T') stopped (gate_timeout ${MAX}s, not started)" >> "$LOOPLOG"
    exit 5
  fi
  sleep "$EVERY"
done
