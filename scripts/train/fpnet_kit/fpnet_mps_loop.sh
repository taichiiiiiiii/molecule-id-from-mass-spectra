#!/bin/bash
# #82 FPNet（キットの train_fpnet.py の MPS 版）本番の再開ループ。切り離して起動する（PPID 1）。
# 使い方: bash fpnet_mps_loop.sh <out dir> <run dir> <sha256 list> -- <python> <train_fpnet_mps.py> <args...>
#   各回の前: (1) この run dir の学習の python が残っていれば止める（自分のプロセスだけ。引数に --run <run dir> を含むもの）
#             (2) 入力の sha256 を <sha256 list>（shasum -a 256 -c の形式）で確かめる。合わなければ終了 2
#             (3) vm_stat の空き（free+inactive+speculative）≥ MIN_AVAIL_GB（既定 10）まで 5 分おきに待つ。上限 WAIT_MAX_S（既定 3600）。超えたら終了 5
#   各回: MIN_FREE_PCT=${GUARD_PCT:-15} memguard_run.sh <out>/train.log.<n> <cmd...>（PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO=0.6/0.5）
#   判定: 終了コード 0 かつログに ALL_DONE → 終了 0
#         ログに "non-finite loss" → 終了 2（再開しない）
#         それ以外（99 = memguard の停止、1 = /usr/bin/time の子が signal で死んだ、143 など）→ 再開
#   再開は最大 MAX_RESTARTS 回（既定 5）。超えたら終了 3。判定は 1 回ごとに <out>/loop.log に 1 行。
# 実行中のこのファイルを書き換えない（直すなら別名で作る）。
set -u
OUT="$1"; RUN="$2"; SHA="$3"; shift 3
[ "${1:-}" = "--" ] && shift
GUARD=${MEMGUARD:-/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh}
MAX_RESTARTS=${MAX_RESTARTS:-5}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-10}
WAIT_MAX_S=${WAIT_MAX_S:-3600}
WAIT_STEP_S=${WAIT_STEP_S:-300}
LOOPLOG="$OUT/loop.log"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
say() { echo "$(date '+%F %T') $*" >> "$LOOPLOG"; }

avail_gb() {
  vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.2f", (f+i+s)*ps/1073741824}'
}

leftovers() {   # この run dir の学習の python（/usr/bin/time・bash の行は除く）
  for p in $(pgrep -f "train_fpnet_mps.py.*--run $RUN"); do
    c=$(ps -o command= -p "$p" 2>/dev/null)
    case "$c" in /usr/bin/time*|bash*|/bin/bash*|"") continue;; esac
    echo "$p"
  done
}

prepare() {
  local l; l=$(leftovers | tr '\n' ' ')
  if [ -n "$l" ]; then
    say "leftover train processes: $l -> TERM"; kill -TERM $l 2>/dev/null; sleep 10
    l=$(leftovers | tr '\n' ' '); [ -n "$l" ] && { say "still alive: $l -> KILL"; kill -KILL $l 2>/dev/null; sleep 2; }
  fi
  if ! (cd "$RUN/work" && shasum -a 256 -c "$SHA" > "$OUT/sha_check.last" 2>&1); then
    say "sha256 check FAILED (see $OUT/sha_check.last)"; return 2
  fi
  say "sha256 check ok ($(grep -c ': OK$' "$OUT/sha_check.last") files)"
  local waited=0 a
  while :; do
    a=$(avail_gb)
    if awk -v a="$a" -v m="$MIN_AVAIL_GB" 'BEGIN{exit !(a >= m)}'; then say "vm_stat avail ${a} GB >= ${MIN_AVAIL_GB}"; return 0; fi
    if [ "$waited" -ge "$WAIT_MAX_S" ]; then say "vm_stat avail ${a} GB < ${MIN_AVAIL_GB} after ${waited}s -> give up"; return 5; fi
    say "vm_stat avail ${a} GB < ${MIN_AVAIL_GB} -> wait ${WAIT_STEP_S}s"; sleep "$WAIT_STEP_S"; waited=$((waited + WAIT_STEP_S))
  done
}

n=0; restarts=0
say "loop start pid=$$ ppid=$PPID out=$OUT run=$RUN"
while :; do
  prepare; rc=$?
  [ "$rc" -ne 0 ] && { say "stopped (prepare rc=$rc)"; exit "$rc"; }
  n=$((n + 1)); log="$OUT/train.log.$n"
  say "round $n start"
  (cd "$RUN" && MIN_FREE_PCT=${GUARD_PCT:-15} bash "$GUARD" "$log" "$@")
  rc=$?
  if [ "$rc" -eq 0 ] && grep -q 'ALL_DONE' "$log"; then d="done"
  elif grep -q 'non-finite loss' "$log"; then d="stop non_finite_loss"
  else d="resume rc_$rc"; fi
  say "round $n rc=$rc decision=$d last_step=$(grep -o 'step [0-9]* lr' "$log" | tail -1) log=$log"
  case "$d" in
    done) say "finished"; exit 0;;
    stop*) say "stopped ($d)"; exit 2;;
  esac
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then say "stop max_restarts=$MAX_RESTARTS"; exit 3; fi
  restarts=$((restarts + 1))
done
