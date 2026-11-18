#!/bin/bash
# #152 Engine 2 FP（S2、train_s2_resumable.py）本番の再開ループ。切り離して起動する（PPID 1）。
# 使い方: bash e2fp_loop.sh <out dir> <run dir> <sha256 list> -- <python> <train_s2_resumable.py> <args...>
#   ループの開始時: swap の used（sysctl vm.swapusage）を基準 SWAP0 として 1 回だけ取って固定する（読めなければ終了 4）
#   各回の前: (1) この run dir の学習の python が残っていれば止める（自分のプロセスだけ。引数に --run <run dir> を含むもの）
#             (2) 入力の sha256 を <sha256 list>（shasum -a 256 -c の形式）で確かめる。合わなければ終了 2
#             (3) vm_stat の空き（free+inactive+speculative）≥ MIN_AVAIL_GB（既定 10）かつ swap の used < SWAP0 + SWAP_MAX_INC_MB
#                 （既定 2048）まで 5 分おきに待つ。上限 WAIT_MAX_S（既定 3600）。超えたら終了 5
#   各回: MIN_FREE_PCT=${GUARD_PCT:-15} memguard_run.sh <out>/train.log.<n> <cmd...>（PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO=0.6/0.5）
#         実行中は SWAP_CHECK_S（既定 60）秒おきに swap の used を見て、SWAP0 + SWAP_MAX_INC_MB 以上なら学習の主プロセスに SIGTERM
#         （共用の Mac を守る保険。速い増え方は memguard が 5 秒おきに受ける）
#   判定: ログに "non-finite loss" → 終了 2（再開しない。ALL_DONE より先に見る）
#         終了コード 0 かつログに ALL_DONE → 終了 0
#         swap の止め → 再開（decision=resume swap。次の回の前の (3) で swap が下がるのを待つ）
#         memguard の rc 99（メモリで止めた）→ 終了 99（再開しない。#152 §7）
#         それ以外（1 = /usr/bin/time の子が signal で死んだ、143 など）→ 再開
#   終わるとき（どの終了でも）: HEAVY_LOCK（重いジョブの印のディレクトリ）の owner に自分の pid が書かれていれば印を消す
#   再開は最大 MAX_RESTARTS 回（既定 5。swap の止めも数える）。超えたら終了 3。判定は 1 回ごとに <out>/loop.log に 1 行。
#   テスト用の差し替え: SWAP_CMD（sysctl -n vm.swapusage の代わりに eval するコマンド）
# 実行中のこのファイルを書き換えない（直すなら別名で作る）。
set -u
OUT="$1"; RUN="$2"; SHA="$3"; shift 3
[ "${1:-}" = "--" ] && shift
GUARD=${MEMGUARD:-/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh}
MAX_RESTARTS=${MAX_RESTARTS:-5}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-10}
WAIT_MAX_S=${WAIT_MAX_S:-3600}
WAIT_STEP_S=${WAIT_STEP_S:-300}
SWAP_MAX_INC_MB=${SWAP_MAX_INC_MB:-2048}
SWAP_CHECK_S=${SWAP_CHECK_S:-60}
LOOPLOG="$OUT/loop.log"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
say() { echo "$(date '+%F %T') $*" >> "$LOOPLOG"; }

avail_gb() {
  vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.2f", (f+i+s)*ps/1073741824}'
}

swap_mb() {   # swap の used（MB）
  if [ -n "${SWAP_CMD:-}" ]; then eval "$SWAP_CMD"; else sysctl -n vm.swapusage; fi |
    awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); gsub("M","",v); print v; exit}}'
}

leftovers() {   # この run dir の学習の python（/usr/bin/time・bash の行は除く）
  for p in $(pgrep -f "train_s2_resumable.py.*--run $RUN( |$)"); do
    c=$(ps -o command= -p "$p" 2>/dev/null)
    case "$c" in /usr/bin/time*|bash*|/bin/bash*|"") continue;; esac
    case "$c" in *bin/python*train_s2_resumable.py*) echo "$p";; esac
  done
}

main_pid() {    # leftovers のうち親が学習の python でないもの（fork した worker を除く）
  local p pp c
  for p in $(leftovers); do
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    c=$(ps -o command= -p "$pp" 2>/dev/null)
    case "$c" in /usr/bin/time*|bash*|/bin/bash*|"") echo "$p"; continue;; esac   # 親が time・bash なら主プロセス
    case "$c" in *bin/python*train_s2_resumable.py*) continue;; esac                # 親が学習の python なら worker
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
  local waited=0 a s
  while :; do
    a=$(avail_gb); s=$(swap_mb)
    if awk -v a="$a" -v m="$MIN_AVAIL_GB" -v s="$s" -v b="$SWAP0" -v d="$SWAP_MAX_INC_MB" 'BEGIN{exit !(a >= m && s != "" && s < b + d)}'; then
      say "vm_stat avail ${a} GB >= ${MIN_AVAIL_GB}, swap used ${s} MB < ${SWAP0}+${SWAP_MAX_INC_MB}"; return 0
    fi
    if [ "$waited" -ge "$WAIT_MAX_S" ]; then say "vm_stat avail ${a} GB / swap used ${s} MB (need >= ${MIN_AVAIL_GB} GB and < ${SWAP0}+${SWAP_MAX_INC_MB} MB) after ${waited}s -> give up"; return 5; fi
    say "vm_stat avail ${a} GB, swap used ${s} MB (need >= ${MIN_AVAIL_GB} GB and < ${SWAP0}+${SWAP_MAX_INC_MB} MB) -> wait ${WAIT_STEP_S}s"
    sleep "$WAIT_STEP_S"; waited=$((waited + WAIT_STEP_S))
  done
}

release_lock() {
  if [ -n "${HEAVY_LOCK:-}" ] && grep -q "pid=$$ " "$HEAVY_LOCK/owner" 2>/dev/null; then
    rm -f "$HEAVY_LOCK/owner"; rmdir "$HEAVY_LOCK" 2>/dev/null; say "heavy lock released ($HEAVY_LOCK)"
  fi
}
trap release_lock EXIT
SWAP0=$(swap_mb)
[ -z "$SWAP0" ] && { say "cannot read swap usage -> stop"; exit 4; }
n=0; restarts=0
say "loop start pid=$$ ppid=$PPID out=$OUT run=$RUN swap_base_mb=$SWAP0 swap_max_inc_mb=$SWAP_MAX_INC_MB"
while :; do
  prepare; rc=$?
  [ "$rc" -ne 0 ] && { say "stopped (prepare rc=$rc)"; exit "$rc"; }
  n=$((n + 1)); log="$OUT/train.log.$n"
  say "round $n start"
  (cd "$RUN" && MIN_FREE_PCT=${GUARD_PCT:-15} bash "$GUARD" "$log" "$@") &
  G=$!
  swapstop=0; t=0
  while kill -0 "$G" 2>/dev/null; do
    sleep 1; t=$((t + 1))
    if [ "$swapstop" -eq 0 ] && [ "$t" -ge "$SWAP_CHECK_S" ]; then
      t=0; s=$(swap_mb)
      if [ -n "$s" ] && awk -v s="$s" -v b="$SWAP0" -v d="$SWAP_MAX_INC_MB" 'BEGIN{exit !(s >= b + d)}'; then
        mp=$(main_pid | tr '\n' ' ')
        say "swap used ${s} MB >= ${SWAP0}+${SWAP_MAX_INC_MB} -> SIGTERM main [$mp]"
        swapstop=1; [ -n "$mp" ] && kill -TERM $mp 2>/dev/null
      fi
    fi
  done
  wait "$G"; rc=$?
  if grep -q 'non-finite loss' "$log"; then d="stop non_finite_loss"
  elif [ "$rc" -eq 0 ] && grep -q 'ALL_DONE' "$log"; then d="done"
  elif [ "$rc" -eq 99 ]; then d="stop memguard_rc99"
  elif [ "$swapstop" -eq 1 ]; then d="resume swap"
  else d="resume rc_$rc"; fi
  say "round $n rc=$rc decision=$d last_step=$(grep -o 'step [0-9]* lr' "$log" | tail -1) log=$log"
  case "$d" in
    done) say "finished"; exit 0;;
    "stop memguard_rc99") say "stopped ($d)"; exit 99;;
    stop*) say "stopped ($d)"; exit 2;;
  esac
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then say "stop max_restarts=$MAX_RESTARTS"; exit 3; fi
  restarts=$((restarts + 1))
done
