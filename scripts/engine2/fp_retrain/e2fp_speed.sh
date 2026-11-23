#!/bin/bash
# #152 速さとメモリの測定のドライバー（#82 の fpnet_mps_speed.sh を写して、学習のスクリプト名と途中保存の名前を替えたもの）。
# 使い方: GUARD=<memguard_run.sh> bash e2fp_speed.sh <run dir> -- <python> <train_s2_resumable.py> <args...>
#   <args> には --run <run dir>（同じ絶対パス）を入れる。
#   起動前: swap の used を基準 SWAP0 として 1 回だけ取る（2 回目も同じ基準）。記録 e2fp_monitor.sh を 10 秒おきで起こす
#   回 r = 1..2（1 回目 T1 秒、2 回目 T2 秒で主プロセスに SIGTERM）:
#     前: vm_stat の空き ≥ MIN_AVAIL_GB（既定 10）を 5 分おきに待つ（上限 WAIT_MAX_S=3600、超えたら終了 5）
#         2 回目の前: 途中保存 <run>/state.pt を読んで step・elapsed を記録。読めなければ終了 4
#     中: POLL_S（既定 10）秒おきに swap の used を見て SWAP0 + SWAP_MAX_INC_MB（既定 2048）以上なら主プロセスに SIGTERM して終了 6
#         起動 PID_LOOKUP_S（既定 20）秒後に主プロセスを探し、見つからなければ memguard の木を止めて終了 7
#     後: SIGTERM を送る前に終わった（memguard の rc 99 を含む）→ 終了 8。SIGTERM の後の rc が 143 でない → 終了 9
#         SIGTERM の時点でいた worker が 5 秒後に残っていれば記録して KILL（終了 10）
#   どの止めでも「以降の回を起動しない」と speed_driver.log に書く。正常に 2 回終われば終了 0。
#   テスト用の差し替え: AVAIL_CMD（GB を 1 つ出すコマンド）、SWAP_CMD（vm.swapusage の形の行を出すコマンド）、
#                       RESUME_CHECK（途中保存のパスを引数にとって 1 行出すコマンド）、NO_MONITOR=1
set -u
SP="$1"; shift
[ "${1:-}" = "--" ] && shift
PY="$1"
GUARD=${GUARD:?GUARD (memguard_run.sh) を絶対パスで渡す}
HERE=$(cd "$(dirname "$0")" && pwd)
T1=${T1:-1020}; T2=${T2:-480}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-10}; WAIT_MAX_S=${WAIT_MAX_S:-3600}; WAIT_STEP_S=${WAIT_STEP_S:-300}
SWAP_MAX_INC_MB=${SWAP_MAX_INC_MB:-2048}; POLL_S=${POLL_S:-10}; PID_LOOKUP_S=${PID_LOOKUP_S:-20}
LOG="$SP/speed_driver.log"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

avail_gb() {
  if [ -n "${AVAIL_CMD:-}" ]; then eval "$AVAIL_CMD"; return; fi
  vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.2f", (f+i+s)*ps/1073741824}'
}
swap_mb() {
  if [ -n "${SWAP_CMD:-}" ]; then eval "$SWAP_CMD"; else sysctl -n vm.swapusage; fi |
    awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); gsub("M","",v); print v; exit}}'
}
is_py() {
  local c; c=$(ps -o command= -p "$1" 2>/dev/null)
  case "$c" in /usr/bin/time*|bash*|/bin/bash*|"") return 1;; esac
  case "$c" in *bin/python*train_s2_resumable.py*) return 0;; esac
  return 1
}
main_pid() {   # 学習の python のうち親が学習の python でないもの
  local p pp
  for p in $(pgrep -f "train_s2_resumable.py.*--run $SP( |$)"); do
    is_py "$p" || continue
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    is_py "$pp" || echo "$p"
  done
}
killtree() { local c; for c in $(pgrep -P "$1" 2>/dev/null); do killtree "$c"; done; kill -TERM "$1" 2>/dev/null; }
MON=""
finish() {   # $1 = 終了コード、$2 = 理由
  [ -n "$MON" ] && kill "$MON" 2>/dev/null
  if [ "$1" -ne 0 ]; then say "STOP rc=$1: $2 -> no further rounds"; else say "done: $2"; fi
  exit "$1"
}
resume_check() {
  if [ -n "${RESUME_CHECK:-}" ]; then eval "$RESUME_CHECK \"\$1\""; return; fi
  "$PY" -c "import sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False); print('resume step', c['step'], 'elapsed', c['elapsed'])" "$1"
}

mkdir -p "$SP"
SWAP0=$(swap_mb); [ -z "$SWAP0" ] && finish 4 "cannot read swap usage"
say "start swap_base_mb=$SWAP0 swap_max_inc_mb=$SWAP_MAX_INC_MB T1=$T1 T2=$T2 cmd=$*"
if [ -z "${NO_MONITOR:-}" ]; then
  bash "$HERE/e2fp_monitor.sh" "$SP" "$SP/mem.tsv" 10 & MON=$!
  say "monitor pid $MON"
fi
for r in 1 2; do
  [ $r = 1 ] && T=$T1 || T=$T2
  waited=0
  while :; do
    a=$(avail_gb)
    awk -v a="$a" -v m="$MIN_AVAIL_GB" 'BEGIN{exit !(a >= m)}' && { say "round $r vm_stat avail ${a} GB >= ${MIN_AVAIL_GB}"; break; }
    [ "$waited" -ge "$WAIT_MAX_S" ] && finish 5 "round $r vm_stat avail ${a} GB < ${MIN_AVAIL_GB} after ${waited}s"
    say "round $r vm_stat avail ${a} GB < ${MIN_AVAIL_GB} -> wait ${WAIT_STEP_S}s"; sleep "$WAIT_STEP_S"; waited=$((waited + WAIT_STEP_S))
  done
  if [ $r = 2 ]; then
    out=$(resume_check "$SP/state.pt" 2>&1) || finish 4 "round 2: cannot load $SP/state.pt: $(echo "$out" | tail -1)"
    say "before round 2: $out"
  fi
  say "round $r start (T=${T}s) swap_used_mb=$(swap_mb)"
  (cd "$SP" && MIN_FREE_PCT=15 bash "$GUARD" "$SP/train.log.$r" "$@") & G=$!
  t=0; mp=""; ended=0
  while [ "$t" -lt "$T" ]; do
    sleep "$POLL_S"; t=$((t + POLL_S))
    kill -0 "$G" 2>/dev/null || { ended=1; break; }
    if [ -z "$mp" ] && [ "$t" -ge "$PID_LOOKUP_S" ]; then
      mp=$(main_pid | head -1)
      if [ -z "$mp" ]; then killtree "$G"; wait "$G"; finish 7 "round $r: main training pid not found ${t}s after start (memguard tree stopped)"; fi
      say "round $r main pid=$mp kids=$(pgrep -P "$mp" | wc -l | tr -d ' ')"
    fi
    s=$(swap_mb)
    if [ -n "$s" ] && awk -v s="$s" -v b="$SWAP0" -v d="$SWAP_MAX_INC_MB" 'BEGIN{exit !(s >= b + d)}'; then
      p=${mp:-$(main_pid | head -1)}
      say "round $r swap used ${s} MB >= ${SWAP0}+${SWAP_MAX_INC_MB} -> SIGTERM main [$p]"
      if [ -n "$p" ]; then kill -TERM "$p" 2>/dev/null; else killtree "$G"; fi
      wait "$G"; finish 6 "round $r: swap used ${s} MB >= ${SWAP0}+${SWAP_MAX_INC_MB} (memguard rc=$?)"
    fi
  done
  if [ "$ended" = 1 ]; then
    wait "$G"; rc=$?
    finish 8 "round $r ended before the driver's SIGTERM (memguard rc=$rc; 99 = memguard stop)"
  fi
  mp=${mp:-$(main_pid | head -1)}
  if [ -z "$mp" ] || ! kill -0 "$mp" 2>/dev/null; then killtree "$G"; wait "$G"; finish 7 "round $r: main training pid not found at SIGTERM time"; fi
  kids=$(pgrep -P "$mp" 2>/dev/null | tr '\n' ' ')
  say "round $r SIGTERM main=$mp kids=[$kids]"
  kill -TERM "$mp"
  wait "$G"; rc=$?
  say "round $r memguard rc=$rc"
  sleep 5; left=""; for k in $kids; do kill -0 "$k" 2>/dev/null && left="$left $k"; done
  if [ -n "$left" ]; then kill -KILL $left 2>/dev/null; finish 10 "round $r leftover workers:[$left] (killed)"; fi
  say "round $r leftover workers:[]"
  [ "$rc" -ne 143 ] && finish 9 "round $r rc=$rc after SIGTERM (expected 143)"
done
finish 0 "2 rounds"
