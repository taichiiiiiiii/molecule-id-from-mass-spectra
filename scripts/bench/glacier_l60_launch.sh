#!/bin/bash
# Issue #242: GLACIER L60 の手元 CPU 生成（scripts/bench/glacier_l60.py run）を、関門を通ってから memguard の下で回す。
#   bash scripts/bench/glacier_l60_launch.sh <code_dir> <out_dir>
# code_dir: git archive の写し（data/・outputs/ はクローンへのリンク）。out_dir: クローンの実ディレクトリ。
# 関門（5 分おき、最大 2 h）: memory_pressure の空き ≥ 30%、vm_stat の free+inactive+speculative ≥ 10 GB、
#   swap の使用 ≤ 7.5 GB、RSS ≥ 3 GiB の python（/bin/python・Python.app）が他に無い。通らなければ記録して exit 3。
# 1 回目は shard 1 本だけ（/usr/bin/time -l で peak を測る）、2 回目で残り全部。memguard の停止（rc 99）なら止める。
set -uo pipefail
C="$1"; O="$2"
PY=/Users/taichi/work/molecule-id-dev/.venv/bin/python
MG=/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh
LOG="$O/launch.log"
export NUMBA_CACHE_DIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/numba
export PYTHONPYCACHEPREFIX=/Users/taichi/.hermes/profiles/massspec/cache/scratch/pyc_gl60
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

gate() {
  local mp vm sw hv
  mp=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  vm=$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.1f", (f+i+s)*ps/1e9}')
  sw=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); gsub("M","",v); print v}}')
  hv=$(ps -Ao pid=,rss=,command= | awk -v me=$$ '$2 >= 3*1024*1024 && (tolower($0) ~ /\/bin\/python/ || tolower($0) ~ /python\.app/) && $1 != me {print $1":"int($2/1024)"MB"}' | tr '\n' ' ')
  say "gate: memory_pressure_free=${mp}% vm_free_inactive_spec=${vm}GB swap_used=${sw}MB heavy_python=[${hv}]"
  [ -n "$mp" ] && [ "$mp" -ge 30 ] || return 1
  awk -v v="$vm" 'BEGIN{exit !(v >= 10)}' || return 1
  awk -v v="$sw" 'BEGIN{exit !(v <= 7680)}' || return 1
  [ -z "$hv" ] || return 1
  return 0
}

say "start code_dir=${C} out_dir=${O} pid=$$ ppid=${PPID}"
ok=0
for k in $(seq 1 25); do
  if gate; then ok=1; break; fi
  [ "$k" -lt 25 ] && sleep 300
done
if [ "$ok" != 1 ]; then say "gate_timeout（2 h 待って通らない）→ 起動しない"; exit 3; fi
cd "$C" || exit 4
say "gate ok → run 1（shard 1 本、/usr/bin/time -l）"
MIN_FREE_PCT=15 /bin/bash "$MG" "$O/run_first.log" "$PY" scripts/bench/glacier_l60.py run --out-dir "$O" \
  --pkg "$O/env/pkg" --site "$O/env/site" --max-shards 1
rc=$?; say "run 1 rc=${rc}"
[ "$rc" = 0 ] || { say "run 1 が失敗 → 止める"; exit "$rc"; }
say "run 2（残りの shard 全部）"
MIN_FREE_PCT=15 /bin/bash "$MG" "$O/run_rest.log" "$PY" scripts/bench/glacier_l60.py run --out-dir "$O" \
  --pkg "$O/env/pkg" --site "$O/env/site"
rc=$?; say "run 2 rc=${rc}"
n=$(ls "$O"/shards/shard_*.json 2>/dev/null | wc -l | tr -d ' ')
say "done rc=${rc} shard_json=${n}"
exit "$rc"
