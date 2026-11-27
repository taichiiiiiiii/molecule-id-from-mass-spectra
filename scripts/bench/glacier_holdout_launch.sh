#!/bin/bash
# Issue #290: 判定用 4 集合（holdout_v3）の GLACIER の予測（glacier_l60.py run）を、関門を通ってから memguard
# （MIN_FREE_PCT=15）の下で回す。glacier_tier_launch.sh（#268）と同じ形で、プール（pool）→ 層（tier）の 2 つの out-dir を順に。
#   bash scripts/bench/glacier_holdout_launch.sh <code_dir> <out_root> <env_dir>      （すべて絶対パス）
# code_dir: git archive の写し（data/・outputs/ はクローンへのリンク）。out_root: クローンの実ディレクトリ
#   outputs/local_eval/glacier_holdout（下に pool/・tier/。どちらも list の段の出力が先にあること）。
# env_dir: dev の予測と同じ pkg・site（outputs/local_eval/glacier_l60/env、読むだけ）。
# 関門（5 分おき、最大 1 h = 13 回）: memory_pressure の空き ≥ 30%、vm_stat の free+inactive+speculative ≥ 10 GB、
#   swap の残り（free）≥ 512 MB、このプロジェクトの重い python（top の MEM ≥ 3 GB）が他に無い。
#   通らなければ記録して exit 3（起動しない）。他のプロジェクトの python は止めも待ちもせず、記録だけする。
# pool の 1 回目は shard 1 本だけ（/usr/bin/time -l で peak を測る）。peak が 8.6 GB（計画の見積もり）を超えたら止める。
# 失敗・memguard の停止（rc 99）なら次へ進まずに止める。予測の値（類似度・要約）は計算しない。
set -uo pipefail
C="$1"; O="$2"; E="$3"
for p in "$C" "$O" "$E"; do case "$p" in /*) ;; *) echo "絶対パスで渡す: $p" >&2; exit 2;; esac; done
PY=/Users/taichi/work/molecule-id-dev/.venv/bin/python
MG=/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh
LOG="$O/launch.log"
THREADS=8
PEAK_MAX=8600000000
export NUMBA_CACHE_DIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/numba
export PYTHONPYCACHEPREFIX=/Users/taichi/.hermes/profiles/massspec/cache/scratch/pyc_glhold
export TMPDIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/tmp_glhold
mkdir -p "$TMPDIR"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# top の MEM（例 22G・1534M・512K、末尾の +/- は無視）→ GB
heavy_pythons() {
  local own="$1"
  top -l 1 -stats pid,mem,command | awk '$1 ~ /^[0-9]+$/ && tolower($3) ~ /^python/ {m=$2; gsub(/[+-]$/,"",m); u=substr(m,length(m)); v=substr(m,1,length(m)-1)+0;
      g=(u=="G")?v:(u=="M")?v/1024:(u=="K")?v/1048576:v/1073741824; if (g>=3) print $1, g}' |
  while read -r pid g; do
    cmd=$(ps -o command= -p "$pid" 2>/dev/null)
    if printf '%s' "$cmd" | grep -qiE 'molecule-id-dev|Molecule ID From Mass Spectra'; then is_own=1; else is_own=0; fi
    if [ "$is_own" = "$own" ]; then printf '%s:%.1fGB:%s ' "$pid" "$g" "$(printf '%s' "$cmd" | cut -c1-90)"; fi
  done
}

swap_free_mb() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="free") {v=$(i+2); gsub("M","",v); print v}}'; }

gate() {
  local mp vm sf hv other np
  mp=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  vm=$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.1f", (f+i+s)*ps/1e9}')
  sf=$(swap_free_mb)
  np=$(top -l 1 -stats pid,mem,command | awk '$1 ~ /^[0-9]+$/' | wc -l | tr -d ' ')
  hv=$(heavy_pythons 1)
  other=$(heavy_pythons 0)
  say "gate: top_pid_lines=${np} memory_pressure_free=${mp}% vm_free_inactive_spec=${vm}GB swap=[$(sysctl -n vm.swapusage)] swap_free_mb=${sf} own_heavy_python=[${hv}] other_projects_python_ge3GB=[${other}]"
  [ "${np:-0}" -ge 50 ] || return 1          # top が読めない・形が変わった → 通さない（fail-closed）
  [ -n "$mp" ] && [ "$mp" -ge 30 ] || return 1
  [ -n "$vm" ] && awk -v v="$vm" 'BEGIN{exit !(v >= 10)}' || return 1
  [ -n "$sf" ] && awk -v v="$sf" 'BEGIN{exit !(v >= 512)}' || return 1
  [ -z "$hv" ] || return 1
  return 0
}

say "start argv=[$0 $*] code_dir=${C} out_root=${O} env_dir=${E} pid=$$ ppid=${PPID} threads=${THREADS} NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR} PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX} TMPDIR=${TMPDIR}"
for d in pool tier; do
  [ -f "$O/$d/manifest.json" ] && [ -f "$O/$d/jobs.parquet" ] || { say "$O/$d に list の段の出力が無い → 止める"; exit 2; }
done
ok=0
for k in $(seq 1 13); do
  if gate; then ok=1; break; fi
  [ "$k" -lt 13 ] && sleep 300
done
if [ "$ok" != 1 ]; then say "gate_timeout（1 h 待って通らない）→ 起動しない"; exit 3; fi
cd "$C" || exit 4
say "gate ok → pool run 1（shard 1 本、/usr/bin/time -l）"
MIN_FREE_PCT=15 /bin/bash "$MG" "$O/pool/run_first.log" "$PY" scripts/bench/glacier_l60.py run --out-dir "$O/pool" \
  --pkg "$E/pkg" --site "$E/site" --threads "$THREADS" --max-shards 1
rc=$?; say "pool run 1 rc=${rc} swap=$(sysctl -n vm.swapusage)"
[ "$rc" = 0 ] || { say "pool run 1 が失敗 → 止める"; exit "$rc"; }
pk=$(awk '/maximum resident set size/{print $1}' "$O/pool/run_first.log" | tail -1)
fp=$(awk '/peak memory footprint/{print $1}' "$O/pool/run_first.log" | tail -1)
say "pool run 1 peak: maxrss_bytes=${pk:-NA} peak_footprint_bytes=${fp:-NA}"
[ -n "$pk" ] || { say "pool run 1 の peak が読めない → 止める"; exit 9; }
awk -v v="$pk" -v m="$PEAK_MAX" 'BEGIN{exit !(v <= m)}' || { say "pool run 1 の peak が 8.6 GB を超えた → 止める（計画の見積もり）"; exit 9; }
for d in pool tier; do
  say "${d} run（残りの shard 全部）"
  MIN_FREE_PCT=15 /bin/bash "$MG" "$O/$d/run_rest.log" "$PY" scripts/bench/glacier_l60.py run --out-dir "$O/$d" \
    --pkg "$E/pkg" --site "$E/site" --threads "$THREADS"
  rc=$?; say "${d} run rc=${rc} swap=$(sysctl -n vm.swapusage)"
  [ "$rc" = 0 ] || { say "${d} run が失敗（rc 99 は memguard）→ 止める"; exit "$rc"; }
  r=$("$PY" scripts/bench/glacier_check_shards.py --out-dir "$O/$d"); rc=$?
  say "check ${d}: rc=${rc} ${r}"
  [ "$rc" = 0 ] || { say "${d} の予測の本数・shard の点検が通らない → 止める"; exit 7; }
done
n=$(ls "$O"/pool/shards/shard_*.json "$O"/tier/shards/shard_*.json 2>/dev/null | wc -l | tr -d ' ')
say "done rc=0 shard_json=${n}"
exit 0
