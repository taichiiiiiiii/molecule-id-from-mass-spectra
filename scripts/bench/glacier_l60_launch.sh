#!/bin/bash
# Issue #242: GLACIER L60 の手元 CPU 生成（scripts/bench/glacier_l60.py run）を、関門を通ってから memguard の下で回す。
#   bash scripts/bench/glacier_l60_launch.sh <code_dir> <out_dir>      （どちらも絶対パス）
# code_dir: git archive の写し（data/・outputs/ はクローンへのリンク）。out_dir: クローンの実ディレクトリ。
# 関門（5 分おき、最大 2 h）: memory_pressure の空き ≥ 30%、vm_stat の free+inactive+speculative ≥ 10 GB、
#   swap の使用 ≤ 7.5 GB、このプロジェクトの重い python（phys footprint ≥ 3 GB）が他に無い。通らなければ記録して exit 3。
#   重い python の判定は top の MEM（phys footprint。MLX/MPS の unified memory は RSS に出ないため）。他のプロジェクトの
#   python は止めも待ちもせず、footprint を記録だけする（メモリは memory_pressure・vm_stat・swap の関門と memguard が守る）。
# 1 回目は shard 1 本だけ（/usr/bin/time -l で peak を測る）、2 回目で残り全部。memguard の停止（rc 99）なら止める。
set -uo pipefail
C="$1"; O="$2"
case "$C" in /*) ;; *) echo "code_dir は絶対パス" >&2; exit 2;; esac
case "$O" in /*) ;; *) echo "out_dir は絶対パス" >&2; exit 2;; esac
PY=/Users/taichi/work/molecule-id-dev/.venv/bin/python
MG=/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh
LOG="$O/launch.log"
THREADS=8
export NUMBA_CACHE_DIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/numba
export PYTHONPYCACHEPREFIX=/Users/taichi/.hermes/profiles/massspec/cache/scratch/pyc_gl60
export TMPDIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/tmp_gl60
mkdir -p "$TMPDIR"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# top の MEM（例 22G・1534M・512K、末尾の +/- は無視）→ GB
heavy_pythons() {
  local own="$1"
  top -l 1 -stats pid,mem,command -n 1000 | awk '$1 ~ /^[0-9]+$/ && tolower($3) ~ /^python/ {m=$2; gsub(/[+-]$/,"",m); u=substr(m,length(m)); v=substr(m,1,length(m)-1)+0;
      g=(u=="G")?v:(u=="M")?v/1024:(u=="K")?v/1048576:v/1073741824; if (g>=3) print $1, g}' |
  while read -r pid g; do
    cmd=$(ps -o command= -p "$pid" 2>/dev/null)
    if printf '%s' "$cmd" | grep -qiE 'molecule-id-dev|Molecule ID From Mass Spectra'; then is_own=1; else is_own=0; fi
    if [ "$is_own" = "$own" ]; then printf '%s:%.1fGB:%s ' "$pid" "$g" "$(printf '%s' "$cmd" | cut -c1-90)"; fi
  done
}

gate() {
  local mp vm sw hv other
  mp=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  vm=$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.1f", (f+i+s)*ps/1e9}')
  sw=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); gsub("M","",v); print v}}')
  hv=$(heavy_pythons 1)
  other=$(heavy_pythons 0)
  say "gate: memory_pressure_free=${mp}% vm_free_inactive_spec=${vm}GB swap_used=${sw}MB own_heavy_python=[${hv}] other_projects_python_ge3GB=[${other}]"
  [ -n "$mp" ] && [ "$mp" -ge 30 ] || return 1
  [ -n "$vm" ] && awk -v v="$vm" 'BEGIN{exit !(v >= 10)}' || return 1
  [ -n "$sw" ] && awk -v v="$sw" 'BEGIN{exit !(v <= 7680)}' || return 1
  [ -z "$hv" ] || return 1
  return 0
}

say "start argv=[$0 $*] code_dir=${C} out_dir=${O} pid=$$ ppid=${PPID} threads=${THREADS} NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR} PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX} TMPDIR=${TMPDIR}"
ok=0
for k in $(seq 1 25); do
  if gate; then ok=1; break; fi
  [ "$k" -lt 25 ] && sleep 300
done
if [ "$ok" != 1 ]; then say "gate_timeout（2 h 待って通らない）→ 起動しない"; exit 3; fi
cd "$C" || exit 4
say "gate ok → run 1（shard 1 本、/usr/bin/time -l）"
MIN_FREE_PCT=15 /bin/bash "$MG" "$O/run_first.log" "$PY" scripts/bench/glacier_l60.py run --out-dir "$O" \
  --pkg "$O/env/pkg" --site "$O/env/site" --threads "$THREADS" --max-shards 1
rc=$?; say "run 1 rc=${rc} swap=$(sysctl -n vm.swapusage)"
[ "$rc" = 0 ] || { say "run 1 が失敗 → 止める"; exit "$rc"; }
say "run 2（残りの shard 全部）"
MIN_FREE_PCT=15 /bin/bash "$MG" "$O/run_rest.log" "$PY" scripts/bench/glacier_l60.py run --out-dir "$O" \
  --pkg "$O/env/pkg" --site "$O/env/site" --threads "$THREADS"
rc=$?; say "run 2 rc=${rc} swap=$(sysctl -n vm.swapusage)"
n=$(ls "$O"/shards/shard_*.json 2>/dev/null | wc -l | tr -d ' ')
say "done rc=${rc} shard_json=${n}"
exit "$rc"
