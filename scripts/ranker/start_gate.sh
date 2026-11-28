#!/bin/bash
# 重いジョブの共通の起動条件（AND、#296 ほか）: memory_pressure の空き ≥ 30%、vm_stat の free+inactive+speculative
# ≥ 10 GB、swap の残り（free）≥ 512 MB、このプロジェクトの重い python（top の MEM ≥ 3 GB）が他に無い。
# 満たすまで 5 分おきに見直し、<最大の秒数>（既定 3600）で満たさなければ exit 3（起動しない）。値が読めなければ通さない。
#   /bin/bash scripts/ranker/start_gate.sh [最大の秒数]
set -uo pipefail
MAXW="${1:-3600}"
say() { echo "$(date '+%F %T') gate: $*"; }
heavy_own() {
  top -l 1 -stats pid,mem,command | awk '$1 ~ /^[0-9]+$/ && tolower($3) ~ /^python/ {m=$2; gsub(/[+-]$/,"",m); u=substr(m,length(m)); v=substr(m,1,length(m)-1)+0;
      g=(u=="G")?v:(u=="M")?v/1024:(u=="K")?v/1048576:v/1073741824; if (g>=3) print $1, g}' |
  while read -r pid g; do
    cmd=$(ps -o command= -p "$pid" 2>/dev/null)
    if printf '%s' "$cmd" | grep -qiE 'molecule-id-dev|Molecule ID From Mass Spectra'; then printf '%s:%.1fGB ' "$pid" "$g"; fi
  done
}
check() {
  local mp vm sf hv
  mp=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  vm=$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.1f", (f+i+s)*ps/1e9}')
  sf=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="free") {v=$(i+2); gsub("M","",v); print v}}')
  hv=$(heavy_own)
  say "memory_pressure_free=${mp}% vm_free_inactive_spec=${vm}GB swap_free_mb=${sf} own_heavy_python=[${hv}]"
  [ -n "$mp" ] && [ "$mp" -ge 30 ] || return 1
  [ -n "$vm" ] && awk -v v="$vm" 'BEGIN{exit !(v >= 10)}' || return 1
  [ -n "$sf" ] && awk -v v="$sf" 'BEGIN{exit !(v >= 512)}' || return 1
  [ -z "$hv" ] || return 1
  return 0
}
t0=$(date +%s)
while true; do
  if check; then say "pass"; exit 0; fi
  now=$(date +%s)
  if [ $((now - t0 + 300)) -gt "$MAXW" ]; then say "満たさない（${MAXW} s）→ 起動しない"; exit 3; fi
  sleep 300
done
