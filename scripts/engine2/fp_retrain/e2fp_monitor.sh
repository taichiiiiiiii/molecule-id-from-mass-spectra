#!/bin/bash
# #152 Engine 2 FP（S2）の MPS 学習のメモリの記録（読むだけ。プロセスには触らない）。
# 使い方: bash e2fp_monitor.sh <run dir> <out.tsv> [間隔秒=10]
# 列（最後に memory_pressure の空き % を足した）: 時刻 / vm_stat の空き GB（free+inactive+speculative）/ 使用 GB（hw.memsize − 空き）/ 主プロセスの pid /
#     主プロセスの footprint MB / そのうち IOAccelerator (graphics) MB（MPS の確保が入る区分）/ プロセスの木の RSS の合計 MB（記録だけ）/
#     swap の使用 MB
# 主プロセス = 引数に「train_s2_resumable.py」と「--run <run dir>」を含む python のうち親が python でないもの。
RUN="$1"; OUT="$2"; DT="${3:-10}"
TOTAL=$(sysctl -n hw.memsize)
is_py() {   # 学習の python 本体か（/usr/bin/time・bash の行は除く。venv のパスに空白があるので先頭で判定）
  local c; c=$(ps -o command= -p "$1" 2>/dev/null)
  case "$c" in /usr/bin/time*|bash*|/bin/bash*|"") return 1;; esac
  case "$c" in *bin/python*train_s2_resumable.py*) return 0;; esac
  return 1
}
[ -f "$OUT" ] || printf 'time\tavail_gb\tused_gb\tpid\tfootprint_mb\tgraphics_mb\ttree_rss_mb\tswap_used_mb\tmp_free_pct\n' > "$OUT"
while :; do
  read -r avail <<< "$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.0f", (f+i+s)*ps}')"
  pid=""; fp=""; gr=""; tree=""
  for p in $(pgrep -f "train_s2_resumable.py.*--run $RUN"); do
    is_py "$p" || continue
    pp=$(ps -o ppid= -p "$p" | tr -d ' ')
    is_py "$pp" || pid=$p          # 親が学習の python なら worker
  done
  if [ -n "$pid" ]; then
    f=$(footprint -f bytes -p "$pid" 2>/dev/null)
    fp=$(echo "$f" | awk '/Footprint:/{for(i=1;i<=NF;i++) if($i=="Footprint:") {printf "%.0f", $(i+1)/1048576}}')
    gr=$(echo "$f" | awk '/IOAccelerator \(graphics\)/{printf "%.0f", $1/1048576}')
    kids=$(pgrep -P "$pid" | tr '\n' ',')
    tree=$(ps -o rss= -p "$pid,${kids%,}" 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s/1024}')
  fi
  sw=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); gsub("M","",v); print v}}')
  mpf=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  printf '%s\t%.2f\t%.2f\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$(echo "$avail/1073741824" | bc -l)" \
    "$(echo "($TOTAL-$avail)/1073741824" | bc -l)" "$pid" "$fp" "$gr" "$tree" "$sw" "$mpf" >> "$OUT"
  sleep "$DT"
done
