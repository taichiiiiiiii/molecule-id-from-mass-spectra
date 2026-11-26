#!/bin/bash
# Issue #268: glacier_tier_launch.sh の終わりを待ち、予測の本数を確かめてから add_glacier_match.py で
# プール＋層の両方の予測から dev_b8_cc_gl2.parquet を作る watcher（detached で起動する）。
#   bash scripts/bench/glacier_tier_watch.sh <code_dir> <launcher_pid>      （code_dir は絶対パス）
# 順番: (1) launcher の pid が消えるまで 60 s おきに待つ（最大 6 h） (2) launch.log に "done rc=0" があること
#   (3) glacier_check_shards.py で層（34 shard・673,214 本）とプール（28 shard・544,886 本）がすべて完了
#   (4) メモリの関門（memory_pressure の空き ≥ 30%、vm_stat の free+inactive+speculative ≥ 10 GB、swap の残り ≥ 512 MB。
#   5 分おき、最大 2 h。当てはめなど他のカードの軽い作業とは並行してよいので、このプロジェクトの python は数えない）
#   (5) memguard（MIN_FREE_PCT=15）の下で add_glacier_match.py (6) report の行数・分子数を確かめる。
# どこかで止まったら watch.log に理由を書いて非 0 で終わる。当てはめ exp013・台帳・PR は次の作業者が行う。
set -uo pipefail
C="$1"; LPID="$2"
case "$C" in /*) ;; *) echo "code_dir は絶対パス" >&2; exit 2;; esac
D=/Users/taichi/work/molecule-id-dev
PY=$D/.venv/bin/python
MG=/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh
OT=$D/outputs/local_eval/glacier_tier
OP=$D/outputs/local_eval/glacier_l60
F=$D/outputs/local_eval/features
LOG=$OT/watch.log
export NUMBA_CACHE_DIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/numba
export PYTHONPYCACHEPREFIX=/Users/taichi/.hermes/profiles/massspec/cache/scratch/pyc_gltier
export TMPDIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/tmp_gltier
mkdir -p "$TMPDIR"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "start argv=[$0 $*] pid=$$ ppid=${PPID} launcher_pid=${LPID}"
k=0
while kill -0 "$LPID" 2>/dev/null; do
  k=$((k + 1))
  if [ "$k" -gt 360 ]; then say "launcher が 6 h で終わらない → 止める"; exit 5; fi
  sleep 60
done
say "launcher の終わりを確認: $(tail -1 "$OT/launch.log")"
grep -q "done rc=0 " "$OT/launch.log" || { say "launch.log に done rc=0 が無い → 止める"; exit 6; }
cd "$C" || exit 4
for o in "$OT" "$OP"; do
  r=$("$PY" scripts/bench/glacier_check_shards.py --out-dir "$o"); rc=$?
  say "check ${o}: rc=${rc} ${r}"
  [ "$rc" = 0 ] || { say "予測の本数・shard の点検が通らない → 止める"; exit 7; }
done

gate() {
  local mp vm sf
  mp=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  vm=$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} END{gsub("\\.","",f);gsub("\\.","",i);gsub("\\.","",s); printf "%.1f", (f+i+s)*ps/1e9}')
  sf=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="free") {v=$(i+2); gsub("M","",v); print v}}')
  say "gate: memory_pressure_free=${mp}% vm_free_inactive_spec=${vm}GB swap=[$(sysctl -n vm.swapusage)]"
  [ -n "$mp" ] && [ "$mp" -ge 30 ] || return 1
  [ -n "$vm" ] && awk -v v="$vm" 'BEGIN{exit !(v >= 10)}' || return 1
  [ -n "$sf" ] && awk -v v="$sf" 'BEGIN{exit !(v >= 512)}' || return 1
  return 0
}
ok=0
for k in $(seq 1 25); do
  if gate; then ok=1; break; fi
  [ "$k" -lt 25 ] && sleep 300
done
[ "$ok" = 1 ] || { say "gate_timeout（2 h 待って通らない）→ 表を作らない"; exit 3; }
say "gate ok → add_glacier_match.py（プール＋層）"
MIN_FREE_PCT=15 /bin/bash "$MG" "$OT/add_match.log" "$PY" scripts/features/add_glacier_match.py \
  --in "$F/dev_b8_cc.parquet" --out "$F/dev_b8_cc_gl2.parquet" --gl-dir "$OP" --gl-dir-tier "$OT" --workers 4
rc=$?; say "add_glacier_match rc=${rc} swap=$(sysctl -n vm.swapusage)"
[ "$rc" = 0 ] || { say "add_glacier_match が失敗（rc 99 は memguard）→ 止める"; exit "$rc"; }
r=$("$PY" -c "import json; r=json.load(open('$F/dev_b8_cc_gl2_report.json')); assert r['issue']==268 and r['n_rows']==430979 and r['n_molecules']==3901 and r['n_nonfinite_has_gl1']==0, r; print(json.dumps({k: r[k] for k in ('output_sha256','n_rows','n_has_gl1','n_has_gl1_tier','n_has_gl0_tier','n_has_gl0_pool','secs')}))"); rc=$?
say "report: rc=${rc} ${r}"
[ "$rc" = 0 ] || { say "report の点検が通らない → 止める"; exit 8; }
say "WATCH_DONE ok（次: 当てはめ exp013・台帳・PR は次の作業者）"
exit 0
