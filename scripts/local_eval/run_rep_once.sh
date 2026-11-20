#!/bin/bash
# Issue #171 §4: rep 段の本番の採点を 1 回だけ回す起動スクリプト（カード 2c で 1 回だけ起動する。2b では回さない）。
#
#   /bin/bash scripts/local_eval/run_rep_once.sh
#
# 5 つの集合を順に回す:
#   v3_c1・v3_c2・enveda_v1・v3_c2n（採否用）: run.py --stages score,prior,rep,frag（submission と窓全体の rep の点の npz を
#     同じ 1 本のプロセスで書く）
#   e180_c2k（§5 の Δ_dom 用だけ。採否に使わない）: rep.py window（npz だけ。submission は作らない）
# 集合ごとに outputs/local_eval/<h>/rep_once.done（開始・終了の時刻、commit sha、コマンド、ピーク RSS）を書く。
# .done のある集合は飛ばす（回し直さない）。途中で切れたら、.done の無い集合だけが同じ commit で回し直される。
#
# メモリ（補足 8 と補足 2-2、レビュー記録の決めごと 1・2）:
#   - 集合ごとの python 1 本を MIN_FREE_PCT=15 /bin/bash memguard_run.sh <集合ごとのログ> .venv/bin/python ... で包む
#     （間に uv run・caffeinate・env・bash -c を挟まない）
#   - 終了コードが 99（memguard が止めた）か 0 以外なら、その集合の .done を書かずにスクリプト全体を止める
#   - 各集合を始める前に memory_pressure の空き ≥ 35% かつ swap 空き ≥ 500 MB（M1 の本番が動いている間）、
#     M1 の終了後は memory_pressure ≥ 25%（swap 空き ≥ 500 MB はそのまま）。満たさなければ 5 分おきに最大 1 h 待ち、
#     だめなら終了コード 3（「M1 の終了待ち」。2c のカードを blocked で終える）
#   - M1 が動いているか: logs/.heavy_job.d/owner の pid が生きているか
set -u
cd "$(dirname "$0")/../.." || exit 2
REPO=$(pwd)
PY=.venv/bin/python
MEMGUARD=${MEMGUARD:-/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh}
SETS=(v3_c1 v3_c2 enveda_v1 v3_c2n e180_c2k)
WAIT_STEP=300
WAIT_MAX=3600

[ -x "$PY" ] || { echo "python が無い: $REPO/$PY"; exit 2; }
[ -f "$MEMGUARD" ] || { echo "memguard_run.sh が無い: $MEMGUARD"; exit 2; }
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "追跡しているファイルに未コミットの変更がある（commit sha がコードを表さない）"; exit 2
fi
SHA=$(git rev-parse HEAD)
mkdir -p logs

m1_running() {
  local o=logs/.heavy_job.d/owner pid
  [ -f "$o" ] || return 1
  pid=$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$o" | head -1)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

free_pct() { memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'; }
swap_free_mb() { sysctl -n vm.swapusage | sed -n 's/.*free = \([0-9.]*\)M.*/\1/p' | cut -d. -f1; }

wait_memory() {
  local waited=0 f s need
  while :; do
    if m1_running; then need=35; else need=25; fi
    f=$(free_pct); s=$(swap_free_mb)
    echo "$(date '+%F %T') gate: free ${f}% (need ${need}%), swap free ${s} MB (need 500), M1 $(m1_running && echo running || echo done)"
    if [ -n "$f" ] && [ -n "$s" ] && [ "$f" -ge "$need" ] && [ "$s" -ge 500 ]; then return 0; fi
    if [ "$waited" -ge "$WAIT_MAX" ]; then return 1; fi
    sleep "$WAIT_STEP"; waited=$((waited + WAIT_STEP))
  done
}

for h in "${SETS[@]}"; do
  out=outputs/local_eval/$h
  done_f=$out/rep_once.done
  if [ -e "$done_f" ]; then echo "$h: rep_once.done がある → 飛ばす"; continue; fi
  [ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "途中で HEAD が変わった"; exit 2; }
  if ! wait_memory; then echo "M1 の終了待ち: 1 h 待っても空きの条件を満たさない（$h は未実行）"; exit 3; fi
  log=logs/rep_once_$h.log
  if [ "$h" = e180_c2k ]; then
    cmd=("$PY" scripts/local_eval/rep.py window --holdout "$h")
    outputs=("$out/rep_scores.npz" "$out/rep_scores.json")
  else
    cmd=("$PY" scripts/local_eval/run.py --holdout "$h" --stages score,prior,rep,frag)
    outputs=("$out/submission_score+prior+rep+frag.csv" "$out/result_score+prior+rep+frag.json" "$out/rep_scores.npz")
  fi
  mkdir -p "$out"
  for p in "${outputs[@]}"; do
    if [ -e "$p" ]; then mv "$p" "$p.stale_$(date +%Y%m%d%H%M%S)"; fi   # .done の無い前の回の出力は残さない（記録用に名前を変える）
  done
  start=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  printf '%s\n' "${cmd[*]}" > "logs/rep_once_$h.cmd"
  echo "$start $h: MIN_FREE_PCT=15 /bin/bash $MEMGUARD $log ${cmd[*]}"
  MIN_FREE_PCT=15 /bin/bash "$MEMGUARD" "$log" "${cmd[@]}"
  rc=$?
  end=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  [ -f "$log" ] || { echo "$h: ジョブのログが無い（$log）"; exit 2; }
  if [ "$rc" -ne 0 ]; then
    echo "$end $h: rc=$rc（99 = memguard が止めた）→ .done を書かずに止める。ログ: $log"; exit "$rc"
  fi
  for p in "${outputs[@]}"; do
    [ -s "$p" ] || { echo "$h: 出力が無い $p → .done を書かずに止める"; exit 2; }
  done
  rss=$(awk '/maximum resident set size/{print $1}' "$log" | tail -1)
  "$PY" - "$done_f" "$h" "$start" "$end" "$SHA" "${cmd[*]}" "$log" "${rss:-}" <<'PYEOF'
import json, os, sys
p, h, start, end, sha, cmd, log, rss = sys.argv[1:9]
d = dict(issue=171, holdout=h, start=start, end=end, commit=sha, command=cmd, log=log,
         max_rss_bytes=int(rss) if rss else None)
tmp = p + ".tmp"
with open(tmp, "w") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
os.replace(tmp, p)
print(json.dumps(d, ensure_ascii=False))
PYEOF
done
echo "全 5 集合の rep_once.done が揃った（commit ${SHA}）"
