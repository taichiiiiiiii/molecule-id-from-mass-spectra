#!/bin/bash
# 特徴群を 1 つずつ外した台帳の行 exp004〜exp006 の当てはめ（Issue #257）。1 本ずつ順に、memguard の下で。
# 評価（evaluate.py）は作業ツリーが clean な状態で別に回す（git_commit に dirty の印を付けないため）。
#   /bin/bash scripts/ranker/run_group_ablation.sh <log_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra/.venv/bin/python"
MG="/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh"
APPROVAL="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/228#issuecomment-7167884263"
TABLE="$ROOT/outputs/local_eval/features/dev_b8_cc.parquet"
LOGDIR="${1:?log dir}"
mkdir -p "$LOGDIR"
cd "$ROOT"
for e in exp004 exp005 exp006; do
  if [ -d "outputs/local_eval/ranker/lambdarank/$e" ]; then echo "skip ${e}（出力あり）"; continue; fi
  read -r -a ARGS <<< "$("$PY" scripts/ranker/feature_groups.py --args "$e")"
  echo "start ${e} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  set +e   # memguard は /usr/bin/time -l を付けて走らせる（peak RSS はログ）。rc 99 = memguard の停止
  MIN_FREE_PCT=15 /bin/bash "$MG" "$LOGDIR/${e}.log" "$PY" scripts/ranker/lambdarank_oof.py \
    --table "$TABLE" --exp-id "$e" --approval "$APPROVAL" --rows all "${ARGS[@]}"
  rc=$?
  set -e
  echo "end ${e} rc=${rc} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  [ "$rc" -eq 0 ] && [ -f "outputs/local_eval/ranker/lambdarank/$e/meta.json" ] || { echo "止める: ${e} rc=${rc}"; exit 1; }
done
echo "all done"
