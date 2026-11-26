#!/bin/bash
# GLACIER の群の台帳の行 exp007・exp008 の当てはめ（Issue #259）。1 本ずつ順に、memguard の下で。
# 評価（evaluate.py）は作業ツリーが clean な状態で別に回す。
#   /bin/bash scripts/ranker/run_glacier_ledger.sh <log_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra/.venv/bin/python"
MG="/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh"
APPROVAL="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/228#issuecomment-7167884263"
TABLE="$ROOT/outputs/local_eval/features/dev_b8_cc_gl.parquet"
EXCL="$ROOT/docs/experiments/msg1_exclude.csv"
EVAL="$ROOT/docs/experiments/msg0_eval.csv"
GL="gl_cos_mean,gl_cos_max,gl_ent_mean,gl_ent_max,has_gl,gl_cos_mean_dmean,gl_cos_mean_z,gl_cos_mean_dmax,gl_cos_mean_softmax,gl_cos_mean_sd,gl_ent_mean_dmean,gl_ent_mean_z,gl_ent_mean_dmax,gl_ent_mean_softmax,gl_ent_mean_sd"
LOGDIR="${1:?log dir}"
mkdir -p "$LOGDIR"
cd "$ROOT"
for e in exp007 exp008; do
  if [ -d "outputs/local_eval/ranker/lambdarank/$e" ]; then echo "skip ${e}（出力あり）"; continue; fi
  EXTRA="cc,cc_z"
  [ "$e" = exp008 ] && EXTRA="cc,cc_z,${GL}"
  echo "start ${e} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  set +e   # rc 99 = memguard の停止
  MIN_FREE_PCT=15 /bin/bash "$MG" "$LOGDIR/${e}.log" "$PY" scripts/ranker/lambdarank_oof.py \
    --table "$TABLE" --exp-id "$e" --approval "$APPROVAL" --rows all \
    --exclude-features prior_unverified_pop --extra-features "$EXTRA" \
    --exclude-train-molecules "$EXCL" --eval-molecules "$EVAL"
  rc=$?
  set -e
  echo "end ${e} rc=${rc} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  [ "$rc" -eq 0 ] && [ -f "outputs/local_eval/ranker/lambdarank/$e/meta.json" ] || { echo "止める: ${e} rc=${rc}"; exit 1; }
done
echo "all done"
