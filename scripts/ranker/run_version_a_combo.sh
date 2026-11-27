#!/bin/bash
# 版 A の候補の組（N1+GLACIER）の 3 seed の当てはめ exp014〜exp018（Issue #282、#272 の改定）。1 本ずつ順に、memguard の下で。
# 対照 seed 0 は既存の exp007。評価（evaluate.py）は作業ツリーが clean な状態で別に回す。
#   /bin/bash scripts/ranker/run_version_a_combo.sh <log_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra/.venv/bin/python"
MG="/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh"
APPROVAL="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/228#issuecomment-7167884263"
T_CTRL="$ROOT/outputs/local_eval/features/dev_b8_cc_gl.parquet"
T_CAND="$ROOT/outputs/local_eval/features/dev_b8_cc_gl2_n1.parquet"
EXCL="$ROOT/docs/experiments/msg1_exclude.csv"
EVAL="$ROOT/docs/experiments/msg0_eval.csv"
GL="gl_cos_mean,gl_cos_max,gl_ent_mean,gl_ent_max,has_gl,gl_cos_mean_dmean,gl_cos_mean_z,gl_cos_mean_dmax,gl_cos_mean_softmax,gl_cos_mean_sd,gl_ent_mean_dmean,gl_ent_mean_z,gl_ent_mean_dmax,gl_ent_mean_softmax,gl_ent_mean_sd"
N1="nbr_ms_mean,nbr_tani_mean"
LOGDIR="${1:?log dir}"
mkdir -p "$LOGDIR"
cd "$ROOT"
# exp_id seed 表 追加の列
RUNS=(
  "exp014 1 ctrl"
  "exp015 2 ctrl"
  "exp016 0 cand"
  "exp017 1 cand"
  "exp018 2 cand"
)
for r in "${RUNS[@]}"; do
  read -r e seed kind <<<"$r"
  if [ -f "outputs/local_eval/ranker/lambdarank/$e/meta.json" ]; then echo "skip ${e}（meta.json あり）"; continue; fi
  if [ -d "outputs/local_eval/ranker/lambdarank/$e" ]; then echo "止める: ${e} の dir があるが meta.json が無い"; exit 1; fi
  if [ "$kind" = ctrl ]; then TABLE="$T_CTRL"; EXTRA="cc,cc_z"; else TABLE="$T_CAND"; EXTRA="cc,cc_z,${GL},${N1}"; fi
  echo "start ${e} seed=${seed} kind=${kind} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  set +e   # rc 99 = memguard の停止
  MIN_FREE_PCT=15 /bin/bash "$MG" "$LOGDIR/${e}.log" "$PY" scripts/ranker/lambdarank_oof.py \
    --table "$TABLE" --exp-id "$e" --seed "$seed" --approval "$APPROVAL" --rows all \
    --exclude-features prior_unverified_pop --extra-features "$EXTRA" \
    --exclude-train-molecules "$EXCL" --eval-molecules "$EVAL"
  rc=$?
  set -e
  echo "end ${e} rc=${rc} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  [ "$rc" -eq 0 ] && [ -f "outputs/local_eval/ranker/lambdarank/$e/meta.json" ] || { echo "止める: ${e} rc=${rc}"; exit 1; }
done
echo "all done"
