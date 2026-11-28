#!/bin/bash
# H3（断片の構造の統計 pfs_*、Issue #296）の 3 seed の当てはめ exp022〜exp024。版 A の候補の組（exp016〜018 と同じ
# 81 列）に H3 の 4 列を足した 85 列。1 本ずつ順に、memguard の下で。各本の前に共通の起動条件（AND）を確かめ、
# 満たさなければ 5 分おきに見直して 1 h で止める。評価（evaluate.py）は作業ツリーが clean な状態で別に回す。
#   /bin/bash scripts/ranker/run_h3.sh <log_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra/.venv/bin/python"
MG="/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh"
GATE="$ROOT/scripts/ranker/start_gate.sh"
APPROVAL="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/228#issuecomment-7167884263"
T_H3="$ROOT/outputs/local_eval/features/dev_b8_cc_gl2_n1_h3.parquet"
EXCL="$ROOT/docs/experiments/msg1_exclude.csv"
EVAL="$ROOT/docs/experiments/msg0_eval.csv"
GL="gl_cos_mean,gl_cos_max,gl_ent_mean,gl_ent_max,has_gl,gl_cos_mean_dmean,gl_cos_mean_z,gl_cos_mean_dmax,gl_cos_mean_softmax,gl_cos_mean_sd,gl_ent_mean_dmean,gl_ent_mean_z,gl_ent_mean_dmax,gl_ent_mean_softmax,gl_ent_mean_sd"
N1="nbr_ms_mean,nbr_tani_mean"
H3="pfs_ll,pfs_ll_per_peak,pfs_seen_frac,pfs_ll_fdmax"
EXCLUDE="prior_unverified_pop"
EXTRA="cc,cc_z,${GL},${N1},${H3}"
LOGDIR="${1:?log dir}"
mkdir -p "$LOGDIR"
cd "$ROOT"
# exp_id seed
RUNS=(
  "exp022 0"
  "exp023 1"
  "exp024 2"
)
for r in "${RUNS[@]}"; do
  read -r e seed <<<"$r"
  if [ -f "outputs/local_eval/ranker/lambdarank/$e/meta.json" ]; then echo "skip ${e}（meta.json あり）"; continue; fi
  if [ -d "outputs/local_eval/ranker/lambdarank/$e" ]; then echo "止める: ${e} の dir があるが meta.json が無い"; exit 1; fi
  /bin/bash "$GATE" 3600 || { echo "止める: ${e} の起動条件を 1 h 満たさない $(date '+%F %T')"; exit 1; }
  echo "start ${e} seed=${seed} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  set +e   # rc 99 = memguard の停止
  MIN_FREE_PCT=15 /bin/bash "$MG" "$LOGDIR/${e}.log" "$PY" scripts/ranker/lambdarank_oof.py \
    --table "$T_H3" --exp-id "$e" --seed "$seed" --approval "$APPROVAL" --rows all \
    --exclude-features "$EXCLUDE" --extra-features "$EXTRA" \
    --exclude-train-molecules "$EXCL" --eval-molecules "$EVAL"
  rc=$?
  set -e
  echo "end ${e} rc=${rc} $(date '+%F %T') swap: $(sysctl -n vm.swapusage)"
  [ "$rc" -eq 0 ] && [ -f "outputs/local_eval/ranker/lambdarank/$e/meta.json" ] || { echo "止める: ${e} rc=${rc}"; exit 1; }
done
echo "all done"
