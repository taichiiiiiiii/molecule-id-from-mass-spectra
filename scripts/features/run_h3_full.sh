#!/bin/bash
# H3（#296）の全 train の数え上げと dev 20 集合の 4 列を、共通の起動条件（start_gate.sh、5 分おき・最大 1 h）を
# 通ってから memguard（MIN_FREE_PCT=15）の下で 1 回だけ回す。detached で起動する（PPID 1）。
#   /bin/bash scripts/features/run_h3_full.sh <log_dir>
# 出力: outputs/local_eval/features/pfs_counts_full.npz・parts_pfs_dev/<set>.parquet(+json)。再開: 数の表の npz と
# 出力のある集合は飛ばす（add_fragment_stats.py の動き）。値（MRR・ラベル）は作らない。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra/.venv/bin/python"
MG="/Users/taichi/.hermes/profiles/massspec/skills/kaggle-improvement-loop/scripts/memguard_run.sh"
LOGDIR="${1:?log dir}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run_h3_full.log"
export NUMBA_CACHE_DIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/numba
export PYTHONPYCACHEPREFIX=/Users/taichi/.hermes/profiles/massspec/cache/scratch/pyc
cd "$ROOT"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
say "start pid=$$ ppid=${PPID} head=$(git rev-parse HEAD) branch=$(git branch --show-current)"
if ! /bin/bash scripts/ranker/start_gate.sh 3600 >> "$LOG" 2>&1; then say "stopped: 起動条件を 1 h 満たさない（起動しない）"; exit 3; fi
say "launch"
MIN_FREE_PCT=15 /bin/bash "$MG" "$LOGDIR/h3_full_job.log" "$PY" scripts/features/add_fragment_stats.py \
  --all-sets --subset 0 --chunk 10000 \
  --table outputs/local_eval/features/dev_b8_cc_gl2_n1.parquet \
  --counts-cache outputs/local_eval/features/pfs_counts_full.npz \
  --out-dir outputs/local_eval/features/parts_pfs_dev
rc=$?
n=$(ls outputs/local_eval/features/parts_pfs_dev/*.parquet 2>/dev/null | grep -vc tmp)
say "end rc=${rc} parts=${n}"
if [ "$rc" -eq 0 ] && [ "$n" -eq 20 ]; then say "done"; else say "stopped rc=${rc}"; fi
