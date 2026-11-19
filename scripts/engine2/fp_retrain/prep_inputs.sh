#!/bin/bash
# #152: Engine 2 の FP モデル（S2）の学習の入力を、公開の準備コード（upstream/、変えない）と競技の train.parquet・
# COCONUT 09-2026 から作る。各段を memguard_run.sh（MIN_FREE_PCT、既定 15）の下で順に動かし、/usr/bin/time -l の
# ログを <WORK>/logs/<段>.log に残す。段が終わると <WORK>/logs/<段>.done を書き、次に動かしたときは飛ばす（再開できる）。
# どれかの段が rc≠0 で終わったら（memguard の 99 を含む）そこで止まり、次の段を始めない。
# 使い方: bash prep_inputs.sh <CODE dir（このファイルと upstream/ の写し）> <WORK dir> <python> <train.parquet> <coconut csv>
set -u
CODE="$1"; WORK="$2"; PY="$3"; TRAIN="$4"; CSV="$5"
GUARD=${MEMGUARD:-$CODE/memguard_run.sh}
export MIN_FREE_PCT=${MIN_FREE_PCT:-15}
export PYTHONPATH="$CODE/upstream" PYTHONHASHSEED=0
mkdir -p "$WORK/logs"
cd "$WORK" || exit 2
say() { echo "$(date '+%F %T') $*" >> "$WORK/logs/prep.log"; }
swap_mb() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2); gsub("M","",v); print v; exit}}'; }
SWAP0=$(swap_mb); [ -z "$SWAP0" ] && { echo "cannot read swap" >&2; exit 4; }
SWAP_MAX_INC_MB=${SWAP_MAX_INC_MB:-1024}
stage() {   # stage <名前> <cmd...>
  local n="$1"; shift
  if [ -f "logs/$n.done" ]; then say "skip $n (done)"; return 0; fi
  say "start $n: $* (memory_pressure free $(memory_pressure | awk -F': ' '/free percentage/{print $2}'), swap $(sysctl -n vm.swapusage))"
  bash "$GUARD" "logs/$n.log" "$@"; local rc=$?
  say "end $n rc=$rc"
  if [ "$rc" -ne 0 ]; then say "STOP at $n rc=$rc -> no further stages (flat が止まったら data/ を消してから再実行)"; exit "$rc"; fi
  local sw; sw=$(swap_mb)
  if awk -v s="$sw" -v b="$SWAP0" -v d="$SWAP_MAX_INC_MB" 'BEGIN{exit !(s >= b + d)}'; then
    say "STOP after $n: swap used ${sw} MB >= ${SWAP0}+${SWAP_MAX_INC_MB} -> no further stages"; exit 6
  fi
  date '+%F %T' > "logs/$n.done"
}
say "prep start pid=$$ code=$CODE train=$TRAIN csv=$CSV swap_base_mb=$SWAP0"
stage flat "$PY" "$CODE/prep_flat_stream.py" --train "$TRAIN" --output data --err-parquet data/train.parquet
mkdir -p db/coconut
[ -e db/coconut/coconut_csv-09-2026.csv ] || ln -s "$CSV" db/coconut/coconut_csv-09-2026.csv
stage build_fps "$PY" "$CODE/upstream/build_fps.py"
stage build_coconut "$PY" "$CODE/upstream/build_coconut.py"
stage build_dataset "$PY" "$CODE/upstream/build_dataset.py"
stage selected "$PY" "$CODE/upstream/prepare_legacy_exports.py" selected --data data --output up_train
stage build_pool "$PY" "$CODE/upstream/build_pool.py"
shasum -a 256 ds/spec.npz ds/split.pkl up_train/fp_bits.npy up_train/fp_sel.npy up_train/fp_meta.pkl \
  pool_fp.npy pool_mass.npy pool_meta.pkl data/struct_fp.npy data/struct_fp_meta.pkl db/coconut_clean.parquet \
  > logs/INPUTS.sha256
say "prep done"
echo PREP_DONE >> logs/prep.log
