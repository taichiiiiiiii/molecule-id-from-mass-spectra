#!/bin/bash
# B8 振り分け器（#216）の採点ジョブ。コードの写し（git archive）の根から、切り離して起動する。
#   使い方: /bin/bash scripts/research/b8_score_job.sh <集合名の一覧ファイル> <ジョブの log>
# 集合ごとに（完了印 outputs/local_eval/<h>/b8_done があれば飛ばす）:
#   1. run.py --stages score,prior,frag（対照）  2. b8_features.py（特徴・L・Q。PubChem 層の採点はここ）
#   3. run.py --stages score,prior,frag,pubchem（参考。キャッシュを使う）  4. 完了印
# 各段は memguard（MIN_FREE_PCT=15、/usr/bin/time -l）。rc 99（memguard）や 0 以外では止まり、次の集合へ進まない。
# 集合を始める前に memory_pressure の空きが 15% 以上になるまで 5 分おきに待つ（最大 12 回、超えたら止まる）。
set -u
LIST="$1"; JLOG="$2"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
PY=/Users/taichi/work/molecule-id-dev/.venv/bin/python
MG="$HERE/scripts/research/memguard_run.sh"
export NUMBA_CACHE_DIR=/Users/taichi/.hermes/profiles/massspec/cache/scratch/b8n/numba
export PYTHONPYCACHEPREFIX=/Users/taichi/.hermes/profiles/massspec/cache/scratch/b8n/pyc
export MIN_FREE_PCT=15
log() { echo "$(date '+%F %T') $*" >> "$JLOG"; }
log "start root=${HERE} commit=$(cat "$HERE/COMMIT_SHA") pid=$$ ppid=$PPID"
log "untracked_sha256 $(shasum -a 256 data/ext/frag/frag_rescore.py data/ext/v4g/casmi26-glacier/gl_fuse.py data/ext/v4g/casmi26-v3-models/code/casmi/chem.py data/ext/pubchem_tier/pc_mass.npy data/ext/pubchem_tier/pc_off.npy | awk '{printf "%s:%s ", $2, substr($1,1,16)}')"
while read -r H; do
  [ -z "$H" ] && continue
  D="$HERE/outputs/local_eval/$H"
  if [ -f "$D/b8_done" ]; then log "skip ${H}（完了印あり）"; continue; fi
  n=0
  while :; do
    f=$(memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
    [ -n "$f" ] && [ "$f" -ge 15 ] && break
    n=$((n+1)); log "wait ${H} free=${f}%"
    if [ "$n" -ge 12 ]; then log "stopped ${H}: memory gate（1 時間待っても空き 15% 未満）"; exit 98; fi
    sleep 300
  done
  mkdir -p "$D/b8_logs"
  log "begin ${H} free=${f}% logits=${D}/pc_logits.npz"
  i=0
  for step in spf feat spfpc; do
    i=$((i+1))
    case $step in
      spf)   cmd=("$PY" scripts/local_eval/run.py --holdout "$H" --stages score,prior,frag --logits-dir "$D" --truth-dir "$D") ;;
      feat)  cmd=("$PY" scripts/research/b8_features.py --holdout "$H" --dir "$D") ;;
      spfpc) cmd=("$PY" scripts/local_eval/run.py --holdout "$H" --stages score,prior,frag,pubchem --logits-dir "$D" --truth-dir "$D") ;;
    esac
    /bin/bash "$MG" "$D/b8_logs/${i}_${step}.log" "${cmd[@]}"
    rc=$?
    peak=$(awk '/maximum resident set size/{print $1}' "$D/b8_logs/${i}_${step}.log")
    real=$(awk '/ real /{print $1}' "$D/b8_logs/${i}_${step}.log")
    log "${H} ${step} rc=${rc} real=${real}s peak_rss=${peak}"
    if [ "$rc" -ne 0 ]; then log "stopped ${H} ${step} rc=${rc}（次の集合へ進まない）"; exit "$rc"; fi
  done
  date '+%F %T' > "$D/b8_done"
  log "done ${H}"
done < "$LIST"
log "all done"
