#!/usr/bin/env bash
# Kaggle ノートブックの出力を、ページをたどって正規表現に合うものだけ取得する。
# `kaggle kernels output` は 1 ページ（既定 20 件）ずつしか処理しないため。
#   bash scripts/cv/fetch_kernel_output.sh <owner/kernel> <dest> [regex]
set -euo pipefail
K=$1; D=$2; PAT=${3:-'^([^/]+\.(log|json|csv|parquet)|variants/.*\.csv)$'}
mkdir -p "$D"
tok=""
for i in $(seq 1 200); do
  args=(kernels output "$K" -p "$D" --file-pattern "$PAT" --page-size 200 -q)
  [[ -n "$tok" ]] && args+=(--page-token "$tok")
  out=$(kaggle "${args[@]}" 2>&1 | grep -v Warning || true)
  tok=$(printf '%s\n' "$out" | sed -n 's/.*Next Page Token = //p' | tail -1)
  [[ -z "$tok" ]] && break
done
find "$D" -type f | wc -l
