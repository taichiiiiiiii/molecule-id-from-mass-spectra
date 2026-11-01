#!/usr/bin/env bash
# Enveda CASMI 2026 - Molecule ID From Mass Spectra
# データ取得: data/ に sample_submission.csv, test.parquet, train.parquet（約 3 GB）を置く。
#
# 参加確認は `kaggle competitions files` で行う（`competitions list` は一部しか出さないので使わない）。
# 参加済みなら `kaggle competitions list -s casmi` で userHasEntered=True になる。
#
# Usage:
#   bash scripts/download.sh            # 全ファイル
#   bash scripts/download.sh --small    # train.parquet 以外
set -euo pipefail

cd "$(dirname "$0")/.."

SLUG="enveda-CASMI26-molecule-id-mass-spectra"
OUT="data"
FILES=(sample_submission.csv test.parquet train.parquet)
[[ "${1:-}" == "--small" ]] && FILES=(sample_submission.csv test.parquet)

if ! kaggle competitions files -c "$SLUG" >/dev/null 2>&1; then
  echo "==> $SLUG のファイル一覧を取れません。Kaggle サイトで Join し、ルールに同意したか確認してください。" >&2
  exit 1
fi

mkdir -p "$OUT"
for f in "${FILES[@]}"; do
  if [[ -s "$OUT/$f" ]]; then
    echo "==> $f は取得済み"
    continue
  fi
  echo "==> $f を取得中"
  kaggle competitions download -c "$SLUG" -f "$f" -p "$OUT" -q
  # 大きいファイルは zip で届くことがある
  if [[ -f "$OUT/$f.zip" ]]; then
    unzip -o -q "$OUT/$f.zip" -d "$OUT" && rm "$OUT/$f.zip"
  fi
done
ls -la "$OUT"
