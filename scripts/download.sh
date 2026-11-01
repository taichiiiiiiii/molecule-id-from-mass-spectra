#!/usr/bin/env bash
# Envela CASMI 2026 - Molecule ID From Mass Spectra
# Data acquisition (kaggle CLI wallet 対応版)
#
# Kaggle CLIはウォレット内(参加登録済み)のコンペのデータしか取得できない。
# 参加登録はウェブサイトで行う(NOT CLI join)。
#
# Usage:
#   bash scripts/download.sh
#
# Notes:
#   - ウォレット未登録: kaggle competitions list | grep -i "envela" で確認
#   - 再ログイン: kaggle auth login --force
set -euo pipefail

cd "$(dirname "$0")/.."

SLUG="envela-CASMI26-molecule-id-mass-spectra"
OUT="data"

echo "==> [step 1/2] Kaggle CLIウォレット確認 (participant check)..."
if kaggle competitions list --page-size 200 2>/dev/null | grep -qi "envela\|casmi"; then
  echo "==> Competition found in wallet. OK."
else
  echo "==> NOT in wallet. Please complete registration on Kaggle website first."
  echo "==> Commands:"
  echo "==>   kaggle auth login --force   (re-login)"
  echo "==>   bash scripts/download.sh    (retry)"
  exit 1
fi

echo "==> [step 2/2] Downloading data..."
kaggle competitions download -c "$SLUG" -p "$OUT" -o

echo "==> Downloaded files:"
ls -1 "$OUT" 2>/dev/null || echo "(file list unavailable)"
echo "==> Done."
