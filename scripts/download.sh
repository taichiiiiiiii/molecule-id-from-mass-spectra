#!/usr/bin/env bash
# Download competition data for Envela CASMI 2026 - Molecule ID From Mass Spectra.
#
# The Kaggle CLI downloads data only for competitions in the user's wallet
# (subscribed). If this is the first time running, subscribe via the Kaggle
# website (https://www.kaggle.com/competitions/envela-CASMI26-molecule-id-mass-spectra)
# and then run this script again.
set -euo pipefail

cd "$(dirname "$0")/.."

SLUG="envela-CASMI26-molecule-id-mass-spectra"
OUT="data"

echo "==> Downloading competition data (slug: $SLUG)..."
kaggle competitions download -c "$SLUG" -p "$OUT" -o

# The Kaggle CLI also stores a small metadata JSON next to the files.
# Show what was downloaded.
echo "==> Downloaded files:"
ls -1 "$OUT"
