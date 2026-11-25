#!/usr/bin/env bash
# =====================================================================
# download_data.sh — download OpenNeuro ds004771 into data/
#
# Usage (from the repository root):
#   bash download_data.sh
#
# Downloads from OpenNeuro's public Amazon S3 mirror (no account needed);
# installs the AWS command-line tool with pip if it is missing.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [ -f data/participants.tsv ]; then
  echo "data/ already contains the dataset — nothing to do."
  exit 0
fi
mkdir -p data

# OpenNeuro mirrors every dataset on Amazon S3 (public, no account needed).
if ! command -v aws >/dev/null 2>&1; then
  echo "Installing the AWS command-line tool (awscli) with pip ..."
  python -m pip install --quiet awscli
fi
echo "Downloading ds004771 from the OpenNeuro S3 mirror ..."
aws s3 sync --no-sign-request s3://openneuro.org/ds004771 data/

# ---- check that the files the pipeline needs are present -------------
missing=0
[ -f data/participants.tsv ] || { echo "MISSING: data/participants.tsv"; missing=1; }
ls data/stimuli/Stimuli_v4_List*.txt >/dev/null 2>&1 || { echo "MISSING: data/stimuli/Stimuli_v4_List1-4.txt"; missing=1; }
n_sets=$(ls data/sub-*/eeg/*_eeg.set 2>/dev/null | wc -l)
echo "Found $n_sets EEG recordings (expected 61)."
[ "$n_sets" -eq 61 ] || missing=1

if [ "$missing" -eq 0 ]; then
  echo "Download complete. Next: bash run_all.sh"
else
  echo "Download incomplete — see messages above."
  exit 1
fi