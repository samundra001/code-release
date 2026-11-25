#!/usr/bin/env bash
# =====================================================================
# run_all.sh — reproduce every analysis, table and figure in the paper
#
# Usage (from the repository root, after `conda activate mne_stable`
# and after downloading OpenNeuro ds004771 into data/):
#
#   bash run_all.sh            # full run (as in the paper; many hours)
#   QUICK=1 bash run_all.sh    # smoke test: few permutations, short
#                              # EEGNet training; checks that every step
#                              # runs, NOT the reported numbers
#
# Optional: NJOBS=8 bash run_all.sh   (CPU cores for permutations)
# EEGNet steps need a CUDA GPU for reasonable run time.
# Each step's output is logged to outputs/run_all_logs/.
# =====================================================================
set -euo pipefail

cd "$(dirname "$0")"
QUICK="${QUICK:-0}"
NJOBS="${NJOBS:-8}"
SET="step"                      # primary participant set (n = 45)
LOG_DIR="outputs/run_all_logs"
mkdir -p "$LOG_DIR"

if [ ! -f data/participants.tsv ]; then
  echo "ERROR: data/participants.tsv not found. Run: bash download_data.sh"
  exit 1
fi

if [ "$QUICK" = "1" ]; then
  N_PERM_CLASS=10;  N_PERM_CLUSTER=50;  N_PERM_COMPARE=0
  EEG_EPOCHS=2;     EEG_CONFIGS=2
  echo ">>> QUICK mode: reduced permutations/epochs; numbers will NOT match the paper."
else
  N_PERM_CLASS=1000; N_PERM_CLUSTER=2000; N_PERM_COMPARE=200
  EEG_EPOCHS=150;    EEG_CONFIGS=8
fi

step=0
run () {   # run <description> <command...>
  step=$((step + 1))
  local name; name=$(printf "%02d" "$step")
  echo "[$(date +%H:%M:%S)] Step $name: $1"
  shift
  if ! "$@" > "$LOG_DIR/step_${name}.log" 2>&1; then
    echo "  FAILED — see $LOG_DIR/step_${name}.log"
    exit 1
  fi
}

# ---------------------------------------------------------------------
# 1. Preprocessing and trial labels
# ---------------------------------------------------------------------
run "Audit, epoch triplets and labels (all participants)" python src/preprocessing/run_pipeline_batch.py
n_pass=$(wc -l < outputs/batch/pass_subjects.txt 2>/dev/null || echo 0)
echo "  -> $n_pass participants passed technical checks (expected 54)"
if [ "$n_pass" -eq 0 ]; then echo "  FAILED — no participants passed; see $LOG_DIR/step_01.log"; exit 1; fi
run "Check every stimulus line with Python's parser"     python src/preprocessing/verify_labels_by_parsing.py
run "Assign stimulus lists, flag the invalid item"       python src/preprocessing/assign_lists_flag_items.py
run "Artifact rejection and participant exclusion"       python src/preprocessing/artifact_exclusion.py

# ---------------------------------------------------------------------
# 2. Data checks
# ---------------------------------------------------------------------
run "Epoch timing (Methods)"          python src/checks/check_epoch_timing.py
run "Deposited reference (Methods)"   python src/checks/check_deposited_reference.py

# ---------------------------------------------------------------------
# 3. ERP analyses
# ---------------------------------------------------------------------
run "ERP effects, primary set (Table 2)" python src/erp/erp_epoch0.py --set "$SET"
for mask in outputs/artifact_epoch0/trial_mask_*.csv; do
  s=$(basename "$mask" .csv); s=${s#trial_mask_}
  [ "$s" = "$SET" ] && continue
  run "ERP effects, sensitivity set '$s' (Table 3)" python src/erp/erp_epoch0.py --set "$s"
done
run "Fig. 2 (waveforms; also saves per-participant ERPs used below)" python src/figures/figures_epoch0.py --set "$SET"
run "Cluster-based permutation tests (Figs. 5-6)" python src/erp/cluster_epoch.py --set "$SET" --n-perm "$N_PERM_CLUSTER"
run "Experts vs. novices (Table 4)"               python src/erp/expertise_epoch0.py --set "$SET"
run "Scalp topography (Fig. 4)"                   python src/erp/topography_0.py --set "$SET"
run "Data quality by group"                       python src/erp/data_quality_by_group.py --set "$SET"
run "ERP statistics for the statistical table"    python src/erp/stats_table_epoch0.py

# ---------------------------------------------------------------------
# 4. Single-trial classification
# ---------------------------------------------------------------------
run "Single-trial features"  python src/classification/features_epoch0.py --set "$SET"
for fs in all no_early early_only; do
  run "Logistic regression LOSO, features: $fs (Table 5)" \
    python src/classification/classify_loso_epoch0.py --set "$SET" --task all \
      --feature-set "$fs" --n-permutations "$N_PERM_CLASS" --n-jobs "$NJOBS"
done
run "Leave-one-list-out control" \
  python src/classification/classify_by_list_epoch0.py --set "$SET" \
    --n-permutations "$N_PERM_CLASS" --n-jobs "$NJOBS"
run "Classifier comparison (Table 6)" \
  python src/classification/compare_classifiers_epoch0.py --set "$SET" --task all \
    --n-permutations "$N_PERM_COMPARE" --perm-classifiers "Linear SVM" "Random forest" \
    --n-jobs "$NJOBS"
for t in syn sem 3class; do
  run "EEGNet, published settings, task $t" \
    python src/classification/eegnet_simple.py --task "$t" --mode default --max-epochs "$EEG_EPOCHS"
  run "EEGNet, held-out tuning, task $t" \
    python src/classification/eegnet_simple.py --task "$t" --mode holdout \
      --n-configs "$EEG_CONFIGS" --max-epochs "$EEG_EPOCHS"
done
run "EEGNet, nested tuning, syntactic task" \
  python src/classification/eegnet_simple.py --task syn --mode nested \
    --n-configs "$EEG_CONFIGS" --max-epochs "$EEG_EPOCHS"

# ---------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------
run "Fig. 1 (trial structure)"           python src/figures/fig_trial_structure.py
run "Fig. 3 (per-participant effects)"   python src/figures/fig_erp_boxplots.py --set "$SET"

# ---------------------------------------------------------------------
# 6. Check the numbers reported in the manuscript
# ---------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] Final check: numbers reported in the manuscript"
python src/checks/check_manuscript_numbers.py | tee "$LOG_DIR/check_manuscript_numbers.log"

echo "Done. Outputs in outputs/, logs in $LOG_DIR/."