# Single-trial EEG decoding of syntactic and semantic violations in Python code

Analysis code for an independent re-analysis and extension of the EEG dataset of
Kuo & Prat (2024, *Scientific Reports* 14:5404), OpenNeuro **ds004771**.

The code reproduces every number, figure and table in the manuscript from the
public dataset. No data are included in this repository.

---

## 1. Setup

```bash
conda env create -f environment.yml
conda activate mne_stable
```

Main versions: Python 3.10.18, MNE-Python 1.10.2, scikit-learn 1.7.2,
PyTorch 2.10.0 (CUDA 12.8), NumPy 1.26.4, SciPy 1.15.3, pandas 2.3.2,
Matplotlib 3.10.6.

## 2. Data and quick start

Download OpenNeuro ds004771 (version 1.0.0) into `data/`, then run everything:

    bash download_data.sh
    bash run_all.sh            # full analysis (many hours; EEGNet needs a GPU)
    QUICK=1 bash run_all.sh    # fast check that every step runs (numbers will not match the paper)

`run_all.sh` runs all steps below in order and logs each one to
`outputs/run_all_logs/`. All scripts locate the repository root from their own
position and write their results to `outputs/`.

## 3. Run order

Run the scripts in this order from the repository root. Each script's header
lists its options.

### 3.1 Preprocessing and trial labels (`src/preprocessing/`)

| Step | Script | Produces |
|---|---|---|
| 1 | `run_pipeline_batch.py` | Runs `data_audit.py` → `triplet_builder.py` → `visual_and_label.py` for every participant; writes `outputs/labeled/` and `outputs/batch/pass_subjects.txt` (54 participants after technical exclusions) |
| 2 | `verify_labels_by_parsing.py` | Checks every stimulus line with Python's parser; identifies the invalid `class` item |
| 3 | `assign_lists_flag_items.py` | Matches each participant to one of the four stimulus lists; writes `outputs/verification/participant_list_assignment.csv` and `item_exclusions.csv` |
| 4 | `artifact_exclusion.py` | Critical-item epochs, artifact rejection, 25% participant rule; writes `outputs/artifact_epoch0/` (final sample: 45) |

### 3.2 Data checks (`src/checks/`)

| Script | Manuscript |
|---|---|
| `check_epoch_timing.py` | Epoch geometry and 899.7 ms onset spacing (Methods 2.3) |
| `check_deposited_reference.py` | Channels and reference in the deposited files (Methods 2.1, 2.4) |

### 3.3 ERP analyses (`src/erp/`)

| Script | Manuscript |
|---|---|
| `erp_epoch0.py` | N400 and P600 effects (Results 3.1, Table 2); run also for the sensitivity sets (Results 3.4, Table 3) |
| `cluster_epoch.py` | Cluster-based permutation tests (Results 3.3 and 3.5; Figs. 5–6) |
| `expertise_epoch0.py` | Experts vs. novices (Results 3.5, Table 4) |
| `topography_0.py` | Scalp distribution, Cz/Pz vs. Fz (Results 3.2, Fig. 4) |
| `data_quality_by_group.py` | Retained trials and rejection rates by group (Results 3.5) |
| `stats_table_epoch0.py` | Collects the ERP statistics for the statistical table |

### 3.4 Single-trial classification (`src/classification/`)

| Script | Manuscript |
|---|---|
| `features_epoch0.py` | The 13 features per trial (Methods 2.7.1) |
| `classify_loso_epoch0.py` | Logistic regression, leave-one-participant-out, 1,000 permutations (Results 3.6, Table 5); `--feature-set no_early` and `--feature-set early_only` for the control-feature analyses |
| `classify_by_list_epoch0.py` | Leave-one-list-out control (Results 3.6, "Item control") |
| `compare_classifiers_epoch0.py` | Five classifiers (Results 3.6, Table 6); `--n-permutations 200 --perm-classifiers "Linear SVM" "Random forest"` |
| `eegnet_simple.py` | EEGNet: `--mode default` for each task; `--mode holdout` and `--mode nested` for the tuning variants (syntactic task) |

### 3.5 Figures (`src/figures/`)

| Script | Figure |
|---|---|
| `fig_trial_structure.py` | Fig. 1 |
| `figures_epoch0.py` | Fig. 2 (waveforms) |
| `fig_erp_boxplots.py` | Fig. 3 (per-participant effects) |

Figures 4–6 are produced by `topography_0.py` and `cluster_epoch.py`.

### 3.6 Final check

```bash
python src/checks/check_manuscript_numbers.py
```

Recomputes 130 numbers reported in the manuscript from the saved outputs and
reports each as `ok`, `FAIL`, or missing.

## 4. Notes on the dataset

- The deposit stores three epochs per trial: the critical item and the two
  following items. All analyses use the **critical-item** epoch (event codes
  110, 115, 130, 220, 225, 240, 245).
- One item uses `class`, a reserved word in Python, as a variable name and is
  invalid in all versions; it is removed from all conditions.
- The deposited data are sampled at 256 Hz.

## 5. Citation

If you use this code, please cite the manuscript (citation to be added on
publication) and the original study and dataset:

- Kuo CH, Prat CS (2024) Computer programmers show distinct, expertise-dependent
  brain responses to violations in form and meaning when reading code. *Sci Rep*
  14:5404.
- Kuo CH, Prat CS (2023) EEG/ERP data from a Python reading task (v1.0.0).
  OpenNeuro. doi:10.18112/openneuro.ds004771.v1.0.0
