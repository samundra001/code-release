

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


# ==============================================================================
# BLOCK 1 — CONSTANTS AND STYLE
# ------------------------------------------------------------------------------
# All condition names, label mappings, and plot colors are defined once here
# and used everywhere. If the experiment ever adds a fifth condition, you
# change it in one place.
#
# Colors follow a consistent scheme across all seven figures so your eye
# immediately knows which color means which condition without reading legends.
# ==============================================================================

CONDITION_ORDER = [
    "well_formed",
    "semantically_implausible",
    "syntactically_invalid",
    "doubly_anomalous",
]

CONDITION_COLORS = {
    "well_formed":              "#2ecc71",   # green
    "semantically_implausible": "#f39c12",   # orange
    "syntactically_invalid":    "#3498db",   # blue
    "doubly_anomalous":         "#e74c3c",   # red
    "unknown":                  "#95a5a6",   # grey
}

MANIPULATION_COLORS = {
    "variable": "#2980b9",
    "keyword":  "#8e44ad",
    "unknown":  "#7f8c8d",
}

CONDITION_TO_BINARY = {
    "well_formed": 0,
    "semantically_implausible": 1,
    "syntactically_invalid": 1,
    "structurally_invalid": 1,    # alias — same thing
    "doubly_anomalous": 1,
}

CONDITION_TO_MULTICLASS = {
    "well_formed": 0,
    "semantically_implausible": 1,
    "syntactically_invalid": 2,
    "structurally_invalid": 2,    # alias — same thing
    "doubly_anomalous": 3,
}

MULTICLASS_NAMES = {
    0: "well_formed",
    1: "semantically_implausible",
    2: "syntactically_invalid",
    3: "doubly_anomalous",
}


# ==============================================================================
# BLOCK 2 — UTILITIES
# ==============================================================================

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [SAVED] {path.name}  ({len(df)} rows)")


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  [SAVED] {path.name}")


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {path.name}")


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ==============================================================================
# BLOCK 3 — LOAD VERIFIED CODEBOOK
# ------------------------------------------------------------------------------
# Loads configs/verified_codebook.json and normalizes it to always return
# a dict with condition, manipulation, binary_label, and multiclass_label
# regardless of whether the JSON uses the simple string format or the full
# nested dict format.
#
# This is the only place in the pipeline where the verified codebook is read
# and applied as authoritative labels. Every label in the final output table
# is traceable to this file.
# ==============================================================================

def load_codebook(path: Path) -> Dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Verified codebook not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    codebook: Dict[int, dict] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            cond = v
            codebook[int(k)] = {
                "condition":        cond,
                "manipulation":     "unknown",
                "binary_label":     CONDITION_TO_BINARY.get(cond, -1),
                "multiclass_label": CONDITION_TO_MULTICLASS.get(cond, -1),
            }
        elif isinstance(v, dict):
            cond = v.get("condition", "unknown")
            codebook[int(k)] = {
                "condition":        cond,
                "manipulation":     v.get("manipulation", "unknown"),
                "binary_label":     v.get("binary_label",
                                    CONDITION_TO_BINARY.get(cond, -1)),
                "multiclass_label": v.get("multiclass_label",
                                    CONDITION_TO_MULTICLASS.get(cond, -1)),
            }

    return codebook


# ==============================================================================
# BLOCK 4 — LOAD TRIAL FLOW TABLE AND APPLY LABELS
# ------------------------------------------------------------------------------
# Reads trial_flow_table.csv produced by Script 2 and joins the codebook
# onto every row by C_code. The result is the labeled trial table.
#
# Trials whose C_code is not in the codebook get condition="unknown" and
# label=-1. These are flagged in the sanity report and must be resolved
# before EEGNet training.
#
# The epoch2_id column is preserved exactly as Script 2 produced it — it
# is the integer index into the MNE epochs object for EEG signal extraction.
# ==============================================================================

def load_and_label(
    trial_flow_path: Path,
    codebook: Dict[int, dict],
) -> pd.DataFrame:

    df = pd.read_csv(trial_flow_path)

    conditions        = []
    manipulations     = []
    binary_labels     = []
    multiclass_labels = []

    for c_code in df["C_code"]:
        if pd.isna(c_code):
            conditions.append("unknown")
            manipulations.append("unknown")
            binary_labels.append(-1)
            multiclass_labels.append(-1)
            continue

        entry = codebook.get(int(c_code))
        if entry is None:
            conditions.append("unknown")
            manipulations.append("unknown")
            binary_labels.append(-1)
            multiclass_labels.append(-1)
        else:
            conditions.append(entry["condition"])
            manipulations.append(entry["manipulation"])
            binary_labels.append(int(entry["binary_label"]))
            multiclass_labels.append(int(entry["multiclass_label"]))

    df["condition"]        = conditions
    df["manipulation"]     = manipulations
    df["binary_label"]     = binary_labels
    df["multiclass_label"] = multiclass_labels

    return df


# ==============================================================================
# BLOCK 5 — BUILD FINAL LABELED TRIAL TABLE
# ------------------------------------------------------------------------------
# Selects and orders the columns that EEGNet and downstream analysis need.
# The column order matters — the most important columns (subject, trial_id,
# epoch2_id, labels) come first so the file is readable without scrolling.
#
# epoch2_id is the single most important column in this table. It is the
# integer you pass to epochs[epoch2_id] to extract the EEG signal tensor
# for that trial. Everything else in the pipeline existed to correctly
# identify this index for every trial.
# ==============================================================================

def build_labeled_table(
    df: pd.DataFrame,
    subject: str,
    sfreq: float,
    n_channels: int,
    n_times: int,
    tmin: float,
    tmax: float,
) -> pd.DataFrame:

    df = df.copy()
    df["subject_id"] = subject
    df["sfreq"]      = sfreq
    df["n_channels"] = n_channels
    df["n_times"]    = n_times
    df["tmin"]       = tmin
    df["tmax"]       = tmax

    # Ordered columns — EEGNet needs the first block; rest is for analysis
    primary_cols = [
        "subject_id",
        "trial_id",
        "epoch2_id",           # <- index into MNE epochs for EEG signal
        "condition",
        "manipulation",
        "binary_label",        # <- 0=well_formed, 1=anomalous
        "multiclass_label",    # <- 0/1/2/3
        "C_code",
        "item_id",
        "A_code", "B_code",
        "A_onset_sec", "B_onset_sec", "C_onset_sec",
        "dAB_sec", "dBC_sec",
        "epoch0_id", "epoch1_id",
        "pattern_ok", "chain_ok", "all_epochs_matched", "timing_ok",
        "trial_qc_pass",
        "sfreq", "n_channels", "n_times", "tmin", "tmax",
    ]

    available = [c for c in primary_cols if c in df.columns]
    extra     = [c for c in df.columns if c not in available]
    return df[available + extra]


# ==============================================================================
# BLOCK 6 — LABEL SANITY REPORT
# ------------------------------------------------------------------------------
# Prints and saves a text report confirming the label distribution is correct
# before any model training starts. This is the last human checkpoint.
#
# What to look for:
#   - No "unknown" conditions (would mean codebook is incomplete)
#   - Condition counts sum to 160 (all trials labeled)
#   - Binary: ~40 negative (well_formed), ~120 positive (three anomalous types)
#   - Multiclass: roughly 40 per class (counterbalanced design)
#   - QC pass rate: ideally 160/160
# ==============================================================================

def build_sanity_report(
    df: pd.DataFrame,
    subject: str,
    codebook: Dict[int, dict],
) -> str:

    lines = []
    sep   = "=" * 60
    dash  = "-" * 60

    lines.append(sep)
    lines.append(f"LABEL SANITY REPORT — {subject}")
    lines.append(sep)
    lines.append("")

    n_total   = len(df)
    n_unknown = int((df["condition"] == "unknown").sum())
    n_qc_pass = int(df["trial_qc_pass"].sum()) if "trial_qc_pass" in df.columns else n_total

    lines.append(f"Total trials        : {n_total}")
    lines.append(f"QC passing trials   : {n_qc_pass}")
    lines.append(f"Unknown condition   : {n_unknown}")
    lines.append("")

    if n_unknown > 0:
        lines.append("WARNING: Unknown C_codes found — add to verified_codebook.json:")
        for c in sorted(df[df["condition"] == "unknown"]["C_code"].dropna().unique()):
            lines.append(f"  C_code {int(c)}")
        lines.append("")

    lines.append("Condition counts:")
    lines.append(dash)
    for cond in CONDITION_ORDER + ["unknown"]:
        count = int((df["condition"] == cond).sum())
        if count == 0:
            continue
        bar = "█" * min(count, 40)
        lines.append(f"  {cond:<30} {count:>4}  {bar}")
    lines.append("")

    lines.append("Manipulation split per condition:")
    lines.append(dash)
    for cond in CONDITION_ORDER:
        cdf = df[df["condition"] == cond]
        if len(cdf) == 0:
            continue
        var = int((cdf["manipulation"] == "variable").sum())
        kw  = int((cdf["manipulation"] == "keyword").sum())
        unk = int((cdf["manipulation"] == "unknown").sum())
        lines.append(f"  {cond:<30}  variable={var}  keyword={kw}  unknown={unk}")
    lines.append("")

    lines.append("Binary label counts (0=well_formed, 1=anomalous):")
    lines.append(dash)
    for label, name in [(0, "negative / well_formed"), (1, "positive / anomalous")]:
        count = int((df["binary_label"] == label).sum())
        lines.append(f"  {label} ({name:<25}) : {count}")
    lines.append("")

    lines.append("Multiclass label counts (0/1/2/3):")
    lines.append(dash)
    for label, name in MULTICLASS_NAMES.items():
        count = int((df["multiclass_label"] == label).sum())
        lines.append(f"  {label} ({name:<30}) : {count}")
    lines.append("")

    lines.append("C_code -> condition mapping used:")
    lines.append(dash)
    for c_code, entry in sorted(codebook.items()):
        lines.append(
            f"  {c_code:<6}  {entry['condition']:<30} "
            f"manipulation={entry['manipulation']:<10} "
            f"binary={entry['binary_label']}  multiclass={entry['multiclass_label']}"
        )
    lines.append("")
    lines.append("Verify this matches your experiment documentation before training.")

    return "\n".join(lines)


# ==============================================================================
# BLOCK 7 — SEVEN DIAGNOSTIC FIGURES
# ------------------------------------------------------------------------------
# These figures exist to catch problems before EEGNet training. Each one
# answers a specific question:
#
# Fig 1 — Condition balance bar chart
#   Are there roughly 40 trials per condition? If not, something is wrong
#   with the labeling or the subject had an unusually structured session.
#
# Fig 2 — Trial timeline colored by condition
#   Are conditions interleaved randomly throughout the recording, or do they
#   cluster in blocks? Clustering would suggest the randomization failed.
#
# Fig 3 — dBC timing gaps across all trials
#   Is every trial's gap exactly 1.3046875s? Outliers mean the C-epoch is
#   not properly time-locked and should be excluded from training.
#
# Fig 4 — C_code scatter by condition and manipulation
#   Visual confirmation that the 7 C_codes cluster into the expected groups.
#   well_formed and anomalous types should form distinct bands.
#
# Fig 5 — Binary label class distribution
#   Shows the 40/120 imbalance (well_formed vs all anomalous) that EEGNet
#   needs to handle via class weights or oversampling.
#
# Fig 6 — Manipulation split within each condition
#   Confirms the variable/keyword split is 50/50 within each applicable
#   condition. An asymmetric split would bias any manipulation analysis.
#
# Fig 7 — epoch2_id coverage across the full recording
#   Shows that C-epochs are spread across the entire 10-minute session, not
#   concentrated at the start or end. Concentration would suggest the subject
#   stopped attending midway through.
# ==============================================================================

def make_figure_1_condition_balance(df: pd.DataFrame, subject: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = [c for c in CONDITION_ORDER if c in df["condition"].values]
    counts     = [int((df["condition"] == c).sum()) for c in conditions]
    colors     = [CONDITION_COLORS[c] for c in conditions]

    bars = ax.bar(conditions, counts, color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(40, linestyle="--", color="#7f8c8d", linewidth=1, label="Expected = 40")
    ax.set_title(f"{subject} — Trial count per condition", fontsize=13, pad=12)
    ax.set_ylabel("Number of trials")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.legend()
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    fig.tight_layout()
    return fig


def make_figure_2_trial_timeline(df: pd.DataFrame, subject: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))

    for _, row in df.iterrows():
        color = CONDITION_COLORS.get(str(row["condition"]), "#95a5a6")
        onset = float(row["C_onset_sec"]) if pd.notna(row.get("C_onset_sec")) else 0
        ax.axvline(onset, color=color, alpha=0.6, linewidth=1.2)

    patches = [mpatches.Patch(color=CONDITION_COLORS[c], label=c)
               for c in CONDITION_ORDER]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_xlabel("Recording time (seconds)")
    ax.set_title(f"{subject} — Trial timeline (each line = one C-epoch onset)", fontsize=12)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def make_figure_3_timing_gaps(df: pd.DataFrame, subject: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))

    dbc = df["dBC_sec"].dropna().values
    dab = df["dAB_sec"].dropna().values
    x   = np.arange(len(dbc))

    ax.plot(x, dbc, color="#3498db", linewidth=0.8, label="dBC (B->C gap)")
    ax.plot(x, dab, color="#e67e22", linewidth=0.8, alpha=0.6, label="dAB (A->B gap)")

    expected = float(pd.Series(dbc).round(6).mode().iloc[0]) if len(dbc) > 0 else 1.3046875
    ax.axhline(expected, linestyle="--", color="#2c3e50", linewidth=1,
               label=f"Expected SOA = {expected:.7f}s")

    # Highlight outliers
    tol      = 0.01
    outliers = np.where(np.abs(dbc - expected) > tol)[0]
    if len(outliers) > 0:
        ax.scatter(outliers, dbc[outliers], color="#e74c3c", zorder=5,
                   s=40, label=f"Outliers (n={len(outliers)})")

    ax.set_xlabel("Trial index")
    ax.set_ylabel("Time gap (seconds)")
    ax.set_title(f"{subject} — Triplet timing gaps (should be flat line)", fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_figure_4_ccode_scatter(df: pd.DataFrame, subject: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    for _, row in df.iterrows():
        c_code    = int(row["C_code"])   if pd.notna(row["C_code"])   else 0
        trial_id  = int(row["trial_id"]) if pd.notna(row["trial_id"]) else 0
        condition = str(row["condition"])
        color     = CONDITION_COLORS.get(condition, "#95a5a6")
        marker    = "o" if str(row.get("manipulation")) == "variable" else "^"
        ax.scatter(c_code, trial_id, color=color, marker=marker, alpha=0.5, s=20)

    patches = [mpatches.Patch(color=CONDITION_COLORS[c], label=c)
               for c in CONDITION_ORDER]
    circle  = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                         markersize=8, label="variable")
    tri     = plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="grey",
                         markersize=8, label="keyword")
    ax.legend(handles=patches + [circle, tri], fontsize=8, loc="upper right")
    ax.set_xlabel("C_code")
    ax.set_ylabel("Trial index")
    ax.set_title(f"{subject} — C_code scatter by condition and manipulation", fontsize=12)
    fig.tight_layout()
    return fig


def make_figure_5_class_imbalance(df: pd.DataFrame, subject: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Binary
    ax = axes[0]
    binary_counts = [int((df["binary_label"] == 0).sum()),
                     int((df["binary_label"] == 1).sum())]
    ax.bar(["0: well_formed\n(negative)", "1: anomalous\n(positive)"],
           binary_counts,
           color=["#2ecc71", "#e74c3c"], edgecolor="white")
    for i, v in enumerate(binary_counts):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Binary label distribution", fontsize=11)
    ax.set_ylabel("Trial count")

    # Multiclass
    ax = axes[1]
    mc_labels = [0, 1, 2, 3]
    mc_counts = [int((df["multiclass_label"] == l).sum()) for l in mc_labels]
    mc_names  = [f"{l}: {MULTICLASS_NAMES[l][:12]}" for l in mc_labels]
    mc_colors = [CONDITION_COLORS[MULTICLASS_NAMES[l]] for l in mc_labels]
    bars = ax.bar(mc_names, mc_counts, color=mc_colors, edgecolor="white")
    for bar, count in zip(bars, mc_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("Multiclass label distribution", fontsize=11)
    ax.set_xticklabels(mc_names, rotation=20, ha="right", fontsize=8)

    fig.suptitle(f"{subject} — Class distributions", fontsize=12)
    fig.tight_layout()
    return fig


def make_figure_6_manipulation_split(df: pd.DataFrame, subject: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))

    conditions = [c for c in CONDITION_ORDER if c in df["condition"].values]
    x          = np.arange(len(conditions))
    width      = 0.35

    var_counts = [int(((df["condition"] == c) & (df["manipulation"] == "variable")).sum())
                  for c in conditions]
    kw_counts  = [int(((df["condition"] == c) & (df["manipulation"] == "keyword")).sum())
                  for c in conditions]

    bars1 = ax.bar(x - width/2, var_counts, width, label="variable",
                   color=MANIPULATION_COLORS["variable"], edgecolor="white")
    bars2 = ax.bar(x + width/2, kw_counts,  width, label="keyword",
                   color=MANIPULATION_COLORS["keyword"],  edgecolor="white")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Trial count")
    ax.set_title(f"{subject} — Variable vs keyword split per condition", fontsize=12)
    ax.legend()
    fig.tight_layout()
    return fig


def make_figure_7_epoch2id_coverage(df: pd.DataFrame, subject: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))

    for _, row in df.iterrows():
        epoch2_id = int(row["epoch2_id"]) if pd.notna(row.get("epoch2_id")) else 0
        trial_id  = int(row["trial_id"])
        condition = str(row["condition"])
        color     = CONDITION_COLORS.get(condition, "#95a5a6")
        ax.scatter(epoch2_id, trial_id, color=color, alpha=0.6, s=15)

    patches = [mpatches.Patch(color=CONDITION_COLORS[c], label=c)
               for c in CONDITION_ORDER]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_xlabel("epoch2_id (MNE epoch index for EEG signal extraction)")
    ax.set_ylabel("Trial index")
    ax.set_title(
        f"{subject} — epoch2_id coverage across full recording\n"
        f"(should be spread evenly — not clustered at start or end)",
        fontsize=11
    )
    fig.tight_layout()
    return fig


# ==============================================================================
# BLOCK 8 — ARGUMENT PARSER AND MAIN
# ------------------------------------------------------------------------------
# Orchestrates all blocks in order. The console output is structured so you
# can read it top to bottom and know immediately whether to proceed to
# EEGNet or stop and investigate.
#
# The only stopping condition is --strict mode with unknown labels.
# All other issues are warnings — they are logged in the sanity report but
# the script completes and saves all outputs so you can inspect them.
# ==============================================================================

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Script 3: Label and visualize trials")
    p.add_argument("--subject",      default="sub-001")
    p.add_argument("--triplets-dir", default="../../outputs/triplets",
                   help="Directory where Script 2 wrote trial_flow_table.csv")
    p.add_argument("--codebook",     default="../../configs/c_code_to_condition_verified.json")
    p.add_argument("--output-dir",   default="../../outputs/labeled",
                   help="Where to write labeled_trial_table.csv and figures")
    p.add_argument("--sfreq",        type=float, default=256.0,
                   help="Sampling rate (read from audit_summary.json if available)")
    p.add_argument("--n-channels",   type=int,   default=34)
    p.add_argument("--n-times",      type=int,   default=333)
    p.add_argument("--tmin",         type=float, default=-0.1015625)
    p.add_argument("--tmax",         type=float, default=1.19921875)
    p.add_argument("--strict",       action="store_true",
                   help="Exit with error if any trials have unknown labels")
    return p.parse_args()


def maybe_load_audit_meta(audit_dir: Path, subject: str) -> dict:
    """Try to load sfreq and recording metadata from audit_summary.json."""
    path = audit_dir / subject / f"{subject}_audit_summary.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return summary.get("recording", {})
    return {}


def main() -> None:
    args         = build_args()
    subject      = args.subject
    triplets_dir = Path(args.triplets_dir)
    output_dir   = Path(args.output_dir)
    fig_dir      = output_dir / "figures"
    codebook_path = Path(args.codebook)

    print(f"\n  Subject    : {subject}")
    print(f"  Codebook   : {codebook_path.name}")
    print(f"  Output dir : {output_dir}")

    # Try to pull recording metadata from audit_summary.json
    audit_dir = triplets_dir.parent / "audit"
    meta      = maybe_load_audit_meta(audit_dir, subject)
    sfreq      = meta.get("sfreq",      args.sfreq)
    n_channels = meta.get("n_channels", args.n_channels)
    n_times    = meta.get("n_times",    args.n_times)
    tmin       = meta.get("tmin",       args.tmin)
    tmax       = meta.get("tmax",       args.tmax)

    # ------------------------------------------------------------------
    section("STEP 1 — Load verified codebook")
    codebook = load_codebook(codebook_path)
    print(f"  {len(codebook)} C_codes loaded:")
    for c_code, entry in sorted(codebook.items()):
        print(f"    {c_code} -> {entry['condition']:<30} "
              f"manipulation={entry['manipulation']:<10} "
              f"binary={entry['binary_label']}  multiclass={entry['multiclass_label']}")

    # ------------------------------------------------------------------
    section("STEP 2 — Load trial flow table and apply labels")
    trial_flow_path = triplets_dir / subject / f"{subject}_trial_flow_table.csv"
    if not trial_flow_path.exists():
        print(f"  ERROR: {trial_flow_path} not found — run Script 2 first")
        sys.exit(1)

    df = load_and_label(trial_flow_path, codebook)
    print(f"  {len(df)} trials labeled")

    n_unknown = int((df["condition"] == "unknown").sum())
    if n_unknown > 0:
        print(f"  WARNING: {n_unknown} trials have unknown condition labels")
        if args.strict:
            print("  [STRICT] Exiting. Fix verified_codebook.json before proceeding.")
            sys.exit(1)
    else:
        print("  All trials labeled successfully — no unknown conditions")

    # ------------------------------------------------------------------
    section("STEP 3 — Build final labeled trial table")
    labeled_df = build_labeled_table(
        df, subject, sfreq, n_channels, n_times, tmin, tmax
    )
    save_csv(labeled_df, output_dir / f"{subject}_labeled_trial_table.csv")

    # ------------------------------------------------------------------
    section("STEP 4 — Label sanity report")
    report = build_sanity_report(labeled_df, subject, codebook)
    print(report)
    save_text(report, output_dir / f"{subject}_label_sanity_report.txt")

    # ------------------------------------------------------------------
    section("STEP 5 — Generate diagnostic figures")

    fig1 = make_figure_1_condition_balance(labeled_df, subject)
    save_fig(fig1, fig_dir / f"{subject}_fig1_condition_balance.png")

    fig2 = make_figure_2_trial_timeline(labeled_df, subject)
    save_fig(fig2, fig_dir / f"{subject}_fig2_trial_timeline.png")

    fig3 = make_figure_3_timing_gaps(labeled_df, subject)
    save_fig(fig3, fig_dir / f"{subject}_fig3_timing_gaps.png")

    fig4 = make_figure_4_ccode_scatter(labeled_df, subject)
    save_fig(fig4, fig_dir / f"{subject}_fig4_ccode_scatter.png")

    fig5 = make_figure_5_class_imbalance(labeled_df, subject)
    save_fig(fig5, fig_dir / f"{subject}_fig5_class_imbalance.png")

    fig6 = make_figure_6_manipulation_split(labeled_df, subject)
    save_fig(fig6, fig_dir / f"{subject}_fig6_manipulation_split.png")

    fig7 = make_figure_7_epoch2id_coverage(labeled_df, subject)
    save_fig(fig7, fig_dir / f"{subject}_fig7_epoch2id_coverage.png")

    # ------------------------------------------------------------------
    section("DONE")

    n_qc_pass = int(labeled_df["trial_qc_pass"].sum()) \
                if "trial_qc_pass" in labeled_df.columns else len(labeled_df)
    n_unknown = int((labeled_df["condition"] == "unknown").sum())

    print(f"\n  Subject           : {subject}")
    print(f"  Trials total      : {len(labeled_df)}")
    print(f"  Trials QC pass    : {n_qc_pass}")
    print(f"  Unknown labels    : {n_unknown}")
    print(f"\n  Condition counts  :")
    for cond in CONDITION_ORDER:
        count = int((labeled_df["condition"] == cond).sum())
        print(f"    {cond:<30} : {count}")
    print(f"\n  Primary output:")
    print(f"    {subject}_labeled_trial_table.csv")
    print(f"    -> use epoch2_id column to index MNE epochs for EEG signal extraction")
    print(f"\n  Check the 7 figures in {fig_dir}")
    print(f"  before proceeding to feature extraction and EEGNet training.\n")

    if n_unknown > 0:
        print(f"  WARNING: Resolve {n_unknown} unknown labels before training.")
        print(f"  Add missing C_codes to configs/verified_codebook.json\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise