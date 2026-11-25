

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ==============================================================================
# BLOCK 1 — CONFIGURATION
# ------------------------------------------------------------------------------
# All paths and expected values are defined here. If your directory structure
# differs from the default, change these constants rather than editing the
# logic blocks below.
#
# EXPECTED_TRIALS, EXPECTED_PER_CONDITION, and QC_WARN_THRESHOLD define what
# a healthy subject looks like. These come directly from the experiment design:
#   - 160 trials per subject (40 snippets × 4 conditions)
#   - 40 trials per condition (balanced design)
#   - 100% QC pass rate expected
#
# WARN_THRESHOLD is the number of QC failures that triggers WARN vs PASS.
# Zero failures = PASS. 1-4 failures = WARN. 5+ failures = FAIL.
# ==============================================================================

# Paths — adjust if your project structure differs
SCRIPTS_DIR        = Path(__file__).parent
DATA_DIR           = SCRIPTS_DIR.parent.parent / "data"
OUTPUT_BASE        = SCRIPTS_DIR.parent.parent / "outputs"
AUDIT_DIR          = OUTPUT_BASE / "audit"
TRIPLETS_DIR       = OUTPUT_BASE / "triplets"
LABELED_DIR        = OUTPUT_BASE / "labeled"
BATCH_DIR          = OUTPUT_BASE / "batch"
CODEBOOK_PATH      = SCRIPTS_DIR.parent.parent / "configs" / "c_code_to_condition_verified.json"

# Script paths — adjust names if you renamed the scripts
SCRIPT_1 = SCRIPTS_DIR / "data_audit.py"
SCRIPT_2 = SCRIPTS_DIR / "triplet_builder.py"
SCRIPT_3 = SCRIPTS_DIR / "visual_and_label.py"

# Expected values for a healthy subject
EXPECTED_TRIALS          = 160
EXPECTED_PER_CONDITION   = 40
QC_WARN_THRESHOLD        = 5    # QC failures below this = WARN, above = FAIL
CONDITION_NAMES          = [
    "well_formed",
    "semantically_implausible",
    "syntactically_invalid",
    "doubly_anomalous",
]


# ==============================================================================
# BLOCK 2 — SUBJECT DISCOVERY
# ------------------------------------------------------------------------------
# Scans the data directory for subject folders matching the sub-XXX pattern.
# Returns them sorted so the batch runs in a predictable order.
#
# Each subject must have a sub-XXX/eeg/ folder containing at least one .set
# file and one _events.tsv file. Folders that exist but are missing these
# files are reported as SKIP rather than causing a crash mid-batch.
# ==============================================================================

def discover_subjects(data_dir: Path) -> List[str]:
    """Find all sub-XXX folders in data_dir that have EEG data."""
    subjects = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.startswith("sub-"):
            continue
        eeg_dir = folder / "eeg"
        if not eeg_dir.exists():
            continue
        has_set = any(eeg_dir.glob("*.set"))
        has_tsv = any(eeg_dir.glob("*_events.tsv"))
        if has_set and has_tsv:
            subjects.append(folder.name)

    return subjects


def validate_subject_data(data_dir: Path, subject: str) -> Tuple[bool, str]:
    """
    Check that a subject folder has the required files before running.
    Returns (valid, reason_if_invalid).
    """
    eeg_dir = data_dir / subject / "eeg"
    if not eeg_dir.exists():
        return False, f"EEG directory not found: {eeg_dir}"

    set_files = list(eeg_dir.glob("*.set"))
    if not set_files:
        return False, "No .set file found in eeg directory"

    tsv_files = list(eeg_dir.glob("*_events.tsv"))
    if not tsv_files:
        return False, "No *_events.tsv file found in eeg directory"

    return True, ""


# ==============================================================================
# BLOCK 3 — SCRIPT RUNNER
# ------------------------------------------------------------------------------
# Runs one script as a subprocess and captures stdout, stderr, and return code.
# Using subprocess rather than importing the scripts directly keeps each run
# completely isolated — a crash in Script 1 for one subject does not affect
# the batch loop for the next subject.
#
# The timeout is set generously at 300 seconds (5 minutes) per script per
# subject. Script 1 is the slowest because it loads MNE — on our compute cluster
# cluster with the .set files on fast storage it typically takes 30-60 seconds.
# Scripts 2 and 3 are pure pandas and finish in under 5 seconds each.
#
# stdout and stderr are captured and saved to per-subject log files so you
# can inspect exactly what happened for any subject that failed.
# ==============================================================================

def run_script(
    script_path: Path,
    subject: str,
    extra_args: List[str],
    log_dir: Path,
    timeout: int = 300,
) -> Tuple[bool, str, float]:
    """
    Run a pipeline script for one subject as a subprocess.

    Returns:
        success  — True if returncode == 0
        message  — stdout tail or error message
        duration — wall clock time in seconds
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{subject}_{script_path.stem}.log"

    cmd = [sys.executable, str(script_path), "--subject", subject] + extra_args

    t_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - t_start

        # Save full log
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"COMMAND: {' '.join(cmd)}\n")
            f.write(f"RETURN CODE: {result.returncode}\n")
            f.write(f"DURATION: {duration:.1f}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        if result.returncode == 0:
            # Return last 3 non-empty lines of stdout as summary
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            summary = " | ".join(lines[-3:]) if lines else "OK"
            return True, summary, duration
        else:
            # Return last error line
            err_lines = [l for l in result.stderr.splitlines() if l.strip()]
            error_msg = err_lines[-1] if err_lines else "Unknown error"
            return False, error_msg, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - t_start
        return False, f"TIMEOUT after {timeout}s", duration

    except Exception as e:
        duration = time.time() - t_start
        return False, str(e), duration


# ==============================================================================
# BLOCK 4 — RESULT PARSER
# ------------------------------------------------------------------------------
# After Script 3 runs, this block reads the labeled_trial_table.csv and
# label_sanity_report.txt to extract the counts that determine subject status.
#
# Reading from the output files rather than parsing stdout means the parser
# works regardless of what was printed to the console during the run. It is
# also more robust — the CSV is structured data, not text to regex over.
#
# If the expected output files don't exist (because a script crashed), the
# parser returns safe defaults so the status logic can assign FAIL cleanly.
# ==============================================================================

def parse_subject_results(subject: str) -> dict:
    """
    Parse the outputs of all three scripts for one subject.
    Returns a dict of counts and flags used to determine status.
    """
    result = {
        "subject":          subject,
        "n_trials":         0,
        "n_qc_pass":        0,
        "n_qc_fail":        0,
        "n_unknown":        0,
        "well_formed":      0,
        "semantically_implausible": 0,
        "syntactically_invalid":    0,
        "doubly_anomalous": 0,
        "binary_0":         0,
        "binary_1":         0,
        "match_rate":       0.0,
        "chain_ok_rate":    0.0,
        "expected_soa":     None,
        "n_timing_fail":    0,
        "labeled_csv_exists": False,
        "triplet_json_exists": False,
        "audit_json_exists":   False,
        "parse_error":      None,
    }

    try:
        # --- Check labeled trial table ---
        labeled_path = LABELED_DIR / f"{subject}_labeled_trial_table.csv"
        if labeled_path.exists():
            result["labeled_csv_exists"] = True
            df = pd.read_csv(labeled_path)
            result["n_trials"] = len(df)

            if "trial_qc_pass" in df.columns:
                result["n_qc_pass"] = int(df["trial_qc_pass"].sum())
                result["n_qc_fail"] = int((~df["trial_qc_pass"]).sum())
            else:
                result["n_qc_pass"] = len(df)

            if "condition" in df.columns:
                result["n_unknown"] = int((df["condition"] == "unknown").sum())
                for cond in CONDITION_NAMES:
                    result[cond] = int((df["condition"] == cond).sum())

            if "binary_label" in df.columns:
                result["binary_0"] = int((df["binary_label"] == 0).sum())
                result["binary_1"] = int((df["binary_label"] == 1).sum())

        # --- Check triplet summary JSON ---
        triplet_path = TRIPLETS_DIR / subject / f"{subject}_triplet_summary.json"
        if triplet_path.exists():
            result["triplet_json_exists"] = True
            with open(triplet_path, "r", encoding="utf-8") as f:
                triplet_data = json.load(f)
            result["expected_soa"]  = triplet_data.get("expected_soa_sec")
            result["n_timing_fail"] = int(triplet_data.get("n_trials_qc_fail", 0))

        # --- Check audit summary JSON ---
        audit_path = AUDIT_DIR / subject / f"{subject}_audit_summary.json"
        if audit_path.exists():
            result["audit_json_exists"] = True
            with open(audit_path, "r", encoding="utf-8") as f:
                audit_data = json.load(f)
            matching = audit_data.get("matching", {})
            n_total   = matching.get("n_epochs_total", 0)
            n_matched = matching.get("n_epochs_matched", 0)
            result["match_rate"] = (n_matched / n_total) if n_total > 0 else 0.0

        # --- Check triplet flow table for chain consistency ---
        flow_path = TRIPLETS_DIR / subject / f"{subject}_trial_flow_table.csv"
        if flow_path.exists():
            flow_df = pd.read_csv(flow_path)
            if "chain_ok" in flow_df.columns:
                n_chain = int(flow_df["chain_ok"].sum())
                result["chain_ok_rate"] = n_chain / len(flow_df) if len(flow_df) > 0 else 0.0

    except Exception as e:
        result["parse_error"] = str(e)

    return result


# ==============================================================================
# BLOCK 5 — STATUS ASSIGNMENT
# ------------------------------------------------------------------------------
# Takes the parsed result dict and assigns one of three statuses:
#   PASS / WARN / FAIL
#
# The rules are conservative — it is better to flag a subject for human
# review than to silently include a bad subject in EEGNet training.
#
# Each rule that fires contributes a reason string to the status list.
# The final status is the worst of all triggered rules.
# This means a subject can trigger both WARN and FAIL rules — it gets FAIL.
# ==============================================================================

def assign_status(result: dict) -> Tuple[str, List[str]]:
    """
    Assign PASS, WARN, or FAIL status based on parsed result counts.
    Returns (status, list_of_reasons).
    """
    fail_reasons = []
    warn_reasons = []

    # FAIL conditions
    if not result["labeled_csv_exists"]:
        fail_reasons.append("labeled_trial_table.csv not found — script may have crashed")

    if result["n_trials"] < 100:
        fail_reasons.append(f"Too few trials: {result['n_trials']} (expected {EXPECTED_TRIALS})")

    if result["n_unknown"] > 0:
        fail_reasons.append(f"Unknown labels: {result['n_unknown']} trials")

    if result["parse_error"]:
        fail_reasons.append(f"Parse error: {result['parse_error']}")

    if result["labeled_csv_exists"]:
        for cond in CONDITION_NAMES:
            if result[cond] == 0:
                fail_reasons.append(f"Condition '{cond}' has zero trials")

    if result["match_rate"] < 0.95 and result["audit_json_exists"]:
        fail_reasons.append(f"Low epoch match rate: {result['match_rate']:.1%}")

    if result["chain_ok_rate"] < 0.95 and result["triplet_json_exists"]:
        fail_reasons.append(f"Low chain consistency: {result['chain_ok_rate']:.1%}")

    if result["n_qc_fail"] >= QC_WARN_THRESHOLD:
        fail_reasons.append(f"Too many QC failures: {result['n_qc_fail']}")

    # WARN conditions
    if result["n_trials"] != EXPECTED_TRIALS and result["n_trials"] >= 100:
        warn_reasons.append(f"Unexpected trial count: {result['n_trials']} (expected {EXPECTED_TRIALS})")

    if result["n_qc_fail"] > 0 and result["n_qc_fail"] < QC_WARN_THRESHOLD:
        warn_reasons.append(f"Minor QC failures: {result['n_qc_fail']}")

    for cond in CONDITION_NAMES:
        count = result[cond]
        if 0 < count and abs(count - EXPECTED_PER_CONDITION) > 2:
            warn_reasons.append(f"Condition imbalance: {cond}={count} (expected {EXPECTED_PER_CONDITION})")

    if result["match_rate"] < 1.0 and result["match_rate"] >= 0.95:
        warn_reasons.append(f"Match rate not 100%: {result['match_rate']:.1%}")

    # Assign final status
    if fail_reasons:
        return "FAIL", fail_reasons + warn_reasons
    if warn_reasons:
        return "WARN", warn_reasons
    return "PASS", []


# ==============================================================================
# BLOCK 6 — PROGRESS PRINTER
# ------------------------------------------------------------------------------
# Prints a clean one-line summary for each subject as it completes.
# Uses ANSI colors so PASS/WARN/FAIL are immediately scannable in a terminal.
# The line format is fixed-width so columns stay aligned even for long names.
# ==============================================================================

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

STATUS_COLORS = {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED}

def print_subject_result(
    subject: str,
    status: str,
    result: dict,
    reasons: List[str],
    script_times: Dict[str, float],
) -> None:
    color     = STATUS_COLORS.get(status, "")
    total_sec = sum(script_times.values())

    cond_str = (
        f"wf={result['well_formed']:>2}  "
        f"sem={result['semantically_implausible']:>2}  "
        f"syn={result['syntactically_invalid']:>2}  "
        f"dbl={result['doubly_anomalous']:>2}"
    )

    print(
        f"  {color}{BOLD}{status:<4}{RESET}  "
        f"{subject:<10}  "
        f"trials={result['n_trials']:>3}  "
        f"qc_fail={result['n_qc_fail']:>3}  "
        f"{cond_str}  "
        f"{DIM}{total_sec:.0f}s{RESET}"
    )

    if reasons:
        for r in reasons:
            print(f"         {DIM}-> {r}{RESET}")


# ==============================================================================
# BLOCK 7 — MASTER SUMMARY BUILDER
# ------------------------------------------------------------------------------
# After all subjects complete, this block assembles the master CSV and
# text report. The CSV is machine-readable for further analysis. The text
# report is human-readable for quick review before feature extraction.
#
# The failed_subjects.txt file contains just the subject IDs that need
# attention — paste it into an email, a GitHub issue, or your lab notebook.
# ==============================================================================

def build_master_summary(
    all_results: List[dict],
    all_statuses: Dict[str, Tuple[str, List[str]]],
    batch_dir: Path,
    start_time: float,
) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Build summary DataFrame
    rows = []
    for result in all_results:
        subj   = result["subject"]
        status, reasons = all_statuses[subj]
        rows.append({
            "subject":                  subj,
            "status":                   status,
            "n_trials":                 result["n_trials"],
            "n_qc_pass":                result["n_qc_pass"],
            "n_qc_fail":                result["n_qc_fail"],
            "n_unknown":                result["n_unknown"],
            "well_formed":              result["well_formed"],
            "semantically_implausible": result["semantically_implausible"],
            "syntactically_invalid":    result["syntactically_invalid"],
            "doubly_anomalous":         result["doubly_anomalous"],
            "binary_0":                 result["binary_0"],
            "binary_1":                 result["binary_1"],
            "match_rate":               round(result["match_rate"], 4),
            "chain_ok_rate":            round(result["chain_ok_rate"], 4),
            "n_timing_fail":            result["n_timing_fail"],
            "labeled_csv_exists":       result["labeled_csv_exists"],
            "reasons":                  " | ".join(reasons) if reasons else "",
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(batch_dir / "batch_summary.csv", index=False)
    print(f"\n  [SAVED] batch_summary.csv")

    # Count statuses
    n_pass = int((summary_df["status"] == "PASS").sum())
    n_warn = int((summary_df["status"] == "WARN").sum())
    n_fail = int((summary_df["status"] == "FAIL").sum())
    total  = len(summary_df)
    elapsed = time.time() - start_time

    # Text report
    sep  = "=" * 80
    dash = "-" * 80
    lines = []
    lines.append(sep)
    lines.append("BATCH PIPELINE REPORT")
    lines.append(f"Run date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    lines.append(sep)
    lines.append("")
    lines.append(f"SUMMARY: {total} subjects processed")
    lines.append(f"  {GREEN}PASS{RESET} : {n_pass}")
    lines.append(f"  {YELLOW}WARN{RESET} : {n_warn}")
    lines.append(f"  {RED}FAIL{RESET} : {n_fail}")
    lines.append("")

    for status_filter in ["FAIL", "WARN", "PASS"]:
        group = summary_df[summary_df["status"] == status_filter]
        if len(group) == 0:
            continue

        color = STATUS_COLORS[status_filter]
        lines.append(dash)
        lines.append(f"{color}{status_filter} SUBJECTS ({len(group)}){RESET}")
        lines.append(dash)

        for _, row in group.iterrows():
            lines.append(
                f"  {row['subject']:<12}  "
                f"trials={row['n_trials']:>3}  "
                f"qc_fail={row['n_qc_fail']:>3}  "
                f"wf={row['well_formed']:>2}  "
                f"sem={row['semantically_implausible']:>2}  "
                f"syn={row['syntactically_invalid']:>2}  "
                f"dbl={row['doubly_anomalous']:>2}  "
                f"unknown={row['n_unknown']:>2}  "
                f"match={row['match_rate']:.1%}"
            )
            if row["reasons"]:
                for reason in row["reasons"].split(" | "):
                    lines.append(f"    -> {reason}")

    lines.append("")
    lines.append(dash)
    lines.append("SUBJECTS READY FOR FEATURE EXTRACTION (PASS only):")
    lines.append(dash)
    pass_subjects = summary_df[summary_df["status"] == "PASS"]["subject"].tolist()
    for subj in pass_subjects:
        lines.append(f"  {subj}")
    lines.append("")
    lines.append(f"Total PASS subjects: {len(pass_subjects)}")
    lines.append("Use these subject IDs for feature extraction and EEGNet training.")

    report_text = "\n".join(lines)
    (batch_dir / "batch_report.txt").write_text(
        report_text.replace(GREEN, "").replace(YELLOW, "").replace(RED, "")
        .replace(RESET, "").replace(BOLD, "").replace(DIM, ""),
        encoding="utf-8"
    )
    print(f"  [SAVED] batch_report.txt")

    # Failed subjects list
    fail_warn = summary_df[summary_df["status"].isin(["FAIL", "WARN"])]["subject"].tolist()
    if fail_warn:
        (batch_dir / "failed_subjects.txt").write_text(
            "\n".join(fail_warn) + "\n", encoding="utf-8"
        )
        print(f"  [SAVED] failed_subjects.txt  ({len(fail_warn)} subjects need review)")

    # Pass subjects list — use this for feature extraction
    (batch_dir / "pass_subjects.txt").write_text(
        "\n".join(pass_subjects) + "\n", encoding="utf-8"
    )
    print(f"  [SAVED] pass_subjects.txt  ({len(pass_subjects)} subjects ready)")

    # Print final console summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  BATCH COMPLETE{RESET}")
    print(f"{'='*60}")
    print(f"  Total subjects : {total}")
    print(f"  {GREEN}PASS{RESET}           : {n_pass}")
    print(f"  {YELLOW}WARN{RESET}           : {n_warn}")
    print(f"  {RED}FAIL{RESET}           : {n_fail}")
    print(f"  Total time     : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")


# ==============================================================================
# BLOCK 8 — ARGUMENT PARSER AND MAIN
# ------------------------------------------------------------------------------
# Orchestrates the full batch run. For each subject it:
#   1. Validates data exists
#   2. Runs Script 1 -> Script 2 -> Script 3 in sequence
#   3. Stops running scripts for that subject on first failure
#   4. Parses outputs and assigns status
#   5. Prints per-subject result immediately
#
# The --stage flag lets you rerun just one stage across all subjects without
# rerunning the earlier stages. This is useful when Script 3 changes (e.g.
# you update the codebook) and you want to relabel all subjects without
# reloading EEG data.
#
# Log files are written to outputs/batch/logs/<subject>_<script>.log
# so you can inspect the full stdout/stderr for any subject that failed.
# ==============================================================================

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch pipeline runner for all subjects")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Specific subjects to process (default: all found in data dir)")
    p.add_argument("--data-dir", default=str(DATA_DIR),
                   help="Root data directory containing sub-XXX folders")
    p.add_argument("--stage", type=int, default=None, choices=[1, 2, 3],
                   help="Run only this stage (1=audit, 2=triplets, 3=label). Default: all.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip subjects that already have labeled_trial_table.csv")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Stop the batch immediately if any subject fails")
    p.add_argument("--dry-run", action="store_true",
                   help="Show which subjects would be processed without running anything")
    p.add_argument("--timeout", type=int, default=300,
                   help="Timeout per script per subject in seconds (default: 300)")
    return p.parse_args()


def main() -> None:
    args     = build_args()
    data_dir = Path(args.data_dir)
    log_dir  = BATCH_DIR / "logs"
    start_time = time.time()

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(data_dir)

    if not subjects:
        print(f"  No subjects found in {data_dir}")
        sys.exit(1)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  BATCH PIPELINE RUNNER{RESET}")
    print(f"{'='*60}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Subjects    : {len(subjects)}")
    print(f"  Stage       : {args.stage if args.stage else 'all (1 -> 2 -> 3)'}")
    print(f"  Skip exist  : {args.skip_existing}")
    print(f"  Stop on fail: {args.stop_on_fail}")
    print(f"  Scripts dir : {SCRIPTS_DIR}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("  DRY RUN — subjects that would be processed:")
        for subj in subjects:
            valid, reason = validate_subject_data(data_dir, subj)
            flag = "OK" if valid else f"SKIP ({reason})"
            print(f"    {subj}  [{flag}]")
        return

    # Verify scripts exist
    script_map = {1: SCRIPT_1, 2: SCRIPT_2, 3: SCRIPT_3}
    stages_to_run = [args.stage] if args.stage else [1, 2, 3]

    for stage in stages_to_run:
        if not script_map[stage].exists():
            print(f"  ERROR: Script {stage} not found: {script_map[stage]}")
            print(f"  Make sure the script names match what is set in BLOCK 1 of this file.")
            sys.exit(1)

    # Build script extra args
    script_extra_args = {
        1: [
            "--data-dir",   str(data_dir),
            "--output-dir", str(AUDIT_DIR),
        ],
        2: [
            "--audit-dir",  str(AUDIT_DIR),
            "--output-dir", str(TRIPLETS_DIR),
        ],
        3: [
            "--triplets-dir", str(TRIPLETS_DIR),
            "--codebook",     str(CODEBOOK_PATH),
            "--output-dir",   str(LABELED_DIR),
        ],
    }

    # Print column header
    print(
        f"  {'STATUS':<6}  {'SUBJECT':<10}  {'TRIALS':>6}  "
        f"{'QC_FAIL':>7}  {'WF':>4}  {'SEM':>4}  {'SYN':>4}  {'DBL':>4}  TIME"
    )
    print(f"  {'-'*6}  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}")

    all_results  = []
    all_statuses = {}

    for subj in subjects:

        # Skip if already done
        if args.skip_existing:
            labeled_path = LABELED_DIR / f"{subj}_labeled_trial_table.csv"
            if labeled_path.exists():
                result   = parse_subject_results(subj)
                status, reasons = assign_status(result)
                all_results.append(result)
                all_statuses[subj] = (status, reasons)
                print_subject_result(subj, status, result, reasons, {})
                continue

        # Validate data exists
        valid, reason = validate_subject_data(data_dir, subj)
        if not valid:
            result = {"subject": subj, "n_trials": 0, "n_qc_pass": 0,
                      "n_qc_fail": 0, "n_unknown": 0, "match_rate": 0.0,
                      "chain_ok_rate": 0.0, "n_timing_fail": 0,
                      "labeled_csv_exists": False, "triplet_json_exists": False,
                      "audit_json_exists": False, "parse_error": reason,
                      "well_formed": 0, "semantically_implausible": 0,
                      "syntactically_invalid": 0, "doubly_anomalous": 0,
                      "binary_0": 0, "binary_1": 0, "expected_soa": None}
            all_results.append(result)
            all_statuses[subj] = ("FAIL", [f"Data validation failed: {reason}"])
            print_subject_result(subj, "FAIL", result,
                                 [f"Data validation: {reason}"], {})
            if args.stop_on_fail:
                break
            continue

        # Run each stage
        script_times = {}
        subject_failed = False

        for stage in stages_to_run:
            script_path = script_map[stage]
            extra_args  = script_extra_args[stage]

            success, message, duration = run_script(
                script_path, subj, extra_args, log_dir, args.timeout
            )
            script_times[f"script{stage}"] = duration

            if not success:
                subject_failed = True
                result = parse_subject_results(subj)
                status = "FAIL"
                reasons = [f"Script {stage} failed: {message}"]
                all_results.append(result)
                all_statuses[subj] = (status, reasons)
                print_subject_result(subj, status, result, reasons, script_times)
                break

        if subject_failed:
            if args.stop_on_fail:
                print(f"\n  --stop-on-fail triggered. Stopping batch.")
                break
            continue

        # Parse results and assign status
        result  = parse_subject_results(subj)
        status, reasons = assign_status(result)
        all_results.append(result)
        all_statuses[subj] = (status, reasons)
        print_subject_result(subj, status, result, reasons, script_times)

        if args.stop_on_fail and status == "FAIL":
            print(f"\n  --stop-on-fail triggered. Stopping batch.")
            break

    # Build and save master summary
    print(f"\n{'='*60}")
    print(f"  Building master summary...")
    build_master_summary(all_results, all_statuses, BATCH_DIR, start_time)


if __name__ == "__main__":
    main()