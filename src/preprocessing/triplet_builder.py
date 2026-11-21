


from __future__ import annotations
 
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
 
warnings.filterwarnings("ignore")


# Utility functions - parse_list_string , 
# redunctant function bits 8 and last 2_bits
"""
Redefines the bit-parsing helpers from Script 1 independently so Script 2 has zero dependency on Script 1's code. The literal_to_list function exists specifically because pandas turns [245, 246] into the string "[245, 246]" when saving to CSV — this silently breaks any downstream arithmetic if you don't convert it back.
"""


def bits8(x: object) -> Optional[str]:
    try:
        return format(int(x), "08b")
    except (TypeError, ValueError):
        return None
 
 
def last2_bits(x: object) -> Optional[int]:
    try:
        return int(x) & 0b11
    except (TypeError, ValueError):
        return None
 
 
def upper6_bits_item_id(x: object) -> Optional[int]:
    try:
        return int(x) >> 2
    except (TypeError, ValueError):
        return None
 
 
def literal_to_list(x: object) -> List[int]:
    """Convert a value that is either already a list or a stringified list back to List[int]."""
    if isinstance(x, list):
        return [int(v) for v in x]
    if pd.isna(x):
        return []
    s = str(x).strip()
    # Strip brackets and split on comma
    s = s.replace("[", "").replace("]", "")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return []
 
 
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [SAVED] {path.name}  ({len(df)} rows)")
 
 
def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {path.name}")
 
 
def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# Argument parser and path_setup

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Script 2: Triplet builder for one BIDS subject")
    p.add_argument("--subject", default="sub-001")
    p.add_argument("--audit-dir", default="../../outputs/audit",
                   help="Directory where Script 1 wrote its outputs")
    p.add_argument("--output-dir", default="../../outputs/triplets",
                   help="Directory where Script 2 will write its outputs")
    p.add_argument("--strict-qc", action="store_true",
                   help="Raise an error if any triplet fails chain or timing checks")
    p.add_argument("--timing-tolerance-sec", type=float, default=0.01,
                   help="Allowed deviation from expected dBC gap in seconds (default 0.01)")
    p.add_argument("--verbose", action="store_true",
                   help="Print full tables to console")
    return p.parse_args()


#load script 1 outputs
"""
 The column validation check is important — if Script 1 was 
 modified and dropped a column, you get a clear error here rather 
 than a cryptic KeyError three steps later when it actually matters.
 This block loads Script 1’s outputs and verifies required fields—especially
sfreq from the audit JSON—so Script 2 can compute time gaps 
  correctly without hardcoding the sampling rate.
"""

def load_script1_outputs(
    audit_dir: Path,
    subject: str,
) -> Tuple[pd.DataFrame, float]:
    """
    Load epoch_audit_table.csv and audit_summary.json from Script 1.
    Returns the epoch DataFrame and sfreq as a float.
    """
    audit_table_path   = audit_dir / subject / f"{subject}_epoch_audit_table.csv"
    audit_summary_path = audit_dir / subject / f"{subject}_audit_summary.json"
 
    if not audit_table_path.exists():
        raise FileNotFoundError(f"Script 1 output not found: {audit_table_path}")
    if not audit_summary_path.exists():
        raise FileNotFoundError(f"Script 1 summary not found: {audit_summary_path}")
 
    epoch_df = pd.read_csv(audit_table_path)
 
    with open(audit_summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
 
    sfreq = float(summary["recording"]["sfreq"])
 
    required_cols = {
        "epoch_id", "triplet_id_guess", "triplet_pos_guess",
        "event_string_type", "all_codes_in_string",
        "first_code_in_string", "last_code_in_string",
        "matched_onset_sec", "matched_sample", "match_status",
    }
    missing = required_cols - set(epoch_df.columns)
    if missing:
        raise ValueError(f"epoch_audit_table.csv is missing columns: {missing}")
 
    # Convert the stringified list column back to actual Python lists
    epoch_df["all_codes_in_string"] = epoch_df["all_codes_in_string"].apply(literal_to_list)
 
    print(f"  Loaded {len(epoch_df)} epochs from Script 1")
    print(f"  sfreq from audit summary: {sfreq} Hz")
 
    return epoch_df, sfreq


# Triple bulder
"""
The scientific core. The chain consistency check — B
 must match across the A/B and B/C boundary — is what proves 
 these three epochs actually belong to the same stimulus chain. 
 Without this check you could be assigning a C-epoch's EEG signal to 
 the wrong trial's condition label.
 This block groups every three consecutive epochs
  into a trial and verifies their pattern and code-chain 
  consistency to ensure they form a valid stimulus sequence 
  before extracting A, B, and C codes.
"""
def build_triplets(epoch_df: pd.DataFrame, sfreq: float) -> pd.DataFrame:
    """
    Group 480 epochs into 160 triplets and verify structure for each.
    Returns a DataFrame with one row per trial.
    """
    n_epochs   = len(epoch_df)
    n_triplets = n_epochs // 3
    remainder  = n_epochs % 3
 
    print(f"  {n_epochs} epochs -> {n_triplets} triplets (remainder: {remainder})")
    if remainder != 0:
        print(f"  WARNING: epoch count is not divisible by 3 — last {remainder} epochs will be ignored")
 
    rows = []
 
    for tid in range(n_triplets):
        # Pull the three epochs for this trial
        e0 = epoch_df.iloc[tid * 3].copy()      # A/B boundary epoch
        e1 = epoch_df.iloc[tid * 3 + 1].copy()  # B/C boundary epoch
        e2 = epoch_df.iloc[tid * 3 + 2].copy()  # C single epoch
 
        codes0 = literal_to_list(e0["all_codes_in_string"])
        codes1 = literal_to_list(e1["all_codes_in_string"])
        codes2 = literal_to_list(e2["all_codes_in_string"])
 
        # --- Pattern check ---
        # Each position must have the correct event_string_type
        pattern_ok = (
            str(e0["event_string_type"]) == "transition_2codes"
            and str(e1["event_string_type"]) == "transition_2codes"
            and str(e2["event_string_type"]) == "single_1code"
        )
 
        # --- Chain consistency check and code extraction ---
        chain_ok   = False
        A_code = B_code = C_code = np.nan
 
        if len(codes0) == 2 and len(codes1) == 2 and len(codes2) == 1:
            A, B_from_0      = codes0       # A ended, B started
            B_from_1, C_from_1 = codes1    # B ended, C started
            C_from_2           = codes2[0] # C alone
 
            # B must match across the boundary and C must match between epoch 1 and epoch 2
            chain_ok = (B_from_0 == B_from_1) and (C_from_1 == C_from_2)
 
            A_code = A
            B_code = B_from_0
            C_code = C_from_2
 
        # --- Match status for all three epochs ---
        # A triplet is only as trustworthy as its weakest epoch match
        all_matched = (
            e0["match_status"] == "matched"
            and e1["match_status"] == "matched"
            and e2["match_status"] == "matched"
        )
 
        # --- Timing extraction ---
        # Use matched_sample (from events.tsv) rather than mne_sample for timing
        # because matched_sample is the ground truth from the recording log
        A_onset = float(e0["matched_onset_sec"]) if pd.notna(e0["matched_onset_sec"]) else np.nan
        B_onset = float(e1["matched_onset_sec"]) if pd.notna(e1["matched_onset_sec"]) else np.nan
        C_onset = float(e2["matched_onset_sec"]) if pd.notna(e2["matched_onset_sec"]) else np.nan
 
        A_sample = float(e0["matched_sample"]) if pd.notna(e0["matched_sample"]) else np.nan
        B_sample = float(e1["matched_sample"]) if pd.notna(e1["matched_sample"]) else np.nan
        C_sample = float(e2["matched_sample"]) if pd.notna(e2["matched_sample"]) else np.nan
 
        # Convert sample differences to seconds using sfreq from audit_summary.json
        dAB_sec = (B_sample - A_sample) / sfreq if not (np.isnan(A_sample) or np.isnan(B_sample)) else np.nan
        dBC_sec = (C_sample - B_sample) / sfreq if not (np.isnan(B_sample) or np.isnan(C_sample)) else np.nan
 
        # --- Condition decoding from C_code ---
        # This is what the entire pipeline has been building toward.
        # The lower 2 bits of C_code encode the condition.
        # The upper 6 bits encode which snippet (item) was shown.
        C_last2      = last2_bits(C_code)     if not pd.isna(C_code) else np.nan
        C_item_id    = upper6_bits_item_id(C_code) if not pd.isna(C_code) else np.nan
        C_binary8    = bits8(C_code)          if not pd.isna(C_code) else None
        A_binary8    = bits8(A_code)          if not pd.isna(A_code) else None
        B_binary8    = bits8(B_code)          if not pd.isna(B_code) else None
 
        rows.append({
            # Trial identity
            "trial_id":             int(tid),
            "subject_id":           e0.get("subject_id", ""),
 
            # Epoch IDs — used later to index back into MNE for signal extraction
            "epoch0_id":            int(e0["epoch_id"]),
            "epoch1_id":            int(e1["epoch_id"]),
            "epoch2_id":            int(e2["epoch_id"]),   # THE ONE THAT MATTERS FOR EEGNet
 
            # MNE raw strings for traceability
            "epoch0_mne_string":    str(e0.get("mne_raw_string", "")),
            "epoch1_mne_string":    str(e1.get("mne_raw_string", "")),
            "epoch2_mne_string":    str(e2.get("mne_raw_string", "")),
 
            # QC flags
            "pattern_ok":           bool(pattern_ok),
            "chain_ok":             bool(chain_ok),
            "all_epochs_matched":   bool(all_matched),
 
            # Extracted trigger codes
            "A_code":               int(A_code) if not pd.isna(A_code) else np.nan,
            "B_code":               int(B_code) if not pd.isna(B_code) else np.nan,
            "C_code":               int(C_code) if not pd.isna(C_code) else np.nan,
 
            # Binary representations (useful for manual inspection)
            "A_binary8":            A_binary8,
            "B_binary8":            B_binary8,
            "C_binary8":            C_binary8,
 
            # Condition and item decoded from C_code
            "condition_bits":       int(C_last2)   if not pd.isna(C_last2)   else np.nan,
            "item_id":              int(C_item_id) if not pd.isna(C_item_id) else np.nan,
 
            # Timing
            "A_onset_sec":          A_onset,
            "B_onset_sec":          B_onset,
            "C_onset_sec":          C_onset,
            "dAB_sec":              dAB_sec,
            "dBC_sec":              dBC_sec,
        })
 
    return pd.DataFrame(rows)


# TIMING CONSISTENCY ANALYSIS

"""
Uses the mode of observed dBC values as the
 expected SOA rather than hardcoding 1.3046875s.
  This means if you run this on a different dataset with a
different SOA it self-calibrates rather than silently flagging every 
trial as wrong. block infers the expected SOA from the data and flags
trials where timing gaps—especially dBC—deviate, ensuring the critical
C-epoch remains properly time-locked for reliable analysis.(stimulus
onset asynchrony) of
exactly 1.3046875 seconds between each chunk. 
This value is not arbitrary —
it is 334 samples at 256 Hz (334/256 = 1.3046875). 
Every trial should show exactly this gap for both dAB and dBC.

"""
def flag_timing_outliers(
    trial_df: pd.DataFrame,
    tolerance_sec: float,
) -> pd.DataFrame:
    """
    Compute expected SOA from the data and flag trials where dBC deviates.
    Adds columns: expected_soa_sec, dBC_deviation_sec, timing_ok.
    """
    dBC_values = trial_df["dBC_sec"].dropna()
    dAB_values = trial_df["dAB_sec"].dropna()
 
    if len(dBC_values) == 0:
        trial_df["expected_soa_sec"]  = np.nan
        trial_df["dBC_deviation_sec"] = np.nan
        trial_df["dAB_deviation_sec"] = np.nan
        trial_df["timing_ok"]         = False
        return trial_df
 
    # Use mode (most common value) as expected SOA — robust to a few outliers
    expected_soa = float(dBC_values.round(6).mode().iloc[0])
 
    trial_df["expected_soa_sec"]  = expected_soa
    trial_df["dBC_deviation_sec"] = (trial_df["dBC_sec"] - expected_soa).abs()
    trial_df["dAB_deviation_sec"] = (trial_df["dAB_sec"] - expected_soa).abs()
    trial_df["timing_ok"] = trial_df["dBC_deviation_sec"] <= tolerance_sec
 
    n_timing_fail = int((~trial_df["timing_ok"]).sum())
    print(f"  Expected SOA: {expected_soa:.7f}s  (= {expected_soa*256:.1f} samples at 256 Hz)")
    print(f"  dBC range: {dBC_values.min():.7f}s to {dBC_values.max():.7f}s")
    print(f"  dAB range: {dAB_values.min():.7f}s to {dAB_values.max():.7f}s")
    print(f"  Timing failures (dBC deviation > {tolerance_sec}s): {n_timing_fail}")
 
    return trial_df

#OVERALL TRIAL QC FLAG
"""
This block combines all checks into one flag (trial_qc_pass) and marks which trials are valid, while keeping failed ones for transparency.

Explanation:

A trial passes only if all checks are true (pattern, chain, match, timing)
Failed trials are not deleted — they’re just marked as False
You later train your model using only valid trials, but still keep the rest to:
track what went wrong
ensure reproducibility without needing extra files
"""

def apply_overall_qc(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all QC flags into a single trial_qc_pass column.
    Also adds a human-readable qc_fail_reason string for failed trials.
    """
    trial_df["trial_qc_pass"] = (
        trial_df["pattern_ok"]
        & trial_df["chain_ok"]
        & trial_df["all_epochs_matched"]
        & trial_df["timing_ok"]
    )
 
    # Build a readable failure reason string for diagnostic use
    def fail_reason(row: pd.Series) -> str:
        reasons = []
        if not row["pattern_ok"]:
            reasons.append("pattern_fail")
        if not row["chain_ok"]:
            reasons.append("chain_fail")
        if not row["all_epochs_matched"]:
            reasons.append("match_fail")
        if not row["timing_ok"]:
            reasons.append("timing_fail")
        return "|".join(reasons) if reasons else "pass"
 
    trial_df["qc_fail_reason"] = trial_df.apply(fail_reason, axis=1)
 
    n_pass = int(trial_df["trial_qc_pass"].sum())
    n_fail = int((~trial_df["trial_qc_pass"]).sum())
    print(f"  Trial QC: {n_pass} pass, {n_fail} fail")
 
    if n_fail > 0:
        fail_breakdown = trial_df[~trial_df["trial_qc_pass"]]["qc_fail_reason"].value_counts()
        print(f"  Failure reasons:\n{fail_breakdown.to_string()}")
 
    return trial_df

#C_CODE DISTRIBUTION TABLE
"""
This block only reports how many trials occur for each unique C_code
 and is used to verify that all expected codes are present and correctly
  distributed before labeling. It avoids inferring condition or 
  manipulation from bits, since that mapping is handled later via the 
  verified codebook, ensuring the counts are purely descriptive and not 
  interpreted incorrectly.

"""
def build_c_code_distribution(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count trials per unique C_code. No condition interpretation — raw counts only.
    Returns one row per unique C_code observed in the trial data.
    """
    clean = trial_df[trial_df["C_code"].notna()].copy()
    clean["C_code"] = clean["C_code"].astype(int)
 
    rows = []
    for c_code, group in clean.groupby("C_code"):
        rows.append({
            "C_code":          int(c_code),
            "C_binary8":       bits8(c_code),
            "item_id":         upper6_bits_item_id(c_code),
            "trial_count":     len(group),
            "qc_pass_count":   int(group["trial_qc_pass"].sum())
                               if "trial_qc_pass" in group.columns else np.nan,
            "qc_fail_count":   int((~group["trial_qc_pass"]).sum())
                               if "trial_qc_pass" in group.columns else np.nan,
            "example_trial_ids": sorted(group["trial_id"].tolist())[:5],
        })
 
    df = pd.DataFrame(rows).sort_values("C_code").reset_index(drop=True)
 
    print(f"\n  Raw C_code counts (no condition labels — apply verified_codebook.json in Script 3):")
    print(f"  {'C_code':<10} {'item_id':<10} {'trials':<10} {'qc_pass':<10}")
    print(f"  {'-'*40}")
    for _, row in df.iterrows():
        print(f"  {int(row['C_code']):<10} {int(row['item_id']):<10} "
              f"{int(row['trial_count']):<10} {int(row['qc_pass_count']):<10}")
    print(f"\n  Total unique C_codes : {len(df)}")
    print(f"  Total trials counted : {int(df['trial_count'].sum())}")
 
    return df


#TRIPLET SUMMARY JSON
"""
exports QC results and C_code counts into a JSON file for Script 3, 
without attempting any label decoding.

Explanation:

It stores only validation stats and raw C_code counts
It avoids guessing conditions from bits (to prevent silent labeling errors)
Instead, it tells Script 3 to use a manually verified codebook for correct condition mapping
"""

def build_triplet_summary(
    subject: str,
    trial_df: pd.DataFrame,
    c_code_df: pd.DataFrame,
    sfreq: float,
    remainder: int,
) -> dict:
    """
    Build scalar QC summary for Script 3.
    Contains raw C_code counts and QC statistics only — no condition labels.
    """
    n_trials   = len(trial_df)
    n_qc_pass  = int(trial_df["trial_qc_pass"].sum()) \
                 if "trial_qc_pass" in trial_df.columns else n_trials
    n_qc_fail  = n_trials - n_qc_pass
 
    expected_soa = float(
        trial_df["expected_soa_sec"].dropna().iloc[0]
    ) if "expected_soa_sec" in trial_df.columns else np.nan
 
    # Raw C_code -> trial count (no condition interpretation)
    raw_c_code_counts: Dict[str, int] = {
        str(int(row["C_code"])): int(row["trial_count"])
        for _, row in c_code_df.iterrows()
        if not pd.isna(row["C_code"])
    }
 
    # QC breakdown by failure reason
    qc_fail_reasons: Dict[str, int] = {}
    if "qc_fail_reason" in trial_df.columns:
        fail_counts = (
            trial_df[~trial_df["trial_qc_pass"]]["qc_fail_reason"]
            .value_counts()
            .to_dict()
        )
        qc_fail_reasons = {str(k): int(v) for k, v in fail_counts.items()}
 
    return {
        "subject":            subject,
        "sfreq":              sfreq,
        "n_trials_total":     n_trials,
        "n_trials_qc_pass":   n_qc_pass,
        "n_trials_qc_fail":   n_qc_fail,
        "remainder_epochs":   remainder,
        "expected_soa_sec":   expected_soa,
        "qc_fail_reasons":    qc_fail_reasons,
        "raw_c_code_counts":  raw_c_code_counts,
        "condition_labeling": (
            "NOT applied in Script 2. "
            "Apply configs/verified_codebook.json in Script 3. "
            "Codebook must map each C_code to condition name and manipulation type "
            "as documented in the experiment files — do not derive from bit patterns."
        ),
    }

def main() -> None:
    args    = build_args()
    subject = args.subject
 
    audit_dir  = Path(args.audit_dir)
    output_dir = Path(args.output_dir) / subject
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"\nSubject    : {subject}")
    print(f"Audit dir  : {audit_dir / subject}")
    print(f"Output dir : {output_dir}")
 
    # ------------------------------------------------------------------
    section("STEP 1 — Load Script 1 outputs")
    epoch_df, sfreq = load_script1_outputs(audit_dir, subject)
 
    n_epochs   = len(epoch_df)
    n_triplets = n_epochs // 3
    remainder  = n_epochs % 3
 
    # ------------------------------------------------------------------
    section("STEP 2 — Build triplets and verify chain consistency")
    trial_df = build_triplets(epoch_df, sfreq)
 
    chain_pass = int(trial_df["chain_ok"].sum())
    pattern_pass = int(trial_df["pattern_ok"].sum())
    print(f"  Chain consistency: {chain_pass}/{n_triplets} triplets passed")
    print(f"  Pattern check:     {pattern_pass}/{n_triplets} triplets passed")
 
    # ------------------------------------------------------------------
    section("STEP 3 — Timing consistency analysis")
    trial_df = flag_timing_outliers(trial_df, args.timing_tolerance_sec)
 
    # ------------------------------------------------------------------
    section("STEP 4 — Apply overall trial QC")
    trial_df = apply_overall_qc(trial_df)
 
    # ------------------------------------------------------------------
    section("STEP 5 — Build C_code distribution table")
    c_code_df = build_c_code_distribution(trial_df)
 
    # ------------------------------------------------------------------
    section("STEP 6 — Save all outputs")
 
    # Main output — one row per trial
    save_csv(trial_df, output_dir / f"{subject}_trial_flow_table.csv")
 
    # C_code distribution reference
    save_csv(c_code_df, output_dir / f"{subject}_c_code_counts.csv")
 
    # Suspicious trials (QC failures) in their own file for easy inspection
    suspicious = trial_df[~trial_df["trial_qc_pass"]].copy()
    if len(suspicious) > 0:
        save_csv(suspicious, output_dir / f"{subject}_suspicious_trials.csv")
    else:
        print("  No suspicious trials — all 160 passed QC.")
 
    # Summary JSON for Script 3
    summary = build_triplet_summary(subject, trial_df, c_code_df, sfreq, remainder)
    save_json(summary, output_dir / f"{subject}_triplet_summary.json")
 
    # ------------------------------------------------------------------
    section("DONE")
 
    print(f"\n  Outputs written to: {output_dir}")
    print(f"\n  Files produced:")
    print(f"    1. {subject}_trial_flow_table.csv     <- PRIMARY output for Script 3")
    print(f"    2. {subject}_triplet_summary.json     <- scalar QC + auto codebook for Script 3")
    print(f"    3. {subject}_c_code_counts.csv        <- C_code distribution (use for manual stimulus check)")
    if len(suspicious) > 0:
        print(f"    4. {subject}_suspicious_trials.csv   <- trials that failed QC")
 
    n_pass = int(trial_df["trial_qc_pass"].sum())
    print(f"\n  QC summary:")
    print(f"    Trials total        : {n_triplets}")
    print(f"    Trials QC pass      : {n_pass}")
    print(f"    Trials QC fail      : {n_triplets - n_pass}")
    print(f"    Chain consistency   : {chain_pass}/{n_triplets}")
    print(f"    Pattern correct     : {pattern_pass}/{n_triplets}")
    print(f"    Expected SOA        : {trial_df['expected_soa_sec'].dropna().iloc[0]:.7f}s")
 
    print(f"\n  Raw C_code counts (condition labels applied in Script 3):")
    print(f"  {'C_code':<10} {'trials':<10}")
    for c_code, count in summary["raw_c_code_counts"].items():
        print(f"    {c_code:<10} {count:<10}")
 
    print(f"\n  Next step: run 03_visualize_and_label.py --subject {subject}")
 
    if args.strict_qc and (n_triplets - n_pass) > 0:
        raise RuntimeError(
            f"[STRICT QC] {n_triplets - n_pass} trials failed QC. "
            f"Check {subject}_suspicious_trials.csv before proceeding."
        )
 
 
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
 