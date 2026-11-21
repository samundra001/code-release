#!/usr/bin/env python3
"""
assign_lists_and_flag_items.py

1. Assigns every participant to a stimulus list by matching their full
   160-trial C-code sequence (sorted by onset) against the 4 list files.
   Every participant should match exactly ONE list. This confirms labels
   and trial order for all participants, not just sub-001.
2. Finds the item whose critical line uses the reserved keyword `class`
   (invalid Python even in its "well-formed"/"semantic" versions) and writes
   its trial index for every participant to an exclusion file.

Outputs (outputs/verification/):
  participant_list_assignment.csv   subject, list, n_trials, match
  item_exclusions.csv               subject, list, trial_index, c_code, reason

Run Phase 2 (artifact exclusion) with item_exclusions.csv folded into the
trial mask.
"""

import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
PARSE_CSV = BASE / "outputs/verification/condition_label_parse_check_lines.csv"
OUT_DIR = BASE / "outputs/verification"

# EDIT THIS: glob for the per-subject label CSVs written by visual_and_label.py
LABEL_GLOB = "outputs/labeled/sub-*_labeled*.csv"
CODE_COL = "C_code"        # trigger code column in the label CSV
ONSET_COL = "C_onset_sec"  # used to put trials in presentation order

BAD_ITEM = re.compile(r"\bclass\b")  # reserved keyword used as a variable


def main():
    stim = pd.read_csv(PARSE_CSV)
    # File order within each list = presentation order (practice already removed)
    stim["trial_index"] = stim.groupby("list").cumcount()
    list_seqs = {l: g["c_code"].astype(int).tolist() for l, g in stim.groupby("list")}

    bad = stim[stim["critical_line"].str.contains(BAD_ITEM)]
    print(f"Items flagged by `class` keyword: {len(bad)} (expect 1 per list)")
    print(bad[["list", "trial_index", "c_code", "critical_line"]].to_string(index=False), "\n")

    files = sorted(BASE.glob(LABEL_GLOB))
    if not files:
        print(f"No label files found for {BASE / LABEL_GLOB}. Edit LABEL_GLOB.")
        return

    assign, excl = [], []
    for f in files:
        df = pd.read_csv(f)
        subj = re.search(r"sub-\d+", f.name).group(0)
        n_missing = df[CODE_COL].isna().sum()
        df = df.dropna(subset=[CODE_COL])
        seq = df.sort_values(ONSET_COL)[CODE_COL].astype(int).tolist()
        if n_missing:
            print(f"  {subj}: {n_missing} rows with no C_code skipped")
        hits = [l for l, s in list_seqs.items() if s == seq]
        match = "OK" if len(hits) == 1 else ("NONE" if not hits else "MULTIPLE")
        lst = hits[0] if len(hits) == 1 else None
        assign.append({"subject": subj, "list": lst, "n_trials": len(seq), "match": match})
        if lst is not None:
            for _, r in bad[bad["list"] == lst].iterrows():
                excl.append({"subject": subj, "list": lst,
                             "trial_index": int(r["trial_index"]),
                             "c_code": int(r["c_code"]),
                             "reason": "reserved keyword `class` used as variable"})

    a = pd.DataFrame(assign)
    print("Participants per list:", a["list"].value_counts().sort_index().to_dict())
    print("Match status:", a["match"].value_counts().to_dict())
    if (a["match"] != "OK").any():
        print("\nParticipants without a unique match:")
        print(a[a["match"] != "OK"].to_string(index=False))

    a.to_csv(OUT_DIR / "participant_list_assignment.csv", index=False)
    pd.DataFrame(excl).to_csv(OUT_DIR / "item_exclusions.csv", index=False)
    print(f"\n[SAVED] {OUT_DIR / 'participant_list_assignment.csv'}")
    print(f"[SAVED] {OUT_DIR / 'item_exclusions.csv'}  ({len(excl)} trials)")


if __name__ == "__main__":
    main()