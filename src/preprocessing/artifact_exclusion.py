#!/usr/bin/env python3
"""
artifact_exclusion.py  —  Phase 2: which trials and participants to keep

Applies Kuo & Prat's (2024) artifact rule to the CRITICAL-WORD epoch
(epoch0_id) of every trial, removes the broken `class` item, and decides
which participants to keep.

Kuo & Prat's rule (their Methods):
  1. Reject a trial if Cz changes by more than 100 uV within any 200 ms
     sliding window.
  2. Reject a trial if any electrode goes beyond -65 uV or +65 uV.
  3. Drop a participant whose rejection rate is above 25%.
  Their kept participants averaged 6.8% rejection (by condition: WF 7.5,
  SEM 6.4, SYN 6.9, DBL 6.4).

Our implementation choices (state these in the Methods):
  - Rules are applied on the full stored epoch (-101.6 to +1203 ms), which
    matches their -100 to 1205 ms epoch, after a -100 to 0 ms baseline.
  - Rule 2 is read as a simple amplitude threshold (|V| > 65 uV).
  - Primary analysis includes the EOG channels in rule 2 ("any electrode");
    an EEG-only variant is reported alongside.
  - The `class` item is a stimulus problem, not an artifact: it is removed
    but does not count toward the 25% rejection rate.

Starting sample: the 54 participants in outputs/batch/pass_subjects.txt.

Outputs (outputs/artifact_epoch0/):
  trial_metrics.csv                 per-trial artifact measures (all variants
                                    can be recomputed from this file)
  trial_mask_<variant>.csv          keep/drop per trial, with reason
  kept_participants_<variant>.txt   same format as pass_subjects.txt, so it
                                    can be passed to the ERP script
  subject_rejection_rates.csv       per-participant rates for every variant
  artifact_summary.txt              the report

Variants:
  primary         65 uV, EOG included, drop participants > 25%
  eeg_only        65 uV, EEG channels only, drop participants > 25%
  all54           primary trial rejection, no participant dropped
  thresh100       100 uV, EOG included, drop participants > 25%

USAGE:
  python artifact_exclusion.py
  python artifact_exclusion.py --subjects sub-001 sub-002     (quick test)
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPTS_DIR = Path(__file__).parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
DATA_DIR    = BASE_DIR / "data"
LABELED_DIR = BASE_DIR / "outputs" / "labeled"
PASS_LIST   = BASE_DIR / "outputs" / "batch" / "pass_subjects.txt"
ITEM_EXCL   = BASE_DIR / "outputs" / "verification" / "item_exclusions.csv"
OUT_DIR     = BASE_DIR / "outputs" / "artifact_epoch0"
PARTICIPANTS_TSV_CANDIDATES = [
    DATA_DIR / "participants.tsv",
    BASE_DIR / "participants.tsv",
]

EOG_NAMES = {"LIO", "ROC", "RIO", "LOC", "VEOG", "HEOG"}  # this cap: LIO, ROC

EPOCH_COL = "epoch0_id"      # critical-word epoch
ORDER_COL = "C_onset_sec"    # presentation order
CODE_COL  = "C_code"

BASELINE      = (-0.1, 0.0)
STEP_WINDOW_S = 0.200        # ERPLAB-style step detector: moving window
STEP_STEP_S   = 0.050        # ... moved 50 ms at a time
CZ_WINDOW_S   = 0.200
CZ_P2P_UV     = 100.0
ABS_UV        = 65.0
SENS_ABS_UV   = 100.0
MAX_REJECT    = 0.25

PAPER_RATE    = 6.8
PAPER_BY_COND = {"well_formed": 7.5, "semantically_implausible": 6.4,
                 "syntactically_invalid": 6.9, "doubly_anomalous": 6.4}
CONDITIONS    = list(PAPER_BY_COND)

VARIANTS = {
    #  name            rule 2   threshold     include EOG  drop participants
    "primary":       ("abs",   ABS_UV,       True,        True),
    "eeg_only":      ("abs",   ABS_UV,       False,       True),
    "all54":         ("abs",   ABS_UV,       True,        False),
    "thresh100":     ("abs",   SENS_ABS_UV,  True,        True),
    "step":          ("step",  ABS_UV,       True,        True),
    "step_eeg_only": ("step",  ABS_UV,       False,       True),
    "step_all54":    ("step",  ABS_UV,       True,        False),   # NEW
}
# rule 2 "abs":  any sample beyond +/-65 uV
# rule 2 "step": ERPLAB-style voltage step -- in a 200 ms window moved in
#                50 ms steps, |mean(second half) - mean(first half)| > 65 uV


# ==============================================================================
# HELPERS
# ==============================================================================

def load_subjects(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    return [l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def find_set(subject: str) -> Path | None:
    eeg_dir = DATA_DIR / subject / "eeg"
    files = sorted(eeg_dir.glob("*.set")) if eeg_dir.exists() else []
    return files[0] if files else None


def split_channels(epochs):
    """Return (eeg_idx, eog_idx, cz_idx). EOG = typed 'eog' or named EOG/EXG."""
    names, types = epochs.ch_names, epochs.get_channel_types()
    eeg, eog = [], []
    for i, (n, t) in enumerate(zip(names, types)):
        if t == "eog" or n.upper() in EOG_NAMES or re.search(r"EOG|EXG", n, re.I):
            eog.append(i)
        elif t == "eeg":
            eeg.append(i)
    upper = {n.upper(): i for i, n in enumerate(names)}
    cz = upper.get("CZ")
    return eeg, eog, cz


def epoch_codes(epochs) -> np.ndarray:
    """Event code attached to each epoch, parsed as an integer (-1 if not)."""
    inv = {v: k for k, v in epochs.event_id.items()}
    out = []
    for c in epochs.events[:, 2]:
        m = re.search(r"\d+", str(inv.get(c, c)))
        out.append(int(m.group()) if m else -1)
    return np.array(out)


def epoch_names(epochs) -> np.ndarray:
    """Raw event name attached to each epoch, as stored in the .set file."""
    inv = {v: k for k, v in epochs.event_id.items()}
    return np.array([str(inv.get(c, c)) for c in epochs.events[:, 2]])


def max_step(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    ERPLAB-style step detector. x: (n_epochs, n_channels, n_times) in uV.
    For each 200 ms window (moved 50 ms at a time), take
    |mean(second half) - mean(first half)|; return the largest value per
    epoch across all windows and channels.
    """
    if x.shape[1] == 0:
        return np.full(x.shape[0], np.nan)
    w = int(round(STEP_WINDOW_S * sfreq))
    h = w // 2
    s = max(1, int(round(STEP_STEP_S * sfreq)))
    cs = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    starts = np.arange(0, x.shape[-1] - w + 1, s)
    first  = (cs[..., starts + h] - cs[..., starts]) / h
    second = (cs[..., starts + w] - cs[..., starts + w - h]) / h
    return np.abs(second - first).max(axis=(1, 2))


def load_expertise() -> pd.Series | None:
    for p in PARTICIPANTS_TSV_CANDIDATES:
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            id_col = next((c for c in df.columns if "participant" in c.lower()
                           or c.lower() in ("subject", "sub")), None)
            ex_col = next((c for c in df.columns if re.search(
                r"expert|group|level|proficien", c, re.I)), None)
            if id_col and ex_col:
                return df.set_index(id_col)[ex_col]
            print(f"  participants.tsv found but expertise column unclear: "
                  f"{list(df.columns)}")
            return None
    print("  participants.tsv not found; expertise split skipped.")
    return None


# ==============================================================================
# PER-SUBJECT ARTIFACT MEASURES
# ==============================================================================

def measure_subject(subject: str, excl: pd.DataFrame, chan_info: dict):
    import mne
    mne.set_log_level("ERROR")

    set_path = find_set(subject)
    csv_path = LABELED_DIR / f"{subject}_labeled_trial_table.csv"
    if set_path is None or not csv_path.exists():
        print(f"SKIP (missing {'.set' if set_path is None else 'label csv'})")
        return None

    epochs = mne.read_epochs_eeglab(str(set_path), verbose=False)
    epochs.apply_baseline(BASELINE, verbose=False)
    data = epochs.get_data() * 1e6                       # -> uV
    sfreq = epochs.info["sfreq"]

    eeg_idx, eog_idx, cz = split_channels(epochs)
    if not chan_info:
        chan_info.update(
            names=epochs.ch_names, eeg=[epochs.ch_names[i] for i in eeg_idx],
            eog=[epochs.ch_names[i] for i in eog_idx], sfreq=sfreq,
            tmin=epochs.tmin, tmax=epochs.tmax, n_epochs=len(epochs))
    if cz is None:
        print("FAIL (no Cz channel)")
        return None

    # Rule 1: max peak-to-peak at Cz within any 200 ms sliding window
    w = int(round(CZ_WINDOW_S * sfreq))
    win = sliding_window_view(data[:, cz, :], w, axis=1)
    cz_p2p = (win.max(-1) - win.min(-1)).max(axis=1)

    # Rule 2: max absolute voltage, EEG and EOG separately
    max_eeg = np.abs(data[:, eeg_idx, :]).max(axis=(1, 2))
    max_eog = (np.abs(data[:, eog_idx, :]).max(axis=(1, 2))
               if eog_idx else np.full(len(data), np.nan))

    # Rule 2, step reading: largest voltage step, EEG and EOG separately
    step_eeg = max_step(data[:, eeg_idx, :], sfreq)
    step_eog = max_step(data[:, eog_idx, :], sfreq)

    codes = epoch_codes(epochs)
    names_ev = epoch_names(epochs)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[EPOCH_COL]).sort_values(ORDER_COL).reset_index(drop=True)
    df["trial_order"] = np.arange(len(df))
    ep = df[EPOCH_COL].astype(int).values
    if ep.max() >= len(data):
        print(f"FAIL (epoch0_id {ep.max()} out of range for {len(data)} epochs)")
        return None

    bad_orders = set(excl.loc[excl["subject"] == subject, "trial_index"].astype(int))

    out = pd.DataFrame({
        "subject":     subject,
        "trial_order": df["trial_order"],
        "epoch0_id":   ep,
        "condition":   df["condition"],
        "C_code":      df[CODE_COL].astype(int),
        "epoch_code":  codes[ep],
        "epoch_event_name": names_ev[ep],
        "cz_p2p_uv":   cz_p2p[ep].round(2),
        "max_abs_eeg_uv": max_eeg[ep].round(2),
        "max_abs_eog_uv": max_eog[ep].round(2),
        "max_step_eeg_uv": step_eeg[ep].round(2),
        "max_step_eog_uv": step_eog[ep].round(2),
        "class_item":  df["trial_order"].isin(bad_orders),
    })
    print(f"OK  {len(out)} trials")
    return out


# ==============================================================================
# APPLY RULES
# ==============================================================================

def apply_variant(m: pd.DataFrame, method: str, abs_uv: float, use_eog: bool, drop: bool):
    pre = "max_abs" if method == "abs" else "max_step"
    amp = m[f"{pre}_eeg_uv"]
    if use_eog:
        amp = np.fmax(amp, m[f"{pre}_eog_uv"])
    rej_cz  = m["cz_p2p_uv"] > CZ_P2P_UV
    rej_abs = amp > abs_uv
    artifact = rej_cz | rej_abs

    mask = m[["subject", "trial_order", "epoch0_id", "condition", "C_code"]].copy()
    mask["reason"] = ""
    mask.loc[rej_abs, "reason"] = f"{method}>{abs_uv:g}uV"
    mask.loc[rej_cz, "reason"] = mask.loc[rej_cz, "reason"].where(
        mask.loc[rej_cz, "reason"] == "", mask.loc[rej_cz, "reason"] + "+") + "cz_p2p>100uV"
    mask.loc[m["class_item"], "reason"] = "class_item"

    # Rejection rate excludes the class item (stimulus problem, not artifact)
    valid = ~m["class_item"]
    rates = (artifact[valid].groupby(m.loc[valid, "subject"]).mean() * 100)
    kept = rates.index[rates <= MAX_REJECT * 100].tolist() if drop else rates.index.tolist()

    mask["keep"] = (~artifact) & (~m["class_item"]) & mask["subject"].isin(kept)
    return mask, rates, kept, rej_cz, rej_abs


def describe(name, m, mask, rates, kept, rej_cz, rej_abs, expertise, method="abs"):
    lines = [f"--- {name} " + "-" * (60 - len(name))]
    lines.append(f"  Participants kept   : {len(kept)} of {rates.size}")
    dropped = sorted(set(rates.index) - set(kept))
    if dropped:
        lines.append("  Dropped (>25%)      : " + ", ".join(
            f"{s} ({rates[s]:.1f}%)" for s in dropped))
    if expertise is not None and kept:
        grp = expertise.reindex(kept).value_counts(dropna=False).to_dict()
        lines.append(f"  Expertise split     : {grp}")

    km = m["subject"].isin(kept) & ~m["class_item"]
    art = (rej_cz | rej_abs)
    if km.any():
        lines.append(f"  Mean rejection (kept): {rates[kept].mean():.1f}%   "
                     f"(Kuo & Prat: {PAPER_RATE}%)")
        by_cond = art[km].groupby(m.loc[km, "condition"]).mean() * 100
        lines.append("  By condition        : " + "  ".join(
            f"{c[:3].upper()} {by_cond.get(c, np.nan):.1f}% (paper {PAPER_BY_COND[c]})"
            for c in CONDITIONS))
        lines.append(f"  Trials hit by rule  : Cz 100uV/200ms {rej_cz[km].mean()*100:.1f}%   "
                     f"{'abs threshold' if method == 'abs' else 'voltage step'} "
                     f"{rej_abs[km].mean()*100:.1f}%")
        lines.append(f"  Trials kept         : {int(mask['keep'].sum())}")
    lines.append("")
    return lines


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or load_subjects(PASS_LIST)
    if not ITEM_EXCL.exists():
        sys.exit(f"ERROR: {ITEM_EXCL} not found. Run assign_lists_flag_items.py first.")
    excl = pd.read_csv(ITEM_EXCL)

    print(f"\nPhase 2 artifact rule on {EPOCH_COL}  |  {len(subjects)} participants\n")
    chan_info, parts = {}, []
    for i, s in enumerate(subjects, 1):
        print(f"  [{i:>2}/{len(subjects)}] {s} ...", end=" ", flush=True)
        r = measure_subject(s, excl, chan_info)
        if r is not None:
            parts.append(r)
    if not parts:
        sys.exit("No participants processed.")

    m = pd.concat(parts, ignore_index=True)
    m.to_csv(OUT_DIR / "trial_metrics.csv", index=False)

    expertise = load_expertise()
    rep = ["=" * 76, "PHASE 2 — ARTIFACT REJECTION ON THE CRITICAL-WORD EPOCH (epoch0)",
           "=" * 76, ""]

    # Sanity checks
    rep.append("SANITY CHECKS")
    rep.append(f"  Epoch window        : {chan_info['tmin']*1000:.1f} to "
               f"{chan_info['tmax']*1000:.1f} ms, sfreq {chan_info['sfreq']:g} Hz, "
               f"{chan_info['n_epochs']} epochs/file")
    rep.append(f"  EEG channels ({len(chan_info['eeg'])})   : {', '.join(chan_info['eeg'])}")
    rep.append(f"  EOG channels ({len(chan_info['eog'])})    : {', '.join(chan_info['eog']) or 'NONE FOUND'}")
    if len(chan_info["eog"]) != 2:
        rep.append("  WARNING: expected 2 EOG channels. Check names above; "
                   "the primary/eeg_only split depends on this.")
    rep.append(f"  Participants        : {m['subject'].nunique()} processed")
    rep.append(f"  class item flagged  : {int(m['class_item'].sum())} trials "
               f"(expect 1 per participant)")
    known = m["epoch_code"] >= 0
    match = (m.loc[known, "epoch_code"] == m.loc[known, "C_code"] - 2)
    if known.any() and match.mean() > 0.5:
        rep.append(f"  epoch0 event code = C_code - 2 : {match.mean()*100:.1f}% of trials "
                   f"(confirms epoch0 is the critical word)")
    else:
        rep.append("  epoch0 event-code check not usable; raw event names on the")
        rep.append("  selected (epoch0) epochs, with counts:")
        counts = m["epoch_event_name"].value_counts()
        for nm, ct in counts.head(10).items():
            rep.append(f"      {nm!r:<30} {ct}")
        rep.append("  If these names mark word position (e.g. x0/x1/x2), every selected")
        rep.append("  epoch must show position 0. Anything else: STOP and check.")
    rep.append("")

    rates_all = {}
    for name, (method, abs_uv, use_eog, drop) in VARIANTS.items():
        mask, rates, kept, rej_cz, rej_abs = apply_variant(m, method, abs_uv, use_eog, drop)
        rates_all[name] = rates
        mask.to_csv(OUT_DIR / f"trial_mask_{name}.csv", index=False)
        (OUT_DIR / f"kept_participants_{name}.txt").write_text(
            f"# {name}: {method}>{abs_uv:g}uV, EOG={'yes' if use_eog else 'no'}, "
            f"drop>{int(MAX_REJECT*100)}%={'yes' if drop else 'no'}\n"
            + "\n".join(kept) + "\n")
        rep += describe(name, m, mask, rates, kept, rej_cz, rej_abs, expertise, method)

    pd.DataFrame(rates_all).round(2).rename_axis("subject").to_csv(
        OUT_DIR / "subject_rejection_rates.csv")

    rep += ["HOW TO READ THIS",
            f"  Kuo & Prat: 16 of 61 participants dropped (26%), kept participants",
            f"  averaged {PAPER_RATE}% rejection. Compare 'primary' (absolute reading of",
            "  their 65 uV rule) with 'step' (voltage-step reading). The reading that",
            "  lands near their numbers is the likelier match to what they ran.",
            "  Decide the primary rule from this report, before any ERP results.",
            "=" * 76]
    report = "\n".join(rep)
    (OUT_DIR / "artifact_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n[SAVED] {OUT_DIR}")


if __name__ == "__main__":
    main()