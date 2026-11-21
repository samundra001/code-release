#!/usr/bin/env python3
"""
features_epoch0.py  —  Phase 8: single-trial ERP features at the critical word

Updated version of the old feature extraction script for the corrected pipeline.

Changes from the old script
  1. epoch0_id (critical word) instead of epoch2_id (critical word + 2).
  2. Participants = Phase 2 kept list; trials = Phase 2 trial mask (keep == True),
     so artifact trials and the broken `class` item are already removed.
  3. The old ad hoc 100 uV rejection on feature values is removed: artifact
     rejection now happens once, in Phase 2, with Kuo & Prat's rule.
  4. Feature set redefined around the effects actually present (N400, P600), on
     the same electrodes as the ERP analysis (Fz, Cz, Pz). LAN and right-frontal
     features are removed.
  5. Features are fixed HERE, before any classifier is run on epoch0 data.
  6. Output goes to a new folder; the old features_extended/ stays untouched.

Feature set (13 per trial, all from Fz/Cz/Pz, -100 to 0 ms baseline):
   0 early_mean        mean 100-300 ms, avg of Fz/Cz/Pz   (control window)
   1 n400_mean         mean 300-500 ms, avg of Fz/Cz/Pz
   2 n400_peak_amp     most negative value 300-500 ms
   3 n400_peak_lat_ms  latency of that minimum
   4 p600_mean         mean 500-800 ms, avg of Fz/Cz/Pz
   5 p600_peak_amp     most positive value 500-800 ms
   6 p600_peak_lat_ms  latency of that maximum
   7 n400_Fz   8 n400_Cz   9 n400_Pz      per-electrode means, 300-500 ms
  10 p600_Fz  11 p600_Cz  12 p600_Pz      per-electrode means, 500-800 ms

Labels: well_formed = 0, semantically_implausible = 1, syntactically_invalid = 2
(doubly anomalous trials are not used, as before).

Outputs (outputs/features_epoch0/<set>/):
  X_features.npy, y_labels.npy, trial_metadata.csv, feature_names.txt,
  feature_summary.txt

USAGE:
  python features_epoch0.py                (primary: step)
  python features_epoch0.py --set step_all54
  python features_epoch0.py --subjects sub-001 sub-002   (quick test)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── CONFIG ────────────────────────────────────────────────────────────────

SCRIPTS_DIR  = Path(__file__).resolve().parent
BASE_DIR     = SCRIPTS_DIR.parent.parent
DATA_DIR     = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "outputs" / "artifact_epoch0"
OUT_BASE     = BASE_DIR / "outputs" / "features_epoch0"
PARTICIPANTS_TSV = DATA_DIR / "participants.tsv"

EPOCH_COL = "epoch0_id"
TMIN, TMAX = -0.1, 0.8
BASELINE   = (-0.1, 0.0)

CHANNELS = ["Fz", "Cz", "Pz"]                 # same as the ERP analysis
EARLY_WIN = (0.100, 0.300)
N400_WIN  = (0.300, 0.500)
P600_WIN  = (0.500, 0.800)

FEATURE_NAMES = [
    "early_mean", "n400_mean", "n400_peak_amp", "n400_peak_lat_ms",
    "p600_mean", "p600_peak_amp", "p600_peak_lat_ms",
    "n400_Fz", "n400_Cz", "n400_Pz", "p600_Fz", "p600_Cz", "p600_Pz",
]

KEEP_CONDITIONS = ["well_formed", "semantically_implausible", "syntactically_invalid"]
LABEL_MAP = {"well_formed": 0, "semantically_implausible": 1, "syntactically_invalid": 2}


# ── HELPERS ───────────────────────────────────────────────────────────────

def load_list(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run artifact_exclusion.py first.")
    return [l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def load_expertise() -> dict[str, str]:
    if not PARTICIPANTS_TSV.exists():
        return {}
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    id_col = next((c for c in df.columns if "participant" in c.lower()), df.columns[0])
    for c in df.columns:
        v = df[c].dropna().astype(str).str.strip()
        if v.isin(["Expert", "Novice"]).mean() > 0.9:
            return dict(zip(df[id_col].astype(str).str.strip(), v))
    return {}


def win_idx(times, win):
    return (int(np.searchsorted(times, win[0], "left")),
            int(np.searchsorted(times, win[1], "right")))


# ── PER-SUBJECT FEATURES ──────────────────────────────────────────────────

def extract_subject(subject: str, set_path: Path, subj_mask: pd.DataFrame):
    import mne
    mne.set_log_level("ERROR")

    ep = mne.read_epochs_eeglab(str(set_path), verbose=False)
    ep.crop(TMIN, TMAX).apply_baseline(BASELINE, verbose=False)
    data, times = ep.get_data() * 1e6, ep.times
    upper = {c.upper(): i for i, c in enumerate(ep.ch_names)}
    ch_idx = [upper[c.upper()] for c in CHANNELS]

    trials = subj_mask[subj_mask["keep"] & subj_mask["condition"].isin(KEEP_CONDITIONS)]
    trials = trials[trials[EPOCH_COL].astype(int) < len(data)].reset_index(drop=True)
    if trials.empty:
        return None

    x = data[trials[EPOCH_COL].astype(int).values][:, ch_idx, :]   # (n, 3, T)
    avg = x.mean(axis=1)                                            # (n, T)

    e0, e1 = win_idx(times, EARLY_WIN)
    n0, n1 = win_idx(times, N400_WIN)
    p0, p1 = win_idx(times, P600_WIN)

    n_seg, p_seg = avg[:, n0:n1], avg[:, p0:p1]
    X = np.column_stack([
        avg[:, e0:e1].mean(1),
        n_seg.mean(1), n_seg.min(1), times[n0:n1][n_seg.argmin(1)] * 1000,
        p_seg.mean(1), p_seg.max(1), times[p0:p1][p_seg.argmax(1)] * 1000,
        x[:, :, n0:n1].mean(2),                                     # n400 Fz, Cz, Pz
        x[:, :, p0:p1].mean(2),                                     # p600 Fz, Cz, Pz
    ]).astype(np.float32)

    y = trials["condition"].map(LABEL_MAP).values.astype(np.int32)
    meta = pd.DataFrame({
        "subject": subject,
        "trial_order": trials["trial_order"].values,
        "epoch0_id": trials[EPOCH_COL].values,
        "condition": trials["condition"].values,
        "label": y,
    })
    for i, f in enumerate(FEATURE_NAMES):
        meta[f] = np.round(X[:, i], 4)
    return X, y, meta


# ── REPORT ────────────────────────────────────────────────────────────────

def summary(X, y, meta, failed, set_name, out_dir):
    sep, dash = "=" * 72, "-" * 72
    L = [sep, "SINGLE-TRIAL FEATURES AT THE CRITICAL WORD (epoch0)", sep, "",
         f"Participant set     : {set_name}",
         f"Subjects processed  : {meta['subject'].nunique()}   failed/skipped: {len(failed)}",
         f"Trials              : {len(X)}   features per trial: {X.shape[1]}",
         f"Electrodes          : {', '.join(CHANNELS)}",
         f"Windows             : early {EARLY_WIN}, N400 {N400_WIN}, P600 {P600_WIN} s", ""]
    if "group" in meta and meta["group"].ne("").any():
        g = meta.drop_duplicates("subject")["group"].value_counts().to_dict()
        L += [f"Expertise           : {g}", ""]

    L += [dash, "TRIAL COUNTS", dash]
    for cond, lab in LABEL_MAP.items():
        L.append(f"  {cond:<28} label={lab}  n={int((y == lab).sum())}")
    L.append("")

    L += [dash, "FEATURE MEANS BY CONDITION (all trials pooled)", dash]
    for i, f in enumerate(FEATURE_NAMES):
        unit = "ms" if "lat" in f else "uV"
        L.append(f"  {f:<18} wf={X[y==0,i].mean():+8.3f}  sem={X[y==1,i].mean():+8.3f}  "
                 f"syn={X[y==2,i].mean():+8.3f} {unit}")
    L.append("")

    n4 = FEATURE_NAMES.index("n400_mean")
    p6 = FEATURE_NAMES.index("p600_mean")
    L += [dash, "DIRECTION CHECKS (should match the ERP analysis)", dash,
          f"  N400: sem < wf ?  {'YES' if X[y==1,n4].mean() < X[y==0,n4].mean() else 'NO'}"
          f"   (sem - wf = {X[y==1,n4].mean() - X[y==0,n4].mean():+.2f} uV)",
          f"  P600: syn > wf ?  {'YES' if X[y==2,p6].mean() > X[y==0,p6].mean() else 'NO'}"
          f"   (syn - wf = {X[y==2,p6].mean() - X[y==0,p6].mean():+.2f} uV)",
          "  Pooled-trial differences can differ slightly from the subject-average",
          "  ERP results (-1.86 / +3.44 uV) because subjects contribute unequal trials.",
          sep]
    rep = "\n".join(L)
    (out_dir / "feature_summary.txt").write_text(rep, encoding="utf-8")
    return rep


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Single-trial ERP features at epoch0")
    ap.add_argument("--set", default="step")
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()

    out_dir = OUT_BASE / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or load_list(ARTIFACT_DIR / f"kept_participants_{args.set}.txt")
    mask = pd.read_csv(ARTIFACT_DIR / f"trial_mask_{args.set}.csv")
    mask["keep"] = mask["keep"].astype(str).str.lower().isin(["true", "1"])
    groups = load_expertise()

    print(f"\n  Features at epoch0 | set '{args.set}' | {len(subjects)} subjects\n")
    allX, ally, allm, failed = [], [], [], []
    for i, s in enumerate(subjects, 1):
        print(f"  [{i:>2}/{len(subjects)}] {s} ...", end=" ", flush=True)
        files = sorted((DATA_DIR / s / "eeg").glob("*.set"))
        sm = mask[mask["subject"] == s]
        if not files or sm.empty:
            print("SKIP"); failed.append(s); continue
        try:
            r = extract_subject(s, files[0], sm)
        except Exception as e:
            print(f"FAIL ({e})"); failed.append(s); continue
        if r is None:
            print("SKIP (no trials)"); failed.append(s); continue
        X, y, m = r
        m["group"] = groups.get(s, "")
        allX.append(X); ally.append(y); allm.append(m)
        print(f"OK  n={len(X):>3}  wf={(y==0).sum()} sem={(y==1).sum()} syn={(y==2).sum()}")

    if not allX:
        sys.exit("\n  ERROR: no subjects processed.")
    X, y = np.concatenate(allX), np.concatenate(ally)
    meta = pd.concat(allm, ignore_index=True)

    np.save(out_dir / "X_features.npy", X)
    np.save(out_dir / "y_labels.npy", y)
    meta.to_csv(out_dir / "trial_metadata.csv", index=False)
    (out_dir / "feature_names.txt").write_text("\n".join(FEATURE_NAMES), encoding="utf-8")

    print(f"\n  X {X.shape}   y {y.shape}\n")
    print(summary(X, y, meta, failed, args.set, out_dir))
    print(f"\n  [SAVED] {out_dir}")


if __name__ == "__main__":
    main()