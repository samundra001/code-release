#!/usr/bin/env python3
"""
check_epoch_timing.py

The deposited .set files are epoched (n_times samples per epoch) and the
events.tsv 'sample' column indexes the CONCATENATED epochs. This script
checks what that implies for the "constant 1.3046875 s" interval and
reports the only real timing information the deposit contains: the
position of secondary triggers inside each epoch.

Checks:
  1. Epoch geometry (n_times, pre-stimulus samples, sfreq) and whether
     n_times / sfreq equals the 1.3046875 s interval.
  2. Does events.tsv 'onset' equal sample / sfreq? (if yes, 'onset' is a
     position in the concatenated epochs, not elapsed recording time)
  3. Does every epoch have a time-locking trigger at relative sample 0?
     Is the spacing between consecutive time-locking triggers always one
     epoch length?
  4. Real timing: for triggers that are NOT at relative sample 0, how far
     from the epoch's zero point do they sit, by trial position and code?

USAGE (mne_stable environment):
  python check_epoch_timing.py
  python check_epoch_timing.py --data-dir D --pass-list L --out-dir O
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
CODE_RE = re.compile(r"\((\d+)\)")


def geometry(data_dir: Path, subject: str):
    """n_times, n_pre, sfreq from the first subject's .set, else defaults."""
    try:
        import mne
        mne.set_log_level("ERROR")
        f = sorted((data_dir / subject / "eeg").glob("*.set"))[0]
        ep = mne.read_epochs_eeglab(str(f), verbose=False)
        sfreq = float(ep.info["sfreq"])
        return (int(len(ep.times)), int(round(-ep.tmin * sfreq)), sfreq,
                f"read from {f.name}")
    except Exception as e:
        return 334, 26, 256.0, f"DEFAULTS (could not read .set: {type(e).__name__})"


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    rows = []
    for _, r in df.iterrows():
        for code in CODE_RE.findall(str(r["value"])):
            rows.append({"sample": int(r["sample"]), "onset": float(r["onset"]),
                         "code": int(code)})
    return pd.DataFrame(rows)


def analyse_subject(ev: pd.DataFrame, n_times: int, n_pre: int, sfreq: float, sub: str):
    ev = ev.copy()
    ev["epoch"] = ev["sample"] // n_times
    ev["rel"] = ev["sample"] - (ev["epoch"] * n_times + n_pre)
    ev["pos"] = ev["epoch"] % 3
    onset_gap = float(np.abs(ev["onset"] - ev["sample"] / sfreq).max())

    zero = ev[ev["rel"] == 0]
    n_epochs = int(ev["epoch"].max()) + 1
    per_epoch = zero.groupby("epoch").size()
    exactly_one = int((per_epoch == 1).sum())
    spacing = np.diff(np.sort(zero["sample"].values))

    zero_code = zero.drop_duplicates("epoch").set_index("epoch")["code"]
    other = ev[ev["rel"] != 0].copy()
    other["zero_code"] = other["epoch"].map(zero_code)
    other["subject"] = sub
    other["rel_ms"] = other["rel"] / sfreq * 1000.0
    return {
        "n_epochs": n_epochs, "exactly_one_zero": exactly_one,
        "spacing": spacing, "onset_gap": onset_gap,
        "n_events": len(ev), "n_other": len(other),
    }, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(BASE_DIR / "data"))
    ap.add_argument("--pass-list", default=str(BASE_DIR / "outputs" / "batch" / "pass_subjects.txt"))
    ap.add_argument("--out-dir", default=str(BASE_DIR / "outputs" / "verification"))
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects = [l.strip() for l in Path(args.pass_list).read_text().splitlines()
                if l.strip() and not l.startswith("#")]

    n_times, n_pre, sfreq, how = geometry(data_dir, subjects[0])
    interval = n_times / sfreq
    lines = ["=" * 76, "EPOCH TIMING CHECK", "=" * 76, "",
             f"Epoch geometry ({how}):",
             f"  n_times = {n_times} samples, pre-stimulus = {n_pre} samples "
             f"({n_pre / sfreq * 1000:.1f} ms), sfreq = {sfreq:g} Hz",
             f"  n_times / sfreq = {interval:.7f} s",
             f"  -> equals the reported 'B-to-C interval' of 1.3046875 s: "
             f"{abs(interval - 1.3046875) < 1e-9}", ""]

    others, spacings, onset_gaps = [], [], []
    tot_epochs = tot_one = 0
    done = 0
    for sub in subjects:
        files = sorted((data_dir / sub / "eeg").glob("*_events.tsv"))
        if not files:
            lines.append(f"  {sub}: no events.tsv, skipped")
            continue
        info, other = analyse_subject(load_events(files[0]), n_times, n_pre, sfreq, sub)
        tot_epochs += info["n_epochs"]
        tot_one += info["exactly_one_zero"]
        spacings.append(info["spacing"])
        onset_gaps.append(info["onset_gap"])
        others.append(other)
        done += 1

    allsp = np.concatenate(spacings) if spacings else np.array([])
    lines += [
        f"Participants analysed: {done}",
        "",
        "Check 2, is events.tsv 'onset' just sample / sfreq?",
        f"  largest |onset - sample/sfreq| over all participants = {max(onset_gaps):.9f} s",
        "  -> if ~0, 'onset' is a position in the concatenated epochs, "
        "not elapsed recording time",
        "",
        "Check 3, time-locking trigger of each epoch (relative sample 0):",
        f"  epochs with exactly one trigger at relative sample 0: {tot_one} of {tot_epochs}",
        f"  spacing between consecutive time-locking triggers (samples), unique values: "
        f"{sorted(set(int(x) for x in allsp))}",
        f"  -> if the only value is {n_times}, consecutive epochs are exactly one epoch "
        "length apart, so that spacing carries no information about presentation timing",
        "",
    ]

    oth = pd.concat(others, ignore_index=True) if others else pd.DataFrame()
    if len(oth):
        oth.to_csv(out_dir / "epoch_timing_secondary_triggers.csv", index=False)
        tab = (oth.groupby(["pos", "zero_code", "code"])["rel_ms"]
                  .agg(n="size", mean_ms="mean", sd_ms="std", min_ms="min", max_ms="max")
                  .reset_index())
        lines += ["Check 4, REAL timing: secondary triggers inside each epoch, "
                  "relative to that epoch's zero trigger.",
                  "  pos = epoch's position within the trial (0, 1, 2); "
                  "zero_code = trigger at time zero; code = the secondary trigger.",
                  tab.to_string(index=False, float_format=lambda x: f"{x:.1f}"), "",
                  "Distribution of offsets in samples (all participants):",
                  oth["rel"].value_counts().sort_index().to_string(), ""]
        lines += [
            "HOW TO READ CHECK 4",
            f"  Offsets near {0.9 * sfreq:.0f} samples (~900 ms) match one item period "
            "(700 ms display + 200 ms ISI).",
            "  A constant offset across conditions and trials means the triggers are "
            "spaced one fixed step apart. Whether the trigger at time zero marks the "
            "critical token still has to be confirmed in PYerp2.sce / the Methods.",
        ]
    lines.append("=" * 76)

    report = "\n".join(lines)
    print(report)
    (out_dir / "epoch_timing_check.txt").write_text(report, encoding="utf-8")
    print(f"\n[SAVED] {out_dir / 'epoch_timing_check.txt'}")


if __name__ == "__main__":
    main()