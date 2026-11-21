#!/usr/bin/env python3
"""
check_deposited_reference.py

Two checks on what reference the deposited .set files appear to use,
run on several participants (not just one):

  Check 1, flat channel: a single-electrode reference is ~0 uV at every
           sample. Prints the 5 channels with the smallest peak amplitude.
  Check 2, average reference: if the scalp channels were re-referenced to
           their own average, they sum to ~0 at every sample. Prints the
           largest absolute cross-channel mean, next to a typical amplitude.

USAGE (on itiger, in mne_stable):
  python check_deposited_reference.py
"""

import glob
from pathlib import Path

import numpy as np
import mne

mne.set_log_level("ERROR")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SUBJECTS = ["sub-001", "sub-002", "sub-008", "sub-030"]   # add more if you like
EOG = ["LIO", "ROC"]

for sub in SUBJECTS:
    files = sorted(glob.glob(str(DATA_DIR / sub / "eeg" / "*.set")))
    if not files:
        print(f"{sub}: no .set file found\n")
        continue

    ep = mne.read_epochs_eeglab(files[0], verbose=False)
    data = ep.get_data() * 1e6                      # (epochs, channels, times), uV
    chs = ep.ch_names
    scalp = [i for i, c in enumerate(chs) if c not in EOG]

    # Check 1: any flat channel?
    peak = np.abs(data).max(axis=(0, 2))
    smallest = sorted(zip(chs, peak), key=lambda x: x[1])[:5]

    # Check 2: do the scalp channels sum to ~0 at every sample?
    cross_mean = np.abs(data[:, scalp, :].mean(axis=1)).max()
    typical = np.abs(data[:, scalp, :]).mean()

    print(f"{sub}  ({Path(files[0]).name}, {len(chs)} channels)")
    print("  Check 1, 5 smallest peak amplitudes:")
    for ch, a in smallest:
        print(f"    {ch:6s} {a:12.4f} uV")
    print(f"  Check 2, max |mean across scalp channels| = {cross_mean:.6f} uV "
          f"(typical |amplitude| = {typical:.2f} uV)\n")

print("HOW TO READ THIS")
print("  One channel ~0 uV                 -> single-electrode reference (that channel)")
print("  Check 2 near 0 (<< typical)       -> average reference over the scalp channels")
print("  Neither                           -> reference is not a channel in the file and not the")
print("                                       scalp average; consistent with linked mastoids")
print("                                       (or another offline reference), but the file")
print("                                       alone cannot prove which.")