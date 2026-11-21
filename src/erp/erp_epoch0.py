#!/usr/bin/env python3
"""
erp_epoch0.py  —  Phase 3: averaged ERP analysis at the critical word

Same pipeline as average_erp_analysis.py, with these changes:
  1. Uses epoch0_id (critical word) instead of epoch2_id (critical word + 2).
  2. Participants come from the Phase 2 kept list for the chosen set.
  3. Only trials marked keep == True in the Phase 2 trial mask are used
     (artifact trials and the broken `class` item are removed).
  4. --set chooses the participant/trial set: step (primary), all54 or
     thresh100 (sensitivity). Results go to outputs/erp_epoch0/<set>/.
  5. Report header states the electrodes actually used (Fz, Cz, Pz).
  6. Paired tests match conditions by subject explicitly.
  7. Trial counts per subject x condition are saved and low counts flagged.

Unchanged: three conditions (well-formed, semantic, syntactic), epoch -100 to
800 ms, baseline -100 to 0 ms, N400 300-500 ms, P600 500-800 ms, average within
subject first, then paired t-tests across subjects.

Inputs (from Phase 2, outputs/artifact_epoch0/):
  kept_participants_<set>.txt
  trial_mask_<set>.csv       columns: subject, trial_order, epoch0_id,
                             condition, C_code, reason, keep

Outputs (outputs/erp_epoch0/<set>/):
  averaged_erp_table.csv     one row per subject x condition
  trial_counts.csv           trials used per subject x condition
  erp_statistics.txt         the statistics report
  grand_average_waveforms.csv

USAGE:
  python erp_epoch0.py                    (primary: step)
  python erp_epoch0.py --set all54        (sensitivity)
  python erp_epoch0.py --set thresh100    (sensitivity)
  python erp_epoch0.py --subjects sub-001 sub-002   (quick test)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPTS_DIR  = Path(__file__).parent
BASE_DIR     = SCRIPTS_DIR.parent.parent
DATA_DIR     = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "outputs" / "artifact_epoch0"
OUT_BASE     = BASE_DIR / "outputs" / "erp_epoch0"

EPOCH_COL = "epoch0_id"

TMIN     = -0.1
TMAX     =  0.8
BASELINE = (-0.1, 0.0)

N400_CHANNELS = ["Fz", "Cz", "Pz"]   # midline electrodes, as in Kuo & Prat
P600_CHANNELS = ["Fz", "Cz", "Pz"]

N400_TMIN, N400_TMAX = 0.300, 0.500
P600_TMIN, P600_TMAX = 0.500, 0.800

KEEP_CONDITIONS = ["well_formed", "semantically_implausible", "syntactically_invalid"]
CONDITION_SHORT = {"well_formed": "WF", "semantically_implausible": "SEM",
                   "syntactically_invalid": "SYN"}

LOW_TRIALS = 20   # flag subject x condition averages built from fewer trials


# ==============================================================================
# HELPERS
# ==============================================================================

def load_list(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run artifact_exclusion.py first.")
    return [l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def find_set(subject: str) -> Path | None:
    eeg_dir = DATA_DIR / subject / "eeg"
    files = sorted(eeg_dir.glob("*.set")) if eeg_dir.exists() else []
    return files[0] if files else None


def time_to_idx(times, tmin, tmax):
    return (int(np.searchsorted(times, tmin, side="left")),
            int(np.searchsorted(times, tmax, side="right")))


def resolve_channels(available, targets):
    upper = {c.upper(): c for c in available}
    return [upper[t.upper()] for t in targets if t.upper() in upper]


def cohens_d(diff: np.ndarray) -> float:
    return diff.mean() / diff.std(ddof=1)


# ==============================================================================
# PER-SUBJECT AVERAGING
# ==============================================================================

def process_subject(set_path: Path, subj_mask: pd.DataFrame,
                    n400_idx: list[int], p600_idx: list[int]) -> dict | None:
    import mne
    mne.set_log_level("ERROR")
    try:
        epochs = mne.read_epochs_eeglab(str(set_path), verbose=False)
        epochs = epochs.crop(tmin=TMIN, tmax=TMAX)
        epochs = epochs.apply_baseline(baseline=BASELINE, verbose=False)
        eeg = epochs.get_data() * 1e6          # (480, n_channels, n_times), uV
        times = epochs.times
        n4 = time_to_idx(times, N400_TMIN, N400_TMAX)
        p6 = time_to_idx(times, P600_TMIN, P600_TMAX)

        results = {}
        for cond in KEEP_CONDITIONS:
            idx = subj_mask.loc[(subj_mask["condition"] == cond) & subj_mask["keep"],
                                EPOCH_COL].astype(int).values
            idx = idx[idx < eeg.shape[0]]
            if len(idx) == 0:
                continue
            erp = eeg[idx].mean(axis=0)        # average trials -> (n_ch, n_times)
            results[cond] = {
                "n400_uv":  float(erp[n400_idx, n4[0]:n4[1]].mean()),
                "p600_uv":  float(erp[p600_idx, p6[0]:p6[1]].mean()),
                "n_trials": int(len(idx)),
                "waveform": erp,
                "times":    times,
            }
        return results
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        return None


# ==============================================================================
# STATISTICS
# ==============================================================================

def paired_test(wide: pd.DataFrame, measure: str, cond: str) -> dict:
    """cond minus well_formed on one measure, paired by subject."""
    a = wide[(measure, cond)].values
    b = wide[(measure, "well_formed")].values
    diff = a - b
    n = len(diff)
    t, p = stats.ttest_rel(a, b)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=diff.mean(), scale=stats.sem(diff))
    return {"n": n, "mean": diff.mean(), "se": diff.std(ddof=1) / np.sqrt(n),
            "ci": (lo, hi), "t": t, "p": p, "d": cohens_d(diff)}


def fmt_test(title, expect, r, label, sign_ok):
    return [
        "-" * 70, title, expect, "-" * 70,
        f"  Mean difference ({label})  : {r['mean']:+.4f} uV",
        f"  SE of difference           : {r['se']:.4f} uV",
        f"  95% CI                     : [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] uV",
        f"  t({r['n']-1})                     : {r['t']:.4f}",
        f"  p-value (two-tailed)       : {r['p']:.6f}",
        f"  Cohen's d                  : {r['d']:.4f}",
        f"  Significant (p<0.05)       : {'YES' if r['p'] < 0.05 else 'NO'}",
        f"  Direction as expected      : {'YES' if sign_ok else 'NO'}",
        "",
    ]


def run_statistics(table: pd.DataFrame, set_name: str) -> str:
    wide = table.pivot(index="subject", columns="condition", values=["n400_uv", "p600_uv"])
    complete = wide.dropna()
    n_drop = len(wide) - len(complete)
    n = len(complete)

    sep = "=" * 70
    lines = [
        sep, "AVERAGED ERP ANALYSIS AT THE CRITICAL WORD (epoch0)", sep, "",
        f"Participant set : {set_name}",
        f"Subjects        : {n}" + (f"  ({n_drop} dropped: missing a condition)" if n_drop else ""),
        f"Conditions      : well_formed, semantically_implausible, syntactically_invalid",
        f"N400            : mean amplitude at {N400_CHANNELS}, "
        f"{int(N400_TMIN*1000)}-{int(N400_TMAX*1000)} ms after the critical word",
        f"P600            : mean amplitude at {P600_CHANNELS}, "
        f"{int(P600_TMIN*1000)}-{int(P600_TMAX*1000)} ms after the critical word",
        f"Baseline        : {int(BASELINE[0]*1000)} to {int(BASELINE[1]*1000)} ms",
        f"Trials/condition: mean {table['n_trials'].mean():.1f}, "
        f"min {table['n_trials'].min()} (after artifact and item exclusion)",
        "",
        "-" * 70, "DESCRIPTIVE STATISTICS (mean +/- SE across subjects, uV)", "-" * 70,
        f"  {'Condition':<28} {'N400':>10} {'SE':>8} {'P600':>10} {'SE':>8}",
    ]
    for cond in KEEP_CONDITIONS:
        n4 = complete[("n400_uv", cond)]
        p6 = complete[("p600_uv", cond)]
        lines.append(f"  {cond:<28} {n4.mean():>+10.4f} {n4.sem():>8.4f} "
                     f"{p6.mean():>+10.4f} {p6.sem():>8.4f}")
    lines.append("")

    n400 = paired_test(complete, "n400_uv", "semantically_implausible")
    p600 = paired_test(complete, "p600_uv", "syntactically_invalid")
    x_n4 = paired_test(complete, "n400_uv", "syntactically_invalid")
    x_p6 = paired_test(complete, "p600_uv", "semantically_implausible")

    lines += fmt_test("N400 EFFECT: semantically_implausible vs well_formed",
                      "Expected: semantic more negative than well-formed",
                      n400, "sem - wf", n400["mean"] < 0)
    lines += fmt_test("P600 EFFECT: syntactically_invalid vs well_formed",
                      "Expected: syntactic more positive than well-formed",
                      p600, "syn - wf", p600["mean"] > 0)
    lines += fmt_test("CROSS-CHECK: syntactically_invalid vs well_formed, N400 window",
                      "Descriptive; not a primary test",
                      x_n4, "syn - wf", True)
    lines += fmt_test("CROSS-CHECK: semantically_implausible vs well_formed, P600 window",
                      "Descriptive; not a primary test",
                      x_p6, "sem - wf", True)

    n400_ok = n400["p"] < 0.05 and n400["mean"] < 0
    p600_ok = p600["p"] < 0.05 and p600["mean"] > 0
    lines += [sep, "SUMMARY", sep,
              f"  N400 (semantic)  : {'replicated' if n400_ok else 'not replicated'}",
              f"  P600 (syntactic) : {'replicated' if p600_ok else 'not replicated'}",
              "  Compare against Kuo & Prat (2024), 45 participants, linear mixed model.",
              "  Our test is a paired t-test on subject averages, so effect sizes are",
              "  comparable in direction and magnitude, not identical in form.",
              sep]
    return "\n".join(lines)


# ==============================================================================
# GRAND AVERAGE WAVEFORMS
# ==============================================================================

def grand_averages(all_waves: dict, times, n400_idx, p600_idx) -> pd.DataFrame:
    ga = {c: np.stack(w).mean(axis=0) for c, w in all_waves.items() if w}
    rows = []
    for t_i, t_ms in enumerate(times * 1000):
        row = {"time_ms": round(t_ms, 2)}
        for c, g in ga.items():
            s = CONDITION_SHORT[c]
            row[f"{s}_n400_cluster"] = float(g[n400_idx, t_i].mean())
            row[f"{s}_p600_cluster"] = float(g[p600_idx, t_i].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Averaged ERP at the critical word (epoch0)")
    ap.add_argument("--set", default="step",
                    help="Phase 2 set: step (primary), all54, thresh100, ...")
    ap.add_argument("--subjects", nargs="+", default=None)
    args = ap.parse_args()

    kept_path = ARTIFACT_DIR / f"kept_participants_{args.set}.txt"
    mask_path = ARTIFACT_DIR / f"trial_mask_{args.set}.csv"
    out_dir = OUT_BASE / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or load_list(kept_path)
    if not mask_path.exists():
        sys.exit(f"ERROR: {mask_path} not found. Run artifact_exclusion.py first.")
    mask = pd.read_csv(mask_path)
    mask["keep"] = mask["keep"].astype(str).str.lower().isin(["true", "1"])

    import mne
    mne.set_log_level("ERROR")
    first = next((find_set(s) for s in subjects if find_set(s)), None)
    if first is None:
        sys.exit("ERROR: no .set file found")
    ch_names = mne.read_epochs_eeglab(str(first), verbose=False).ch_names
    n400_names = resolve_channels(ch_names, N400_CHANNELS)
    p600_names = resolve_channels(ch_names, P600_CHANNELS)
    n400_idx = [ch_names.index(c) for c in n400_names]
    p600_idx = [ch_names.index(c) for c in p600_names]

    print(f"\n{'='*70}\n  ERP AT THE CRITICAL WORD (epoch0)  |  set: {args.set}\n{'='*70}")
    print(f"  Subjects : {len(subjects)}")
    print(f"  N400     : {n400_names}, {int(N400_TMIN*1000)}-{int(N400_TMAX*1000)} ms")
    print(f"  P600     : {p600_names}, {int(P600_TMIN*1000)}-{int(P600_TMAX*1000)} ms")
    print(f"  Output   : {out_dir}\n{'='*70}\n")

    rows, counts, failed = [], [], []
    all_waves = {c: [] for c in KEEP_CONDITIONS}
    times_ref = None

    for i, subj in enumerate(subjects, 1):
        print(f"  [{i:>2}/{len(subjects)}] {subj} ...", end=" ", flush=True)
        set_path = find_set(subj)
        subj_mask = mask[mask["subject"] == subj]
        if set_path is None or subj_mask.empty:
            print("SKIP (missing .set or no rows in trial mask)")
            failed.append(subj)
            continue
        res = process_subject(set_path, subj_mask, n400_idx, p600_idx)
        if not res:
            print("FAIL")
            failed.append(subj)
            continue
        times_ref = times_ref if times_ref is not None else next(iter(res.values()))["times"]
        summ = []
        for cond in KEEP_CONDITIONS:
            if cond not in res:
                continue
            e = res[cond]
            rows.append({"subject": subj, "condition": cond, "n_trials": e["n_trials"],
                         "n400_uv": round(e["n400_uv"], 4),
                         "p600_uv": round(e["p600_uv"], 4)})
            counts.append({"subject": subj, "condition": cond, "n_trials": e["n_trials"]})
            all_waves[cond].append(e["waveform"])
            summ.append(f"{CONDITION_SHORT[cond]}({e['n_trials']}): "
                        f"N400={e['n400_uv']:+.2f} P600={e['p600_uv']:+.2f}")
        print("OK  " + "  ".join(summ))

    if not rows:
        sys.exit("\n  ERROR: no subjects processed.")

    table = pd.DataFrame(rows)
    cnt = pd.DataFrame(counts)
    table.to_csv(out_dir / "averaged_erp_table.csv", index=False)
    cnt.to_csv(out_dir / "trial_counts.csv", index=False)

    low = cnt[cnt["n_trials"] < LOW_TRIALS]
    print(f"\n  Subjects processed : {table['subject'].nunique()}   failed: {len(failed)}")
    if len(low):
        print(f"  WARNING: {len(low)} subject x condition averages use < {LOW_TRIALS} trials:")
        print(low.to_string(index=False))

    report = run_statistics(table, args.set)
    (out_dir / "erp_statistics.txt").write_text(report, encoding="utf-8")

    if times_ref is not None:
        grand_averages(all_waves, times_ref, n400_idx, p600_idx).to_csv(
            out_dir / "grand_average_waveforms.csv", index=False)

    print(f"\n{report}\n\n  [SAVED] {out_dir}")


if __name__ == "__main__":
    main()