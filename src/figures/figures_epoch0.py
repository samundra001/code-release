#!/usr/bin/env python3
"""
figures_epoch0.py  —  Phase 5: figures at the critical word (epoch0)

Replaces plot_waveform.py / plot_topo.py / topo_by_expert.py for the corrected
pipeline. Uses exactly the same data as erp_epoch0.py: epoch0_id, the Phase 2
kept participants, and the trial mask (artifacts + `class` item removed).

Figures (outputs/figures_epoch0/<set>/):
  fig1_waveforms        Grand-average ERPs at Fz, Cz, Pz (the analyzed electrodes),
                        3 conditions, +/- 1 within-subject SE, N400 and P600
                        windows shaded. Negative plotted up (ERP convention).
  fig2_difference_waves Difference waves (sem - wf, syn - wf), Fz/Cz/Pz average,
                        with 95% CI bands.
  fig3_topographies     Scalp maps: N400 effect (sem - wf, 300-500 ms) and
                        P600 effect (syn - wf, 500-800 ms).
  fig4_expertise        syn - wf and sem - wf difference waves, Experts vs Novices.
  fig5_topo_expertise   P600 topography split by expertise.

Each figure is saved as .png (for viewing) and .tiff (300 dpi, for submission).

Per-subject ERPs are cached in subject_erps.npz, so re-plotting is fast. Delete the
cache (or use --recompute) after changing the participant set or trial mask.

USAGE:
  python figures_epoch0.py                (primary: step)
  python figures_epoch0.py --set step_all54
  python figures_epoch0.py --recompute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR  = Path(__file__).resolve().parent
BASE_DIR     = SCRIPTS_DIR.parent.parent
DATA_DIR     = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "outputs" / "artifact_epoch0"
OUT_BASE     = BASE_DIR / "outputs" / "figures_epoch0"
PARTICIPANTS_TSV = DATA_DIR / "participants.tsv"

EPOCH_COL = "epoch0_id"
TMIN, TMAX = -0.1, 0.8
BASELINE   = (-0.1, 0.0)
NON_EEG    = ["LIO", "ROC"]

ANALYSIS_CHANNELS = ["Fz", "Cz", "Pz"]           # same as erp_epoch0.py
N400_WIN = (0.300, 0.500)
P600_WIN = (0.500, 0.800)

WF, SEM, SYN = "well_formed", "semantically_implausible", "syntactically_invalid"
CONDS  = [WF, SEM, SYN]
LABELS = {WF: "Well-formed", SEM: "Semantic violation", SYN: "Syntactic violation"}
COLORS = {WF: "#222222", SEM: "#c0392b", SYN: "#2471a3"}
STYLES = {WF: "-", SEM: "--", SYN: "-."}
GROUP_COLORS = {"Expert": "#2471a3", "Novice": "#d35400"}


# ==============================================================================
# DATA
# ==============================================================================

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
        vals = df[c].dropna().astype(str).str.strip()
        if vals.isin(["Expert", "Novice"]).mean() > 0.9:
            return dict(zip(df[id_col].astype(str).str.strip(), vals))
    return {}


def compute_subject_erps(subjects, mask):
    """Return erps (n_subj, 3 conds, n_ch, n_times), times, ch_names, kept subjects."""
    import mne
    mne.set_log_level("ERROR")
    erps, kept, times, ch_names = [], [], None, None
    for i, s in enumerate(subjects, 1):
        print(f"  [{i:>2}/{len(subjects)}] {s} ...", end=" ", flush=True)
        files = sorted((DATA_DIR / s / "eeg").glob("*.set"))
        m = mask[mask["subject"] == s]
        if not files or m.empty:
            print("SKIP")
            continue
        ep = mne.read_epochs_eeglab(str(files[0]), verbose=False)
        ep.drop_channels([c for c in NON_EEG if c in ep.ch_names])
        ep.crop(TMIN, TMAX).apply_baseline(BASELINE, verbose=False)
        data = ep.get_data() * 1e6
        sub = []
        for c in CONDS:
            idx = m.loc[(m["condition"] == c) & m["keep"], EPOCH_COL].astype(int).values
            idx = idx[idx < len(data)]
            sub.append(data[idx].mean(axis=0) if len(idx) else None)
        if any(x is None for x in sub):
            print("SKIP (empty condition)")
            continue
        erps.append(np.stack(sub))
        kept.append(s)
        times, ch_names = ep.times, ep.ch_names
        print("OK")
    return np.stack(erps), times, ch_names, kept


def get_erps(set_name: str, recompute: bool):
    out = OUT_BASE / set_name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "subject_erps.npz"
    if cache.exists() and not recompute:
        z = np.load(cache, allow_pickle=True)
        print(f"Loaded cached per-subject ERPs: {cache}")
        return z["erps"], z["times"], list(z["ch_names"]), list(z["subjects"]), out
    subjects = load_list(ARTIFACT_DIR / f"kept_participants_{set_name}.txt")
    mask = pd.read_csv(ARTIFACT_DIR / f"trial_mask_{set_name}.csv")
    mask["keep"] = mask["keep"].astype(str).str.lower().isin(["true", "1"])
    erps, times, ch, kept = compute_subject_erps(subjects, mask)
    np.savez(cache, erps=erps, times=times, ch_names=np.array(ch), subjects=np.array(kept))
    return erps, times, ch, kept, out


# ==============================================================================
# HELPERS
# ==============================================================================

def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}.png / .tiff")


def within_se(x):
    """Within-subject SE (Cousineau-Morey). x: (n_subj, n_cond, n_times)."""
    n, k = x.shape[0], x.shape[1]
    norm = x - x.mean(axis=1, keepdims=True) + x.mean(axis=(0, 1), keepdims=True)
    return norm.std(axis=0, ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))


def ci95(d):
    from scipy import stats
    n = d.shape[0]
    return stats.t.ppf(0.975, n - 1) * d.std(axis=0, ddof=1) / np.sqrt(n)


def style_axis(ax, title, ylabel=True):
    ax.axvspan(N400_WIN[0]*1000, N400_WIN[1]*1000, color="#8e44ad", alpha=0.08)
    ax.axvspan(P600_WIN[0]*1000, P600_WIN[1]*1000, color="#e67e22", alpha=0.08)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7, ls=":")
    ax.invert_yaxis()
    ax.set_xlim(TMIN*1000, TMAX*1000)
    ax.set_xlabel("Time (ms)")
    if ylabel:
        ax.set_ylabel("Amplitude (µV, negative up)")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.2)


def window_mean(x, times, win):
    s = (times >= win[0]) & (times <= win[1])
    return x[..., s].mean(axis=-1)


# ==============================================================================
# FIGURES
# ==============================================================================

def fig_waveforms(erps, times, ch, out, n):
    t = times * 1000
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, c in zip(axes, ANALYSIS_CHANNELS):
        x = erps[:, :, ch.index(c), :]
        se = within_se(x)
        for k, cond in enumerate(CONDS):
            m = x[:, k].mean(axis=0)
            ax.plot(t, m, color=COLORS[cond], ls=STYLES[cond], lw=1.2, label=LABELS[cond])
            ax.fill_between(t, m - se[k], m + se[k], color=COLORS[cond], alpha=0.15, lw=0)
        style_axis(ax, c, ylabel=(c == ANALYSIS_CHANNELS[0]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig1_waveforms")


def fig_difference(erps, times, ch, out, n):
    t = times * 1000
    idx = [ch.index(c) for c in ANALYSIS_CHANNELS]
    x = erps[:, :, idx, :].mean(axis=2)                # (n, 3, times)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, cond in ((1, SEM), (2, SYN)):
        d = x[:, k] - x[:, 0]
        m, c = d.mean(axis=0), ci95(d)
        ax.plot(t, m, color=COLORS[cond], ls=STYLES[cond], lw=2,
                label=f"{LABELS[cond]} − well-formed")
        ax.fill_between(t, m - c, m + c, color=COLORS[cond], alpha=0.15, lw=0)
    style_axis(ax, f"Difference waves, mean of Fz/Cz/Pz (95% CI, n = {n})")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig2_difference_waves")


def make_info(ch):
    import mne
    info = mne.create_info(ch, sfreq=256.0, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="warn")
    return info


def topo_pair(fig, axes, maps, info, titles, vlim):
    import mne
    im = None
    for ax, m, title in zip(axes, maps, titles):
        im, _ = mne.viz.plot_topomap(m, info, axes=ax, show=False, cmap="RdBu_r",
                                     vlim=(-vlim, vlim), contours=0,
                                     sensors=True)
        ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def fig_topographies(erps, times, ch, out, n):
    info = make_info(ch)
    n400 = window_mean(erps[:, 1] - erps[:, 0], times, N400_WIN).mean(axis=0)
    p600 = window_mean(erps[:, 2] - erps[:, 0], times, P600_WIN).mean(axis=0)
    vlim = np.ceil(max(np.abs(n400).max(), np.abs(p600).max()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    im = topo_pair(fig, axes, [n400, p600], info,
                   ["N400 effect: semantic − well-formed\n300–500 ms",
                    "P600 effect: syntactic − well-formed\n500–800 ms"], vlim)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.04)
    cb.set_label("µV")
    fig.suptitle(f"Scalp distribution of effects (n = {n})", fontweight="bold")
    save(fig, out, "fig3_topographies")

    order = np.argsort(n400)[:5]
    order_p = np.argsort(p600)[::-1][:5]
    print("    Most negative N400-effect channels:",
          ", ".join(f"{ch[i]} ({n400[i]:+.2f})" for i in order))
    print("    Most positive P600-effect channels:",
          ", ".join(f"{ch[i]} ({p600[i]:+.2f})" for i in order_p))


def fig_expertise(erps, times, ch, subjects, groups, out):
    t = times * 1000
    idx = [ch.index(c) for c in ANALYSIS_CHANNELS]
    x = erps[:, :, idx, :].mean(axis=2)
    g = np.array([groups.get(s, "") for s in subjects])
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, (k, cond, title) in zip(axes, [(2, SYN, "Syntactic − well-formed (P600)"),
                                          (1, SEM, "Semantic − well-formed (N400)")]):
        for grp in ("Expert", "Novice"):
            sel = g == grp
            if sel.sum() < 2:
                continue
            d = x[sel, k] - x[sel, 0]
            m, c = d.mean(axis=0), ci95(d)
            ax.plot(t, m, color=GROUP_COLORS[grp], lw=2, label=f"{grp} (n = {sel.sum()})")
            ax.fill_between(t, m - c, m + c, color=GROUP_COLORS[grp], alpha=0.15, lw=0)
        style_axis(ax, title, ylabel=(k == 2))
        ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("Difference waves by expertise, mean of Fz/Cz/Pz (95% CI)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, out, "fig4_expertise")


def fig_topo_expertise(erps, times, ch, subjects, groups, out):
    info = make_info(ch)
    g = np.array([groups.get(s, "") for s in subjects])
    maps, titles = [], []
    for grp in ("Expert", "Novice"):
        sel = g == grp
        if sel.sum() < 2:
            return
        maps.append(window_mean(erps[sel, 2] - erps[sel, 0], times, P600_WIN).mean(axis=0))
        titles.append(f"{grp} (n = {sel.sum()})\nsyntactic − well-formed, 500–800 ms")
    vlim = np.ceil(max(np.abs(m).max() for m in maps))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    im = topo_pair(fig, axes, maps, info, titles, vlim)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.04)
    cb.set_label("µV")
    fig.suptitle("P600 effect by expertise", fontweight="bold")
    save(fig, out, "fig5_topo_expertise")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="step")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    erps, times, ch, subjects, out = get_erps(args.set, args.recompute)
    n = len(subjects)
    print(f"\nPlotting {n} participants, set '{args.set}' -> {out}")

    fig_waveforms(erps, times, ch, out, n)
    fig_difference(erps, times, ch, out, n)
    fig_topographies(erps, times, ch, out, n)

    groups = load_expertise()
    if groups:
        fig_expertise(erps, times, ch, subjects, groups, out)
        fig_topo_expertise(erps, times, ch, subjects, groups, out)
    else:
        print("  Expertise labels not found; skipped figures 4-5.")

    # Consistency check against erp_epoch0.py
    idx = [ch.index(c) for c in ANALYSIS_CHANNELS]
    x = erps[:, :, idx, :].mean(axis=2)
    n4 = window_mean(x[:, 1] - x[:, 0], times, N400_WIN).mean()
    p6 = window_mean(x[:, 2] - x[:, 0], times, P600_WIN).mean()
    print(f"\nCheck vs erp_epoch0.py: N400 effect {n4:+.2f} uV, P600 effect {p6:+.2f} uV "
          f"(primary run gave -1.86 and +3.44)")


if __name__ == "__main__":
    main()