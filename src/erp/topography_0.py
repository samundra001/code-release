#!/usr/bin/env python3
"""
topography_epoch0.py  —  scalp topography of the N400 and P600 effects (epoch0)

Uses exactly the same data as the ERP
statistics: the per-participant averages cached by figures_epoch0.py
(epoch0, Phase 2 kept participants, cleaned trials, `class` item removed).

Changes from the old script
  1. epoch0 (critical item) instead of epoch2.
  2. No separate artifact rule. The old 150 uV / 15% / 30 uV criteria are gone;
     participants and trials come from Phase 2 (Kuo & Prat's rule), so the
     topographies use the same 45 participants as every other analysis.
  3. Both effects: N400 (semantic - well-formed, 300-500 ms) and
     P600 (syntactic - well-formed, 500-800 ms).
  4. Montage = 'biosemi32' (the cap actually used) instead of standard_1020.
  5. Per-channel statistics: effect, t, p and FDR-corrected p at every electrode,
     so the distribution can be described with numbers, not only pictures.
  6. One simple distribution test per effect: is the effect larger at the
     posterior/central site than at Fz (paired t-test)?
  7. Contour lines are off by default (contourpy library problem on some systems);
     use --contours 6 if your environment supports them.

Outputs (outputs/topography_epoch0/<set>/):
  topo_main.png/.tiff        conditions + difference maps for both effects
  topo_timecourse.png/.tiff  difference maps in successive time windows
  channel_effects.csv        per-channel effect statistics
  topography_report.txt

USAGE:
  python topography_epoch0.py
  python topography_epoch0.py --set step_all54
  python topography_epoch0.py --contours 6
"""
import mne
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
CACHE_BASE  = BASE_DIR / "outputs" / "figures_epoch0"
OUT_BASE    = BASE_DIR / "outputs" / "topography_epoch0"

WF, SEM, SYN = 0, 1, 2          # condition order in the cache
EFFECTS = {
    "N400": {"cond": SEM, "win": (0.300, 0.500), "label": "Semantic", "ref": "Cz"},
    "P600": {"cond": SYN, "win": (0.500, 0.800), "label": "Syntactic", "ref": "Pz"},
}
TIME_WINDOWS = [(0.100, 0.200), (0.200, 0.300), (0.300, 0.400), (0.400, 0.500),
                (0.500, 0.650), (0.650, 0.800)]


def fdr_bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def win_mean(x, times, win):
    m = (times >= win[0]) & (times <= win[1])
    return x[..., m].mean(axis=-1)


def topo(ax, values, info, vlim, title, contours, mask=None):
    
    kw = {}
    if mask is not None:
        kw = dict(mask=mask, mask_params=dict(marker="o", markerfacecolor="white",
                                              markeredgecolor="black", markersize=3))
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-vlim, vlim), contours=contours, sensors=True, **kw)
    ax.set_title(title, fontsize=8, fontweight="bold")
    return im


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(out / f"{name}.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {name}.png / .tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="step")
    ap.add_argument("--contours", type=int, default=0)
    args = ap.parse_args()

    import mne
    mne.set_log_level("ERROR")

    cache = CACHE_BASE / args.set / "subject_erps.npz"
    if not cache.exists():
        sys.exit(f"ERROR: {cache} not found. Run figures_epoch0.py --set {args.set} first.")
    z = np.load(cache, allow_pickle=True)
    erps, times = z["erps"], z["times"]           # (n_subj, 3, n_ch, n_times)
    ch, subjects = list(z["ch_names"]), list(z["subjects"])
    n = len(subjects)
    out = OUT_BASE / args.set
    out.mkdir(parents=True, exist_ok=True)

    info = mne.create_info(ch, sfreq=256.0, ch_types="eeg")
    info.set_montage("biosemi32", on_missing="warn")
    print(f"Topography | set '{args.set}' | n = {n} | {len(ch)} channels | montage biosemi32")

    # ---- per-channel statistics ----
    rows, L = [], []
    sep = "=" * 74
    L += [sep, "SCALP TOPOGRAPHY OF THE N400 AND P600 EFFECTS (epoch0)", sep,
          f"Set: {args.set}   n = {n}   montage: biosemi32", ""]
    maps = {}
    sigs = {}
    for name, e in EFFECTS.items():
        diff = win_mean(erps[:, e["cond"]] - erps[:, WF], times, e["win"])   # (n_subj, n_ch)
        t, p = stats.ttest_1samp(diff, 0, axis=0)
        q = fdr_bh(p)
        mean = diff.mean(axis=0)
        maps[name] = mean
        sigs[name] = q < 0.05
        for i, c in enumerate(ch):
            rows.append({"effect": name, "channel": c, "effect_uV": round(mean[i], 3),
                         "t": round(t[i], 3), "p": p[i], "p_fdr": q[i]})
        order = np.argsort(mean) if name == "N400" else np.argsort(mean)[::-1]
        sig = (q < 0.05)
        L += [f"{name}: {e['label'].lower()} - well-formed, "
              f"{int(e['win'][0]*1000)}-{int(e['win'][1]*1000)} ms",
              f"  Largest effect at: " + ", ".join(f"{ch[i]} ({mean[i]:+.2f})" for i in order[:6]),
              f"  Significant electrodes (FDR q < 0.05): {sig.sum()}/{len(ch)}"]
        # simple distribution test: reference site vs Fz
        ri, fi = ch.index(e["ref"]), ch.index("Fz")
        tt, pp = stats.ttest_rel(diff[:, ri], diff[:, fi])
        dd = diff[:, ri] - diff[:, fi]
        dz = dd.mean() / dd.std(ddof=1)
        lo, hi = stats.t.interval(0.95, n - 1, loc=dd.mean(), scale=stats.sem(dd))
        L += [f"  {e['ref']} vs Fz: {mean[ri]:+.2f} vs {mean[fi]:+.2f} uV, "
              f"paired t({n-1}) = {tt:.2f}, p = {pp:.4f}, "
              f"d_z = {dz:.2f}, diff = {dd.mean():+.2f} uV, 95% CI [{lo:.2f}, {hi:.2f}]", ""]
    pd.DataFrame(rows).to_csv(out / "channel_effects.csv", index=False)

    # ---- figure 1: conditions + difference for each effect ----
    plt.rcParams.update({"font.size": 7})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for r, (name, e) in enumerate(EFFECTS.items()):
        wf = win_mean(erps[:, WF], times, e["win"]).mean(axis=0)
        vi = win_mean(erps[:, e["cond"]], times, e["win"]).mean(axis=0)
        vc = np.ceil(max(np.abs(wf).max(), np.abs(vi).max()))
        vd = np.ceil(np.abs(maps[name]).max())
        w = f"{int(e['win'][0]*1000)}–{int(e['win'][1]*1000)} ms"
        im1 = topo(axes[r, 0], wf, info, vc, f"Well-formed\n{w}", args.contours)
        topo(axes[r, 1], vi, info, vc, f"{e['label']}\n{w}", args.contours)
        im3 = topo(axes[r, 2], maps[name], info, vd,
                   f"{name} effect\n{e['label']} − well-formed", args.contours,
                   mask=sigs[name])
        fig.colorbar(im1, ax=axes[r, :2], shrink=0.6, label="µV")
        cb = fig.colorbar(im3, ax=axes[r, 2], shrink=0.6, label="µV")
        cb.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True, symmetric=True))
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}".replace("-", "\u2212")))
    save(fig, out, "topo_main")

    # ---- figure 2: difference maps over time ----
    fig, axes = plt.subplots(2, len(TIME_WINDOWS), figsize=(2.6 * len(TIME_WINDOWS), 5.6))
    for r, (name, e) in enumerate(EFFECTS.items()):
        series = [win_mean(erps[:, e["cond"]] - erps[:, WF], times, w).mean(axis=0)
                  for w in TIME_WINDOWS]
        v = np.ceil(max(np.abs(s).max() for s in series))
        for c, (w, s) in enumerate(zip(TIME_WINDOWS, series)):
            im = topo(axes[r, c], s, info, v, f"{int(w[0]*1000)}-{int(w[1]*1000)} ms",
                      args.contours)
        axes[r, 0].text(-0.35, 0.5, f"{e['label']}\n- well-formed", transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=axes[r, :], shrink=0.7, label="µV")
    fig.suptitle(f"Difference maps over time (n = {n})", fontweight="bold")
    save(fig, out, "topo_timecourse")

    rep = "\n".join(L + ["Per-channel values: channel_effects.csv", sep])
    (out / "topography_report.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep + f"\n\n[SAVED] {out}")


if __name__ == "__main__":
    main()