#!/usr/bin/env python3
"""
cluster_epoch0.py  —  Phase 6: cluster-based permutation tests over time (epoch0)

Instead of pre-chosen windows, tests every time point from 0 to 800 ms and corrects
for multiple comparisons with cluster-based permutation (Maris & Oostenveld, 2007).
Signal = mean of Fz, Cz, Pz, the same electrodes as the window analysis.

Reads the per-subject ERPs cached by figures_epoch0.py (no EEG re-reading):
  outputs/figures_epoch0/<set>/subject_erps.npz

Tests
  A. Condition effects (all participants, paired / sign-flip permutation):
       A1  semantic  - well-formed   (expect an N400-like negative cluster)
       A2  syntactic - well-formed   (expect a P600-like positive cluster)
  B. Expertise effects (Expert vs Novice, label permutation):
       B1  syntactic - well-formed difference wave
       B2  semantic  - well-formed difference wave

Clusters: adjacent time points with |t| above the p < .05 threshold, split by sign;
cluster mass = sum of t. Each cluster's p = share of permutations whose largest
|cluster mass| is at least as large.

Changes in this version (statistics unchanged; same seed and test order, so the
cluster results are identical to the previous run):
  - Two figures instead of one, sized for print (eNeuro two-column width):
      cluster_conditions  A1 + A2  -> Results, Time Course
      cluster_expertise   B1 + B2  -> Results, Expertise (replaces fig4_expertise)
  - Significant clusters shaded gray (purple/orange mark the ERP windows elsewhere).
  - Expertise groups drawn in teal/orange (blue means "syntactic" in Figs 1-2).
  - No figure title; proper minus signs; report lists ALL clusters.

Outputs (outputs/cluster_epoch0/<set>/):
  cluster_report.txt, cluster_results.csv
  cluster_conditions.png / .tiff
  cluster_expertise.png / .tiff

USAGE:
  python cluster_epoch0.py
  python cluster_epoch0.py --set step_all54 --n-perm 5000
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
from scipy import stats

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
FIG_DIR     = BASE_DIR / "outputs" / "figures_epoch0"
OUT_BASE    = BASE_DIR / "outputs" / "cluster_epoch0"
PARTICIPANTS_TSV = BASE_DIR / "data" / "participants.tsv"

CHANNELS     = ["Fz", "Cz", "Pz"]
TEST_WIN     = (0.0, 0.800)
CLUSTER_P    = 0.05
SEED         = 42
WF, SEM, SYN = 0, 1, 2          # condition order in the cache

MINUS = "−"
TITLES = {
    "A1 semantic - well-formed":            f"Semantic {MINUS} well-formed",
    "A2 syntactic - well-formed":           f"Syntactic {MINUS} well-formed",
    "B1 Expert vs Novice: syntactic - wf":  f"Syntactic {MINUS} well-formed",
    "B2 Expert vs Novice: semantic - wf":   f"Semantic {MINUS} well-formed",
}
GROUP_COLORS = {"Expert": "#1b7c7d", "Novice": "#d35400"}
CLUSTER_SHADE = dict(color="#9e9e9e", alpha=0.25, lw=0)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.unicode_minus": True})


# ==============================================================================
# STATISTICS
# ==============================================================================

def t_one_sample(x):
    """x: (n, T) -> t (T,)"""
    n = x.shape[0]
    return x.mean(0) / (x.std(0, ddof=1) / np.sqrt(n))


def t_two_sample(a, b):
    """Welch t, a: (na, T), b: (nb, T)"""
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb)


def find_clusters(t, thr):
    """Return list of (start, stop, mass) for runs with t > thr or t < -thr."""
    out = []
    for sign in (1, -1):
        above = sign * t > thr
        i = 0
        while i < len(t):
            if above[i]:
                j = i
                while j < len(t) and above[j]:
                    j += 1
                out.append((i, j, float(t[i:j].sum())))
                i = j
            else:
                i += 1
    return out


def max_mass(t, thr):
    c = find_clusters(t, thr)
    return max((abs(m) for _, _, m in c), default=0.0)


def paired_cluster_test(d, n_perm, rng):
    """d: (n, T) within-subject differences; sign-flip permutation."""
    n = d.shape[0]
    thr = stats.t.ppf(1 - CLUSTER_P / 2, n - 1)
    t_obs = t_one_sample(d)
    null = np.empty(n_perm)
    for k in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=(n, 1))
        null[k] = max_mass(t_one_sample(d * flips), thr)
    return t_obs, thr, null


def group_cluster_test(a, b, n_perm, rng):
    """a, b: (na, T), (nb, T); label permutation."""
    allx = np.vstack([a, b])
    na = len(a)
    thr = stats.t.ppf(1 - CLUSTER_P / 2, len(a) + len(b) - 2)
    t_obs = t_two_sample(a, b)
    null = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(len(allx))
        null[k] = max_mass(t_two_sample(allx[idx[:na]], allx[idx[na:]]), thr)
    return t_obs, thr, null


def summarize(name, t_obs, thr, null, times_ms):
    rows = []
    for s, e, mass in find_clusters(t_obs, thr):
        p = (np.sum(null >= abs(mass)) + 1) / (len(null) + 1)
        peak = s + int(np.argmax(np.abs(t_obs[s:e])))
        rows.append({"test": name, "start_ms": round(times_ms[s]),
                     "end_ms": round(times_ms[e - 1]),
                     "sign": "positive" if mass > 0 else "negative",
                     "mass": round(mass, 2), "peak_t": round(float(t_obs[peak]), 2),
                     "peak_ms": round(times_ms[peak]), "p": p})
    return sorted(rows, key=lambda r: r["start_ms"])


# ==============================================================================
# FIGURES
# ==============================================================================

def draw_panel(ax, ax2, name, curve, res, t_ms, show_ylabel):
    t_obs, thr, data = curve
    if isinstance(data, tuple):
        for arr, grp in ((data[0], "Expert"), (data[1], "Novice")):
            m, se = arr.mean(0), arr.std(0, ddof=1) / np.sqrt(len(arr))
            ax.plot(t_ms, m, color=GROUP_COLORS[grp], lw=1.2,
                    label=f"{grp}s (n = {len(arr)})")
            ax.fill_between(t_ms, m - se, m + se, color=GROUP_COLORS[grp], alpha=0.15, lw=0)
    else:
        m, se = data.mean(0), data.std(0, ddof=1) / np.sqrt(len(data))
        ax.plot(t_ms, m, color="#222222", lw=1.2)
        ax.fill_between(t_ms, m - se, m + se, color="#222222", alpha=0.15, lw=0)

    r = res[res["test"] == name] if len(res) else res
    for _, row in r.iterrows():
        if row["p"] < 0.05:
            for a in (ax, ax2):
                a.axvspan(row["start_ms"], row["end_ms"], **CLUSTER_SHADE)

    ax.axhline(0, color="black", lw=0.6)
    ax.invert_yaxis()
    ax.set_title(TITLES.get(name, name), fontweight="bold")
    ax.grid(alpha=0.2)
    ax2.plot(t_ms, t_obs, color="black", lw=0.9)
    for s in (1, -1):
        ax2.axhline(s * thr, color="gray", ls="--", lw=0.7)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim(t_ms[0], t_ms[-1])
    ax2.grid(alpha=0.2)
    ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
    if show_ylabel:
        ax.set_ylabel("Difference (µV, negative up)")
        ax2.set_ylabel("$t$")


def make_figure(names, curves, res, t_ms, out, fname, legend=False):
    fig, axes = plt.subplots(2, len(names), figsize=(7.2, 3.8),
                             gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 2:
        axes = axes.T
    for j, name in enumerate(names):
        draw_panel(axes[0, j], axes[1, j], name, curves[name], res, t_ms, j == 0)
    if legend:
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(w_pad=1.5)
    for ext in ("png", "tiff"):
        fig.savefig(out / f"{fname}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {fname}.png / .tiff")


# ==============================================================================
# MAIN
# ==============================================================================

def load_groups():
    if not PARTICIPANTS_TSV.exists():
        return {}
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    id_col = next((c for c in df.columns if "participant" in c.lower()), df.columns[0])
    for c in df.columns:
        v = df[c].dropna().astype(str).str.strip()
        if v.isin(["Expert", "Novice"]).mean() > 0.9:
            return dict(zip(df[id_col].astype(str).str.strip(), v))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="step")
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    cache = FIG_DIR / args.set / "subject_erps.npz"
    if not cache.exists():
        sys.exit(f"ERROR: {cache} not found. Run figures_epoch0.py --set {args.set} first.")
    z = np.load(cache, allow_pickle=True)
    erps, times = z["erps"], z["times"]
    ch, subjects = list(z["ch_names"]), list(z["subjects"])

    sel = (times >= TEST_WIN[0]) & (times <= TEST_WIN[1])
    t_ms = times[sel] * 1000
    x = erps[:, :, [ch.index(c) for c in CHANNELS], :].mean(axis=2)[:, :, sel]  # (n,3,T)
    d_sem = x[:, SEM] - x[:, WF]
    d_syn = x[:, SYN] - x[:, WF]

    out = OUT_BASE / args.set
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results, curves = [], {}

    print(f"Set '{args.set}', n = {len(subjects)}, {args.n_perm} permutations, "
          f"{TEST_WIN[0]*1000:.0f}-{TEST_WIN[1]*1000:.0f} ms, mean of {CHANNELS}")

    # Same order as before (A1, A2, B1, B2), so the random permutations and
    # therefore all cluster p-values are identical to the previous run.
    for name, d in (("A1 semantic - well-formed", d_sem),
                    ("A2 syntactic - well-formed", d_syn)):
        print(f"  {name} ...")
        t_obs, thr, null = paired_cluster_test(d, args.n_perm, rng)
        results += summarize(name, t_obs, thr, null, t_ms)
        curves[name] = (t_obs, thr, d)

    groups = load_groups()
    g = np.array([groups.get(s, "") for s in subjects])
    E, N = g == "Expert", g == "Novice"
    if E.sum() > 2 and N.sum() > 2:
        for name, d in (("B1 Expert vs Novice: syntactic - wf", d_syn),
                        ("B2 Expert vs Novice: semantic - wf", d_sem)):
            print(f"  {name} ...")
            t_obs, thr, null = group_cluster_test(d[E], d[N], args.n_perm, rng)
            results += summarize(name, t_obs, thr, null, t_ms)
            curves[name] = (t_obs, thr, (d[E], d[N]))
    else:
        print("  Expertise labels not found; tests B skipped.")

    res = pd.DataFrame(results)
    res.to_csv(out / "cluster_results.csv", index=False)

    # Report: every cluster above the forming threshold, significant or not
    L = ["=" * 76, "CLUSTER-BASED PERMUTATION TESTS AT THE CRITICAL WORD (epoch0)", "=" * 76,
         f"Set: {args.set}   n = {len(subjects)}"
         + (f"  (Experts {E.sum()}, Novices {N.sum()})" if E.sum() else ""),
         f"Signal: mean of {', '.join(CHANNELS)}   Time: 0-800 ms   "
         f"Permutations: {args.n_perm}   Cluster-forming p < {CLUSTER_P}", ""]
    for name in curves:
        L.append(name)
        r = res[res["test"] == name] if len(res) else res
        if r.empty:
            L.append("    no clusters above the forming threshold")
        for _, row in r.iterrows():
            sig = "SIGNIFICANT" if row["p"] < 0.05 else "n.s."
            L.append(f"    {row['start_ms']:>4}-{row['end_ms']:<4} ms  {row['sign']:<8} "
                     f"mass={row['mass']:+8.1f}  peak t={row['peak_t']:+.2f} at {row['peak_ms']} ms  "
                     f"p={row['p']:.4f}  {sig}")
        L.append("")
    L += ["How to read: a significant negative cluster in A1 around 300-500 ms is the N400;",
          "a significant positive cluster in A2 around 500-800 ms is the P600. In B, a",
          "positive cluster means the effect is larger (more positive) in Experts.", "=" * 76]
    report = "\n".join(L)
    (out / "cluster_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    # Figures
    a_names = [n for n in curves if n.startswith("A")]
    b_names = [n for n in curves if n.startswith("B")]
    make_figure(a_names, curves, res, t_ms, out, "cluster_conditions")
    if b_names:
        make_figure(b_names, curves, res, t_ms, out, "cluster_expertise", legend=True)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()