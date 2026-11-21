#!/usr/bin/env python3
"""
fig_erp_boxplots.py  -  per-participant ERP effects (box plots + individual points)

Reads the per-participant averages written by erp_epoch0.py, so the plotted values
are exactly the ones tested in the paper (mean of Fz, Cz, Pz; retained trials only).

  Left panel : N400 window (300-500 ms): semantic - wf, syntactic - wf
  Right panel: P600 window (500-800 ms): semantic - wf, syntactic - wf

Output: outputs/figures_epoch0/<set>/fig2_erp_boxplots.png / .tiff
USAGE:  python fig_erp_boxplots.py            (primary set: step)
        python fig_erp_boxplots.py --set step_all54
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent.parent
ERP_BASE = BASE_DIR / "outputs" / "erp_epoch0"
OUT_BASE = BASE_DIR / "outputs" / "figures_epoch0"

WF, SEM, SYN = "well_formed", "semantically_implausible", "syntactically_invalid"
COLORS = {SEM: "#c0392b", SYN: "#2471a3"}   # same as Figure 1 (figures_epoch0.py)
LABELS = {SEM: "Semantic\n$-$ well-formed", SYN: "Syntactic\n$-$ well-formed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="step")
    args = ap.parse_args()

    table = pd.read_csv(ERP_BASE / args.set / "averaged_erp_table.csv")
    wide = table.pivot(index="subject", columns="condition",
                       values=["n400_uv", "p600_uv"]).dropna()
    n = len(wide)

    panels = [("n400_uv", "N400 window (300–500 ms)"),
              ("p600_uv", "P600 window (500–800 ms)")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    print(f"Set '{args.set}', n = {n} participants")
    for ax, (col, title) in zip(axes, panels):
        data = [wide[(col, c)] - wide[(col, WF)] for c in (SEM, SYN)]
        bp = ax.boxplot(data, widths=0.5, whis=1.5, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1.5))
        for patch, c in zip(bp["boxes"], (SEM, SYN)):
            patch.set_facecolor(COLORS[c]); patch.set_alpha(0.25)
            patch.set_edgecolor(COLORS[c])
        for i, (d, c) in enumerate(zip(data, (SEM, SYN)), start=1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=12,
                       color=COLORS[c], alpha=0.8, edgecolor="none", zorder=3)
            ax.scatter(i, d.mean(), marker="D", s=40, color="white",
                       edgecolor="black", linewidth=1.2, zorder=4)   # mean (tested value)
            print(f"  {col[:4].upper()} {c[:3]} - wf: mean {d.mean():+.2f} uV, "
                  f"median {d.median():+.2f} uV")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xticks([1, 2]); ax.set_xticklabels([LABELS[SEM], LABELS[SYN]], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Effect (µV)")

    out = OUT_BASE / args.set
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "fig2_erp_boxplots.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig2_erp_boxplots.tiff", dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out / 'fig2_erp_boxplots.png'}")


if __name__ == "__main__":
    main()