#!/usr/bin/env python3
"""
stats_table_epoch0.py  —  Phase 7: one statistical table for the manuscript

Builds the eNeuro-style statistical table from outputs already saved by the
epoch0 pipeline. Nothing is re-read from EEG; every number is recomputed from the
saved per-subject values, so the table matches the reports exactly.

Reads:
  outputs/erp_epoch0/<set>/averaged_erp_table.csv        (primary + sensitivity sets)
  outputs/expertise_epoch0/step/expertise_effect_scores.csv
  outputs/cluster_epoch0/step/cluster_results.csv

Writes (outputs/stats_table_epoch0/):
  statistical_table.csv     one row per test, with a superscript letter
  statistical_table.md      same table, ready to paste into notes or a draft
  statistical_table.txt     plain-text version

USAGE:
  python stats_table_epoch0.py
"""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
OUT         = BASE_DIR / "outputs"
OUT_DIR     = OUT / "stats_table_epoch0"

PRIMARY_SET = "step"
SENS_SETS   = {"step_all54": "all 54, same trial rule",
               "thresh100":  "100 uV absolute rule"}
WF, SEM, SYN = "well_formed", "semantically_implausible", "syntactically_invalid"


def paired_row(a, b, label, sample, window):
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    return {"comparison": label, "sample": sample, "window": window, "n": n,
            "data_structure": f"Normal (n = {n})", "test": "Paired t",
            "estimate_uV": d.mean(), "statistic": f"t({n-1}) = {t:.2f}", "p": p,
            "effect_size": f"d = {d.mean()/d.std(ddof=1):.2f}",
            "ci95": f"[{lo:.2f}, {hi:.2f}] uV"}


def onesample_row(x, label, sample, window):
    n = len(x)
    t, p = stats.ttest_1samp(x, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=stats.sem(x))
    return {"comparison": label, "sample": sample, "window": window, "n": n,
            "data_structure": f"Normal (n = {n})", "test": "Paired t (effect vs 0)",
            "estimate_uV": x.mean(), "statistic": f"t({n-1}) = {t:.2f}", "p": p,
            "effect_size": f"d = {x.mean()/x.std(ddof=1):.2f}",
            "ci95": f"[{lo:.2f}, {hi:.2f}] uV"}


def welch_row(a, b, label, window):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    dfw = (va + vb) ** 2 / (va**2 / (len(a)-1) + vb**2 / (len(b)-1))
    diff = a.mean() - b.mean()
    lo, hi = stats.t.interval(0.95, dfw, loc=diff, scale=np.sqrt(va + vb))
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return {"comparison": label, "sample": f"Expert {len(a)} vs Novice {len(b)}",
            "window": window, "n": len(a) + len(b),
            "data_structure": f"Normal (n = {len(a)}, {len(b)})", "test": "Welch t",
            "estimate_uV": diff, "statistic": f"t({dfw:.1f}) = {t:.2f}", "p": p,
            "effect_size": f"d = {diff/pooled:.2f}", "ci95": f"[{lo:.2f}, {hi:.2f}] uV"}


def erp_rows(set_name, sample):
    f = OUT / "erp_epoch0" / set_name / "averaged_erp_table.csv"
    if not f.exists():
        print(f"  (skipped {set_name}: {f.name} not found)")
        return []
    w = pd.read_csv(f).pivot(index="subject", columns="condition",
                             values=["n400_uv", "p600_uv"]).dropna()
    return [
        paired_row(w[("n400_uv", SEM)].values, w[("n400_uv", WF)].values,
                   "N400: semantic - well-formed", sample, "300-500 ms"),
        paired_row(w[("p600_uv", SYN)].values, w[("p600_uv", WF)].values,
                   "P600: syntactic - well-formed", sample, "500-800 ms"),
    ]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. Primary ERP effects + cross-checks
    f = OUT / "erp_epoch0" / PRIMARY_SET / "averaged_erp_table.csv"
    w = pd.read_csv(f).pivot(index="subject", columns="condition",
                             values=["n400_uv", "p600_uv"]).dropna()
    n = len(w)
    prim = f"primary ({n})"
    rows += erp_rows(PRIMARY_SET, prim)
    rows.append(paired_row(w[("n400_uv", SYN)].values, w[("n400_uv", WF)].values,
                           "Cross-check: syntactic - well-formed", prim, "300-500 ms"))
    rows.append(paired_row(w[("p600_uv", SEM)].values, w[("p600_uv", WF)].values,
                           "Cross-check: semantic - well-formed", prim, "500-800 ms"))

    # 2. Sensitivity sets
    for s, desc in SENS_SETS.items():
        rows += erp_rows(s, f"sensitivity: {desc}")

    # 3. Expertise
    ef = OUT / "expertise_epoch0" / PRIMARY_SET / "expertise_effect_scores.csv"
    if ef.exists():
        e = pd.read_csv(ef)
        E, N = e[e["group"] == "Expert"], e[e["group"] == "Novice"]
        for col, label, win in (("n400_effect", "N400 effect", "300-500 ms"),
                                ("p600_effect", "P600 effect", "500-800 ms")):
            rows.append(onesample_row(E[col].values, f"{label}, Experts", f"Experts ({len(E)})", win))
            rows.append(onesample_row(N[col].values, f"{label}, Novices", f"Novices ({len(N)})", win))
            rows.append(welch_row(E[col].values, N[col].values,
                                  f"{label}: Expert vs Novice (interaction)", win))
        rows.append(welch_row(E["syn_n400_win"].values, N["syn_n400_win"].values,
                              "Syntactic - wf: Expert vs Novice (interaction)", "300-500 ms"))
    else:
        print(f"  (skipped expertise: {ef} not found)")

    # 4. Clusters (significant only)
    cf = OUT / "cluster_epoch0" / PRIMARY_SET / "cluster_results.csv"
    if cf.exists():
        c = pd.read_csv(cf)
        for _, r in c[c["p"] < 0.05].iterrows():
            rows.append({"comparison": f"Cluster: {r['test'][3:]}", "sample": prim,
                         "window": f"{r['start_ms']}-{r['end_ms']} ms", "n": n,
                         "data_structure": "Nonparametric", "test": "Cluster permutation",
                         "estimate_uV": np.nan,
                         "statistic": f"mass = {r['mass']:.1f}, peak t = {r['peak_t']:.2f}",
                         "p": r["p"], "effect_size": "-", "ci95": "-"})
    else:
        print(f"  (skipped clusters: {cf} not found)")

    df = pd.DataFrame(rows)
    letters = list(string.ascii_lowercase) + [a + b for a in "ab" for b in string.ascii_lowercase]
    df.insert(0, "letter", letters[:len(df)])
    df["estimate_uV"] = df["estimate_uV"].round(2)
    df["p_fmt"] = df["p"].apply(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}")

    df.to_csv(OUT_DIR / "statistical_table.csv", index=False)
    cols = ["letter", "comparison", "sample", "window", "data_structure", "test",
            "estimate_uV", "statistic", "p_fmt", "effect_size", "ci95"]
    head = ["", "Comparison", "Sample", "Window", "Data structure", "Test",
            "Estimate (uV)", "Statistic", "p", "Effect size", "95% CI"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df[cols].iterrows():
        md.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    (OUT_DIR / "statistical_table.md").write_text("\n".join(md), encoding="utf-8")
    txt = df[cols].rename(columns=dict(zip(cols, head))).to_string(index=False)
    (OUT_DIR / "statistical_table.txt").write_text(txt, encoding="utf-8")

    print(txt)
    print(f"\n[SAVED] {OUT_DIR}  ({len(df)} rows)")


if __name__ == "__main__":
    main()