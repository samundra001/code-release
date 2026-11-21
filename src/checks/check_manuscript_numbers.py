#!/usr/bin/env python3
"""
check_manuscript_numbers.py  --  recompute every key number in the manuscript
from the saved pipeline outputs and compare it with the value in the paper.

Nothing is re-read from EEG and nothing is overwritten; the script only reads
the CSV / TXT / NPY files that the analysis scripts already saved.

USAGE
    python check_manuscript_numbers.py
    python check_manuscript_numbers.py --base /path/to/repository

Output: one line per check   PASS / FAIL / MISSING   plus a summary.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

WF, SEM, SYN = "well_formed", "semantically_implausible", "syntactically_invalid"
results = []


def check(section, label, got, expected, tol):
    if got is None or (isinstance(got, float) and np.isnan(got)):
        results.append((section, label, "MISSING", "-", expected))
        return
    ok = abs(got - expected) <= tol
    results.append((section, label, "PASS" if ok else "FAIL", round(float(got), 4), expected))


def missing(section, label, why):
    results.append((section, label, "MISSING", why, "-"))


def paired(a, b):
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    return d.mean(), t, p, d.mean() / d.std(ddof=1), lo, hi, n


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    diff = a.mean() - b.mean()
    lo, hi = stats.t.interval(0.95, df, loc=diff, scale=np.sqrt(va + vb))
    return diff, t, p, diff / pooled, df, lo, hi


def erp_wide(path):
    return pd.read_csv(path).pivot(index="subject", columns="condition",
                                   values=["n400_uv", "p600_uv"]).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--set", default="step")
    args = ap.parse_args()
    OUT = Path(args.base) / "outputs"
    S = args.set
    print(f"Reading outputs from: {OUT}\n")

    # ---------------- 3.1 ERP effects ----------------
    sec = "3.1 ERP"
    f = OUT / "erp_epoch0" / S / "averaged_erp_table.csv"
    if f.exists():
        w = erp_wide(f)
        check(sec, "n participants", len(w), 45, 0)
        m, t, p, d, lo, hi, n = paired(w[("n400_uv", SEM)].values, w[("n400_uv", WF)].values)
        check(sec, "N400 delta (uV)", m, -1.86, 0.005); check(sec, "N400 t", t, -4.89, 0.005)
        check(sec, "N400 d_z", d, -0.73, 0.005); check(sec, "N400 CI low", lo, -2.63, 0.005)
        check(sec, "N400 CI high", hi, -1.09, 0.005); check(sec, "N400 p (x1e5)", p * 1e5, 1.4, 0.05)
        m, t, p, d, lo, hi, n = paired(w[("p600_uv", SYN)].values, w[("p600_uv", WF)].values)
        check(sec, "P600 delta (uV)", m, 3.44, 0.005); check(sec, "P600 t", t, 5.76, 0.005)
        check(sec, "P600 d_z", d, 0.86, 0.005); check(sec, "P600 CI low", lo, 2.24, 0.005)
        check(sec, "P600 CI high", hi, 4.65, 0.005); check(sec, "P600 p (x1e7)", p * 1e7, 7.6, 0.05)
        m, t, p, d, lo, hi, n = paired(w[("n400_uv", SYN)].values, w[("n400_uv", WF)].values)
        check(sec, "syn N400-window delta", m, 1.13, 0.005); check(sec, "syn N400-window t", t, 2.78, 0.005)
        check(sec, "syn N400-window p", p, 0.008, 0.0005)
        check(sec, "syn N400-window CI low", lo, 0.31, 0.005); check(sec, "syn N400-window CI high", hi, 1.95, 0.005)
        m, t, p, d, lo, hi, n = paired(w[("p600_uv", SEM)].values, w[("p600_uv", WF)].values)
        check(sec, "sem P600-window delta", m, 0.06, 0.005); check(sec, "sem P600-window t", t, 0.12, 0.005)
        check(sec, "sem P600-window p", p, 0.908, 0.0005)
    else:
        missing(sec, "averaged_erp_table.csv", str(f))

    # ---------------- 3.2 Topography ----------------
    sec = "3.2 Scalp"
    f = OUT / "topography_epoch0" / S / "topography_report.txt"
    if f.exists():
        txt = f.read_text()
        sig = re.findall(r"Significant electrodes \(FDR q < 0\.05\): (\d+)/32", txt)
        if len(sig) >= 2:
            check(sec, "N400 sig electrodes", int(sig[0]), 25, 0)
            check(sec, "P600 sig electrodes", int(sig[1]), 28, 0)
        for ref, et, ed, lo_e, hi_e in (("Cz", -4.19, -0.62, -1.79, -0.63), ("Pz", 2.63, 0.39, 0.32, 2.42)):
            mm = re.search(ref + r" vs Fz: .*?t\(44\) = (-?[\d.]+), p = ([\d.]+)"
                           r"(?:, d_z = (-?[\d.]+).*?CI \[(-?[\d.]+), (-?[\d.]+)\])?", txt)
            if mm:
                check(sec, f"{ref} vs Fz t", float(mm.group(1)), et, 0.005)
                if mm.group(3):
                    check(sec, f"{ref} vs Fz d_z", float(mm.group(3)), ed, 0.005)
                    check(sec, f"{ref} vs Fz CI low", float(mm.group(4)), lo_e, 0.005)
                    check(sec, f"{ref} vs Fz CI high", float(mm.group(5)), hi_e, 0.005)
            else:
                missing(sec, f"{ref} vs Fz line", "not found in report")
    else:
        missing(sec, "topography_report.txt", str(f))

    # ---------------- 3.3 / 3.5 Clusters ----------------
    sec = "3.3/3.5 Clusters"
    f = OUT / "cluster_epoch0" / S / "cluster_results.csv"
    if f.exists():
        c = pd.read_csv(f)
        sig = c[c["p"] < 0.05]
        for start, end, p_exp, what in ((316, 496, 0.004, "semantic"), (438, 797, 0.0005, "syntactic"),
                                        (324, 504, 0.013, "expertise early"), (527, 797, 0.004, "expertise late")):
            r = sig[(sig["start_ms"] == start) & (sig["end_ms"] == end)]
            if len(r):
                check(sec, f"{what} cluster {start}-{end} ms p", float(r["p"].iloc[0]), p_exp, 0.0006)
            else:
                missing(sec, f"{what} cluster {start}-{end} ms", "no significant cluster with these bounds")
        other = c[c["p"] >= 0.05]
        if len(other):
            check(sec, "smallest non-significant p", float(other["p"].min()), 0.056, 0.0006)
    else:
        missing(sec, "cluster_results.csv", str(f))

    # ---------------- 3.4 Sensitivity ----------------
    sec = "3.4 Sensitivity"
    expected = {"same rule": (-1.51, 0.002, 3.62), "stricter": (-1.87, None, 2.52)}
    found = {k: False for k in expected}
    for f in sorted((OUT / "erp_epoch0").glob("*/averaged_erp_table.csv")):
        if f.parent.name == S:
            continue
        w = erp_wide(f)
        if len(w) != 54:
            continue
        n4 = paired(w[("n400_uv", SEM)].values, w[("n400_uv", WF)].values)
        p6 = paired(w[("p600_uv", SYN)].values, w[("p600_uv", WF)].values)
        for k, (e4, ep4, e6) in expected.items():
            if abs(n4[0] - e4) < 0.006 and abs(p6[0] - e6) < 0.006:
                found[k] = True
                check(sec, f"{k} ({f.parent.name}) N400", n4[0], e4, 0.005)
                check(sec, f"{k} ({f.parent.name}) P600", p6[0], e6, 0.005)
                if ep4:
                    check(sec, f"{k} N400 p", n4[2], ep4, 0.0005)
    for k, v in found.items():
        if not v:
            missing(sec, f"n=54 set '{k}'", "no 54-participant set with these values")

    # ---------------- 3.5 Expertise (ERP) ----------------
    sec = "3.5 Expertise ERP"
    f = OUT / "expertise_epoch0" / S / "expertise_effect_scores.csv"
    if f.exists():
        e = pd.read_csv(f)
        E, N = e[e["group"] == "Expert"], e[e["group"] == "Novice"]
        check(sec, "n experts", len(E), 27, 0); check(sec, "n novices", len(N), 18, 0)
        for grp, df_, col, em, et in ((("Experts"), E, "p600_effect", 4.85, 7.07),
                                       ("Novices", N, "p600_effect", 1.33, 1.49),
                                       ("Experts", E, "n400_effect", -1.32, -3.35),
                                       ("Novices", N, "n400_effect", -2.67, -3.72)):
            x = df_[col].values
            t, p = stats.ttest_1samp(x, 0)
            check(sec, f"{col} {grp} mean", x.mean(), em, 0.005)
            check(sec, f"{col} {grp} t", t, et, 0.005)
        for col, ed, et, edf, ep, eds in (("p600_effect", 3.52, 3.13, 35.1, 0.004, 0.96),
                                          ("syn_n400_win", 2.52, 3.37, 36.8, 0.002, 1.02),
                                          ("n400_effect", 1.35, 1.65, 27.2, 0.111, 0.54)):
            diff, t, p, ds, df, lo, hi = welch(E[col].values, N[col].values)
            check(sec, f"{col} Expert-Novice diff", diff, ed, 0.005)
            check(sec, f"{col} Welch t", t, et, 0.005)
            check(sec, f"{col} Welch df", df, edf, 0.05)
            check(sec, f"{col} Welch p", p, ep, 0.0005)
            check(sec, f"{col} d_s", ds, eds, 0.005)
        x = E["syn_n400_win"].values; check(sec, "syn N400-window Experts mean", x.mean(), 2.14, 0.005)
        x = N["syn_n400_win"].values; check(sec, "syn N400-window Novices mean", x.mean(), -0.38, 0.005)
    else:
        missing(sec, "expertise_effect_scores.csv", str(f))

    # ---------------- 3.6 Decoding (logistic regression) ----------------
    sec = "3.6 Decoding"
    root = OUT / "classification_epoch0" / S
    exp = {"syn": (59.66, 41, 51.5, 61.89, 56.30, 0.013),
           "sem": (60.67, 42, 51.6, 60.74, 60.55, 0.94),
           "3class": (43.50, 43, 34.6, 44.59, 41.87, 0.22)}
    chance = {"syn": 0.5, "sem": 0.5, "3class": 1 / 3}
    for task, (acc, nabove, p95, ex, nv, pw) in exp.items():
        f = root / task / "loso_per_subject.csv"
        if not f.exists():
            missing(sec, f"{task} loso_per_subject.csv", str(f)); continue
        r = pd.read_csv(f)
        a = r["bal_acc"].values
        check(sec, f"{task} accuracy %", a.mean() * 100, acc, 0.005)
        check(sec, f"{task} participants above chance", int((a > chance[task]).sum()), nabove, 0)
        if "group" in r:
            gE = r.loc[r["group"] == "Expert", "bal_acc"].values
            gN = r.loc[r["group"] == "Novice", "bal_acc"].values
            if len(gE) and len(gN):
                check(sec, f"{task} experts %", gE.mean() * 100, ex, 0.005)
                check(sec, f"{task} novices %", gN.mean() * 100, nv, 0.005)
                check(sec, f"{task} expert vs novice p", stats.ttest_ind(gE, gN, equal_var=False).pvalue,
                      pw, 0.005)
        fn = root / task / "permutation_scores.npy"
        if fn.exists():
            null = np.load(fn)
            check(sec, f"{task} null 95th pct %", np.percentile(null, 95) * 100, p95, 0.05)
            check(sec, f"{task} perm p", (np.sum(null >= a.mean()) + 1) / (len(null) + 1), 1 / 1001, 1e-6)
    for fs, vals in (("features_no_early", {"syn": 59.41, "sem": 59.01, "3class": 42.33}),
                     ("features_early_only", {"syn": 50.58, "sem": 50.25, "3class": 33.44})):
        for task, v in vals.items():
            f = root / fs / task / "loso_per_subject.csv"
            if f.exists():
                r = pd.read_csv(f)
                check(sec, f"{fs} {task} %", r["bal_acc"].mean() * 100, v, 0.005)
                if fs == "features_no_early" and task == "syn" and "group" in r:
                    gE = r.loc[r["group"] == "Expert", "bal_acc"].values
                    gN = r.loc[r["group"] == "Novice", "bal_acc"].values
                    check(sec, "no_early syn expert-novice p",
                          stats.ttest_ind(gE, gN, equal_var=False).pvalue, 0.015, 0.0005)
            else:
                missing(sec, f"{fs}/{task}", str(f))

    # leave-one-list-out
    lol = root / "leave_one_list_out"
    for task, v in {"syn": 59.67, "sem": 60.33, "3class": 43.47}.items():
        f = lol / f"list_per_subject_{task}.csv"
        if f.exists():
            check(sec, f"leave-one-list-out {task} %", pd.read_csv(f)["bal_acc"].mean() * 100, v, 0.005)
        else:
            missing(sec, f"leave-one-list-out {task}", str(f))

    # classifier comparison
    sec = "3.6 Classifiers"
    f = OUT / "classifier_comparison_epoch0" / S / "comparison_per_subject.csv"
    table6 = {"Logistic regression": (59.66, 60.67, 43.50), "Linear SVM": (59.80, 60.32, 43.45),
              "RBF SVM": (58.67, 58.28, 42.72), "Random forest": (56.15, 57.66, 40.41),
              "Gradient boosting": (57.00, 57.12, 41.61)}
    if f.exists():
        c = pd.read_csv(f)
        for clf, vals in table6.items():
            for task, v in zip(("syn", "sem", "3class"), vals):
                x = c[(c["classifier"] == clf) & (c["task"] == task)]["bal_acc"]
                if len(x):
                    check(sec, f"{clf} {task} %", x.mean() * 100, v, 0.005)
                else:
                    missing(sec, f"{clf} {task}", "task/classifier not in CSV")
        s = c[c["task"] == "sem"].pivot(index="subject", columns="classifier", values="bal_acc")
        if {"RBF SVM", "Logistic regression"} <= set(s.columns):
            check(sec, "RBF SVM sem p (x1e4)",
                  stats.ttest_rel(s["RBF SVM"], s["Logistic regression"]).pvalue * 1e4, 4.4, 0.05)
    else:
        missing(sec, "comparison_per_subject.csv", str(f))

    # EEGNet
    sec = "3.6 EEGNet"
    eg = OUT / "eegnet_simple"
    for task, acc, nab in (("syn", 60.72, 44), ("sem", 60.96, 42), ("3class", 44.78, 44)):
        f = eg / f"{task}_default" / "fold_results.csv"
        if f.exists():
            r = pd.read_csv(f)
            a = r["bal_acc"].values
            check(sec, f"default {task} %", a.mean() * 100, acc, 0.005)
            check(sec, f"default {task} above chance", int((a > chance[task]).sum()), nab, 0)
            if task == "syn" and "group" in r:
                gE = r.loc[r["group"] == "Expert", "bal_acc"].values
                gN = r.loc[r["group"] == "Novice", "bal_acc"].values
                check(sec, "default syn experts %", gE.mean() * 100, 62.76, 0.005)
                check(sec, "default syn novices %", gN.mean() * 100, 57.66, 0.005)
            rep = eg / f"{task}_default" / "report.txt"
            if rep.exists():
                mm = re.search(r"x (\d+) samples", rep.read_text())
                if mm:
                    check(sec, "input samples", int(mm.group(1)), 232, 0)
        else:
            missing(sec, f"{task}_default fold_results.csv", str(f))
    for folder, acc, n in (("syn", 60.12, 35), ("syn_nested", 60.42, 45)):
        f = eg / folder / "fold_results.csv"
        if f.exists():
            r = pd.read_csv(f)
            check(sec, f"{folder} n", len(r), n, 0)
            check(sec, f"{folder} %", r["bal_acc"].mean() * 100, acc, 0.005)
        else:
            missing(sec, f"{folder} fold_results.csv", str(f))

    # ---------------- report ----------------
    w1 = max(len(r[1]) for r in results) + 2
    cur = None
    for sec, label, status, got, exp_ in results:
        if sec != cur:
            print(f"\n== {sec} ==")
            cur = sec
        mark = {"PASS": "  ok ", "FAIL": " FAIL", "MISSING": " ????"}[status]
        print(f"{mark}  {label:<{w1}} paper: {exp_!s:<10} output: {got}")
    npass = sum(r[2] == "PASS" for r in results)
    nfail = sum(r[2] == "FAIL" for r in results)
    nmiss = sum(r[2] == "MISSING" for r in results)
    print(f"\nSUMMARY: {npass} pass, {nfail} fail, {nmiss} missing  (of {len(results)} checks)")
    if nfail or nmiss:
        print("Paste the FAIL and ???? lines back for a closer look.")


if __name__ == "__main__":
    main()