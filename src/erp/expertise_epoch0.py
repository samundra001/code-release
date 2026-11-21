#!/usr/bin/env python3
"""
expertise_epoch0.py  —  Phase 4: expertise analysis at the critical word

Combines expert_analysis.py and expertise_condition_interaction_test.py into one
script, updated for the corrected pipeline (epoch0, Phase 2 kept participants and
trial mask). It does NOT re-read the EEG: it uses the per-subject averages that
erp_epoch0.py already saved, so both analyses use exactly the same numbers.

Input:
  outputs/erp_epoch0/<set>/averaged_erp_table.csv   (from erp_epoch0.py)
  data/participants.tsv                             (expertise labels/scores)

Tests (all on N400 = Fz/Cz/Pz 300-500 ms, P600 = Fz/Cz/Pz 500-800 ms):
  1. Within each group (Expert, Novice): paired t-tests
       N400 effect  = semantic  - well-formed  (N400 window)
       P600 effect  = syntactic - well-formed  (P600 window)
       cross-checks = syntactic - wf (N400 window), semantic - wf (P600 window)
  2. Group x condition interaction (the correct test of an expertise effect):
       Welch t-test comparing the per-subject effect scores of Experts vs Novices.
       Optional 2x2 mixed ANOVA cross-check if pingouin is installed.
  3. If participants.tsv has a numeric Python-test score: correlation between
     score and each effect (closest to Kuo & Prat's continuous expertise term).

Kuo & Prat (2024) reported: expertise x syntax interaction in the P600 window
(stronger P600 with more expertise), no expertise x semantics interaction.

Outputs (outputs/expertise_epoch0/<set>/):
  expertise_effect_scores.csv   per-subject effects + group (+ score)
  expertise_report.txt

USAGE:
  python expertise_epoch0.py                 (primary: step)
  python expertise_epoch0.py --set step_all54
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
ERP_DIR     = BASE_DIR / "outputs" / "erp_epoch0"
OUT_BASE    = BASE_DIR / "outputs" / "expertise_epoch0"
PARTICIPANTS_TSV = BASE_DIR / "data" / "participants.tsv"

WF, SEM, SYN = "well_formed", "semantically_implausible", "syntactically_invalid"

# Fallback only (same map as the old scripts: 31 Expert / 23 Novice of the 54).
EXPERTISE_MAP = {
    "sub-001":"Novice","sub-002":"Novice","sub-003":"Novice","sub-004":"Novice",
    "sub-005":"Novice","sub-006":"Novice","sub-007":"Novice","sub-008":"Expert",
    "sub-009":"Novice","sub-010":"Novice","sub-011":"Expert","sub-012":"Expert",
    "sub-013":"Expert","sub-014":"Novice","sub-015":"Expert","sub-016":"Expert",
    "sub-017":"Expert","sub-018":"Novice","sub-019":"Novice","sub-020":"Novice",
    "sub-021":"Expert","sub-022":"Novice","sub-023":"Expert","sub-024":"Expert",
    "sub-025":"Expert","sub-026":"Expert","sub-027":"Expert","sub-028":"Expert",
    "sub-029":"Expert","sub-030":"Expert","sub-031":"Expert","sub-032":"Expert",
    "sub-033":"Expert","sub-034":"Expert","sub-035":"Expert","sub-036":"Novice",
    "sub-037":"Novice","sub-038":"Novice","sub-039":"Expert","sub-040":"Expert",
    "sub-041":"Expert","sub-042":"Expert","sub-043":"Novice","sub-044":"Novice",
    "sub-045":"Expert","sub-046":"Expert","sub-047":"Expert","sub-048":"Expert",
    "sub-049":"Novice","sub-050":"Expert","sub-051":"Expert","sub-052":"Novice",
    "sub-053":"Expert","sub-054":"Novice","sub-055":"Expert","sub-056":"Novice",
    "sub-057":"Expert","sub-058":"Expert","sub-059":"Novice","sub-060":"Novice",
    "sub-061":"Expert",
}


# ==============================================================================
# EXPERTISE LABELS (+ numeric score if available)
# ==============================================================================

def load_expertise(path: Path) -> tuple[pd.DataFrame, str]:
    """Return DataFrame [subject, group, score] and a note on where it came from."""
    if path.exists():
        df = pd.read_csv(path, sep="\t")
        id_col = next((c for c in df.columns if "participant" in c.lower()
                       or c.lower() in ("subject", "subject_id")), df.columns[0])
        grp_col, score_col = None, None
        for c in df.columns:
            if c == id_col:
                continue
            vals = df[c].dropna().astype(str).str.strip()
            if grp_col is None and vals.isin(["Expert", "Novice"]).mean() > 0.9:
                grp_col = c
            elif (score_col is None and pd.api.types.is_numeric_dtype(df[c])
                  and any(k in c.lower() for k in ("score", "python", "test", "accuracy"))):
                score_col = c
        if grp_col:
            out = pd.DataFrame({"subject": df[id_col].astype(str).str.strip(),
                                "group": df[grp_col].astype(str).str.strip()})
            out["score"] = df[score_col].values if score_col else np.nan
            note = f"participants.tsv (group: '{grp_col}'"
            note += f", score: '{score_col}')" if score_col else ", no numeric score column)"
            return out, note
    out = pd.DataFrame({"subject": list(EXPERTISE_MAP), "group": list(EXPERTISE_MAP.values())})
    out["score"] = np.nan
    return out, "hard-coded map (participants.tsv missing or no Expert/Novice column)"


# ==============================================================================
# STATISTICS HELPERS
# ==============================================================================

def one_sample(x: np.ndarray) -> dict:
    n = len(x)
    t, p = stats.ttest_1samp(x, 0.0)
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=x.mean(), scale=stats.sem(x))
    return {"n": n, "mean": x.mean(), "t": t, "p": p, "d": x.mean() / x.std(ddof=1),
            "ci": (lo, hi)}


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return {"t": t, "p": p, "d": (a.mean() - b.mean()) / pooled,
            "diff": a.mean() - b.mean()}


def star(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


EFFECTS = [  # (column, label, expected sign, primary?)
    ("n400_effect",   "N400: semantic - wf (300-500 ms)",          -1, True),
    ("p600_effect",   "P600: syntactic - wf (500-800 ms)",         +1, True),
    ("syn_n400_win",  "cross: syntactic - wf, N400 window",        +1, False),
    ("sem_p600_win",  "cross: semantic - wf, P600 window",          0, False),
]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="step")
    ap.add_argument("--participants-tsv", default=str(PARTICIPANTS_TSV))
    args = ap.parse_args()

    table_path = ERP_DIR / args.set / "averaged_erp_table.csv"
    if not table_path.exists():
        sys.exit(f"ERROR: {table_path} not found. Run erp_epoch0.py --set {args.set} first.")
    out_dir = OUT_BASE / args.set
    out_dir.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(table_path)
    w = t.pivot(index="subject", columns="condition", values=["n400_uv", "p600_uv"]).dropna()
    eff = pd.DataFrame({
        "n400_effect":  w[("n400_uv", SEM)] - w[("n400_uv", WF)],
        "p600_effect":  w[("p600_uv", SYN)] - w[("p600_uv", WF)],
        "syn_n400_win": w[("n400_uv", SYN)] - w[("n400_uv", WF)],
        "sem_p600_win": w[("p600_uv", SEM)] - w[("p600_uv", WF)],
    }).reset_index()

    exp, exp_note = load_expertise(Path(args.participants_tsv))
    eff = eff.merge(exp, on="subject", how="left")
    unknown = eff[~eff["group"].isin(["Expert", "Novice"])]["subject"].tolist()
    eff = eff[eff["group"].isin(["Expert", "Novice"])].reset_index(drop=True)
    eff.to_csv(out_dir / "expertise_effect_scores.csv", index=False)

    E, N = eff[eff["group"] == "Expert"], eff[eff["group"] == "Novice"]
    sep, dash = "=" * 74, "-" * 74
    L = [sep, "EXPERTISE ANALYSIS AT THE CRITICAL WORD (epoch0)", sep, "",
         f"Participant set : {args.set}",
         f"Expertise from  : {exp_note}",
         f"Experts / Novices: {len(E)} / {len(N)}"
         + (f"   (unknown, excluded: {unknown})" if unknown else ""),
         f"Electrodes      : Fz, Cz, Pz (same averages as erp_epoch0.py)", ""]

    # 1. Within-group effects
    L += [dash, "1. EFFECTS WITHIN EACH GROUP (one-sample t-test of the effect vs 0)", dash]
    for col, label, sign, _ in EFFECTS:
        L.append(f"  {label}")
        for name, g in (("Expert", E), ("Novice", N), ("All", eff)):
            r = one_sample(g[col].values)
            L.append(f"    {name:<7} n={r['n']:<3} diff={r['mean']:+.3f} uV  "
                     f"CI[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]  t({r['n']-1})={r['t']:+.2f}  "
                     f"p={r['p']:.5f} {star(r['p']):<4} d={r['d']:+.2f}")
        L.append("")

    # 2. Interaction
    L += [dash, "2. GROUP x CONDITION INTERACTION (Welch t on effect scores, Expert vs Novice)",
          "   This is the correct test of whether expertise changes an effect.", dash]
    inter = {}
    for col, label, sign, primary in EFFECTS:
        r = welch(E[col].values, N[col].values)
        inter[col] = r
        L.append(f"  {label:<40} Expert-Novice = {r['diff']:+.3f} uV  "
                 f"t={r['t']:+.2f}  p={r['p']:.4f} {star(r['p']):<4} d={r['d']:+.2f}"
                 + ("" if primary else "  (cross-check)"))
    L.append("")

    try:
        import pingouin as pg
        L += [dash, "   Cross-check: 2x2 mixed ANOVA (Group x Condition)", dash]
        for name, cond, win in (("N400", SEM, "n400_uv"), ("P600", SYN, "p600_uv")):
            long = t[t["condition"].isin([WF, cond]) & t["subject"].isin(eff["subject"])]
            long = long.merge(eff[["subject", "group"]], on="subject")
            aov = pg.mixed_anova(data=long, dv=win, within="condition",
                                 between="group", subject="subject")
            row = aov[aov["Source"].str.contains("Interaction", case=False)]
            pcol = next((c for c in aov.columns if c.lower() in ("p-unc", "p-val", "pval")), None)
            if len(row) and pcol:
                L.append(f"  {name}: interaction F={row['F'].values[0]:.3f}, "
                         f"p={row[pcol].values[0]:.4f}")
        L.append("")
    except ImportError:
        L += ["  (2x2 mixed ANOVA skipped: pingouin not installed)", ""]

    # 3. Continuous score
    sc = eff.dropna(subset=["score"])
    if len(sc) >= 10:
        L += [dash, "3. CONTINUOUS EXPERTISE: Python test score vs effect size", dash]
        for col, label, _, primary in EFFECTS[:2]:
            r, p = stats.pearsonr(sc["score"], sc[col])
            rs, ps = stats.spearmanr(sc["score"], sc[col])
            L.append(f"  {label:<40} Pearson r={r:+.3f} p={p:.4f} {star(p):<4} "
                     f"| Spearman rho={rs:+.3f} p={ps:.4f}")
        L.append("")
    else:
        L += ["3. Continuous score analysis skipped (no numeric score column found).", ""]

    # Summary
    p6, n4 = inter["p600_effect"], inter["n400_effect"]
    L += [sep, "SUMMARY vs KUO & PRAT (2024)", sep,
          "  They found: expertise x syntax interaction in the P600 window (stronger P600",
          "  with more expertise); no expertise x semantics interaction.",
          f"  Ours: P600 effect Expert-Novice = {p6['diff']:+.2f} uV, p = {p6['p']:.4f} "
          f"-> {'difference detected' if p6['p'] < .05 else 'no reliable group difference'}",
          f"        N400 effect Expert-Novice = {n4['diff']:+.2f} uV, p = {n4['p']:.4f} "
          f"-> {'difference detected' if n4['p'] < .05 else 'no reliable group difference'}",
          "  Note: they used expertise as a continuous z-score in a mixed model; a two-group",
          "  split has less power, so section 3 (if available) is the closer comparison.",
          sep]

    report = "\n".join(L)
    (out_dir / "expertise_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[SAVED] {out_dir}")


if __name__ == "__main__":
    main()