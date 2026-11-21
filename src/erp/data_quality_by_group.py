#!/usr/bin/env python3
"""
data_quality_by_group.py  -  do experts and novices differ in data quality?

Checks whether the expertise effect could be explained by noisier data in one
group. For the final sample (primary set), compares experts and novices on:
  - retained trials per condition (well-formed, semantic, syntactic)
  - retained trials in total (these three conditions)
  - artifact rejection rate (% of trials removed, these three conditions)
Doubly anomalous trials are not analyzed in the paper and are left out here.
The `class` item is not counted as an artifact (it was removed for a stimulus
problem, not for data quality).

Group differences: Welch's t-test (same test as the expertise analyses).

Inputs:
  outputs/artifact_epoch0/kept_participants_<set>.txt
  outputs/artifact_epoch0/trial_mask_<set>.csv
  data/participants.tsv
Output:
  outputs/data_quality_epoch0/<set>/data_quality_by_group.txt (+ .csv)

USAGE:  python data_quality_by_group.py            (primary set: step)
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent.parent
ART_DIR = BASE_DIR / "outputs" / "artifact_epoch0"
OUT_BASE = BASE_DIR / "outputs" / "data_quality_epoch0"
PARTICIPANTS = BASE_DIR / "data" / "participants.tsv"

CONDS = ["well_formed", "semantically_implausible", "syntactically_invalid"]
SHORT = {"well_formed": "wf", "semantically_implausible": "sem",
         "syntactically_invalid": "syn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="step")
    args = ap.parse_args()

    kept = [l.strip() for l in (ART_DIR / f"kept_participants_{args.set}.txt")
            .read_text().splitlines() if l.strip() and not l.startswith("#")]
    p = pd.read_csv(PARTICIPANTS, sep="\t").set_index("participant_id")
    m = pd.read_csv(ART_DIR / f"trial_mask_{args.set}.csv")
    m["keep"] = m["keep"].astype(str).str.lower().isin(["true", "1"])
    m = m[m.subject.isin(kept) & (m.reason.astype(str) != "class_item")
          & m.condition.isin(CONDS)]

    per = m.pivot_table(index="subject", columns="condition", values="keep",
                        aggfunc="sum").rename(columns=SHORT)
    per["total"] = per[["wf", "sem", "syn"]].sum(axis=1)
    n_all = m.groupby("subject").size()
    per["reject_pct"] = 100 * (1 - per["total"] / n_all)
    per["group"] = p.loc[per.index, "python_expertise"]

    E, N = per[per.group == "Expert"], per[per.group == "Novice"]
    L = ["=" * 72, "DATA QUALITY BY EXPERTISE GROUP", "=" * 72,
         f"Set: {args.set}   Experts n = {len(E)}   Novices n = {len(N)}",
         "Conditions: well-formed, semantic, syntactic (class item excluded)", "",
         f"{'Measure':<22}{'Experts M (SD) [min]':>24}{'Novices M (SD) [min]':>24}"
         f"{'Welch t':>9}{'df':>7}{'p':>8}"]
    labels = {"wf": "Trials, well-formed", "sem": "Trials, semantic",
              "syn": "Trials, syntactic", "total": "Trials, total",
              "reject_pct": "Rejection rate (%)"}
    rows = []
    for col, lab in labels.items():
        a, b = E[col].astype(float), N[col].astype(float)
        res = stats.ttest_ind(a, b, equal_var=False)
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        L.append(f"{lab:<22}{a.mean():>10.1f} ({a.std(ddof=1):.1f}) [{a.min():.0f}]"
                 f"{b.mean():>12.1f} ({b.std(ddof=1):.1f}) [{b.min():.0f}]"
                 f"{res.statistic:>9.2f}{df:>7.1f}{res.pvalue:>8.3f}")
        rows.append({"measure": lab, "expert_mean": a.mean(), "expert_sd": a.std(ddof=1),
                     "novice_mean": b.mean(), "novice_sd": b.std(ddof=1),
                     "t": res.statistic, "df": df, "p": res.pvalue})
    L += ["", "If no p < 0.05: the groups did not differ in data quality, so the",
          "expertise effect is not explained by noisier data in one group.", "=" * 72]

    out = OUT_BASE / args.set
    out.mkdir(parents=True, exist_ok=True)
    rep = "\n".join(L)
    (out / "data_quality_by_group.txt").write_text(rep, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "data_quality_by_group.csv", index=False)
    per.to_csv(out / "data_quality_per_participant.csv")
    print(rep)
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()