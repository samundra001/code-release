#!/usr/bin/env python3
"""
classify_by_list_epoch0.py  —  control: leave-one-LIST-out decoding at the critical item

Why: each participant saw one of four stimulus lists, and within a list each item
always appears in the same condition. In the main analysis (leave-one-participant-
out), a test participant from List 1 is classified by a model trained partly on other
List 1 participants, who saw the same items in the same conditions. A classifier could
then, in principle, use item-specific differences (e.g. word length or form) instead
of the response to the violation.

Here the model is trained on three lists and tested on the participants of the fourth
list, for each list in turn. In the test list every item appears in a different
condition than in the training lists, so item-specific information cannot help; only
the response to the violation can. Everything else is identical to the main analysis
(classify_loso_epoch0.py): same 13 features, logistic regression with class-balanced
weights, scaler fit on training data only, balanced accuracy per test participant,
mean across the 45 participants, 95% CI (t), within-participant label permutations.

Also prints (point 2a) the largest feature weights of the MAIN analysis, read from
the existing feature_coefficients.csv files, to check the "largest weights" sentence.

Inputs
  outputs/features_epoch0/<set>/        X_features.npy, y_labels.npy,
                                        trial_metadata.csv, feature_names.txt
  outputs/verification/participant_list_assignment.csv   (subject, list)
  outputs/classification_epoch0/<set>/<task>/loso_per_subject.csv   (optional,
                                        for the paired comparison with the main analysis)
  outputs/classification_epoch0/<set>/<task>/feature_coefficients.csv (optional, 2a)
Outputs  outputs/classification_epoch0/<set>/leave_one_list_out/
  list_results.txt, list_per_subject_<task>.csv, list_permutation_<task>.npy

USAGE
  python classify_by_list_epoch0.py                    (all tasks, 1000 permutations)
  python classify_by_list_epoch0.py --n-jobs 8
  python classify_by_list_epoch0.py --no-permutation   (quick check)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
FEAT_BASE   = BASE_DIR / "outputs" / "features_epoch0"
OUT_BASE    = BASE_DIR / "outputs" / "classification_epoch0"
LIST_FILE   = BASE_DIR / "outputs" / "verification" / "participant_list_assignment.csv"

TASKS = {
    "syn":    {"labels": [0, 2],    "title": "well-formed vs syntactic"},
    "sem":    {"labels": [0, 1],    "title": "well-formed vs semantic"},
    "3class": {"labels": [0, 1, 2], "title": "well-formed vs semantic vs syntactic"},
}
RANDOM_SEED = 42
LR_KWARGS = dict(max_iter=2000, class_weight="balanced", solver="lbfgs",
                 random_state=RANDOM_SEED)


# ── DATA ──────────────────────────────────────────────────────────────────

def load_features(feat_dir: Path):
    for f in ("X_features.npy", "y_labels.npy", "trial_metadata.csv", "feature_names.txt"):
        if not (feat_dir / f).exists():
            sys.exit(f"ERROR: {feat_dir / f} not found. Run features_epoch0.py first.")
    X = np.load(feat_dir / "X_features.npy")
    y = np.load(feat_dir / "y_labels.npy")
    meta = pd.read_csv(feat_dir / "trial_metadata.csv")
    names = (feat_dir / "feature_names.txt").read_text().split()
    return X, y, meta, names


def load_lists(subjects):
    if not LIST_FILE.exists():
        sys.exit(f"ERROR: {LIST_FILE} not found (made by assign_lists_and_flag_items.py).")
    a = pd.read_csv(LIST_FILE)
    lst = dict(zip(a["subject"].astype(str), a["list"]))
    missing = [s for s in subjects if s not in lst or pd.isna(lst[s])]
    if missing:
        sys.exit(f"ERROR: no list assignment for {missing}")
    return {s: int(lst[s]) for s in subjects}


# ── LEAVE-ONE-LIST-OUT ────────────────────────────────────────────────────

def per_subject_scores(X, y, groups, subj_list, lists):
    """Train on three lists, test each participant of the fourth separately."""
    rows = []
    trial_list = np.array([subj_list[s] for s in groups])
    for L in lists:
        tr, te_list = trial_list != L, trial_list == L
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(**LR_KWARGS).fit(sc.transform(X[tr]), y[tr])
        for s in sorted(np.unique(groups[te_list])):
            te = groups == s
            pred = clf.predict(sc.transform(X[te]))
            rows.append({"subject": s, "list": L, "n_test": int(te.sum()),
                         "bal_acc": balanced_accuracy_score(y[te], pred)})
    return pd.DataFrame(rows)


def shuffle_within(y, groups, rng):
    y2 = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        y2[idx] = rng.permutation(y[idx])
    return y2


def permutation_test(X, y, groups, subj_list, lists, real, n_perm, n_jobs, seed):
    seeds = np.random.SeedSequence(seed).spawn(n_perm)

    def one(ss):
        rng = np.random.default_rng(ss)
        y2 = shuffle_within(y, groups, rng)
        return float(per_subject_scores(X, y2, groups, subj_list, lists)["bal_acc"].mean())

    if n_jobs == 1:
        null = np.array([one(ss) for ss in seeds])
    else:
        from joblib import Parallel, delayed
        null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(ss) for ss in seeds))
    p = (np.sum(null >= real) + 1) / (n_perm + 1)
    return p, null


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Leave-one-list-out decoding control")
    ap.add_argument("--set", default="step")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--no-permutation", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    X_all, y_all, meta, names = load_features(FEAT_BASE / args.set)
    subj_all = meta["subject"].astype(str).values
    subjects = sorted(np.unique(subj_all))
    subj_list = load_lists(subjects)
    lists = sorted(set(subj_list.values()))
    grp = (meta.drop_duplicates("subject").set_index("subject")["group"].astype(str)
           if "group" in meta else pd.Series(dtype=str))

    out = OUT_BASE / args.set / "leave_one_list_out"
    out.mkdir(parents=True, exist_ok=True)
    sep, dash = "=" * 74, "-" * 74
    counts = pd.Series(subj_list).value_counts().sort_index()
    L = [sep, "CONTROL: LEAVE-ONE-LIST-OUT DECODING AT THE CRITICAL ITEM", sep,
         f"Set: {args.set}   participants: {len(subjects)}   features: {len(names)}",
         "Participants per list: " + ", ".join(f"List {k}: {v}" for k, v in counts.items()),
         "Train on three lists, test each participant of the held-out list.", ""]

    for task, info in TASKS.items():
        labels = info["labels"]
        keep = np.isin(y_all, labels)
        X, y, groups = X_all[keep], y_all[keep], subj_all[keep]
        chance = 1 / len(labels)
        print(f"\n=== {task}: {info['title']} ===")
        res = per_subject_scores(X, y, groups, subj_list, lists)
        acc = res["bal_acc"].values
        m, n = acc.mean(), len(acc)
        lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=stats.sem(acc))
        L += [dash, f"{info['title'].upper()}  (chance {chance*100:.1f}%)", dash,
              f"  Leave-one-list-out : {m*100:.2f}%   95% CI [{lo*100:.2f}, {hi*100:.2f}]   "
              f"above chance {(acc > chance).sum()}/{n}"]
        for Lh in lists:
            a = res.loc[res["list"] == Lh, "bal_acc"]
            L.append(f"     held-out List {Lh} (n={len(a)}): {a.mean()*100:.2f}%")

        if not args.no_permutation:
            print(f"  permutations ({args.n_permutations}) ...")
            p, null = permutation_test(X, y, groups, subj_list, lists, m,
                                       args.n_permutations, args.n_jobs, RANDOM_SEED)
            np.save(out / f"list_permutation_{task}.npy", null)
            L.append(f"  Permutation p      : {p:.4f}   null mean {null.mean()*100:.2f}%, "
                     f"95th pct {np.percentile(null, 95)*100:.2f}%")

        # comparison with the main (leave-one-participant-out) analysis
        main_f = OUT_BASE / args.set / task / "loso_per_subject.csv"
        if main_f.exists():
            mm = pd.read_csv(main_f)[["subject", "bal_acc"]].rename(columns={"bal_acc": "loso"})
            both = res.merge(mm, on="subject")
            t, pp = stats.ttest_rel(both["bal_acc"], both["loso"])
            L.append(f"  Main analysis      : {both['loso'].mean()*100:.2f}%   difference "
                     f"{(both['bal_acc'] - both['loso']).mean()*100:+.2f} pp, "
                     f"paired t({len(both)-1}) = {t:.2f}, p = {pp:.3f}")
        else:
            L.append("  Main analysis      : loso_per_subject.csv not found, comparison skipped")

        # expertise (descriptive, same folds)
        res["group"] = res["subject"].map(grp).fillna("")
        gE = res.loc[res["group"] == "Expert", "bal_acc"].values
        gN = res.loc[res["group"] == "Novice", "bal_acc"].values
        if len(gE) > 1 and len(gN) > 1:
            t, pg = stats.ttest_ind(gE, gN, equal_var=False)
            L.append(f"  By expertise       : experts {gE.mean()*100:.2f}%, novices "
                     f"{gN.mean()*100:.2f}%, Welch t = {t:.2f}, p = {pg:.3f}")
        L.append("")
        res.to_csv(out / f"list_per_subject_{task}.csv", index=False)

    # point 2a: largest weights in the MAIN analysis
    L += [dash, "LARGEST FEATURE WEIGHTS IN THE MAIN ANALYSIS (mean |coefficient|, "
          "standardized; from feature_coefficients.csv)", dash]
    for task, info in TASKS.items():
        f = OUT_BASE / args.set / task / "feature_coefficients.csv"
        if f.exists():
            c = pd.read_csv(f).sort_values("importance", ascending=False).head(5)
            L.append(f"  {info['title']}: " +
                     ", ".join(f"{r.feature} ({r.importance:.3f})" for r in c.itertuples()))
        else:
            L.append(f"  {info['title']}: feature_coefficients.csv not found")
    L += ["", "How to read: if leave-one-list-out accuracy stays close to the main analysis,",
          "decoding does not depend on item-specific information shared within a list.", sep]

    rep = "\n".join(L)
    (out / "list_results.txt").write_text(rep, encoding="utf-8")
    print("\n" + rep + f"\n\n[SAVED] {out}")


if __name__ == "__main__":
    main()