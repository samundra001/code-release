#!/usr/bin/env python3
"""
compare_classifiers_epoch0.py  —  Phase 10: does the classifier choice matter?

Updated version of the classifier-comparison script for the corrected pipeline.

Changes from the old script
  1. Reads features_epoch0/<set>/ (epoch0, Phase 2 participants and trials).
  2. Runs every decoding task, not only syntactic: --task syn | sem | 3class | all.
  3. Permutation test shuffles labels WITHIN each subject (same as
     classify_loso_epoch0.py); p = (count + 1) / (n + 1); runs in parallel.
  4. Each classifier is compared with logistic regression by a paired t-test on
     the 45 per-subject accuracies (does any model do reliably better?).
  5. Recall is reported per class for any task; hard-coded 54-fold and old
     reference accuracy removed.

Classifiers (unchanged): logistic regression, linear SVM, RBF SVM, random forest,
gradient boosting. Scaler fit on training subjects only; leave-one-subject-out.

Outputs (outputs/classifier_comparison_epoch0/<set>/):
  comparison_<task>.txt, comparison_per_subject.csv, comparison_summary.txt

USAGE:
  python compare_classifiers_epoch0.py                         (all tasks, no perms)
  python compare_classifiers_epoch0.py --n-permutations 200 --n-jobs 8
  python compare_classifiers_epoch0.py --task syn --feature-set no_early
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
FEAT_BASE   = BASE_DIR / "outputs" / "features_epoch0"
OUT_BASE    = BASE_DIR / "outputs" / "classifier_comparison_epoch0"

ALL_LABELS = {0: "well_formed", 1: "semantically_implausible", 2: "syntactically_invalid"}
TASKS = {
    "3class": ([0, 1, 2], "well-formed vs semantic vs syntactic"),
    "syn":    ([0, 2],    "well-formed vs syntactic"),
    "sem":    ([0, 1],    "well-formed vs semantic"),
}
BASELINE = "Logistic regression"

CLASSIFIERS = {
    "Logistic regression": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=42),
    "Linear SVM": lambda: SVC(kernel="linear", class_weight="balanced", random_state=42),
    "RBF SVM": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                           random_state=42),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1),
    "Gradient boosting": lambda: GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}


# ── CORE ──────────────────────────────────────────────────────────────────

def loso(X, y, groups, factory, labels):
    subjects = sorted(np.unique(groups))
    accs, yt, yp = [], [], []
    for s in subjects:
        te, tr = groups == s, groups != s
        sc = StandardScaler().fit(X[tr])
        clf = factory().fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        accs.append(balanced_accuracy_score(y[te], pred))
        yt += y[te].tolist()
        yp += pred.tolist()
    cm = confusion_matrix(yt, yp, labels=labels)
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return np.array(accs), subjects, recall


def shuffle_within(y, groups, rng):
    y2 = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        y2[idx] = rng.permutation(y[idx])
    return y2


def perm_test(X, y, groups, factory, labels, real, n_perm, n_jobs, seed):
    seeds = np.random.SeedSequence(seed).spawn(n_perm)

    def one(ss):
        yy = shuffle_within(y, groups, np.random.default_rng(ss))
        return loso(X, yy, groups, factory, labels)[0].mean()

    if n_jobs == 1:
        null = np.array([one(ss) for ss in seeds])
    else:
        from joblib import Parallel, delayed
        null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(ss) for ss in seeds))
    return (np.sum(null >= real) + 1) / (n_perm + 1)


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Classifier comparison at epoch0")
    ap.add_argument("--set", default="step")
    ap.add_argument("--task", default="all", choices=["all", *TASKS])
    ap.add_argument("--feature-set", default="all", choices=["all", "no_early"])
    ap.add_argument("--n-permutations", type=int, default=0,
                    help="0 skips permutation tests (slow for RBF SVM / boosting)")
    ap.add_argument("--perm-classifiers", nargs="+", default=None,
                    help='limit permutations, e.g. --perm-classifiers "Linear SVM" "Random forest"')
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fd = FEAT_BASE / args.set
    for f in ("X_features.npy", "y_labels.npy", "trial_metadata.csv", "feature_names.txt"):
        if not (fd / f).exists():
            sys.exit(f"ERROR: {fd / f} not found. Run features_epoch0.py first.")
    X_all = np.load(fd / "X_features.npy")
    y_all = np.load(fd / "y_labels.npy")
    meta = pd.read_csv(fd / "trial_metadata.csv")
    names = (fd / "feature_names.txt").read_text().split()
    if args.feature_set == "no_early":
        keep_cols = [i for i, n in enumerate(names) if n != "early_mean"]
        X_all, names = X_all[:, keep_cols], [names[i] for i in keep_cols]

    out = OUT_BASE / args.set
    if args.feature_set != "all":
        out = out / f"features_{args.feature_set}"
    out.mkdir(parents=True, exist_ok=True)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    per_subject, summary = [], []

    for task in tasks:
        labels, title = TASKS[task]
        m = np.isin(y_all, labels)
        X, y, groups = X_all[m], y_all[m], meta["subject"].values[m]
        chance = 1 / len(labels)
        print(f"\n=== {task}: {title} | {len(y)} trials, {len(np.unique(groups))} folds, "
              f"{len(names)} features ===")

        res = {}
        for name, factory in CLASSIFIERS.items():
            print(f"  {name:<22}", end=" ", flush=True)
            accs, subjects, recall = loso(X, y, groups, factory, labels)
            res[name] = (accs, recall)
            print(f"{accs.mean()*100:6.2f}%")
            per_subject += [{"task": task, "classifier": name, "subject": s, "bal_acc": a}
                            for s, a in zip(subjects, accs)]

        pvals = {}
        if args.n_permutations > 0:
            targets = args.perm_classifiers or list(CLASSIFIERS)
            for name, factory in CLASSIFIERS.items():
                if name not in targets:
                    continue
                print(f"  permutations ({args.n_permutations}) for {name} ...", flush=True)
                pvals[name] = perm_test(X, y, groups, factory, labels, res[name][0].mean(),
                                        args.n_permutations, args.n_jobs, args.seed)

        base = res[BASELINE][0]
        L = ["=" * 92, f"CLASSIFIER COMPARISON (epoch0, LOSO) — {title}", "=" * 92,
             f"Set: {args.set}   Features: {len(names)} ({args.feature_set})   "
             f"Trials: {len(y)}   Chance: {chance*100:.1f}%", "",
             f"  {'Classifier':<22} {'Acc':>7} {'95% CI':>17} {'>chance':>8} "
             + "".join(f"{'rec_' + ALL_LABELS[l][:3]:>9}" for l in labels)
             + f" {'perm p':>8} {'vs LogReg':>18}"]
        for name, (accs, recall) in res.items():
            n = len(accs)
            lo, hi = stats.t.interval(0.95, n - 1, loc=accs.mean(), scale=stats.sem(accs))
            if name == BASELINE:
                cmp = "reference"
            else:
                t, p = stats.ttest_rel(accs, base)
                cmp = f"{(accs.mean()-base.mean())*100:+.2f}pp p={p:.3f}"
            pstr = f"{pvals[name]:.4f}" if name in pvals else "-"
            L.append(f"  {name:<22} {accs.mean()*100:6.2f}% [{lo*100:5.2f}, {hi*100:5.2f}] "
                     f"{(accs > chance).sum():>4}/{n:<3}"
                     + "".join(f"{r*100:8.1f}%" for r in recall)
                     + f" {pstr:>8} {cmp:>18}")
            summary.append({"task": task, "classifier": name, "bal_acc": accs.mean(),
                            "ci_lo": lo, "ci_hi": hi, "perm_p": pvals.get(name, np.nan)})
        means = [r[0].mean() for r in res.values()]
        spread = (max(means) - min(means)) * 100
        best = max(res, key=lambda k: res[k][0].mean())
        L += ["", f"  Spread across classifiers: {spread:.2f} pp   Best: {best}",
              "  'vs LogReg' = paired t-test on per-subject accuracies.", "=" * 92]
        rep = "\n".join(L)
        (out / f"comparison_{task}.txt").write_text(rep, encoding="utf-8")
        print("\n" + rep)

    pd.DataFrame(per_subject).to_csv(out / "comparison_per_subject.csv", index=False)
    s = pd.DataFrame(summary)
    piv = s.pivot(index="classifier", columns="task", values="bal_acc").reindex(CLASSIFIERS)
    piv = (piv * 100).round(2)
    txt = "Balanced accuracy (%) by classifier and task\n" + piv.to_string()
    (out / "comparison_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt + f"\n\n[SAVED] {out}")


if __name__ == "__main__":
    main()