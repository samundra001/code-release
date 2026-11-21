#!/usr/bin/env python3
"""
classify_loso_epoch0.py  —  Phase 9: single-trial decoding at the critical word

Updated version of the LOSO logistic-regression script for the corrected pipeline.
One script now runs all three decoding problems (previously separate scripts):

  --task 3class   well-formed vs semantic vs syntactic   (chance 33.3%)
  --task syn      well-formed vs syntactic               (chance 50%)
  --task sem      well-formed vs semantic                (chance 50%)  NEW
  --task all      runs all three (default)

Changes from the old script
  1. Reads features_epoch0/<set>/ (epoch0, Phase 2 participants and trials);
     feature names are read from feature_names.txt, not hard-coded.
  2. Semantic-vs-well-formed added: with a real N400 at epoch0 this is now a
     meaningful test (it was pointless before).
  3. Permutation test shuffles labels WITHIN each subject, so each subject keeps
     its own class balance; p = (count + 1) / (n + 1) so p is never exactly 0.
  4. 95% CI uses the t-distribution across subject folds.
  5. Accuracy also reported separately for Experts and Novices (from per-subject
     folds; descriptive, plus a Welch t-test).
  6. Fixed the coefficient-column bug in the old report (coef_WEL).
  7. Permutations can run in parallel (--n-jobs).

Unchanged: logistic regression (class_weight balanced), StandardScaler fit on the
training subjects only, leave-one-subject-out, balanced accuracy averaged over folds.

Outputs (outputs/classification_epoch0/<set>/<task>/):
  loso_results.txt, loso_per_subject.csv, confusion_matrix.csv,
  feature_coefficients.csv, permutation_scores.npy
plus outputs/classification_epoch0/<set>/summary.txt when --task all.

USAGE:
  python classify_loso_epoch0.py                          (all tasks, 1000 perms)
  python classify_loso_epoch0.py --task syn --n-jobs 8
  python classify_loso_epoch0.py --no-permutation         (quick check)
  python classify_loso_epoch0.py --feature-set no_early   (control: without 100-300 ms)
  python classify_loso_epoch0.py --feature-set early_only (control: 100-300 ms alone)
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR    = SCRIPTS_DIR.parent.parent
FEAT_BASE   = BASE_DIR / "outputs" / "features_epoch0"
OUT_BASE    = BASE_DIR / "outputs" / "classification_epoch0"

ALL_LABELS = {0: "well_formed", 1: "semantically_implausible", 2: "syntactically_invalid"}
TASKS = {
    "3class": {"labels": [0, 1, 2], "title": "well-formed vs semantic vs syntactic"},
    "syn":    {"labels": [0, 2],    "title": "well-formed vs syntactic"},
    "sem":    {"labels": [0, 1],    "title": "well-formed vs semantic"},
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
    if X.shape[1] != len(names):
        sys.exit(f"ERROR: X has {X.shape[1]} columns but {len(names)} feature names.")
    return X, y, meta, names


# ── LOSO ──────────────────────────────────────────────────────────────────

def loso_scores(X, y, groups, subjects):
    """Mean balanced accuracy over LOSO folds (used for permutations)."""
    accs = []
    for s in subjects:
        te, tr = groups == s, groups != s
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(**LR_KWARGS).fit(sc.transform(X[tr]), y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
    return float(np.mean(accs))


def run_loso(X, y, groups, subjects, labels):
    folds, all_true, all_pred, coefs = [], [], [], []
    for s in subjects:
        te, tr = groups == s, groups != s
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(**LR_KWARGS).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        cm = confusion_matrix(y[te], pred, labels=labels)
        rec = np.where(cm.sum(1) > 0, cm.diagonal() / np.maximum(cm.sum(1), 1), np.nan)
        row = {"subject": s, "n_test": int(te.sum()),
               "bal_acc": balanced_accuracy_score(y[te], pred)}
        for lab, r in zip(labels, rec):
            row[f"recall_{ALL_LABELS[lab][:3]}"] = r
        folds.append(row)
        all_true += y[te].tolist()
        all_pred += pred.tolist()
        coefs.append(clf.coef_)
    return pd.DataFrame(folds), np.array(all_true), np.array(all_pred), np.stack(coefs)


def shuffle_within(y, groups, rng):
    y2 = y.copy()
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        y2[idx] = rng.permutation(y[idx])
    return y2


def permutation_test(X, y, groups, subjects, real, n_perm, n_jobs, seed):
    seeds = np.random.SeedSequence(seed).spawn(n_perm)

    def one(ss):
        rng = np.random.default_rng(ss)
        return loso_scores(X, shuffle_within(y, groups, rng), groups, subjects)

    if n_jobs == 1:
        null = []
        for i, ss in enumerate(seeds, 1):
            null.append(one(ss))
            if i % 100 == 0:
                print(f"    {i}/{n_perm}  null mean so far: {np.mean(null):.4f}")
        null = np.array(null)
    else:
        from joblib import Parallel, delayed
        null = np.array(Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(ss) for ss in seeds))
    p = (np.sum(null >= real) + 1) / (n_perm + 1)
    return p, null


# ── REPORT ────────────────────────────────────────────────────────────────

def coef_table(coefs, names, labels):
    mean = coefs.mean(0)                     # (n_classes or 1, n_feat)
    rows = []
    for j, f in enumerate(names):
        row = {"feature": f, "importance": float(np.abs(mean[:, j]).mean())}
        if mean.shape[0] == 1:                # binary: positive -> second class
            row[f"coef_toward_{ALL_LABELS[labels[1]][:3]}"] = float(mean[0, j])
        else:
            for k, lab in enumerate(labels):
                row[f"coef_{ALL_LABELS[lab][:3]}"] = float(mean[k, j])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def run_task(task, X_all, y_all, meta, names, args, out_root):
    labels = TASKS[task]["labels"]
    keep = np.isin(y_all, labels)
    X, y = X_all[keep], y_all[keep]
    groups = meta["subject"].values[keep]
    subjects = sorted(np.unique(groups))
    chance = 1 / len(labels)
    out = out_root / task
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  === {task}: {TASKS[task]['title']} | {len(y)} trials, {len(subjects)} folds ===")
    folds, yt, yp, coefs = run_loso(X, y, groups, subjects, labels)
    acc = folds["bal_acc"].values
    m, n = acc.mean(), len(acc)
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=stats.sem(acc))
    print(f"  balanced accuracy {m*100:.2f}%  (chance {chance*100:.1f}%)")

    if args.no_permutation:
        p, null = float("nan"), np.array([])
    else:
        print(f"  permutation test ({args.n_permutations}, labels shuffled within subject)...")
        p, null = permutation_test(X, y, groups, subjects, m, args.n_permutations,
                                   args.n_jobs, args.random_seed)
        print(f"  p = {p:.4f}")

    ctab = coef_table(coefs, names, labels)
    cm = confusion_matrix(yt, yp, labels=labels)
    cr = classification_report(yt, yp, labels=labels,
                               target_names=[ALL_LABELS[l] for l in labels], output_dict=True)

    # Expertise split (descriptive, from the same folds)
    grp = meta.drop_duplicates("subject").set_index("subject")["group"] \
        if "group" in meta else pd.Series(dtype=str)
    folds["group"] = folds["subject"].map(grp).fillna("")
    gE = folds.loc[folds["group"] == "Expert", "bal_acc"].values
    gN = folds.loc[folds["group"] == "Novice", "bal_acc"].values

    sep, dash = "=" * 72, "-" * 72
    L = [sep, f"LOSO DECODING AT THE CRITICAL WORD (epoch0) — {TASKS[task]['title']}", sep, "",
         f"Participant set : {args.set}",
         f"Classifier      : logistic regression, class-balanced; scaler fit on training subjects",
         f"Features ({len(names)})   : {', '.join(names)}",
         f"Trials / folds  : {len(y)} / {n}",
         f"Chance          : {chance*100:.1f}%", "",
         dash, "PRIMARY RESULT", dash,
         f"  Mean balanced accuracy : {m*100:.2f}%  (SD across subjects {acc.std(ddof=1)*100:.2f}%)",
         f"  95% CI (t, across folds): [{lo*100:.2f}%, {hi*100:.2f}%]",
         f"  Above chance           : {(m - chance)*100:+.2f} percentage points",
         f"  Subjects above chance  : {(acc > chance).sum()}/{n}"]
    if len(null):
        L += [f"  Permutation p          : {p:.4f}  ({len(null)} within-subject shuffles)",
              f"  Null mean / 95th pct   : {null.mean()*100:.2f}% / {np.percentile(null, 95)*100:.2f}%"]
    L.append("")

    L += [dash, "PER-CONDITION RESULTS (pooled predictions)", dash]
    for l in labels:
        r = cr[ALL_LABELS[l]]
        L.append(f"  {ALL_LABELS[l]:<28} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
                 f"F1 {r['f1-score']:.3f}  n={int(r['support'])}")
    L += ["", "  Confusion matrix (rows = true, cols = predicted):",
          "  " + " " * 28 + "".join(f"{ALL_LABELS[l][:8]:>12}" for l in labels)]
    for i, l in enumerate(labels):
        L.append(f"  {ALL_LABELS[l]:<28}" + "".join(f"{v:>12}" for v in cm[i]))
    L.append("")

    L += [dash, "FEATURE IMPORTANCE (mean |coefficient| across folds, standardized)", dash]
    ccols = [c for c in ctab.columns if c.startswith("coef_")]
    for _, r in ctab.iterrows():
        L.append(f"  {r['feature']:<18} {r['importance']:.4f}   " +
                 "  ".join(f"{c[5:]}={r[c]:+.3f}" for c in ccols))
    L.append("")

    if len(gE) > 1 and len(gN) > 1:
        t, pg = stats.ttest_ind(gE, gN, equal_var=False)
        L += [dash, "BY EXPERTISE (descriptive; same LOSO folds)", dash,
              f"  Experts (n={len(gE)}) : {gE.mean()*100:.2f}%   above chance {(gE > chance).sum()}/{len(gE)}",
              f"  Novices (n={len(gN)}) : {gN.mean()*100:.2f}%   above chance {(gN > chance).sum()}/{len(gN)}",
              f"  Welch t = {t:.2f}, p = {pg:.4f}", ""]
    L.append(sep)
    rep = "\n".join(L)

    (out / "loso_results.txt").write_text(rep, encoding="utf-8")
    folds.to_csv(out / "loso_per_subject.csv", index=False)
    pd.DataFrame(cm, index=[f"true_{ALL_LABELS[l]}" for l in labels],
                 columns=[f"pred_{ALL_LABELS[l]}" for l in labels]).to_csv(out / "confusion_matrix.csv")
    ctab.to_csv(out / "feature_coefficients.csv", index=False)
    if len(null):
        np.save(out / "permutation_scores.npy", null)
    print(rep)
    return {"task": task, "title": TASKS[task]["title"], "trials": len(y), "chance": chance,
            "bal_acc": m, "ci_lo": lo, "ci_hi": hi, "p": p,
            "expert": gE.mean() if len(gE) else np.nan, "novice": gN.mean() if len(gN) else np.nan}


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LOSO decoding at epoch0")
    ap.add_argument("--set", default="step")
    ap.add_argument("--task", default="all", choices=["all", *TASKS])
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--no-permutation", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--feature-set", default="all", choices=["all", "no_early", "early_only"],
                    help="control analysis: drop the 100-300 ms feature, or use it alone")
    args = ap.parse_args()

    X, y, meta, names = load_features(FEAT_BASE / args.set)
    if args.feature_set != "all":
        cols = [i for i, n in enumerate(names)
                if (n == "early_mean") == (args.feature_set == "early_only")]
        X, names = X[:, cols], [names[i] for i in cols]
    out_root = OUT_BASE / args.set
    if args.feature_set != "all":
        out_root = out_root / f"features_{args.feature_set}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n  Features: {X.shape} ({args.feature_set}), "
          f"{meta['subject'].nunique()} subjects, set '{args.set}'")

    tasks = list(TASKS) if args.task == "all" else [args.task]
    res = [run_task(t, X, y, meta, names, args, out_root) for t in tasks]

    if len(res) > 1:
        L = ["=" * 72, "DECODING SUMMARY (epoch0, logistic regression, LOSO)", "=" * 72,
             f"  {'Task':<38} {'Acc':>7} {'Chance':>7} {'95% CI':>17} {'p':>8} {'Exp':>7} {'Nov':>7}"]
        for r in res:
            pstr = "-" if np.isnan(r["p"]) else f"{r['p']:.4f}"
            L.append(f"  {r['title']:<38} {r['bal_acc']*100:6.2f}% {r['chance']*100:6.1f}% "
                     f"[{r['ci_lo']*100:5.2f}, {r['ci_hi']*100:5.2f}] {pstr:>8} "
                     f"{r['expert']*100:6.2f}% {r['novice']*100:6.2f}%")
        L.append("=" * 72)
        s = "\n".join(L)
        (out_root / "summary.txt").write_text(s, encoding="utf-8")
        print("\n" + s)
    print(f"\n  [SAVED] {out_root}")


if __name__ == "__main__":
    main()