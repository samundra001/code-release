#!/usr/bin/env python3
"""
eegnet_simple.py  —  EEGNet on the raw EEG at the critical word (epoch0)

Three evaluation modes (--mode):
  holdout  tune on 10 participants, test on the other 35 (default; original run)
  default  no tuning, published EEGNet settings, test on all 45
  nested   tune inside every fold (10 of the 44 training participants pick the
           best config), test on all 45
No subject fine-tuning (separate step).

Data: same as every other epoch0 analysis
  - epoch0_id, Phase 2 kept participants and trial mask (artifacts and the
    `class` item removed)
  - -100 to 800 ms, baseline -100 to 0 ms, 32 scalp channels (EOG excluded)

Steps
  1. Set aside 10 tuning participants (stratified by expertise).
  2. Try N random hyperparameter settings. Each is trained on the other
     participants and scored ONLY on the 10 tuning participants. Keep the best.
  3. With the best setting, leave-one-participant-out over the remaining
     participants: train on everyone else, test on the held-out one.
  4. Report mean balanced accuracy, 95% CI, binomial test vs chance.

Early stopping: 4 training participants are held out as a validation group in
every training run; the epoch with the best validation accuracy is kept.

Outputs (outputs/eegnet_simple/<task>/):
  eegnet_data.npz (cache, in the parent folder), tuning_subjects.txt,
  search_results.csv, best_config.json, fold_results.csv, report.txt

USAGE:
  pip install torch
  python eegnet_simple.py --task syn                         (holdout, 35)
  python eegnet_simple.py --task syn --mode default          (B: all 45)
  python eegnet_simple.py --task syn --mode nested          (C: all 45)
  python eegnet_simple.py --task sem
  python eegnet_simple.py --task syn --n-configs 2 --max-epochs 10   (quick test)
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

# ── CONFIG ────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "outputs" / "artifact_epoch0"
OUT_BASE     = BASE_DIR / "outputs" / "eegnet_simple"
SET_NAME     = "step"

TMIN, TMAX = -0.1, 0.8
BASELINE   = (-0.1, 0.0)
EOG        = ["LIO", "ROC"]
LABELS     = {"well_formed": 0, "semantically_implausible": 1, "syntactically_invalid": 2}
TASKS      = {"syn": [0, 2], "sem": [0, 1], "3class": [0, 1, 2]}

N_TUNING   = 10
N_VAL      = 4
BATCH      = 64
SEED       = 42

SEARCH_SPACE = {
    "lr":       [1e-3, 5e-4, 1e-4],
    "dropout":  [0.25, 0.5],
    "F1":       [8, 16],
    "kern":     [32, 64],
    "patience": [10, 15, 20],
}


INT_KEYS = ("F1", "kern", "patience")


def clean_cfg(cfg):
    """Settings read back from pandas/JSON can come back as floats (16.0);
    layer sizes and patience must be ints."""
    out = {k: (v.item() if hasattr(v, "item") else v) for k, v in cfg.items()}
    for k in INT_KEYS:
        out[k] = int(round(out[k]))
    out["lr"] = float(out["lr"])
    out["dropout"] = float(out["dropout"])
    return out


# ── DATA ──────────────────────────────────────────────────────────────────

def load_data():
    cache = OUT_BASE / "eegnet_data.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"  Loaded cached data: {z['X'].shape}")
        return z["X"], z["y"], z["subj"]

    import mne
    mne.set_log_level("ERROR")
    subjects = [l.strip() for l in
                (ARTIFACT_DIR / f"kept_participants_{SET_NAME}.txt").read_text().splitlines()
                if l.strip() and not l.startswith("#")]
    mask = pd.read_csv(ARTIFACT_DIR / f"trial_mask_{SET_NAME}.csv")
    mask["keep"] = mask["keep"].astype(str).str.lower().isin(["true", "1"])

    Xs, ys, ss = [], [], []
    for i, s in enumerate(subjects, 1):
        print(f"  [{i:>2}/{len(subjects)}] {s} ...", end=" ", flush=True)
        f = sorted((DATA_DIR / s / "eeg").glob("*.set"))[0]
        ep = mne.read_epochs_eeglab(str(f), verbose=False)
        ep.drop_channels([c for c in EOG if c in ep.ch_names])
        ep.crop(TMIN, TMAX).apply_baseline(BASELINE, verbose=False)
        d = ep.get_data() * 1e6
        m = mask[(mask["subject"] == s) & mask["keep"] & mask["condition"].isin(LABELS)]
        idx = m["epoch0_id"].astype(int).values
        Xs.append(d[idx].astype(np.float32))
        ys.append(m["condition"].map(LABELS).values.astype(np.int64))
        ss.append(np.array([s] * len(idx)))
        print(f"OK {len(idx)} trials")

    X, y, subj = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, subj=subj)
    return X, y, subj


def load_groups():
    tsv = DATA_DIR / "participants.tsv"
    if not tsv.exists():
        return {}
    df = pd.read_csv(tsv, sep="\t")
    idc = next((c for c in df.columns if "participant" in c.lower()), df.columns[0])
    for c in df.columns:
        v = df[c].dropna().astype(str).str.strip()
        if v.isin(["Expert", "Novice"]).mean() > 0.9:
            return dict(zip(df[idc].astype(str).str.strip(), v))
    return {}


# ── MODEL ─────────────────────────────────────────────────────────────────

def build_eegnet(n_ch, n_t, n_cls, cfg):
    import torch
    import torch.nn as nn
    F1, D, kern, drop = cfg["F1"], 2, cfg["kern"], cfg["dropout"]
    F2 = F1 * D

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # temporal filters
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                nn.BatchNorm2d(F1),
                # spatial (depthwise) filters
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(drop),
                # separable convolution
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, 1, bias=False),
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(drop),
            )
            with torch.no_grad():
                n_out = self.features(torch.zeros(1, 1, n_ch, n_t)).numel()
            self.classify = nn.Linear(n_out, n_cls)

        def forward(self, x):                     # x: (batch, channels, time)
            return self.classify(self.features(x.unsqueeze(1)).flatten(1))

    return EEGNet()


# ── TRAINING ──────────────────────────────────────────────────────────────

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, X):
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(get_device())
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_model(X, y, subj, train_subjects, cfg, max_epochs, seed):
    """Train on train_subjects; 4 of them are a validation group for early stopping.
    Returns (model, normalize_fn, epochs_used)."""
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = get_device()

    train_subjects = np.array(sorted(train_subjects))
    val_subj = rng.choice(train_subjects, size=N_VAL, replace=False)
    fit_m = np.isin(subj, train_subjects) & ~np.isin(subj, val_subj)
    val_m = np.isin(subj, val_subj)

    # channel-wise standardization from the training data only
    mu = X[fit_m].mean(axis=(0, 2), keepdims=True)
    sd = X[fit_m].std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda Z: ((Z - mu) / sd).astype(np.float32)
    Xtr, ytr = norm(X[fit_m]), y[fit_m]
    Xva, yva = norm(X[val_m]), y[val_m]

    model = build_eegnet(X.shape[1], X.shape[2], int(y.max()) + 1, cfg).to(dev)
    counts = np.bincount(ytr).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_acc, best_state, wait, used = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            xb = torch.from_numpy(Xtr[b]).to(dev)
            yb = torch.from_numpy(ytr[b]).to(dev)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        used = epoch + 1
        val_acc = balanced_accuracy_score(yva, predict(model, Xva))
        if val_acc > best_acc:
            best_acc, best_state, wait = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, norm, used


# ── TUNING HELPERS ────────────────────────────────────────────────────────

# Standard EEGNet settings (Lawhern et al., 2018): F1 = 8, D = 2, F2 = 16,
# temporal kernel = half the sampling rate (256 Hz -> 128 samples), dropout 0.25
# for cross-subject decoding, Adam lr = 1e-3. Patience is not specified in the
# paper; 15 epochs is used here.
DEFAULT_CFG = {"lr": 1e-3, "dropout": 0.25, "F1": 8, "kern": 128, "patience": 15}


def pick_tuning(subjects, groups, n, rng):
    """n participants, stratified by expertise when labels are available."""
    E = [s for s in subjects if groups.get(s) == "Expert"]
    N = [s for s in subjects if groups.get(s) == "Novice"]
    if E and N:
        nE = min(len(E), max(1, round(n * len(E) / len(subjects))))
        t = list(rng.choice(E, nE, replace=False)) + \
            list(rng.choice(N, min(len(N), n - nE), replace=False))
    else:
        t = list(rng.choice(subjects, n, replace=False))
    return sorted(str(x) for x in t)


def sample_configs(n, seed):
    rng = np.random.default_rng(seed)
    return [clean_cfg({k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()})
            for _ in range(n)]


def search(X, y, subj, train_subj, tune_subj, configs, max_epochs, seed, verbose=True):
    """Train each config on train_subj, score on tune_subj; return best cfg + table."""
    tm = np.isin(subj, tune_subj)
    rows = []
    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        model, norm, used = train_model(X, y, subj, train_subj, cfg, max_epochs, seed)
        pred = predict(model, norm(X[tm]))
        acc = np.mean([balanced_accuracy_score(y[tm][subj[tm] == s], pred[subj[tm] == s])
                       for s in tune_subj])
        rows.append({**cfg, "tuning_bal_acc": acc, "epochs_used": used})
        if verbose:
            print(f"    config {i}/{len(configs)}: {cfg} -> {acc*100:.2f}% "
                  f"({used} epochs, {time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows).sort_values("tuning_bal_acc", ascending=False)
    return clean_cfg({k: res.iloc[0][k] for k in SEARCH_SPACE}), res


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="syn", choices=list(TASKS))
    ap.add_argument("--mode", default="holdout", choices=["holdout", "default", "nested"],
                    help="holdout: tune on 10 participants, test on the other 35 | "
                         "default: published EEGNet settings, no tuning, test on all 45 | "
                         "nested: tune inside every fold, test on all 45")
    ap.add_argument("--n-configs", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=150)
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("ERROR: PyTorch not installed. Run: pip install torch")

    suffix = "" if args.mode == "holdout" else f"_{args.mode}"
    out = OUT_BASE / f"{args.task}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n  EEGNet | task {args.task} | mode {args.mode} | device {get_device()}")

    X, y, subj = load_data()
    labels = TASKS[args.task]
    keep = np.isin(y, labels)
    X, subj = X[keep], subj[keep]
    y = np.searchsorted(labels, y[keep]).astype(np.int64)      # relabel to 0..k-1
    chance = 1 / len(labels)
    subjects = sorted(np.unique(subj))
    groups = load_groups()
    print(f"  {len(y)} trials, {len(subjects)} participants, "
          f"{X.shape[1]} channels x {X.shape[2]} samples, chance {chance*100:.1f}%")

    # ---- settings per mode ----
    tuning, best = [], None
    if args.mode == "holdout":
        tfile = OUT_BASE / "tuning_subjects.txt"
        if tfile.exists():
            tuning = tfile.read_text().split()
        else:
            tuning = pick_tuning(subjects, groups, N_TUNING, np.random.default_rng(SEED))
            tfile.write_text("\n".join(tuning))
        eval_subjects = [s for s in subjects if s not in tuning]
        print(f"  Tuning participants: {', '.join(tuning)}")
        cfg_file = out / "best_config.json"
        if cfg_file.exists():
            best = clean_cfg(json.loads(cfg_file.read_text()))
            print(f"  Using saved best config: {best}")
        else:
            print("  Hyperparameter search on the tuning participants:")
            best, res = search(X, y, subj, eval_subjects, tuning,
                               sample_configs(args.n_configs, SEED), args.max_epochs, SEED)
            res.to_csv(out / "search_results.csv", index=False)
            cfg_file.write_text(json.dumps(best, indent=2))
            print(f"  Best config: {best}")
        design = f"{len(tuning)} tuning participants, {args.n_configs} random configs"
    elif args.mode == "default":
        best = dict(DEFAULT_CFG)
        eval_subjects = subjects
        design = "no tuning: published EEGNet settings"
    else:
        eval_subjects = subjects
        configs = sample_configs(args.n_configs, SEED)      # same candidates in every fold
        design = (f"nested: in every fold, {N_TUNING} of the training participants pick "
                  f"the best of {args.n_configs} configs")
    print(f"  Evaluation participants: {len(eval_subjects)}  ({design})")

    # ---- leave-one-participant-out (resumable) ----
    fold_file = out / "fold_results.csv"
    rows = pd.read_csv(fold_file).to_dict("records") if fold_file.exists() else []
    done = {r["subject"] for r in rows}
    for k, s in enumerate(eval_subjects, 1):
        if s in done:
            continue
        t0 = time.time()
        others = [o for o in subjects if o != s]
        cfg = best
        if args.mode == "nested":
            rng = np.random.default_rng(SEED + k)
            inner_tune = pick_tuning(others, groups, N_TUNING, rng)
            inner_train = [o for o in others if o not in inner_tune]
            print(f"  [{k:>2}/{len(eval_subjects)}] {s}: inner search ...")
            cfg, _ = search(X, y, subj, inner_train, inner_tune, configs,
                            args.max_epochs, SEED + k, verbose=False)
        model, norm, used = train_model(X, y, subj, others, cfg, args.max_epochs, SEED + k)
        te = subj == s
        pred = predict(model, norm(X[te]))
        acc = balanced_accuracy_score(y[te], pred)
        rows.append({"subject": s, "group": groups.get(s, ""), "n_test": int(te.sum()),
                     "n_correct": int((pred == y[te]).sum()), "bal_acc": acc,
                     "epochs_used": used, "config": json.dumps(cfg)})
        pd.DataFrame(rows).to_csv(fold_file, index=False)
        print(f"  [{k:>2}/{len(eval_subjects)}] {s}  {acc*100:5.1f}%  "
              f"({used} epochs, {time.time()-t0:.0f}s)"
              + (f"  config {cfg}" if args.mode == "nested" else ""))

    # ---- statistics ----
    r = pd.DataFrame(rows)
    a = r["bal_acc"].values
    n = len(a)
    lo, hi = stats.t.interval(0.95, n - 1, loc=a.mean(), scale=stats.sem(a))
    t_p = stats.ttest_1samp(a, chance, alternative="greater").pvalue
    k_corr, n_tot = int(r["n_correct"].sum()), int(r["n_test"].sum())
    b_p = stats.binomtest(k_corr, n_tot, chance, alternative="greater").pvalue

    L = ["=" * 72, f"EEGNET — task: {args.task} — mode: {args.mode}", "=" * 72,
         f"Input        : {X.shape[1]} scalp channels x {X.shape[2]} samples (-100 to 800 ms)",
         f"Design       : {design}",
         f"Settings     : {best if best else 'chosen per fold (see fold_results.csv)'}",
         f"Evaluated on : {n} participants",
         f"Chance       : {chance*100:.1f}%", "",
         f"  Mean balanced accuracy : {a.mean()*100:.2f}%",
         f"  95% CI (across folds)  : [{lo*100:.2f}%, {hi*100:.2f}%]",
         f"  Above chance           : {(a > chance).sum()}/{n} participants",
         f"  t-test vs chance       : one-sided p = {t_p:.3g}",
         f"  Binomial test (pooled) : {k_corr}/{n_tot} correct, one-sided p = {b_p:.3g}"]
    gE = r.loc[r["group"] == "Expert", "bal_acc"].values
    gN = r.loc[r["group"] == "Novice", "bal_acc"].values
    if len(gE) > 1 and len(gN) > 1:
        t, p = stats.ttest_ind(gE, gN, equal_var=False)
        L.append(f"  Experts {gE.mean()*100:.2f}% (n={len(gE)}) vs Novices "
                 f"{gN.mean()*100:.2f}% (n={len(gN)}): Welch t = {t:.2f}, p = {p:.4f}")
    lr_file = BASE_DIR / "outputs" / "classification_epoch0" / SET_NAME / args.task / \
        "loso_per_subject.csv"
    if lr_file.exists():
        lr = pd.read_csv(lr_file)[["subject", "bal_acc"]].rename(columns={"bal_acc": "lr"})
        m = r.merge(lr, on="subject")
        if len(m) > 2:
            t, p = stats.ttest_rel(m["bal_acc"], m["lr"])
            L.append(f"  vs logistic regression, same {len(m)} participants: EEGNet "
                     f"{m['bal_acc'].mean()*100:.2f}% vs LR {m['lr'].mean()*100:.2f}%, "
                     f"paired t = {t:.2f}, p = {p:.4f}")
    if args.mode == "nested":
        L.append(f"  Settings chosen per fold: "
                 f"{r['config'].value_counts().to_dict()}")
    L.append("=" * 72)
    rep = "\n".join(L)
    (out / "report.txt").write_text(rep)
    print("\n" + rep + f"\n\n  [SAVED] {out}")


if __name__ == "__main__":
    main()