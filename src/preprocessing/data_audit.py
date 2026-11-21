
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import mne

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


# ── BioSemi trigger string parsing ──────────────────────────
# Example: "B2,4,8,16(240)/B18,20(241)" -> codes [240, 241]

def parse_all_codes(value: object) -> List[int]:
    return [int(m) for m in re.findall(r"\((\d+)\)", str(value))]


def parse_first_code(value: object) -> Optional[int]:
    codes = parse_all_codes(value)
    return codes[0] if codes else None


def parse_last_code(value: object) -> Optional[int]:
    codes = parse_all_codes(value)
    return codes[-1] if codes else None


def classify_event_string(codes: List[int]) -> str:
    """single_1code = C-epoch; transition_2codes = A/B or B/C boundary."""
    n = len(codes)
    if n == 1:
        return "single_1code"
    if n == 2:
        return "transition_2codes"
    return f"other_{n}codes"


def bits8(x: object) -> Optional[str]:
    try:
        return format(int(x), "08b")
    except (TypeError, ValueError):
        return None


def upper6_bits_item_id(x: object) -> Optional[int]:
    """Upper 6 bits of trigger code = item/snippet ID (1-63)."""
    try:
        return int(x) >> 2
    except (TypeError, ValueError):
        return None


# ── File I/O helpers ─────────────────────────────────────────

def find_single_file(folder: Path, pattern: str, required: bool = True) -> Optional[Path]:
    files = sorted(folder.glob(pattern))
    if not files:
        if required:
            raise FileNotFoundError(f"No file matching '{pattern}' in {folder}")
        return None
    if len(files) > 1:
        for f in files:
            if "_task-PY_" in f.name:
                return f
    return files[0]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [SAVED] {path.name}  ({len(df)} rows)")


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {path.name}")


def section(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ── Args ──────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Script 1: EEG data audit for one BIDS subject")
    p.add_argument("--subject", default="sub-001")
    p.add_argument("--data-dir", default="../../data")
    p.add_argument("--output-dir", default="../../outputs/audit")
    p.add_argument("--offset-search-range", type=int, default=3)
    p.add_argument("--strict-match", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ── events.tsv loader ────────────────────────────────────────

def load_events_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")

    required = {"value", "sample"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events.tsv is missing required columns: {missing}")

    df = df.copy().reset_index(drop=True)
    df["event_row_id"] = df.index.astype(int)

    df["all_codes"]  = df["value"].apply(parse_all_codes)
    df["n_codes"]     = df["all_codes"].apply(len)
    df["first_code"]  = df["all_codes"].apply(lambda x: x[0]  if x else np.nan)
    df["last_code"]   = df["all_codes"].apply(lambda x: x[-1] if x else np.nan)
    df["has_slash"]   = df["value"].astype(str).str.contains("/", regex=False)
    df["event_type"]  = df["all_codes"].apply(classify_event_string)

    df["last_code_binary8"] = df["last_code"].apply(bits8)
    df["last_code_item_id"] = df["last_code"].apply(upper6_bits_item_id)

    if "onset" in df.columns:
        df["dt_from_prev_onset_sec"] = df["onset"].diff()

    return df


# ── MNE loader ────────────────────────────────────────────────

def load_eeg_set(path: Path) -> Tuple[np.ndarray, Dict[str, int], dict]:
    epochs = mne.io.read_epochs_eeglab(str(path), verbose=False)

    meta = {
        "sfreq":      float(epochs.info["sfreq"]),
        "tmin":       float(epochs.tmin),
        "tmax":       float(epochs.tmax),
        "n_channels": int(len(epochs.ch_names)),
        "n_times":    int(len(epochs.times)),
        "n_epochs":   int(len(epochs)),
        "ch_names":   list(epochs.ch_names),
    }

    events_array  = epochs.events.copy()
    event_id_dict = dict(epochs.event_id)

    return events_array, event_id_dict, meta


def build_mne_mapping_table(event_id_dict: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for raw_string, mne_id in event_id_dict.items():
        codes = parse_all_codes(raw_string)
        rows.append({
            "mne_id":           int(mne_id),
            "raw_string":       raw_string,
            "event_string_type": classify_event_string(codes),
            "n_codes":          len(codes),
            "all_codes":        codes,
            "first_code":       codes[0]  if codes else np.nan,
            "last_code":        codes[-1] if codes else np.nan,
            "has_slash":        "/" in str(raw_string),
            "last_code_binary8": bits8(codes[-1]) if codes else None,
            "last_code_item_id": upper6_bits_item_id(codes[-1]) if codes else None,
        })

    return pd.DataFrame(rows).sort_values("mne_id").reset_index(drop=True)


# ── Epoch <-> events.tsv matching ───────────────────────────

def match_epochs_to_events(
    events_array: np.ndarray,
    mne_mapping: pd.DataFrame,
    df_events: pd.DataFrame,
    sfreq: float,
    offset_search_range: int = 3,
) -> pd.DataFrame:
    """
    Match each epoch to events.tsv via sample proximity (±offset_search_range)
    plus anchor-code agreement (first_code of the MNE string must equal the
    last_code of the events.tsv row at that sample).
    """
    sample_to_rows: Dict[int, List[int]] = {}
    for idx, row in df_events.iterrows():
        s = int(row["sample"])
        sample_to_rows.setdefault(s, []).append(idx)

    mne_id_to_info: Dict[int, dict] = {
        int(r["mne_id"]): r.to_dict()
        for _, r in mne_mapping.iterrows()
    }

    consumed: Dict[Tuple[int, int], int] = {}
    epoch_rows = []

    for epoch_idx in range(len(events_array)):
        mne_sample = int(events_array[epoch_idx, 0])
        mne_id     = int(events_array[epoch_idx, 2])

        info = mne_id_to_info.get(mne_id, {})
        raw_string  = info.get("raw_string", "")
        event_type  = info.get("event_string_type", "unknown")
        all_codes   = info.get("all_codes", [])
        first_code  = info.get("first_code", np.nan)
        last_code   = info.get("last_code", np.nan)

        anchor_code = int(first_code) if not pd.isna(first_code) else None

        matched_row_idx = None
        best_offset     = None

        for offset in range(-(offset_search_range), offset_search_range + 1):
            candidate_sample = mne_sample - 1 + offset
            candidates = sample_to_rows.get(candidate_sample, [])

            for c_idx in candidates:
                c_row = df_events.loc[c_idx]
                c_last = c_row.get("last_code", np.nan)

                if anchor_code is not None and not pd.isna(c_last):
                    if int(c_last) == anchor_code:
                        key = (candidate_sample, anchor_code)
                        n_used = consumed.get(key, 0)
                        all_matching = [
                            i for i in candidates
                            if not pd.isna(df_events.loc[i, "last_code"])
                            and int(df_events.loc[i, "last_code"]) == anchor_code
                        ]
                        if n_used < len(all_matching):
                            matched_row_idx = all_matching[n_used]
                            consumed[key] = n_used + 1
                            best_offset = offset
                            break
            if matched_row_idx is not None:
                break

        if matched_row_idx is not None:
            m = df_events.loc[matched_row_idx]
            matched_fields = {
                "match_status":          "matched",
                "match_sample_offset":   best_offset,
                "matched_event_row_id":  int(m["event_row_id"]),
                "matched_sample":        int(m["sample"]),
                "matched_onset_sec":     float(m["onset"]) if "onset" in m else np.nan,
                "matched_value_string":  str(m["value"]),
                "matched_last_code":     int(m["last_code"]) if not pd.isna(m["last_code"]) else np.nan,
            }
        else:
            matched_fields = {
                "match_status":          "failed",
                "match_sample_offset":   np.nan,
                "matched_event_row_id":  np.nan,
                "matched_sample":        np.nan,
                "matched_onset_sec":     np.nan,
                "matched_value_string":  None,
                "matched_last_code":     np.nan,
            }

        epoch_rows.append({
            "epoch_id":              epoch_idx,
            "triplet_id_guess":      epoch_idx // 3,
            "triplet_pos_guess":     (epoch_idx % 3) + 1,

            "mne_sample":            mne_sample,
            "mne_id":                mne_id,
            "mne_raw_string":        raw_string,
            "event_string_type":     event_type,
            "n_codes_in_string":     len(all_codes),
            "all_codes_in_string":   all_codes,
            "first_code_in_string":  int(first_code) if not pd.isna(first_code) else np.nan,
            "last_code_in_string":   int(last_code)  if not pd.isna(last_code)  else np.nan,
            "last_code_binary8":     bits8(last_code),
            "last_code_item_id":     upper6_bits_item_id(last_code),

            "epoch_onset_sec_mne_estimate": (mne_sample - 1) / sfreq,

            **matched_fields,
        })

    return pd.DataFrame(epoch_rows)


def report_match_quality(epoch_audit_df: pd.DataFrame, subject: str, output_dir: Path) -> dict:
    n_total    = len(epoch_audit_df)
    n_matched  = int((epoch_audit_df["match_status"] == "matched").sum())
    n_failed   = int((epoch_audit_df["match_status"] == "failed").sum())
    match_rate = n_matched / n_total if n_total > 0 else 0.0

    print(f"\n  Match results: {n_matched}/{n_total} matched ({match_rate:.1%})")

    if n_failed > 0:
        print(f"  WARNING: {n_failed} epochs could not be matched to events.tsv")
        fail_df = epoch_audit_df[epoch_audit_df["match_status"] == "failed"].copy()
        save_csv(fail_df, output_dir / f"{subject}_match_failures.csv")
    else:
        print("  All epochs matched successfully.")

    offset_counts = (
        epoch_audit_df["match_sample_offset"]
        .dropna().astype(int).value_counts().sort_index()
    )
    if len(offset_counts) > 0:
        print("\n  Sample offset distribution:")
        for offset, count in offset_counts.items():
            bar = "█" * min(count, 40)
            print(f"    offset {offset:+3d}: {count:5d} epochs  {bar}")

    return {
        "n_epochs_total":    n_total,
        "n_epochs_matched":  n_matched,
        "n_epochs_failed":   n_failed,
        "match_rate":        round(match_rate, 6),
        "offset_distribution": offset_counts.to_dict(),
    }


def build_raw_lookup(df_events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for val, group in df_events.groupby("value", sort=False):
        codes = parse_all_codes(val)
        rows.append({
            "raw_value_string":    val,
            "count_in_events_tsv": len(group),
            "event_string_type":   classify_event_string(codes),
            "n_codes":             len(codes),
            "all_codes":           codes,
            "first_code":          codes[0]  if codes else np.nan,
            "last_code":           codes[-1] if codes else np.nan,
            "last_code_binary8":   bits8(codes[-1]) if codes else None,
            "last_code_item_id":   upper6_bits_item_id(codes[-1]) if codes else None,
        })
    return pd.DataFrame(rows).sort_values(["event_string_type", "last_code"]).reset_index(drop=True)


def build_audit_summary(
    subject: str, eeg_set_path: Path, events_tsv_path: Path, meta: dict,
    df_events: pd.DataFrame, mne_mapping: pd.DataFrame, match_quality: dict,
    output_dir: Path,
) -> dict:
    n_epochs   = meta["n_epochs"]
    n_triplets = n_epochs // 3
    remainder  = n_epochs % 3

    return {
        "subject": subject,
        "input_files": {"eeg_set": str(eeg_set_path), "events_tsv": str(events_tsv_path)},
        "recording": {
            "sfreq": meta["sfreq"], "tmin": meta["tmin"], "tmax": meta["tmax"],
            "n_channels": meta["n_channels"], "n_times": meta["n_times"],
            "n_epochs": n_epochs, "n_triplets_guessed": n_triplets,
            "remainder_epochs": remainder,
        },
        "events_tsv": {
            "n_rows": int(len(df_events)),
            "unique_values": int(df_events["value"].nunique()),
        },
        "mne_event_ids": {
            "n_unique": int(len(mne_mapping)),
            "type_counts": mne_mapping["event_string_type"].value_counts().to_dict(),
        },
        "matching": match_quality,
        "outputs_dir": str(output_dir),
    }


def main() -> None:
    args = build_args()
    subject    = args.subject
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) / subject
    output_dir.mkdir(parents=True, exist_ok=True)

    eeg_dir = data_dir / subject / "eeg"
    if not eeg_dir.exists():
        raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")

    eeg_set_path    = find_single_file(eeg_dir, "*.set", required=True)
    events_tsv_path = find_single_file(eeg_dir, "*_events.tsv", required=True)

    print(f"\nSubject   : {subject}")
    print(f"EEG dir   : {eeg_dir}")
    print(f".set file : {eeg_set_path.name}")
    print(f"events    : {events_tsv_path.name}")
    print(f"Output    : {output_dir}")

    section("STEP 1 — Parse events.tsv")
    df_events = load_events_tsv(events_tsv_path)
    print(f"  Loaded {len(df_events)} trigger rows")
    print(f"  Unique value strings: {df_events['value'].nunique()}")
    print(f"  Event type breakdown:\n{df_events['event_type'].value_counts().to_string()}")
    if args.verbose:
        print(df_events.head(20).to_string(index=False))

    section("STEP 2 — Load .set file with MNE (metadata only)")
    events_array, event_id_dict, meta = load_eeg_set(eeg_set_path)
    print(f"  Epochs     : {meta['n_epochs']}")
    print(f"  Channels   : {meta['n_channels']}")
    print(f"  Sfreq      : {meta['sfreq']} Hz")
    print(f"  Time window: {meta['tmin']:.3f}s to {meta['tmax']:.3f}s")
    print(f"  MNE event_id entries: {len(event_id_dict)}")

    section("STEP 3 — Build MNE event_id mapping table")
    mne_mapping = build_mne_mapping_table(event_id_dict)
    print(f"  {len(mne_mapping)} unique MNE event IDs")
    save_csv(mne_mapping, output_dir / f"{subject}_mne_eventid_mapping.csv")

    section("STEP 4 — Build raw value lookup table")
    raw_lookup = build_raw_lookup(df_events)
    save_csv(raw_lookup, output_dir / f"{subject}_raw_value_lookup.csv")
    if args.verbose:
        print(raw_lookup.to_string(index=False))

    section("STEP 5 — Match epochs to events.tsv rows")
    epoch_audit_df = match_epochs_to_events(
        events_array=events_array, mne_mapping=mne_mapping, df_events=df_events,
        sfreq=meta["sfreq"], offset_search_range=args.offset_search_range,
    )
    match_quality = report_match_quality(epoch_audit_df, subject, output_dir)

    if args.strict_match and match_quality["n_epochs_failed"] > 0:
        raise RuntimeError(
            f"[STRICT] {match_quality['n_epochs_failed']} epochs failed to match. "
            f"See {subject}_match_failures.csv"
        )

    save_csv(epoch_audit_df, output_dir / f"{subject}_epoch_audit_table.csv")

    preview_cols = [
        "epoch_id", "triplet_id_guess", "triplet_pos_guess",
        "event_string_type", "first_code_in_string", "last_code_in_string",
        "last_code_item_id", "match_status", "matched_onset_sec", "matched_value_string",
    ]
    n_preview = len(epoch_audit_df) if args.verbose else 15
    print(f"\n  Epoch audit table preview (first {n_preview} rows):")
    print(epoch_audit_df[preview_cols].head(n_preview).to_string(index=False))

    section("STEP 6 — Save events audit and audit summary")
    save_csv(df_events, output_dir / f"{subject}_events_audit.csv")

    summary = build_audit_summary(
        subject=subject, eeg_set_path=eeg_set_path, events_tsv_path=events_tsv_path,
        meta=meta, df_events=df_events, mne_mapping=mne_mapping,
        match_quality=match_quality, output_dir=output_dir,
    )
    save_json(summary, output_dir / f"{subject}_audit_summary.json")

    section("DONE")
    print(f"\n  Outputs written to: {output_dir}")
    print(f"  Match rate : {match_quality['match_rate']:.1%}")
    print(f"  N epochs   : {match_quality['n_epochs_total']}")
    print(f"  N triplets : {summary['recording']['n_triplets_guessed']}")
    remainder = summary["recording"]["remainder_epochs"]
    print("  Epoch count divisible by 3: OK" if remainder == 0
          else f"  WARNING: remainder={remainder}")
    print(f"\n  Next step: run 02_triplet_builder.py --subject {subject}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise