#!/usr/bin/env python3
"""
verify_labels_by_parsing.py

Checks the condition labels by parsing the stimulus lines with Python itself.
Syntactically invalid lines (222/227) and doubly anomalous lines (242/247)
should fail to parse; well-formed (112/117) and semantic (132) lines should
parse.

Why list level, not trial level: the trigger code identifies a CONDITION, not
a line (20 trials share each code), so a per-trial (C-code, text) table cannot
be built. Each participant saw one full list (160 trials = 160 lines), so
checking every line in the four list files covers every stimulus anyone saw.

What it does:
  1. Parses Stimuli_v4_List1..4.txt with a tolerant block parser
     (digit, digit, code lines, 3-digit condition code).
  2. Reports blocks parsed vs 3-digit code lines found (should be equal).
  3. Finds the offset between the file's condition codes and the EEG C-codes
     (expected +2: the events.tsv triplets are base, base+1, base+2).
  4. Checks each list's code counts against the EEG profile
     (112:20, 117:20, 132:40, 222:20, 227:20, 242:20, 247:20).
  5. Reports the share of lines that parse, per C-code and condition.

Parse rule: the LAST line of each block (the line holding the critical item)
is parsed alone; a second, looser test parses the whole block. A trailing
"pass" is tried for headers ending in ':'.
Limitation: this checks syntax only. Semantic plausibility is not testable.

USAGE:
  python verify_labels_by_parsing.py
  python verify_labels_by_parsing.py --stimuli-dir /path/to/stimuli
  python verify_labels_by_parsing.py --offset 2
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
STIM_CANDIDATES = [
    BASE_DIR / "data" / "stimuli",
    Path(__file__).resolve().parents[2] / "data" / "stimuli",
]
DEFAULT_CODEBOOK = BASE_DIR / "configs" / "c_code_to_condition_verified.json"
DEFAULT_OUT = BASE_DIR / "outputs" / "verification"

FALLBACK_CODEBOOK = {
    112: "well_formed", 117: "well_formed", 132: "semantically_implausible",
    222: "syntactically_invalid", 227: "syntactically_invalid",
    242: "doubly_anomalous", 247: "doubly_anomalous",
}
# Trials per participant for each C-code (sub-001, Script 2 output)
EEG_PROFILE = {112: 20, 117: 20, 132: 40, 222: 20, 227: 20, 242: 20, 247: 20}
EXPECT_PARSE = {"well_formed": True, "semantically_implausible": True,
                "syntactically_invalid": False, "doubly_anomalous": False}
PRACTICE_CODES = {101, 201}
ONE_DIGIT = re.compile(r"^\d$")
ANY_NUM   = re.compile(r"^\d+$")
CODE_LINE = re.compile(r"^\d{3,}$")


def load_codebook(path: Path) -> dict[int, str]:
    if not path.exists():
        print(f"Codebook not found at {path}; using the built-in 7-code table.")
        return dict(FALLBACK_CODEBOOK)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        if str(k).startswith("_"):
            continue
        out[int(k)] = v if isinstance(v, str) else v.get("condition", "unknown")
    return out


def parse_list_file(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [l.rstrip() for l in raw]
    blocks, i, n = [], 0, len(lines)
    while i < n:
        a = lines[i].strip()
        if ONE_DIGIT.match(a) and i + 1 < n and ONE_DIGIT.match(lines[i + 1].strip()):
            j, code_lines = i + 2, []
            while j < n and not ANY_NUM.match(lines[j].strip()):
                if lines[j].strip():
                    code_lines.append(lines[j])
                j += 1
            if j < n and CODE_LINE.match(lines[j].strip()) and code_lines:
                blocks.append({
                    "start_line": i + 1,
                    "complexity": int(a),
                    "n_lines_declared": int(lines[i + 1].strip()),
                    "lines": code_lines,
                    "file_code": int(lines[j].strip()),
                })
                i = j + 1
                continue
        i += 1
    n_code_lines = sum(1 for l in lines if CODE_LINE.match(l.strip()))
    return blocks, raw, n_code_lines


def parses(text: str) -> bool:
    text = text.strip("\n")
    for cand in (text, text + "\n    pass", text + "\n        pass"):
        try:
            ast.parse(cand)
            return True
        except (SyntaxError, ValueError):
            continue
    return False


def find_offset(file_codes: set[int], eeg_codes: set[int]):
    hits = [off for off in range(-10, 11)
            if {c + off for c in file_codes} == eeg_codes]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli-dir", default=None)
    ap.add_argument("--codebook", default=str(DEFAULT_CODEBOOK))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--offset", type=int, default=None,
                    help="force file_code + offset = C_code (default: auto-detect)")
    args = ap.parse_args()

    stim_dir = Path(args.stimuli_dir) if args.stimuli_dir else next(
        (p for p in STIM_CANDIDATES if p.exists()), None)
    if stim_dir is None or not stim_dir.exists():
        print("ERROR: stimuli directory not found. Pass --stimuli-dir.")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook = load_codebook(Path(args.codebook))
    eeg_codes = set(codebook)

    rows, first_raw = [], None
    print(f"Stimuli directory: {stim_dir}\n")
    for ln in range(1, 5):
        f = stim_dir / f"Stimuli_v4_List{ln}.txt"
        if not f.exists():
            print(f"List {ln}: {f.name} NOT FOUND")
            continue
        blocks, raw, n_code_lines = parse_list_file(f)
        first_raw = first_raw or (f.name, raw)
        flag = "" if len(blocks) == n_code_lines else "   <-- MISMATCH, parser missed blocks"
        print(f"List {ln}: {len(blocks)} blocks parsed, "
              f"{n_code_lines} three-digit code lines in file{flag}")
        bad_len = sum(1 for b in blocks if len(b["lines"]) != b["n_lines_declared"])
        if bad_len:
            print(f"         {bad_len} blocks have a different line count than declared")
        n_practice = sum(1 for b in blocks if b["file_code"] in PRACTICE_CODES)
        blocks = [b for b in blocks if b["file_code"] not in PRACTICE_CODES]
        print(f"         removed {n_practice} practice blocks, {len(blocks)} experimental blocks left")
        for b in blocks:
            crit = b["lines"][-1].strip()
            rows.append({
                "list": ln, "start_line": b["start_line"], "file_code": b["file_code"],
                "critical_line": crit,
                "critical_parses": parses(crit),
                "whole_parses": parses("\n".join(b["lines"])),
            })

    if first_raw:
        print(f"\nFirst 14 raw lines of {first_raw[0]} (confirm the format looks right):")
        for l in first_raw[1][:14]:
            print(f"   | {l}")

    if not rows:
        print("\nNo blocks parsed. Paste the raw head above and I'll adjust the parser.")
        return
    df = pd.DataFrame(rows)

    file_codes = {int(c) for c in df["file_code"].unique()}
    print(f"\nCondition codes found in the files: {sorted(file_codes)}")
    offset = args.offset if args.offset is not None else find_offset(file_codes, eeg_codes)
    if offset is None:
        print("No constant offset maps the file codes onto the EEG C-codes "
              f"{sorted(eeg_codes)}.\nShowing parse rates by raw file code only.")
        df["c_code"], df["condition"] = pd.NA, "UNMAPPED"
    else:
        print(f"Offset found: C-code = file code {offset:+d}")
        df["c_code"] = df["file_code"] + offset
        df["condition"] = df["c_code"].map(codebook).fillna("NOT_IN_CODEBOOK")

    lines = ["=" * 76, "LABEL CHECK BY PARSING STIMULUS LINES", "=" * 76, ""]

    # Per-list count profile against the EEG profile
    if offset is not None:
        lines.append("Count per C-code in each list vs EEG profile "
                     f"{dict(sorted(EEG_PROFILE.items()))}")
        for ln, g in df.groupby("list"):
            counts = g["c_code"].value_counts().to_dict()
            ok = {c: counts.get(c, 0) for c in EEG_PROFILE} == EEG_PROFILE
            lines.append(f"  List {ln}: {dict(sorted(counts.items()))}   "
                         f"{'MATCHES' if ok else 'DIFFERS'}")
        lines.append("")

    key = "c_code" if offset is not None else "file_code"
    tab = (df.groupby(key)
             .agg(n=("critical_line", "size"),
                  condition=("condition", "first"),
                  critical_parse_rate=("critical_parses", "mean"),
                  whole_parse_rate=("whole_parses", "mean"))
             .reset_index())
    tab["expected_rate"] = tab["condition"].map(
        lambda c: 1.0 if EXPECT_PARSE.get(c) else (0.0 if c in EXPECT_PARSE else float("nan")))
    lines.append("Parse rate by code (critical line alone | whole block):")
    lines.append(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    # Mismatches
    mism_all = 0
    if offset is not None:
        for _, r in df.iterrows():
            exp = EXPECT_PARSE.get(r["condition"])
            if exp is not None and r["critical_parses"] != exp:
                mism_all += 1
        shown = 0
        if mism_all:
            lines.append(f"Lines whose parse result contradicts their label: {mism_all}")
            for _, r in df.iterrows():
                exp = EXPECT_PARSE.get(r["condition"])
                if exp is not None and r["critical_parses"] != exp and shown < 15:
                    lines.append(f"  List {r['list']} line {r['start_line']} "
                                 f"C={r['c_code']} ({r['condition']}), "
                                 f"parses={r['critical_parses']}: {r['critical_line']}")
                    shown += 1
            lines.append("")

        wf_sem = df[df["condition"].isin(["well_formed", "semantically_implausible"])]
        syn_dbl = df[df["condition"].isin(["syntactically_invalid", "doubly_anomalous"])]
        r_ok, r_bad = wf_sem["critical_parses"].mean(), syn_dbl["critical_parses"].mean()
        lines.append("VERDICT")
        if mism_all == 0:
            lines.append("  PASS: every well-formed and semantic line parses; every "
                         "syntactic and doubly anomalous line fails.")
        elif r_ok < 0.1 and r_bad > 0.9:
            lines.append("  LABELS APPEAR SWAPPED: well-formed/semantic lines fail to "
                         "parse and syntactic/doubly anomalous lines parse. Every "
                         "syntactic result would flip sign. Do not publish; investigate.")
        else:
            lines.append(f"  PARTIAL: {mism_all} lines contradict their labels "
                         f"(wf+sem parse rate {r_ok:.3f}, syn+dbl parse rate {r_bad:.3f}). "
                         "Inspect the lines listed above before drawing a conclusion.")
    lines.append("=" * 76)

    report = "\n".join(lines)
    print("\n" + report)
    (out_dir / "condition_label_parse_check.txt").write_text(report, encoding="utf-8")
    df.to_csv(out_dir / "condition_label_parse_check_lines.csv", index=False)
    print(f"\n[SAVED] {out_dir / 'condition_label_parse_check.txt'}")
    print(f"[SAVED] {out_dir / 'condition_label_parse_check_lines.csv'}")


if __name__ == "__main__":
    main()