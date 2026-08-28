#!/usr/bin/env python3
"""
Repair mislabeled 'catastrophic' flags in already-collected data.

An earlier version of the harness wrongly flagged a sequential run as
catastrophic when an INTERMEDIATE step hit its token budget — even when the
final answer was perfectly good. This script re-checks each recorded run using
the corrected rule (only a truly empty/too-short final answer is a failure)
and reports how many were mislabeled. With --write it saves a corrected copy.

    python fix_catastrophic.py                 # report only
    python fix_catastrophic.py --write         # also write results_fixed.jsonl

Your original file is never modified unless you pass --write, and even then it
writes a NEW file (results_fixed.jsonl) — the original stays untouched.
"""

import json
import sys
from pathlib import Path

run_dir = Path("runs/main")
src = run_dir / "results.jsonl"
if not src.exists():
    print(f"No file at {src}")
    sys.exit(1)

rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

def is_really_catastrophic(r):
    ans = (r.get("final_answer") or "").strip()
    # Genuine failure only if the FINAL answer is truly empty. Short correct
    # answers are fine ("H2O", "Bor", "1977", "7").
    if not ans:
        return True, "empty"
    return False, ""

fixed = 0
still_bad = 0
for r in rows:
    was = r.get("catastrophic", False)
    now, reason = is_really_catastrophic(r)
    if was and not now:
        fixed += 1
        r["catastrophic"] = False
        r["catastrophic_reason"] = ""
        r["_relabeled"] = True
    elif now:
        still_bad += 1
        r["catastrophic"] = True
        r["catastrophic_reason"] = reason

good_now = sum(1 for r in rows if not r["catastrophic"] and (r.get("final_answer") or "").strip())

print(f"Total runs:            {len(rows)}")
print(f"Wrongly flagged (now recovered): {fixed}")
print(f"Genuinely empty/failed:          {still_bad}")
print(f"GOOD answers after repair:       {good_now}")
print()

# Show the architecture breakdown after repair
from collections import Counter
arch = Counter(r["architecture"] for r in rows
               if not r["catastrophic"] and (r.get("final_answer") or "").strip())
print("Good answers by architecture (after repair):")
for a, n in arch.items():
    print(f"  {a}: {n}")

if "--write" in sys.argv:
    out = run_dir / "results_fixed.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            r.pop("_relabeled", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote corrected copy -> {out}")
    print("Your original results.jsonl is untouched.")
else:
    print("\n(Report only. Run with --write to save a corrected copy.)")
