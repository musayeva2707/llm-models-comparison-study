#!/usr/bin/env python3
"""
Check how much GOOD data you have so far.

    python check_progress.py

Reads runs/main/results.jsonl and reports:
  - how many runs completed
  - how many are GOOD (real answer) vs EMPTY (catastrophic/rate-limited)
  - where the good data stops
"""

import json
import sys
from pathlib import Path
from collections import Counter

path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/main/results.jsonl")
if not path.exists():
    print(f"No file at {path}")
    sys.exit(1)

rows = []
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if line:
        rows.append(json.loads(line))

total = len(rows)
good = [r for r in rows if not r.get("catastrophic") and r.get("final_answer", "").strip()]
empty = [r for r in rows if r.get("catastrophic") or not r.get("final_answer", "").strip()]

print(f"Total runs recorded:   {total}")
print(f"  GOOD (real answer):  {len(good)}")
print(f"  EMPTY (failed):      {len(empty)}")
print()

# By architecture
arch = Counter(r["architecture"] for r in good)
print("Good answers by architecture:")
for a, n in arch.items():
    print(f"  {a}: {n}")
print()

# By category
cat = Counter(r["category"] for r in good)
print("Good answers by category:")
for c in ["fact", "trans", "reas", "cult"]:
    print(f"  {c}: {cat.get(c, 0)}")
print()

# Where did it go bad? Find the last good run and first sustained empty.
if empty:
    first_empty_idx = next((i for i, r in enumerate(rows)
                            if r.get("catastrophic") or not r.get("final_answer", "").strip()), None)
    print(f"First failure at run #{first_empty_idx + 1} of {total}")
    # count consecutive empties at the end
    tail_empty = 0
    for r in reversed(rows):
        if r.get("catastrophic") or not r.get("final_answer", "").strip():
            tail_empty += 1
        else:
            break
    print(f"Consecutive failures at the end: {tail_empty}")
    print("(A long run of end failures = you hit the rate/daily limit there.)")
