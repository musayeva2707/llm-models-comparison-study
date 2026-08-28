#!/usr/bin/env python3
"""
Safely clear ONLY the empty/failed runs so the harness will retry them.

Your good answers are never touched. This:
  1. Backs up results.jsonl -> results.jsonl.backup
  2. Keeps every GOOD run, drops only the empty/failed ones
  3. Writes the cleaned file back
After running this, re-run your normal experiment command and the harness
will re-attempt only the missing runs.

    python retry_empties.py            # shows what it WOULD do (safe preview)
    python retry_empties.py --write    # actually does it (makes a backup first)
"""
import json, shutil, sys
from pathlib import Path

src = Path("runs/main/results.jsonl")
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

def is_good(r):
    ans = (r.get("final_answer") or "").strip()
    return bool(ans) and not r.get("catastrophic")

good = [r for r in rows if is_good(r)]
empty = [r for r in rows if not is_good(r)]

print(f"Total records:      {len(rows)}")
print(f"  GOOD (keep):      {len(good)}")
print(f"  EMPTY (retry):    {len(empty)}")
print()

# Show which runs will be retried
from collections import Counter
by = Counter((r["architecture"], r["category"]) for r in empty)
print("Empty runs that will be re-attempted:")
for (arch, cat), n in sorted(by.items()):
    print(f"  {arch} / {cat}: {n}")

if "--write" in sys.argv:
    backup = src.with_suffix(".jsonl.backup")
    shutil.copy(src, backup)
    with open(src, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nBacked up original -> {backup}")
    print(f"Wrote {len(good)} good runs back to {src}")
    print("The empty runs are now gone from the file, so the harness will")
    print("retry them. Run your normal experiment command next.")
else:
    print("\n(Preview only. Nothing changed. Run with --write to apply.)")
