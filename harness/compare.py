#!/usr/bin/env python3
"""
See standalone vs pipeline answers side by side, for the same question.

    python compare.py                  # walk through a few from each category
    python compare.py --category cult  # only cultural questions
    python compare.py --item cult_030  # one specific question
    python compare.py --all            # every question with both answers

Reads the repaired data (results_fixed.jsonl) so you see your clean answers.
Nothing is changed — this only reads and prints.
"""
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--category", help="fact / trans / reas / cult")
ap.add_argument("--item", help="a specific item_id, e.g. cult_030")
ap.add_argument("--all", action="store_true", help="show every comparable item")
ap.add_argument("--per-category", type=int, default=2,
                help="how many items per category in the default view")
args = ap.parse_args()

# Prefer the repaired file; fall back to the original.
run_dir = Path("runs/main")
src = run_dir / "results_fixed.jsonl"
if not src.exists():
    src = run_dir / "results.jsonl"
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

def good(r):
    return not r.get("catastrophic") and (r.get("final_answer") or "").strip()

# Group good answers by item_id, keeping one standalone and one sequential each.
by_item = defaultdict(dict)
for r in rows:
    if not good(r):
        continue
    arch = r["architecture"]
    # keep the first good run we see for each architecture
    if arch not in by_item[r["item_id"]]:
        by_item[r["item_id"]][arch] = r

# Only items that have BOTH a standalone and a sequential answer are comparable.
comparable = {iid: d for iid, d in by_item.items()
              if "standalone" in d and "sequential" in d}

# Pick which items to show
def category_of(iid):
    return iid.split("_")[0]

if args.item:
    chosen = [args.item] if args.item in comparable else []
    if not chosen:
        print(f"No comparable pair for {args.item} "
              f"(need both a standalone and a sequential good answer).")
        sys.exit(0)
elif args.all:
    chosen = sorted(comparable)
elif args.category:
    chosen = sorted(i for i in comparable if category_of(i) == args.category)
else:
    # default: a few from each category
    chosen = []
    seen = defaultdict(int)
    for iid in sorted(comparable):
        c = category_of(iid)
        if seen[c] < args.per_category:
            chosen.append(iid)
            seen[c] += 1

CAT_NAME = {"fact": "FACTUAL", "trans": "TRANSLATION",
            "reas": "REASONING", "cult": "CULTURAL"}

def wrap(text, width=78, indent="   "):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(indent + cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(indent + cur)
    return "\n".join(lines)

print(f"\nReading: {src}")
print(f"Comparable items (have both answers): {len(comparable)}")
print(f"Showing: {len(chosen)}\n")

for iid in chosen:
    d = comparable[iid]
    st = d["standalone"]
    sq = d["sequential"]
    cat = CAT_NAME.get(category_of(iid), category_of(iid).upper())

    print("=" * 82)
    print(f"  {iid}   [{cat}]")
    print("=" * 82)
    # The prompt isn't stored in results, so show the reference if present
    ref = st.get("reference") or sq.get("reference") or ""
    if ref:
        print(f"  Expected: {ref}")
        print("-" * 82)

    st_ans = (st.get("final_answer") or "").strip()
    sq_ans = (sq.get("final_answer") or "").strip()
    st_tok = st.get("total_completion_tokens", "?")
    sq_tok = sq.get("total_completion_tokens", "?")

    print(f"\n  SINGLE AI  ({st_tok} tokens):")
    print(wrap(st_ans))
    print(f"\n  TEAM OF AIs  ({sq_tok} tokens):")
    print(wrap(sq_ans))
    print()

# A quick quantitative teaser: average answer length by architecture
print("=" * 82)
print("  QUICK PATTERN: average answer length (tokens)")
print("=" * 82)
tot = defaultdict(list)
for iid, d in comparable.items():
    tot["standalone"].append(d["standalone"].get("total_completion_tokens", 0))
    tot["sequential"].append(d["sequential"].get("total_completion_tokens", 0))
for arch in ("standalone", "sequential"):
    vals = tot[arch]
    if vals:
        avg = sum(vals) / len(vals)
        label = "Single AI" if arch == "standalone" else "Team of AIs"
        print(f"  {label:12s}: {avg:6.0f} tokens avg   (across {len(vals)} items)")
print("\n(This is just a length teaser — real quality scoring comes next.)")
