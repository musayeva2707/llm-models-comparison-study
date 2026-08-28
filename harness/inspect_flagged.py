#!/usr/bin/env python3
"""Show what's actually inside the runs flagged catastrophic — the real
answer text, so we can tell if they're truly empty or wrongly flagged."""
import json
from pathlib import Path

rows = [json.loads(l) for l in open("runs/main/results.jsonl", encoding="utf-8") if l.strip()]
flagged = [r for r in rows if r.get("catastrophic")]

print(f"Total runs: {len(rows)}   flagged catastrophic: {len(flagged)}\n")

# Of the flagged ones, how many actually have a real answer?
has_answer = [r for r in flagged if (r.get("final_answer") or "").strip()]
truly_empty = [r for r in flagged if not (r.get("final_answer") or "").strip()]
print(f"Flagged BUT have a real answer (wrongly flagged?): {len(has_answer)}")
print(f"Flagged AND genuinely empty:                       {len(truly_empty)}\n")

print("=== 5 examples of flagged runs WITH answers ===")
for r in has_answer[:5]:
    ans = (r.get("final_answer") or "").strip()
    print(f"\n[{r['item_id']} {r['architecture']} r{r['run_index']}] "
          f"reason={r.get('catastrophic_reason','?')} completion_tok={r.get('total_completion_tokens')}")
    print(f"  ANSWER: {ans[:160]}")

print("\n=== reasons breakdown ===")
from collections import Counter
print(Counter(r.get("catastrophic_reason","") for r in flagged))
