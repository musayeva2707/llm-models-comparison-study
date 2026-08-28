#!/usr/bin/env python3
"""Figure out WHY runs fail — is it rate limits, or something about the
content? Reads the calls log and the transcripts to see the actual errors."""
import json
from pathlib import Path
from collections import Counter

run_dir = Path("runs/main")
calls = [json.loads(l) for l in open(run_dir/"calls.jsonl", encoding="utf-8") if l.strip()] if (run_dir/"calls.jsonl").exists() else []
results = [json.loads(l) for l in open(run_dir/"results.jsonl", encoding="utf-8") if l.strip()]

# Which calls errored, and what was the error?
errored = [c for c in calls if c.get("error")]
print(f"Total calls logged: {len(calls)}")
print(f"Calls with errors:  {len(errored)}\n")

# What KIND of errors?
print("=== error types (first 200 chars) ===")
kinds = Counter()
for c in errored:
    e = str(c.get("error",""))[:200]
    # bucket by keyword
    if "429" in e or "quota" in e.lower(): kinds["rate_limit_429"] += 1
    elif "400" in e: kinds["bad_request_400"] += 1
    elif "500" in e or "503" in e: kinds["server_error_5xx"] += 1
    elif "timeout" in e.lower(): kinds["timeout"] += 1
    else: kinds["other"] += 1
for k,v in kinds.most_common():
    print(f"  {k}: {v}")

# Show a few actual error messages, especially non-rate-limit ones
print("\n=== sample actual errors (non-429 first) ===")
non429 = [c for c in errored if "429" not in str(c.get("error","")) and "quota" not in str(c.get("error","")).lower()]
for c in (non429[:3] or errored[:3]):
    print(f"\n[{c.get('item_id')} {c.get('agent_role')}] ")
    print(f"  {str(c.get('error'))[:300]}")

# Which agent role fails most? (planner/worker/aggregator)
print("\n=== which pipeline step errors most? ===")
print(Counter(c.get("agent_role") for c in errored))

# Are failures concentrated in reasoning?
print("\n=== failed FINAL runs by category ===")
bad = [r for r in results if r.get("catastrophic")]
print(Counter(r["category"] for r in bad))
