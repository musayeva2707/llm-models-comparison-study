#!/usr/bin/env python3
"""
Layer 1 — automatic checks on every collected answer.

Runs the free, instant, mechanical checks (length, language, script, chrF++
for translations, reference-match for facts, cross-run consistency) on all
answers and writes a per-answer CSV you can open in Excel.

    python auto_score.py

Outputs:
    runs/main/auto_scores.csv      one row per answer, every check
    runs/main/auto_summary.csv     averages by architecture x category
    (also prints the summary to screen)

Reads results_fixed.jsonl (the repaired data) if present, else results.jsonl.
Nothing is modified — this only reads and writes new files.
"""
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, ".")
from llmeval.metrics import score_response, run_consistency

run_dir = Path("runs/main")
src = run_dir / "results_fixed.jsonl"
if not src.exists():
    src = run_dir / "results.jsonl"
    print(f"(results_fixed.jsonl not found — using {src.name})")

rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

# Join back to the prompts to recover reference + direction (not stored in results).
prompts = {}
pf = Path("data/prompts.jsonl")
if pf.exists():
    for l in open(pf, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            prompts[d["item_id"]] = d

def good(r):
    return not r.get("catastrophic") and (r.get("final_answer") or "").strip()

good_rows = [r for r in rows if good(r)]
print(f"Scoring {len(good_rows)} good answers (of {len(rows)} recorded)...\n")

# --- Per-answer scoring ---
per_answer = []
for r in good_rows:
    iid = r["item_id"]
    p = prompts.get(iid, {})
    ref = p.get("reference", "")
    direction = p.get("direction", "")
    ans = r.get("final_answer", "")

    checks = score_response(ans, r["category"], reference=ref, direction=direction)
    row = {
        "item_id": iid,
        "category": r["category"],
        "architecture": r["architecture"],
        "run_index": r.get("run_index", 0),
        "completion_tokens": r.get("total_completion_tokens", ""),
        "wall_clock_s": round(r.get("wall_clock_s", 0), 1),
        **checks,
    }
    per_answer.append(row)

# Write per-answer CSV (union of all keys, stable order)
fieldnames = ["item_id", "category", "architecture", "run_index",
              "completion_tokens", "wall_clock_s",
              "auto_length_chars", "auto_expect_lang", "auto_likely_lang",
              "auto_drift_flag", "auto_script_mixed", "auto_turkish_hits",
              "auto_apostrophe_variants", "auto_empty", "auto_chrf",
              "auto_ref_match"]
out_csv = run_dir / "auto_scores.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in per_answer:
        w.writerow(row)
print(f"Wrote per-answer scores -> {out_csv}")

# --- Consistency across the 3 repeat runs (per item x architecture) ---
groups = defaultdict(list)
for r in good_rows:
    groups[(r["item_id"], r["architecture"])].append(r.get("final_answer", ""))
consistency = {}
for key, answers in groups.items():
    if len(answers) >= 2:
        consistency[key] = run_consistency(answers)

# --- Summary by architecture x category ---
def agg(rows_subset, field):
    vals = [x[field] for x in rows_subset if isinstance(x.get(field), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0

summary = []
print("\n" + "=" * 74)
print(f"{'architecture':<12}{'category':<8}{'n':>5}{'avg_len':>10}{'avg_tok':>10}{'drift%':>9}")
print("=" * 74)
for arch in ("standalone", "sequential"):
    for cat in ("fact", "trans", "reas", "cult"):
        sub = [x for x in per_answer if x["architecture"] == arch and x["category"] == cat]
        if not sub:
            continue
        n = len(sub)
        avg_len = agg(sub, "auto_length_chars")
        avg_tok = agg(sub, "completion_tokens")
        drift = 100 * sum(1 for x in sub if x.get("auto_drift_flag")) / n
        row = {"architecture": arch, "category": cat, "n": n,
               "avg_length_chars": round(avg_len, 1),
               "avg_completion_tokens": round(avg_tok, 1),
               "drift_pct": round(drift, 1)}
        # translation chrF++ average
        chrf_vals = [x["auto_chrf"] for x in sub if isinstance(x.get("auto_chrf"), (int, float))]
        if chrf_vals:
            row["avg_chrf"] = round(sum(chrf_vals) / len(chrf_vals), 3)
        # fact ref-match rate
        rm = [x for x in sub if "auto_ref_match" in x and isinstance(x["auto_ref_match"], bool)]
        if rm:
            row["ref_match_pct"] = round(100 * sum(1 for x in rm if x["auto_ref_match"]) / len(rm), 1)
        summary.append(row)
        print(f"{arch:<12}{cat:<8}{n:>5}{avg_len:>10.0f}{avg_tok:>10.0f}{drift:>8.1f}%")

# Write summary CSV
sum_csv = run_dir / "auto_summary.csv"
sum_fields = ["architecture", "category", "n", "avg_length_chars",
              "avg_completion_tokens", "drift_pct", "avg_chrf", "ref_match_pct"]
with open(sum_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
    w.writeheader()
    for row in summary:
        w.writerow(row)
print("=" * 74)
print(f"\nWrote summary -> {sum_csv}")

# --- The headline check: your "standalone is longer" hunch ---
st = [x["auto_length_chars"] for x in per_answer if x["architecture"] == "standalone"]
sq = [x["auto_length_chars"] for x in per_answer if x["architecture"] == "sequential"]
if st and sq:
    print("\n" + "-" * 60)
    print("YOUR HUNCH, CHECKED across all answers:")
    print(f"  Single AI  average length: {sum(st)/len(st):>6.0f} chars")
    print(f"  Team of AIs average length: {sum(sq)/len(sq):>6.0f} chars")
    if sum(st)/len(st) > sum(sq)/len(sq):
        print("  -> Confirmed: the single model IS longer on average.")
    else:
        print("  -> Overturned: the pipeline is actually longer on average.")
    print("-" * 60)

# Consistency headline
if consistency:
    st_c = [v["mean_pairwise_chrf"] for k, v in consistency.items()
            if k[1] == "standalone" and "mean_pairwise_chrf" in v]
    sq_c = [v["mean_pairwise_chrf"] for k, v in consistency.items()
            if k[1] == "sequential" and "mean_pairwise_chrf" in v]
    if st_c and sq_c:
        print("\nConsistency across the 3 repeat runs (higher = more stable):")
        print(f"  Single AI  : {sum(st_c)/len(st_c):.3f}")
        print(f"  Team of AIs: {sum(sq_c)/len(sq_c):.3f}")

print("\nNext: open auto_scores.csv in Excel to explore per-answer,")
print("or auto_summary.csv for the architecture x category averages.")
