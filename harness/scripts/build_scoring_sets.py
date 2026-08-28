#!/usr/bin/env python3
"""
Build the scoring sets.

Implements the annotation sampling plan:

  1. AUTOMATIC   every response, free, no human
  2. JUDGE       every response, cheap, no human
  3. HUMAN-A     stratified validation sample (~300) -> judge agreement
  4. HUMAN-B     failure investigation sample (~150) -> error taxonomy

Usage
-----
  # automatic checks + build the human sheets
  python scripts/build_scoring_sets.py --run-dir runs/main --items data/prompts.jsonl

  # add judge scores (use a DIFFERENT model family from your backbone)
  python scripts/build_scoring_sets.py --run-dir runs/main --items data/prompts.jsonl \
      --judge-backend anthropic --judge-model claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmeval.metrics import score_response, run_consistency
from llmeval.rubric import DIMENSIONS, RUBRIC_VERSION
from llmeval.judge import judge_response
from llmeval.adapters import build_adapter

SEED = 20260722   # fixed so the sample is reproducible; report it in the paper


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_items(p: Path) -> dict[str, dict]:
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        d = json.loads(line)
        out[d["item_id"]] = d
    return out


def stratified_sample(rows, key_fn, n_total, rng):
    """Draw n_total spread as evenly as possible across strata."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)
    if not buckets:
        return []
    per = max(1, n_total // len(buckets))
    picked = []
    for k in sorted(buckets):
        b = buckets[k]
        rng.shuffle(b)
        picked.extend(b[:per])
    rng.shuffle(picked)
    return picked[:n_total]


def write_sheet(path: Path, rows: list[dict], extra_cols: list[str]) -> None:
    """
    Annotation sheet.

    `architecture` and `config` are deliberately absent. Score blind or the
    rubric measures your expectations.
    """
    cols = ["run_key", "item_id", "category", "task", "response"] \
        + [d.code for d in DIMENSIONS] + extra_cols
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{d.code: "" for d in DIMENSIONS},
                        **{c: "" for c in extra_cols}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--human-validation-n", type=int, default=300)
    ap.add_argument("--human-failure-n", type=int, default=150)
    ap.add_argument("--judge-backend", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    args = ap.parse_args()

    rng = random.Random(SEED)
    run_dir = Path(args.run_dir)
    results = load_jsonl(run_dir / "results.jsonl")
    items = load_items(Path(args.items))
    out_dir = run_dir / "scoring"
    out_dir.mkdir(exist_ok=True)

    print(f"responses: {len(results)}   rubric v{RUBRIC_VERSION}   seed {SEED}")

    # ---------------------------------------------------------------- 1. auto
    enriched = []
    for r in results:
        it = items.get(r["item_id"], {})
        auto = score_response(
            r["final_answer"],
            r["category"],
            reference=it.get("reference", ""),
            direction=it.get("direction", ""),
        )
        enriched.append({**r, **auto,
                         "task": it.get("prompt", ""),
                         "reference": it.get("reference", "")})

    with open(out_dir / "automatic.jsonl", "w", encoding="utf-8") as f:
        for e in enriched:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_drift = sum(e["auto_drift_flag"] for e in enriched)
    n_empty = sum(e["auto_empty"] for e in enriched)
    n_cat = sum(e.get("catastrophic", False) for e in enriched)
    print(f"  drift flagged {n_drift}  empty {n_empty}  catastrophic {n_cat}")

    # ------------------------------------------------- 2. consistency (RQ2)
    groups = defaultdict(list)
    for e in enriched:
        groups[(e["item_id"], e["architecture"], e["config"])].append(e)
    cons = []
    for (iid, arch, cfg), grp in groups.items():
        grp.sort(key=lambda x: x["run_index"])
        c = run_consistency([g["final_answer"] for g in grp])
        cons.append({"item_id": iid, "architecture": arch, "config": cfg, **c})
    with open(out_dir / "consistency.jsonl", "w", encoding="utf-8") as f:
        for c in cons:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  consistency computed for {len(cons)} conditions")

    # ------------------------------------------------------------- 3. judge
    if args.judge_backend and args.judge_model:
        kw = {}
        if args.judge_base_url:
            kw["base_url"] = args.judge_base_url
        adapter = build_adapter(args.judge_backend, args.judge_model, **kw)

        done = set()
        jpath = out_dir / "judge.jsonl"
        if jpath.exists():
            done = {json.loads(l)["run_key"] for l in open(jpath, encoding="utf-8") if l.strip()}

        print(f"  judging with {args.judge_backend}/{args.judge_model} "
              f"({len(done)} already done)")
        fails = 0
        with open(jpath, "a", encoding="utf-8") as f:
            for i, e in enumerate(enriched, 1):
                if e["run_key"] in done:
                    continue
                v = judge_response(
                    adapter, e["run_key"], e["task"], e["final_answer"],
                    e["category"], e.get("reference", ""),
                )
                if not v.parse_ok:
                    fails += 1
                f.write(json.dumps({
                    "run_key": v.run_key, "d1": v.d1, "d2": v.d2, "d3": v.d3,
                    "d4": v.d4, "d5": v.d5, "quality": v.quality,
                    "rationale": v.rationale, "error_codes": v.error_codes,
                    "parse_ok": v.parse_ok, "annotator": "judge",
                }, ensure_ascii=False) + "\n")
                if i % 50 == 0:
                    print(f"    {i}/{len(enriched)}")
        if fails:
            print(f"  WARNING: {fails} judge outputs failed to parse")
    else:
        print("  [judge skipped; pass --judge-backend and --judge-model]")

    # ----------------------------------------- 4. HUMAN-A: validation sample
    val = stratified_sample(
        enriched,
        lambda r: (r["category"], r["architecture"], r["config"]),
        args.human_validation_n,
        rng,
    )
    write_sheet(out_dir / "human_validation.csv", val, ["annotator", "notes"])
    print(f"  HUMAN-A validation sample: {len(val)} rows "
          f"-> scoring/human_validation.csv")

    # --------------------------------------- 5. HUMAN-B: failure investigation
    val_keys = {v["run_key"] for v in val}

    def suspicion(r):
        s = 0
        s += 3 if r.get("catastrophic") else 0
        s += 2 if r.get("auto_drift_flag") else 0
        s += 2 if r.get("auto_empty") else 0
        s += 1 if r.get("auto_script_mixed") else 0
        if r.get("auto_chrf") is not None and r["auto_chrf"] < 0.25:
            s += 2
        if r.get("auto_ref_match") is False:
            s += 1
        return s

    pool = [r for r in enriched if r["run_key"] not in val_keys]
    pool.sort(key=suspicion, reverse=True)
    # Half worst-scoring, half random, so the taxonomy is not built purely
    # from what the automatic screen already knows how to catch.
    half = args.human_failure_n // 2
    fail = pool[:half]
    rest = pool[half:]
    rng.shuffle(rest)
    fail += rest[: args.human_failure_n - half]

    write_sheet(out_dir / "human_failures.csv", fail,
                ["error_codes", "annotator", "notes"])
    print(f"  HUMAN-B failure sample: {len(fail)} rows "
          f"-> scoring/human_failures.csv")

    total_human = len(val) + len(fail)
    print(f"\nHuman workload: {total_human} responses "
          f"(~{total_human * 1.5 / 60:.1f} h at 90 s each)")
    print(f"Automatic + judge cover all {len(enriched)}.")


if __name__ == "__main__":
    main()
