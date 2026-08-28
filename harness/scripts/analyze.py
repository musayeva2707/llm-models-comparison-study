#!/usr/bin/env python3
"""
Analysis.

Produces the tables the paper needs, plus the budget-matching audit that
establishes the comparison was fair. Run after any experiment:

    python scripts/analyze.py --run-dir runs/exp01

Merge in your rubric scores to get the quality tables:

    python scripts/analyze.py --run-dir runs/exp01 --scores data/scores.csv

scores.csv schema:
    run_key,d1,d2,d3,d4,d5,annotator
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def bootstrap_ci(vals, n_boot: int = 5000, alpha: float = 0.05):
    """
    Bootstrap CI on the mean.

    Use this rather than a t-test. With 30 items per category you do not have
    the power for significance claims, and reporting an interval is honest
    about that where a p-value would not be.
    """
    vals = np.asarray([v for v in vals if not pd.isna(v)], dtype=float)
    if len(vals) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(0)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def budget_audit(results: pd.DataFrame) -> pd.DataFrame:
    """
    THE table that makes the comparison credible.

    If the pipeline consumed materially more than the standalone system at the
    same requested budget, any quality advantage is confounded and you must
    say so. Tran & Kiela (2026) found exactly this leakage in some APIs.
    """
    g = results.groupby(["architecture", "config"]).agg(
        n_runs=("run_key", "count"),
        req_budget=("requested_budget", "mean"),
        calls=("n_calls", "mean"),
        prompt_tok=("total_prompt_tokens", "mean"),
        completion_tok=("total_completion_tokens", "mean"),
        reasoning_tok=("total_reasoning_tokens", "mean"),
        latency_s=("wall_clock_s", "mean"),
    ).round(1).reset_index()
    g["budget_overrun"] = (g["reasoning_tok"] / g["req_budget"]).round(2)
    return g


def reliability(results: pd.DataFrame, scores: pd.DataFrame | None) -> pd.DataFrame:
    """RQ2: stability across the k repetitions."""
    df = results.copy()
    if scores is not None:
        df = df.merge(scores, on="run_key", how="left")

    rows = []
    for (arch, cfg, cat), grp in df.groupby(["architecture", "config", "category"]):
        per_item_sd = grp.groupby("item_id")["quality"].std() if "quality" in grp else None
        rows.append({
            "architecture": arch,
            "config": cfg,
            "category": cat,
            "n": len(grp),
            "catastrophic_rate": round(grp["catastrophic"].mean(), 3),
            "mean_within_item_sd": (round(per_item_sd.mean(), 3)
                                    if per_item_sd is not None
                                    and not per_item_sd.isna().all() else None),
        })
    return pd.DataFrame(rows)


def quality_table(results: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """RQ1: quality by architecture x category, with bootstrap CIs."""
    df = results.merge(scores, on="run_key", how="inner")
    rows = []
    for (arch, cfg, cat), grp in df.groupby(["architecture", "config", "category"]):
        lo, hi = bootstrap_ci(grp["quality"].values)
        row = {
            "architecture": arch,
            "config": cfg,
            "category": cat,
            "n": len(grp),
            "quality": round(grp["quality"].mean(), 2),
            "ci_low": round(lo, 2) if not np.isnan(lo) else None,
            "ci_high": round(hi, 2) if not np.isnan(hi) else None,
        }
        for d in ["d1", "d2", "d3", "d4", "d5"]:
            if d in grp:
                row[d] = round(grp[d].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True, na_rep="--")
    return (
        "\\begin{table}[t]\n\\centering\\small\n"
        + body
        + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scores", default=None)
    ap.add_argument("--latex", action="store_true",
                    help="also emit LaTeX tables for the paper")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results = load_jsonl(run_dir / "results.jsonl")
    calls = load_jsonl(run_dir / "calls.jsonl")

    print(f"runs:  {len(results)}")
    print(f"calls: {len(calls)}")

    scores = None
    if args.scores:
        scores = pd.read_csv(args.scores)
        dims = [d for d in ["d1", "d2", "d3", "d4", "d5"] if d in scores.columns]
        scores["quality"] = scores[dims].sum(axis=1)
        # Average across annotators where an item was double-scored.
        scores = scores.groupby("run_key", as_index=False)[dims + ["quality"]].mean()

    print("\n=== BUDGET AUDIT (report this in the paper) ===")
    audit = budget_audit(results)
    print(audit.to_string(index=False))

    print("\n=== PER-AGENT COST ===")
    if not calls.empty:
        per_agent = calls.groupby(["architecture", "agent_role"]).agg(
            n=("run_key", "count"),
            completion_tok=("completion_tokens", "mean"),
            reasoning_tok=("reasoning_tokens", "mean"),
            latency_s=("latency_s", "mean"),
        ).round(1)
        print(per_agent.to_string())

    print("\n=== RELIABILITY (RQ2) ===")
    rel = reliability(results, scores)
    print(rel.to_string(index=False))

    qual = None
    if scores is not None:
        print("\n=== QUALITY (RQ1) ===")
        qual = quality_table(results, scores)
        print(qual.to_string(index=False))
    else:
        print("\n[no --scores supplied; quality tables skipped]")

    out = run_dir / "tables"
    out.mkdir(exist_ok=True)
    audit.to_csv(out / "budget_audit.csv", index=False)
    rel.to_csv(out / "reliability.csv", index=False)
    if qual is not None:
        qual.to_csv(out / "quality.csv", index=False)

    if args.latex:
        with open(out / "tables.tex", "w", encoding="utf-8") as f:
            f.write(to_latex(audit,
                             "Budget audit: requested versus consumed computation "
                             "by architecture.", "tab:budget-audit"))
            f.write("\n")
            f.write(to_latex(rel, "Reliability across repeated runs.",
                             "tab:reliability"))
            if qual is not None:
                f.write("\n")
                f.write(to_latex(qual, "Response quality by architecture and "
                                       "task category.", "tab:quality"))
        print(f"\nLaTeX -> {out / 'tables.tex'}")

    print(f"CSV   -> {out}/")


if __name__ == "__main__":
    main()
