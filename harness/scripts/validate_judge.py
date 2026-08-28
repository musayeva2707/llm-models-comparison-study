#!/usr/bin/env python3
"""
Judge validation.

Computes agreement between the human validation sample and the judge on the
same responses. This number is what licenses using judge scores for the
responses no human read.

Reports quadratic weighted kappa rather than plain kappa, because the scale is
ordinal: scoring a 4 as a 3 is a smaller error than scoring it a 0, and plain
kappa treats those identically.

Interpretation, conventional but worth stating explicitly in the paper:
    < 0.40   poor      -- do not use judge scores as evidence
    0.40-0.60 moderate -- report judge scores with heavy caveats
    0.60-0.80 substantial
    > 0.80   near-perfect

If agreement is poor on d4 (cultural appropriateness) but good on d1
(correctness), that is not a failure of your method. It is a finding: the
judge can check facts in Uzbek but cannot judge cultural fit, which is
precisely the argument your paper makes about critic agents. Report it.

Usage
-----
  python scripts/validate_judge.py --run-dir runs/main \
      --human data/human_validation_filled.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmeval.rubric import DIMENSIONS  # noqa: E402


def quadratic_weighted_kappa(a, b, min_r=0, max_r=4) -> float:
    """Cohen's kappa with quadratic weights, for ordinal ratings."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n = max_r - min_r + 1

    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[x - min_r, y - min_r] += 1

    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)

    ha = np.bincount(a - min_r, minlength=n)
    hb = np.bincount(b - min_r, minlength=n)
    E = np.outer(ha, hb).astype(float)
    E = E * O.sum() / E.sum() if E.sum() else E

    denom = (w * E).sum()
    return 1.0 - (w * O).sum() / denom if denom else float("nan")


def exact_and_adjacent(a, b) -> tuple[float, float]:
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()), float((np.abs(a - b) <= 1).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--human", required=True,
                    help="filled-in human_validation.csv")
    ap.add_argument("--judge", default=None,
                    help="defaults to <run-dir>/scoring/judge.jsonl")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    jpath = Path(args.judge) if args.judge else run_dir / "scoring" / "judge.jsonl"

    human = pd.read_csv(args.human)
    judge = pd.DataFrame(
        [json.loads(l) for l in open(jpath, encoding="utf-8") if l.strip()]
    )
    judge = judge[judge["parse_ok"]]

    dims = [d.code for d in DIMENSIONS]
    human = human.dropna(subset=dims)
    for d in dims:
        human[d] = human[d].astype(int)

    # If two annotators scored a row, average then round for the comparison.
    human = human.groupby("run_key", as_index=False)[dims].mean().round().astype(
        {d: int for d in dims}
    )

    m = human.merge(judge, on="run_key", suffixes=("_h", "_j"))
    print(f"human rows: {len(human)}   judge rows: {len(judge)}   "
          f"overlap: {len(m)}")
    if len(m) < 30:
        print("\nWARNING: overlap under 30. Agreement estimates will be "
              "unstable. Score more of the validation sample.")
    if m.empty:
        return

    print(f"\n{'dim':>6} {'name':<26} {'QWK':>7} {'exact':>7} {'±1':>7} "
          f"{'h_mean':>7} {'j_mean':>7} {'bias':>7}")
    print("-" * 78)

    rows = []
    for d in DIMENSIONS:
        h, j = m[f"{d.code}_h"].values, m[f"{d.code}_j"].values
        k = quadratic_weighted_kappa(h, j)
        ex, adj = exact_and_adjacent(h, j)
        bias = j.mean() - h.mean()
        rows.append({"dimension": d.code, "name": d.name, "qwk": round(k, 3),
                     "exact": round(ex, 3), "adjacent": round(adj, 3),
                     "human_mean": round(h.mean(), 2),
                     "judge_mean": round(j.mean(), 2),
                     "judge_bias": round(bias, 2)})
        print(f"{d.code:>6} {d.name:<26} {k:>7.3f} {ex:>7.3f} {adj:>7.3f} "
              f"{h.mean():>7.2f} {j.mean():>7.2f} {bias:>+7.2f}")

    ht = m[[f"{d}_h" for d in dims]].sum(axis=1)
    jt = m[[f"{d}_j" for d in dims]].sum(axis=1)
    r = float(np.corrcoef(ht, jt)[0, 1])
    print("-" * 78)
    print(f"{'total':>6} {'(0-20 aggregate)':<26} {'':>7} {'':>7} {'':>7} "
          f"{ht.mean():>7.2f} {jt.mean():>7.2f} {jt.mean()-ht.mean():>+7.2f}")
    print(f"\nPearson r on aggregate quality: {r:.3f}")

    worst = min(rows, key=lambda x: x["qwk"])
    print(f"\nWeakest dimension: {worst['dimension']} ({worst['name']}), "
          f"QWK {worst['qwk']}")
    if worst["qwk"] < 0.40:
        print("  -> Judge is not reliable on this dimension. Either restrict "
              "judge-based claims to the other dimensions, or report the "
              "disagreement as a finding about what automated evaluation "
              "misses in a low-resource language.")

    out = run_dir / "tables"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "judge_agreement.csv", index=False)
    print(f"\n-> {out / 'judge_agreement.csv'}")


if __name__ == "__main__":
    main()
