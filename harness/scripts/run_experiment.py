#!/usr/bin/env python3
"""
Experiment driver.

Runs the full grid: items x architectures x configs x budgets x repetitions.
Safe to interrupt and rerun — completed runs are skipped.

Examples
--------
# Debug the harness for free, no network, no API key:
python scripts/run_experiment.py --backend mock --model mock-1 --runs 2

# Fully local, no API:
ollama serve &
ollama pull qwen3:14b
python scripts/run_experiment.py --backend ollama --model qwen3:14b

# Claude API:
export ANTHROPIC_API_KEY=sk-...
python scripts/run_experiment.py --backend anthropic --model claude-sonnet-4-5

# Any OpenAI-compatible endpoint (OpenRouter, vLLM, LM Studio):
python scripts/run_experiment.py --backend openai_compat --model qwen/qwen3-32b \
    --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmeval.adapters import build_adapter, DailyQuotaExceeded
from llmeval.tracking import RunLogger, TrackedClient, make_run_key
from llmeval.architectures import ARCHITECTURES, Item


def load_items(path: Path) -> list[Item]:
    items: list[Item] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            items.append(
                Item(
                    item_id=d["item_id"],
                    category=d["category"],
                    prompt=d["prompt"],
                    reference=d.get("reference", ""),
                    direction=d.get("direction", ""),
                    notes=d.get("notes", ""),
                )
            )
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "ollama", "openai_compat", "anthropic", "gemini"])
    ap.add_argument("--model", default="mock-1")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="Ollama host")

    ap.add_argument("--items", default="data/prompts.jsonl")
    ap.add_argument("--out", default="runs/exp01")

    ap.add_argument("--runs", type=int, default=3,
                    help="k repetitions per condition (reliability analysis)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000],
                    help="thinking-token budgets B to sweep")
    ap.add_argument("--architectures", nargs="+",
                    default=["standalone", "sequential"])
    ap.add_argument("--standalone-rungs", nargs="+",
                    default=["direct", "qdesc", "demo"],
                    help="prompt-strength ladder (Wang et al. control)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of items, for smoke tests")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the call count, then exit")
    args = ap.parse_args()

    items = load_items(Path(args.items))
    if args.limit:
        items = items[: args.limit]

    # Build the condition grid.
    conditions: list[tuple[str, str]] = []
    for arch in args.architectures:
        if arch == "standalone":
            conditions += [("standalone", rung) for rung in args.standalone_rungs]
        else:
            conditions.append((arch, "default"))

    total_runs = len(items) * len(conditions) * len(args.budgets) * args.runs
    est_calls = 0
    for arch, _ in conditions:
        per_run = 1 if arch == "standalone" else (4 if arch == "sequential" else 3)
        est_calls += len(items) * len(args.budgets) * args.runs * per_run

    print(f"items:        {len(items)}")
    print(f"conditions:   {conditions}")
    print(f"budgets:      {args.budgets}")
    print(f"repetitions:  {args.runs}")
    print(f"total runs:   {total_runs}")
    print(f"est. calls:   {est_calls}")

    if args.dry_run:
        return

    adapter_kwargs = {}
    if args.backend == "openai_compat":
        adapter_kwargs = {"base_url": args.base_url, "api_key_env": args.api_key_env}
    elif args.backend == "ollama":
        adapter_kwargs = {"host": args.host}

    adapter = build_adapter(args.backend, args.model, **adapter_kwargs)
    logger = RunLogger(args.out)
    client = TrackedClient(adapter, logger, backend=args.backend)

    done = skipped = failed = 0
    for budget in args.budgets:
        for arch_name, config in conditions:
            cfg_label = f"{config}|B={budget}"
            arch_cls = ARCHITECTURES[arch_name]
            arch = arch_cls(client, config=config)
            arch.label = cfg_label   # logs/run keys carry the budget

            for item in items:
                for k in range(args.runs):
                    key = make_run_key(item.item_id, arch_name, cfg_label, k)
                    if logger.is_done(key):
                        skipped += 1
                        continue
                    try:
                        res = arch.run(item, budget, k)
                        done += 1
                        flag = " CATASTROPHIC" if res.catastrophic else ""
                        print(
                            f"[{done + skipped}/{total_runs}] "
                            f"{item.item_id} {arch_name} {cfg_label} r{k} "
                            f"{res.total_completion_tokens}tok "
                            f"{res.wall_clock_s:.1f}s{flag}"
                        )
                    except DailyQuotaExceeded as e:
                        print("\n" + "=" * 64)
                        print("DAILY FREE QUOTA REACHED — stopping cleanly.")
                        print(str(e))
                        print(f"Progress saved: {done} new this session, "
                              f"{skipped} already done.")
                        print("Come back after the reset and run the SAME "
                              "command to continue.")
                        print("=" * 64)
                        print(f"\ndone={done} skipped={skipped} failed={failed}")
                        print(f"output -> {args.out}/")
                        return
                    except Exception as e:  # noqa: BLE001
                        failed += 1
                        print(f"  FAILED {item.item_id} {arch_name} {cfg_label} r{k}: {e}",
                              file=sys.stderr)

    print(f"\ndone={done} skipped={skipped} failed={failed}")
    print(f"output -> {args.out}/")


if __name__ == "__main__":
    main()
