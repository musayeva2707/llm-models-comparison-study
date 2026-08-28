"""
Instrumentation.

Every model call anywhere in the system passes through TrackedClient, which
writes one JSONL row per call. This is the layer your budget-matching claim
rests on, so it records more than you think you need: requested budget
alongside actual consumption, per-agent attribution, wall-clock latency, and
the exact model version and parameters in force at the time.

Two files per experiment:
  calls.jsonl    one row per model call  (the analysis unit for cost)
  results.jsonl  one row per completed run (the analysis unit for quality)

JSONL rather than CSV because rows are appended atomically, so a crash at call
2,400 of 3,000 costs you nothing — rerun and the harness skips what it already
has.
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import BaseAdapter, LLMResponse


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CallRecord:
    """One model call."""

    run_key: str          # identifies the parent run
    item_id: str
    category: str         # fact | trans | reas | cult
    architecture: str     # standalone | sequential | gcr
    config: str           # e.g. "direct", "demo", "budget=2000"
    run_index: int        # which of the k repetitions

    agent_role: str       # standalone | planner | worker | critic | aggregator
    step_index: int       # order within the run

    backend: str
    model: str
    temperature: float

    requested_max_tokens: int
    requested_thinking_budget: int | None

    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    reasoning_tokens_exact: bool

    latency_s: float
    stop_reason: str
    timestamp: str = field(default_factory=_utc_now)

    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RunResult:
    """One completed run: a prompt through one architecture, once."""

    run_key: str
    item_id: str
    category: str
    architecture: str
    config: str
    run_index: int

    final_answer: str

    n_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int
    requested_budget: int
    wall_clock_s: float

    catastrophic: bool = False
    catastrophic_reason: str = ""
    timestamp: str = field(default_factory=_utc_now)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def make_run_key(
    item_id: str, architecture: str, config: str, run_index: int
) -> str:
    """Stable identifier so reruns can skip completed work."""
    raw = f"{item_id}|{architecture}|{config}|{run_index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class RunLogger:
    """Append-only JSONL writer with resumability."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.out_dir / "calls.jsonl"
        self.results_path = self.out_dir / "results.jsonl"
        self.transcripts_dir = self.out_dir / "transcripts"
        self.transcripts_dir.mkdir(exist_ok=True)
        self._completed = self._load_completed()

    def _load_completed(self) -> set[str]:
        done: set[str] = set()
        if self.results_path.exists():
            with open(self.results_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["run_key"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        return done

    def is_done(self, run_key: str) -> bool:
        return run_key in self._completed

    def log_call(self, rec: CallRecord) -> None:
        with open(self.calls_path, "a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")

    def log_result(self, res: RunResult) -> None:
        with open(self.results_path, "a", encoding="utf-8") as f:
            f.write(res.to_json() + "\n")
        self._completed.add(res.run_key)

    def save_transcript(self, run_key: str, transcript: list[dict[str, Any]]) -> None:
        """Full message trace. Appendix E of the paper comes from these."""
        path = self.transcripts_dir / f"{run_key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)


class TrackedClient:
    """
    Wraps an adapter and records every call.

    The architecture classes hold one of these rather than an adapter
    directly, which is what guarantees no call escapes instrumentation.
    """

    def __init__(self, adapter: BaseAdapter, logger: RunLogger, backend: str):
        self.adapter = adapter
        self.logger = logger
        self.backend = backend
        self._ctx: dict[str, Any] = {}
        self._transcript: list[dict[str, Any]] = []
        self._step = 0
        self.calls: list[CallRecord] = []
        # Pause before every model call, to stay under per-minute rate limits.
        # Free Gemini allows ~15/min; 5s spacing = 12/min, safely under, and it
        # matters most for pipelines that make several calls per run.
        # Override with:  set GEMINI_PACE_SECONDS=8  (Windows: $env:...)
        import os
        self._pace = float(os.environ.get("LLM_PACE_SECONDS", "0") or "0")

    def begin_run(self, **ctx) -> None:
        """Set the run context that every subsequent call inherits."""
        self._ctx = ctx
        self._transcript = []
        self._step = 0
        self.calls = []

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return self._transcript

    def totals(self) -> dict[str, int]:
        return {
            "n_calls": len(self.calls),
            "prompt": sum(c.prompt_tokens for c in self.calls),
            "completion": sum(c.completion_tokens for c in self.calls),
            "reasoning": sum(c.reasoning_tokens for c in self.calls),
        }

    def generate(
        self,
        user: str,
        system: str = "",
        agent_role: str = "standalone",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
    ) -> LLMResponse:
        idx = self._step
        self._step += 1

        # Pace calls to respect per-minute rate limits.
        if self._pace > 0:
            import time as _t
            _t.sleep(self._pace)

        error: str | None = None
        try:
            resp = self.adapter.generate(
                user=user,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking_budget=thinking_budget,
            )
        except Exception as e:  # noqa: BLE001
            error = str(e)
            resp = LLMResponse(
                text="", prompt_tokens=0, completion_tokens=0, model=self.adapter.model
            )

        rec = CallRecord(
            run_key=self._ctx.get("run_key", ""),
            item_id=self._ctx.get("item_id", ""),
            category=self._ctx.get("category", ""),
            architecture=self._ctx.get("architecture", ""),
            config=self._ctx.get("config", ""),
            run_index=self._ctx.get("run_index", 0),
            agent_role=agent_role,
            step_index=idx,
            backend=self.backend,
            model=self.adapter.model,
            temperature=temperature,
            requested_max_tokens=max_tokens,
            requested_thinking_budget=thinking_budget,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            reasoning_tokens=resp.reasoning_tokens,
            reasoning_tokens_exact=resp.reasoning_tokens_exact,
            latency_s=resp.latency_s,
            stop_reason=resp.stop_reason,
            error=error,
        )
        self.logger.log_call(rec)
        self.calls.append(rec)

        self._transcript.append(
            {
                "step": idx,
                "agent_role": agent_role,
                "system": system,
                "user": user,
                "reasoning": resp.reasoning_text,
                "assistant": resp.text,
                "tokens": {
                    "prompt": resp.prompt_tokens,
                    "completion": resp.completion_tokens,
                    "reasoning": resp.reasoning_tokens,
                },
                "latency_s": round(resp.latency_s, 3),
                "error": error,
            }
        )
        return resp
