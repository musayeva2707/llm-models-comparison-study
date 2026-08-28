"""
Architectures.

Every architecture implements the same run() signature and receives the same
global budget B. That symmetry is the experiment: differences in output cannot
be attributed to one system being handed more room to work in.

Budget allocation follows Tran & Kiela (2026): the standalone system gets the
whole of B in one call; pipelines split B across their workers with planning
and aggregation held as close to budget-neutral as the topology allows.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .tracking import TrackedClient, RunResult, make_run_key
from . import prompts as P


# Overhead allowed to planner and aggregator. Kept small and reported, since
# these calls are the main way a pipeline smuggles in extra computation.
COORDINATION_BUDGET = 256

# Room reserved for the actual ANSWER, on top of the thinking budget. Cultural
# and reasoning answers can be long; too small a value clips them mid-sentence.
ANSWER_HEADROOM = 1024


@dataclass
class Item:
    """One prompt from the suite."""

    item_id: str
    category: str      # fact | trans | reas | cult
    prompt: str        # the Uzbek prompt as presented to the model
    reference: str = ""
    direction: str = ""   # for translation items: "uz-en" or "en-uz"
    notes: str = ""


class Architecture(ABC):
    name: str = "base"

    def __init__(self, client: TrackedClient, config: str = "default"):
        self.client = client
        self.config = config      # drives prompt selection (e.g. the rung)
        self.label = config       # what appears in logs and run keys

    @abstractmethod
    def _execute(self, item: Item, budget: int) -> str:
        """Return the final answer text."""

    def run(self, item: Item, budget: int, run_index: int) -> RunResult:
        run_key = make_run_key(item.item_id, self.name, self.label, run_index)
        self.client.begin_run(
            run_key=run_key,
            item_id=item.item_id,
            category=item.category,
            architecture=self.name,
            config=self.label,
            run_index=run_index,
        )

        t0 = time.perf_counter()
        answer = self._execute(item, budget)
        wall = time.perf_counter() - t0

        totals = self.client.totals()
        catastrophic, reason = detect_catastrophic(answer, self.client)

        result = RunResult(
            run_key=run_key,
            item_id=item.item_id,
            category=item.category,
            architecture=self.name,
            config=self.label,
            run_index=run_index,
            final_answer=answer,
            n_calls=totals["n_calls"],
            total_prompt_tokens=totals["prompt"],
            total_completion_tokens=totals["completion"],
            total_reasoning_tokens=totals["reasoning"],
            requested_budget=budget,
            wall_clock_s=wall,
            catastrophic=catastrophic,
            catastrophic_reason=reason,
        )
        self.client.logger.save_transcript(run_key, self.client.transcript)
        self.client.logger.log_result(result)
        return result


def detect_catastrophic(answer: str, client: TrackedClient) -> tuple[bool, str]:
    """
    Automatic screen for the catastrophic-failure category in the paper.

    This catches the mechanical failures only. Wrong-language output that is
    still fluent needs the native-speaker rubric (dimension D3) — do not let
    this function substitute for that.

    IMPORTANT: we judge the FINAL answer, not intermediate pipeline steps.
    In a sequential pipeline a worker often hits its own token budget while
    reasoning — that is normal and does not make the run a failure, as long as
    the aggregator still produced a usable final answer. Only a genuine API
    error, or an empty/near-empty FINAL answer, counts as catastrophic.
    """
    if any(c.error for c in client.calls):
        return True, "api_error"
    if not answer or not answer.strip():
        return True, "empty"
    # Only flag as truncated if truly nothing usable. Short correct answers
    # exist ("H2O", "Bor", "1977", "7"), so 1 char is the real floor, not 5.
    if len(answer.strip()) < 1:
        return True, "truncated"
    # Only the LAST call produces the final answer. If IT was cut off by the
    # token ceiling, the answer may be incomplete; intermediate steps hitting
    # their budget are fine and expected.
    if client.calls and client.calls[-1].stop_reason in ("max_tokens", "length"):
        return True, "final_answer_truncated"
    return False, ""


# --------------------------------------------------------------------------
# Standalone
# --------------------------------------------------------------------------
class Standalone(Architecture):
    """
    Single call, whole budget, one unified context.

    `config` selects the prompt-strength rung, which is the Wang et al. (2024)
    control. Run all of them: reporting only `direct` and declaring the
    pipeline superior is precisely the error the literature has diagnosed.

      direct   bare instruction
      qdesc    detailed task description
      demo     task description plus one worked demonstration
      scaffold structured pre-answer analysis (the SAS-L variant)
    """

    name = "standalone"

    def _execute(self, item: Item, budget: int) -> str:
        system, user = P.standalone_prompt(item, rung=self.config)
        resp = self.client.generate(
            user=user,
            system=system,
            agent_role="standalone",
            max_tokens=budget + ANSWER_HEADROOM,
            thinking_budget=budget,
        )
        return resp.text.strip()


# --------------------------------------------------------------------------
# Sequential pipeline (primary comparator)
# --------------------------------------------------------------------------
class SequentialPipeline(Architecture):
    """
    Planner decomposes, workers solve in order, aggregator synthesizes.

    Chosen as the primary pipeline because it is the cleanest analogue of
    single-agent serial reasoning: both work through one evolving trajectory
    over the whole task, and the only structural difference is that this one
    externalizes intermediate state as explicit messages. That isolates the
    cost and benefit of message passing without also introducing
    specialization, diversity, or adversarial dynamics as confounds.

    Note for §7.3 of the paper: each handoff is a point where information can
    be lost. Instrumenting handoffs individually is what lets you show it.
    """

    name = "sequential"

    def __init__(self, client, config="default", n_steps: int = 2):
        super().__init__(client, config)
        self.n_steps = n_steps

    def _execute(self, item: Item, budget: int) -> str:
        worker_budget = max(128, budget // self.n_steps)

        # --- Planner (budget-neutral) ---
        sys_p, usr_p = P.planner_prompt(item, self.n_steps)
        plan = self.client.generate(
            user=usr_p,
            system=sys_p,
            agent_role="planner",
            max_tokens=COORDINATION_BUDGET,
            thinking_budget=None,
        ).text.strip()

        # --- Workers ---
        intermediate: list[str] = []
        for i in range(self.n_steps):
            sys_w, usr_w = P.worker_prompt(item, plan, intermediate, i, self.n_steps)
            out = self.client.generate(
                user=usr_w,
                system=sys_w,
                agent_role=f"worker_{i}",
                max_tokens=worker_budget + ANSWER_HEADROOM,
                thinking_budget=worker_budget,
            ).text.strip()
            intermediate.append(out)

        # --- Aggregator (budget-neutral) ---
        # The aggregator produces the FINAL answer, so it needs real room —
        # cultural questions can run to 2000+ tokens. Clipping it here was
        # causing long, correct answers to be flagged as truncated failures.
        sys_a, usr_a = P.aggregator_prompt(item, intermediate)
        final = self.client.generate(
            user=usr_a,
            system=sys_a,
            agent_role="aggregator",
            max_tokens=budget + ANSWER_HEADROOM,
            thinking_budget=None,
        ).text.strip()
        return final


# --------------------------------------------------------------------------
# Generator -> Critic -> Refiner (secondary)
# --------------------------------------------------------------------------
class GeneratorCriticRefiner(Architecture):
    """
    The topology that most directly tests the paper's central question.

    A critic built on the same backbone inherits the generator's blind spots.
    If it cannot detect an Uzbek morphological error it would have made
    itself, verification collapses into agreement — and the pipeline returns
    confidence rather than correctness.

    Instrument the critic's verdicts: log whether it flagged anything, and
    whether the refinement improved or degraded the answer. The rate at which
    the critic endorses a wrong answer (error code E11) is a headline number.
    """

    name = "gcr"

    def _execute(self, item: Item, budget: int) -> str:
        share = max(128, budget // 3)

        sys_g, usr_g = P.generator_prompt(item)
        draft = self.client.generate(
            user=usr_g,
            system=sys_g,
            agent_role="generator",
            max_tokens=share + ANSWER_HEADROOM,
            thinking_budget=share,
        ).text.strip()

        sys_c, usr_c = P.critic_prompt(item, draft)
        critique = self.client.generate(
            user=usr_c,
            system=sys_c,
            agent_role="critic",
            max_tokens=share + ANSWER_HEADROOM,
            thinking_budget=share,
        ).text.strip()

        # Cheap proxy for "the critic found nothing". Refine anyway so the
        # budget is comparable across items, but the flag is in the transcript
        # for your E11 analysis.
        sys_r, usr_r = P.refiner_prompt(item, draft, critique)
        final = self.client.generate(
            user=usr_r,
            system=sys_r,
            agent_role="refiner",
            max_tokens=share + ANSWER_HEADROOM,
            thinking_budget=share,
        ).text.strip()
        return final


ARCHITECTURES = {
    "standalone": Standalone,
    "sequential": SequentialPipeline,
    "gcr": GeneratorCriticRefiner,
}
