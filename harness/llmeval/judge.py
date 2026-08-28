"""
LLM-as-judge.

Scores responses against the same rubric the human annotator uses, so the two
are directly comparable and the agreement statistic means something.

Three commitments, each of which matters:

1. The judge never sees which architecture produced the response. If it did,
   you would be measuring its priors about multi-agent systems.
2. The rubric text comes from llmeval.rubric, not from a string in this file.
   One instrument, two raters.
3. The judge should be a DIFFERENT model family from the backbone under test.
   A model scoring its own output shows measurable self-preference. If your
   backbone is a local Qwen and your judge is Claude, you are clean.

The judge does not replace the human annotator. It scales the human's
judgement across responses she will never read, and the validation sample is
what earns you the right to do that.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .adapters import BaseAdapter
from .rubric import DIMENSIONS, judge_rubric_text, RUBRIC_VERSION

JUDGE_SYSTEM = """You are an expert evaluator of Uzbek language model output.
You have native competence in Uzbek and familiarity with Uzbek culture.

You will be shown a task and one response. Score the response on five
dimensions using the rubric provided. Be strict: reserve 4 for output a
native professional would be happy to publish.

Judge only what is in front of you. Do not speculate about how the response
was produced.

Return ONLY a JSON object, no prose, no markdown fences, in this exact form:
{"d1": <0-4>, "d2": <0-4>, "d3": <0-4>, "d4": <0-4>, "d5": <0-4>,
 "rationale": "<one sentence, max 30 words>",
 "error_codes": ["E2", "E6"]}

error_codes may be an empty list. Use only codes from the list supplied."""


@dataclass
class JudgeVerdict:
    run_key: str
    d1: int
    d2: int
    d3: int
    d4: int
    d5: int
    rationale: str
    error_codes: list[str]
    parse_ok: bool
    raw: str = ""

    @property
    def quality(self) -> int:
        return self.d1 + self.d2 + self.d3 + self.d4 + self.d5


def build_judge_prompt(task: str, response: str, category: str,
                       reference: str = "") -> str:
    from .rubric import error_codes_text

    ref_block = f"\nREFERENCE ANSWER (for your information):\n{reference}\n" if reference else ""
    return (
        f"{judge_rubric_text()}\n"
        f"{error_codes_text()}\n\n"
        f"---\n"
        f"TASK CATEGORY: {category}\n"
        f"TASK:\n{task}\n"
        f"{ref_block}"
        f"\nRESPONSE TO SCORE:\n{response}\n"
        f"---\n\n"
        f"Return the JSON object now."
    )


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_response(
    adapter: BaseAdapter,
    run_key: str,
    task: str,
    response: str,
    category: str,
    reference: str = "",
) -> JudgeVerdict:
    if not response or not response.strip():
        return JudgeVerdict(run_key, 0, 0, 0, 0, 0,
                            "Empty response.", ["E8"], parse_ok=True)

    prompt = build_judge_prompt(task, response, category, reference)
    out = adapter.generate(
        user=prompt,
        system=JUDGE_SYSTEM,
        max_tokens=600,
        temperature=0.0,
    )
    data = _extract_json(out.text)

    if not data:
        return JudgeVerdict(run_key, -1, -1, -1, -1, -1,
                            "PARSE FAILURE", [], parse_ok=False, raw=out.text)

    def clamp(v):
        try:
            return max(0, min(4, int(v)))
        except (TypeError, ValueError):
            return -1

    codes = data.get("error_codes", [])
    if not isinstance(codes, list):
        codes = []

    return JudgeVerdict(
        run_key=run_key,
        d1=clamp(data.get("d1")),
        d2=clamp(data.get("d2")),
        d3=clamp(data.get("d3")),
        d4=clamp(data.get("d4")),
        d5=clamp(data.get("d5")),
        rationale=str(data.get("rationale", ""))[:300],
        error_codes=[str(c) for c in codes],
        parse_ok=True,
        raw=out.text,
    )


JUDGE_METADATA = {
    "rubric_version": RUBRIC_VERSION,
    "dimensions": [d.code for d in DIMENSIONS],
}
