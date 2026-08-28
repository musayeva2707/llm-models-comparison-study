"""
Prompt templates.

All prompts live here so Appendix B of the paper can be generated mechanically
rather than transcribed by hand. Nothing in this file should ever be edited
mid-experiment; if you change a template, that is a new experiment.

Design constraint carried through every template: the agent instructions are
in English while the task content is Uzbek. This is deliberate and worth a
sentence in your methodology. Instructing in Uzbek would confound the
architecture comparison with the model's instruction-following ability in a
low-resource language — you would no longer know whether a pipeline failed
because coordination is costly or because the coordinator misread its own
brief. Keeping instructions in English isolates the variable you care about.
"""

from __future__ import annotations

CATEGORY_GUIDANCE = {
    "fact": (
        "This is a factual question. Answer accurately and concisely. "
        "If you are not certain of the fact, say so rather than guessing."
    ),
    "trans": (
        "This is a translation task. Preserve meaning, register, and tone. "
        "Do not translate idioms literally where an equivalent expression "
        "exists in the target language."
    ),
    "reas": (
        "This is a multi-step reasoning task. Work through it carefully "
        "before committing to an answer."
    ),
    "cult": (
        "This task concerns Uzbek culture, customs, or usage. Answer from "
        "within the Uzbek cultural frame rather than by analogy to Western "
        "equivalents. Pay attention to register and to appropriate honorifics."
    ),
}

LANGUAGE_RULE = (
    "Respond in Uzbek (Latin script) unless the task explicitly asks for "
    "another language. Do not mix in Turkish, Russian, or English words "
    "where an Uzbek equivalent exists."
)


def _base_system(item) -> str:
    return (
        "You are assisting with a task in Uzbek.\n"
        + CATEGORY_GUIDANCE.get(item.category, "")
        + "\n"
        + LANGUAGE_RULE
    )


# --------------------------------------------------------------------------
# Standalone: the prompt-strength ladder (Wang et al. 2024 control)
# --------------------------------------------------------------------------

DEMONSTRATIONS = {
    "fact": (
        "Example:\n"
        "Task: O'zbekistonning poytaxti qaysi shahar?\n"
        "Answer: Toshkent.\n"
    ),
    "trans": (
        "Example:\n"
        "Task: Translate to English: Kitob o'qish foydali.\n"
        "Answer: Reading books is beneficial.\n"
    ),
    "reas": (
        "Example:\n"
        "Task: Alisher Karimdan 3 yosh katta. Karim 12 yoshda. "
        "Ular birgalikda necha yoshda?\n"
        "Answer: Alisher 15 yoshda. Birgalikda 27 yoshda.\n"
    ),
    "cult": (
        "Example:\n"
        "Task: Mehmon kelganda dasturxonga birinchi navbatda nima qo'yiladi?\n"
        "Answer: Non va choy. Non dasturxonning eng muhim ramzi hisoblanadi.\n"
    ),
}

SCAFFOLD = (
    "Before answering, work through the following:\n"
    "1. Identify any ambiguity in the task.\n"
    "2. Propose the plausible interpretations.\n"
    "3. Evaluate them and select one.\n"
    "4. Only then give your answer.\n"
    "Give only the final answer in your response; keep the analysis internal."
)


def standalone_prompt(item, rung: str = "direct") -> tuple[str, str]:
    system = _base_system(item)

    if rung == "direct":
        return "", item.prompt

    if rung == "qdesc":
        return system, item.prompt

    if rung == "demo":
        demo = DEMONSTRATIONS.get(item.category, "")
        return system, f"{demo}\nNow the actual task:\n{item.prompt}"

    if rung == "scaffold":
        return system, f"{SCAFFOLD}\n\nTask:\n{item.prompt}"

    raise ValueError(f"Unknown prompt rung: {rung!r}")


# --------------------------------------------------------------------------
# Sequential pipeline
# --------------------------------------------------------------------------

def planner_prompt(item, n_steps: int) -> tuple[str, str]:
    system = (
        "You are a planning agent. Decompose the task into ordered steps for "
        "worker agents. Output only a numbered list of steps, nothing else. "
        "Be brief."
    )
    user = (
        f"Decompose this task into exactly {n_steps} ordered steps.\n\n"
        f"Task category: {item.category}\n"
        f"Task:\n{item.prompt}"
    )
    return system, user


def worker_prompt(item, plan, intermediate, step_idx, n_steps) -> tuple[str, str]:
    system = _base_system(item) + (
        "\nYou are one worker in a pipeline. Complete only your assigned step."
    )
    prior = ""
    if intermediate:
        prior = "Results from earlier steps:\n" + "\n".join(
            f"[Step {i + 1}] {t}" for i, t in enumerate(intermediate)
        )
    user = (
        f"Original task:\n{item.prompt}\n\n"
        f"Plan:\n{plan}\n\n"
        f"{prior}\n\n"
        f"Now complete step {step_idx + 1} of {n_steps}. "
        f"Output only the result of your step."
    )
    return system, user


def aggregator_prompt(item, intermediate) -> tuple[str, str]:
    system = _base_system(item) + (
        "\nYou are an aggregation agent. Synthesize the workers' outputs into "
        "one final answer to the original task. Output only the final answer."
    )
    joined = "\n".join(f"[Step {i + 1}] {t}" for i, t in enumerate(intermediate))
    user = (
        f"Original task:\n{item.prompt}\n\n"
        f"Worker outputs:\n{joined}\n\n"
        f"Produce the final answer."
    )
    return system, user


# --------------------------------------------------------------------------
# Generator -> Critic -> Refiner
# --------------------------------------------------------------------------

def generator_prompt(item) -> tuple[str, str]:
    return _base_system(item), item.prompt


def critic_prompt(item, draft) -> tuple[str, str]:
    system = (
        "You are a reviewing agent with native competence in Uzbek. Evaluate "
        "the draft answer below. Check specifically for:\n"
        "  (a) factual or semantic errors\n"
        "  (b) morphological or agreement errors in the Uzbek\n"
        "  (c) words borrowed from Turkish, Russian, or English where an "
        "Uzbek equivalent exists\n"
        "  (d) inappropriate register or honorifics\n"
        "  (e) cultural framing that is Western rather than Uzbek\n\n"
        "If the draft is correct, say exactly: NO ISSUES FOUND\n"
        "Otherwise list the specific problems. Do not rewrite the answer."
    )
    user = f"Original task:\n{item.prompt}\n\nDraft answer:\n{draft}\n\nYour review:"
    return system, user


def refiner_prompt(item, draft, critique) -> tuple[str, str]:
    system = _base_system(item) + (
        "\nYou are a refinement agent. Apply the reviewer's corrections to the "
        "draft. If the reviewer found no issues, return the draft unchanged. "
        "Output only the final answer."
    )
    user = (
        f"Original task:\n{item.prompt}\n\n"
        f"Draft:\n{draft}\n\n"
        f"Review:\n{critique}\n\n"
        f"Final answer:"
    )
    return system, user
