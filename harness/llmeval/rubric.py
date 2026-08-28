"""
The measurement instruments.

This module is the single source of truth for both the human rubric and the
judge rubric. The annotation sheet, the judge prompt, and the LaTeX appendix
are all generated from the definitions below, so they cannot drift apart. If
they could drift, your judge-validation number would be measuring two
different instruments and telling you nothing.

FREEZE THIS FILE BEFORE SCORING BEGINS.
Any edit after annotation starts invalidates every score collected before it.
Record the version string in your paper.
"""

from __future__ import annotations

from dataclasses import dataclass

RUBRIC_VERSION = "1.0.0"


@dataclass(frozen=True)
class Dimension:
    code: str
    name: str
    question: str
    anchors: dict[int, str]
    applies_to: tuple[str, ...] = ("fact", "trans", "reas", "cult")


# ---------------------------------------------------------------------------
# D1 - Correctness
# ---------------------------------------------------------------------------
D1 = Dimension(
    code="d1",
    name="Correctness",
    question=(
        "Is the content factually right, or (for translation) semantically "
        "faithful to the source?"
    ),
    anchors={
        4: "Fully correct. Nothing to dispute.",
        3: "Correct in substance; a minor detail is imprecise but does not "
           "mislead.",
        2: "Partly correct. Contains a real error alongside correct material, "
           "or answers only part of what was asked.",
        1: "Mostly wrong, but shows some contact with the correct answer.",
        0: "Wrong, or fabricated. Includes confident invention of names, "
           "dates, or sources.",
    },
)

# ---------------------------------------------------------------------------
# D2 - Linguistic quality in Uzbek
# ---------------------------------------------------------------------------
D2 = Dimension(
    code="d2",
    name="Linguistic quality",
    question=(
        "Is the Uzbek morphologically and syntactically well formed, and does "
        "it read naturally?"
    ),
    anchors={
        4: "Natural, fluent Uzbek. A native writer could have produced it.",
        3: "Grammatically sound but slightly stiff or translationese; reads "
           "like careful non-native writing.",
        2: "Noticeable errors in case, possessive, or verb agreement, or "
           "awkward word order. Meaning survives.",
        1: "Frequent morphological errors. Comprehensible only with effort.",
        0: "Broken. Agglutination misapplied, or the text does not parse as "
           "Uzbek.",
    },
)

# ---------------------------------------------------------------------------
# D3 - Language and script fidelity
# ---------------------------------------------------------------------------
D3 = Dimension(
    code="d3",
    name="Language fidelity",
    question=(
        "Did the response stay in the target language and script, without "
        "drifting into Turkish, Russian, or English?"
    ),
    anchors={
        4: "Entirely in the target language and one consistent script.",
        3: "One or two borrowed words that a speaker might use naturally in "
           "this register.",
        2: "Repeated intrusions from another language where an Uzbek "
           "equivalent exists, or inconsistent Latin/Cyrillic mixing.",
        1: "Substantial passages in the wrong language, or heavy Turkish "
           "morphology substituted for Uzbek.",
        0: "The response is in the wrong language entirely.",
    },
)

# ---------------------------------------------------------------------------
# D4 - Cultural appropriateness
# ---------------------------------------------------------------------------
D4 = Dimension(
    code="d4",
    name="Cultural appropriateness",
    question=(
        "Is the register, honorific choice, and cultural framing right for an "
        "Uzbek reader?"
    ),
    anchors={
        4: "Culturally accurate and appropriately registered. Grounded in an "
           "Uzbek frame rather than explained through a foreign one.",
        3: "Appropriate, with a small register slip or a slightly generic "
           "framing.",
        2: "Understands the concept but explains it through a Western "
           "equivalent, or uses the wrong level of formality for the "
           "addressee.",
        1: "Culturally flattened. Recognisably about the right topic but "
           "framed as though for an outsider, or honorifics clearly wrong.",
        0: "Culturally wrong or inappropriate. Would cause offence or "
           "confusion.",
    },
)

# ---------------------------------------------------------------------------
# D5 - Instruction adherence
# ---------------------------------------------------------------------------
D5 = Dimension(
    code="d5",
    name="Instruction adherence",
    question="Did it do what was actually asked, completely and in the right form?",
    anchors={
        4: "Does exactly what was asked, complete, in the requested form.",
        3: "Does what was asked with minor excess (unrequested preamble) or a "
           "small formatting deviation.",
        2: "Partially responsive. Answers a related question, or omits a "
           "requested element.",
        1: "Largely non-responsive, though on topic.",
        0: "Ignores the instruction, refuses, or returns nothing usable.",
    },
)

DIMENSIONS: tuple[Dimension, ...] = (D1, D2, D3, D4, D5)
MAX_SCORE = 4 * len(DIMENSIONS)

# --- Success thresholds (fixed in advance; see §4.2 of the paper) ----------
SUCCESS_MIN = 15
SUCCESS_MIN_DIM = 2
PARTIAL_MIN = 10


def classify(scores: dict[str, int]) -> str:
    """Map a score vector to success / partial / failure."""
    vals = [scores[d.code] for d in DIMENSIONS]
    total = sum(vals)
    if total >= SUCCESS_MIN and min(vals) >= SUCCESS_MIN_DIM:
        return "success"
    if total >= PARTIAL_MIN and min(vals) > 0:
        return "partial"
    return "failure"


# ===========================================================================
# Error taxonomy
# ===========================================================================
@dataclass(frozen=True)
class ErrorCode:
    code: str
    name: str
    definition: str
    architecture: str  # "both" | "pipeline"


ERROR_CODES: tuple[ErrorCode, ...] = (
    ErrorCode("E1", "Factual hallucination",
              "States a fact that is false, or invents a name, date, figure, "
              "or source.", "both"),
    ErrorCode("E2", "Language drift",
              "Produces text in Turkish, Russian, or English where Uzbek was "
              "required. Record which language in the note field.", "both"),
    ErrorCode("E3", "Script inconsistency",
              "Mixes Latin and Cyrillic, or uses inconsistent orthography for "
              "o' and g'.", "both"),
    ErrorCode("E4", "Morphological error",
              "Case, possessive, plural, or verb agreement applied "
              "incorrectly.", "both"),
    ErrorCode("E5", "Idiom literalised",
              "Renders a figurative expression word by word, losing the "
              "meaning.", "both"),
    ErrorCode("E6", "Cultural flattening",
              "Explains an Uzbek concept through a Western equivalent, or "
              "applies a foreign default where a local frame was required.",
              "both"),
    ErrorCode("E7", "Instruction violation",
              "Incomplete, wrong format, or answers a different question.",
              "both"),
    ErrorCode("E8", "Refusal or empty",
              "Declines, returns nothing, or returns only boilerplate.",
              "both"),

    # --- pipeline-specific -------------------------------------------------
    ErrorCode("E9", "Handoff information loss",
              "A detail present in an earlier agent's output is absent from "
              "the final answer, with no correction having been made.",
              "pipeline"),
    ErrorCode("E10", "Error propagation",
              "An error introduced by an early agent survives into the final "
              "answer, and downstream agents build on it.", "pipeline"),
    ErrorCode("E11", "Critic failure",
              "The critic returned NO ISSUES FOUND on a flawed draft, or "
              "'corrected' a draft that was already right. The headline "
              "number for the paper's central argument.", "pipeline"),
    ErrorCode("E12", "False consensus",
              "Agents converge on an answer without any of them having "
              "checked the step that was actually wrong.", "pipeline"),
    ErrorCode("E13", "Aggregator overwrite",
              "A correct intermediate result is discarded or altered for the "
              "worse by the aggregator or refiner.", "pipeline"),
)

ERROR_BY_CODE = {e.code: e for e in ERROR_CODES}


# ===========================================================================
# Rendering
# ===========================================================================
def judge_rubric_text() -> str:
    """The rubric as given to the LLM judge. Identical content to the human sheet."""
    parts = [
        f"SCORING RUBRIC v{RUBRIC_VERSION}",
        "Score each dimension from 0 to 4 using the anchors below.",
        "",
    ]
    for d in DIMENSIONS:
        parts.append(f"{d.code.upper()} - {d.name}: {d.question}")
        for score in sorted(d.anchors, reverse=True):
            parts.append(f"  {score} = {d.anchors[score]}")
        parts.append("")
    return "\n".join(parts)


def error_codes_text() -> str:
    parts = ["ERROR CODES", ""]
    for e in ERROR_CODES:
        tag = " [pipeline only]" if e.architecture == "pipeline" else ""
        parts.append(f"{e.code} {e.name}{tag}: {e.definition}")
    return "\n".join(parts)


def to_latex_rubric() -> str:
    """Appendix C."""
    rows = []
    for d in DIMENSIONS:
        rows.append(
            f"\\multicolumn{{2}}{{l}}{{\\textbf{{{d.code.upper()} --- {d.name}}}: "
            f"\\emph{{{_tex(d.question)}}}}} \\\\[2pt]"
        )
        for s in sorted(d.anchors, reverse=True):
            rows.append(f"\\quad {s} & {_tex(d.anchors[s])} \\\\")
        rows.append("\\addlinespace")
    body = "\n".join(rows)
    return (
        "\\section{Scoring Rubric}\n\\label{app:rubric}\n"
        f"Rubric version {RUBRIC_VERSION}. Frozen prior to annotation.\n\n"
        "\\begin{tabularx}{\\linewidth}{@{}lX@{}}\n\\toprule\n"
        f"{body}\n\\bottomrule\n\\end{{tabularx}}\n"
    )


def to_latex_codebook() -> str:
    """Appendix D."""
    rows = []
    for e in ERROR_CODES:
        tag = "\\textsuperscript{$\\dagger$}" if e.architecture == "pipeline" else ""
        rows.append(
            f"{e.code}{tag} & {_tex(e.name)} & {_tex(e.definition)} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\section{Error Taxonomy}\n\\label{app:codebook}\n"
        "Codes marked \\textsuperscript{$\\dagger$} have no single-agent "
        "analogue.\n\n"
        "\\begin{tabularx}{\\linewidth}{@{}llX@{}}\n\\toprule\n"
        "Code & Name & Definition \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabularx}}\n"
    )


def _tex(s: str) -> str:
    for a, b in [("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")]:
        s = s.replace(a, b)
    return s
