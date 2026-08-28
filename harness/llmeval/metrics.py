"""
Automatic checks.

These run on all 3,600 responses for free and do the work that does not need
a human: catching mechanical failures, screening for language drift, and
scoring translation against references.

Read this carefully: these are SCREENS, not judgements. The language detector
flags candidates for human review; it does not decide D3. A response can pass
every check here and still be culturally wrong, badly registered, or subtly
false — which is exactly why the human sample exists. Do not let a green
light from this module substitute for the rubric.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict


# ---------------------------------------------------------------------------
# Script and language screening  (feeds D3)
# ---------------------------------------------------------------------------

# Uzbek Latin does not use these. Their presence suggests Turkish.
TURKISH_CHARS = set("ıİğĞşŞöÖüÜçÇ") - set("çÇ")  # ç appears in neither; kept explicit
TURKISH_ONLY = set("ıİğĞşŞöÖüÜ")

CYRILLIC_RANGE = (0x0400, 0x04FF)

# High-frequency function words. Deliberately short lists: these are cheap
# signals for a screen, not a language identifier.
UZ_MARKERS = {
    "va", "bilan", "uchun", "bu", "shu", "ham", "lekin", "ammo", "yoki",
    "bo'ldi", "boldi", "bo‘ldi", "qilish", "kerak", "mumkin", "yo'q", "yoq",
    "bor", "juda", "eng", "har", "qanday", "nima", "qachon", "ular", "biz",
}
EN_MARKERS = {
    "the", "and", "is", "are", "was", "were", "this", "that", "with", "for",
    "you", "your", "have", "has", "will", "would", "can", "should", "there",
}
RU_MARKERS = {
    "и", "в", "не", "на", "что", "это", "как", "для", "или", "но", "все",
    "который", "быть", "если",
}
# Uzbek Cyrillic has four letters Russian does not. Their presence means the
# text is Uzbek in the wrong script, which is a different finding from the
# text being Russian — one is an orthography problem, the other is language
# drift, and they belong in different rows of your error table.
UZ_CYRILLIC_CHARS = set("ўЎқҚғҒҳҲ")
UZ_CYRILLIC_MARKERS = {
    "ва", "билан", "учун", "бу", "шу", "ҳам", "лекин", "ёки", "керак",
    "мумкин", "йўқ", "бор", "жуда", "энг", "ҳар", "қандай", "нима",
}
TR_MARKERS = {
    "ve", "bir", "için", "bu", "ile", "olarak", "daha", "çok", "değil",
    "olan", "gibi", "ancak",
}


@dataclass
class ScriptReport:
    latin_chars: int
    cyrillic_chars: int
    script_mixed: bool
    turkish_char_hits: int
    uz_cyrillic_hits: int
    uz_marker_hits: int
    en_marker_hits: int
    ru_marker_hits: int
    tr_marker_hits: int
    likely_language: str
    drift_flag: bool
    apostrophe_variants: int

    def to_dict(self):
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w'’‘ʻ]+", text.lower(), flags=re.UNICODE)


def screen_script(text: str, expect: str = "uz") -> ScriptReport:
    """
    Screen a response for script and language drift.

    `expect` is "uz" or "en" (for uz->en translation targets).
    """
    latin = cyr = 0
    turkish_hits = 0
    uz_cyr_hits = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if CYRILLIC_RANGE[0] <= cp <= CYRILLIC_RANGE[1]:
            cyr += 1
        elif "LATIN" in unicodedata.name(ch, ""):
            latin += 1
        if ch in TURKISH_ONLY:
            turkish_hits += 1
        if ch in UZ_CYRILLIC_CHARS:
            uz_cyr_hits += 1

    toks = set(_tokens(text))
    uz = len(toks & UZ_MARKERS)
    uz_cyr = len(toks & UZ_CYRILLIC_MARKERS) + (1 if uz_cyr_hits else 0)
    en = len(toks & EN_MARKERS)
    ru = len(toks & RU_MARKERS)
    tr = len(toks & TR_MARKERS)

    # uz_cyr scored separately so Uzbek-in-Cyrillic is not mistaken for Russian.
    scores = {
        "uz": uz,
        "uz_cyrillic": uz_cyr * 2,
        "en": en,
        "ru": ru + (cyr // 20) - uz_cyr * 3,
        "tr": tr + turkish_hits,
    }
    likely = max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"

    # Orthographic inconsistency: mixing ' ‘ ’ ʻ for the same letter.
    variants = len({c for c in text if c in "'‘’ʻ"})

    alpha = latin + cyr
    mixed = alpha > 0 and min(latin, cyr) / alpha > 0.10

    # Uzbek in Cyrillic is still Uzbek, but the prompt asked for Latin, so it
    # is flagged for review as an orthography issue rather than language drift.
    wrong_script = expect == "uz" and likely == "uz_cyrillic"
    drift = (
        (likely != expect and not wrong_script)
        or wrong_script
        or mixed
        or turkish_hits > 0
        or (expect == "uz" and en >= 3)
    )

    return ScriptReport(
        latin_chars=latin,
        cyrillic_chars=cyr,
        script_mixed=mixed,
        turkish_char_hits=turkish_hits,
        uz_cyrillic_hits=uz_cyr_hits,
        uz_marker_hits=uz,
        en_marker_hits=en,
        ru_marker_hits=ru,
        tr_marker_hits=tr,
        likely_language=likely,
        drift_flag=drift,
        apostrophe_variants=variants,
    )


# ---------------------------------------------------------------------------
# chrF++  (translation items)
# ---------------------------------------------------------------------------
# Character n-gram F-score. Far better suited to Uzbek than BLEU: BLEU matches
# whole word forms, and in an agglutinative language a correct translation with
# one different suffix scores zero on that token. chrF++ sees the shared stem.

def _ngrams(seq, n):
    return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def _f_score(hyp: Counter, ref: Counter, beta: float) -> float | None:
    if not hyp or not ref:
        return None
    overlap = sum((hyp & ref).values())
    p = overlap / sum(hyp.values())
    r = overlap / sum(ref.values())
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


def chrf_pp(
    hypothesis: str,
    reference: str,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """
    chrF++ in [0, 1]. Character n-grams up to `char_order`, word n-grams up to
    `word_order`, recall weighted beta times precision.
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0

    hc = re.sub(r"\s+", "", hypothesis.lower())
    rc = re.sub(r"\s+", "", reference.lower())
    hw = _tokens(hypothesis)
    rw = _tokens(reference)

    scores = []
    for n in range(1, char_order + 1):
        s = _f_score(_ngrams(hc, n), _ngrams(rc, n), beta)
        if s is not None:
            scores.append(s)
    for n in range(1, word_order + 1):
        s = _f_score(_ngrams(hw, n), _ngrams(rw, n), beta)
        if s is not None:
            scores.append(s)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Factual matching
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Normalise for comparison: fold apostrophe variants, strip punctuation."""
    t = text.lower().strip()
    for v in "‘’ʻ`":
        t = t.replace(v, "'")
    t = re.sub(r"[^\w\s']", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def contains_reference(answer: str, reference: str) -> bool:
    """
    Lenient containment check for factual items.

    Deliberately lenient: the model may wrap the right answer in a sentence.
    A False here means "look at this by hand", not "this is wrong".
    """
    if not reference.strip():
        return False
    return normalize(reference) in normalize(answer)


# ---------------------------------------------------------------------------
# Consistency across repeated runs  (feeds RQ2)
# ---------------------------------------------------------------------------
def run_consistency(answers: list[str]) -> dict:
    """
    How stable were the k runs of one condition?

    `exact_agreement` is the share of runs matching the modal normalised
    answer. `mean_pairwise_chrf` measures near-agreement for open-ended items
    where exact match is meaningless.
    """
    clean = [normalize(a) for a in answers if a and a.strip()]
    if len(clean) < 2:
        return {"n": len(clean), "exact_agreement": None,
                "mean_pairwise_chrf": None, "length_cv": None}

    modal = Counter(clean).most_common(1)[0][1]
    pairs = [
        chrf_pp(answers[i], answers[j])
        for i in range(len(answers))
        for j in range(i + 1, len(answers))
        if answers[i].strip() and answers[j].strip()
    ]
    lens = [len(a) for a in answers if a.strip()]
    mean_len = sum(lens) / len(lens)
    var = sum((x - mean_len) ** 2 for x in lens) / len(lens)
    cv = (var ** 0.5) / mean_len if mean_len else None

    return {
        "n": len(clean),
        "exact_agreement": round(modal / len(clean), 3),
        "mean_pairwise_chrf": round(sum(pairs) / len(pairs), 3) if pairs else None,
        "length_cv": round(cv, 3) if cv is not None else None,
    }


# ---------------------------------------------------------------------------
def score_response(answer: str, item_category: str, reference: str = "",
                   direction: str = "") -> dict:
    """Run every applicable automatic check on one response."""
    expect = "en" if direction == "uz-en" else "uz"
    rep = screen_script(answer, expect=expect)

    out = {
        "auto_expect_lang": expect,
        "auto_likely_lang": rep.likely_language,
        "auto_drift_flag": rep.drift_flag,
        "auto_script_mixed": rep.script_mixed,
        "auto_turkish_hits": rep.turkish_char_hits,
        "auto_apostrophe_variants": rep.apostrophe_variants,
        "auto_length_chars": len(answer),
        "auto_empty": not bool(answer.strip()),
    }
    if item_category == "trans" and reference:
        out["auto_chrf"] = round(chrf_pp(answer, reference), 4)
    if item_category == "fact" and reference:
        out["auto_ref_match"] = contains_reference(answer, reference)
    return out
