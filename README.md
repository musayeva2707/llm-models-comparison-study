# Single-Agent vs. Multi-Agent LLM Architectures for Uzbek

Code, prompt suite, and graded data for the paper **"Does the Single-Agent
Advantage Survive in a Low-Resource Language? A Budget-Matched Comparison of
Standalone and Multi-Agent LLM Architectures for Uzbek"** (Musayeva, 2026).

The study compares three configurations built on a single shared backbone under a
matched generation budget — a direct standalone, a demonstration-augmented
standalone, and a sequential generate–critique–aggregate pipeline — across four
Uzbek task categories (factual, translation, reasoning, cultural), scored on a
five-dimension rubric and a thirteen-code error taxonomy, with a native-speaker
validation subset.

## Repository structure

```
uzbek-study/
├── README.md
├── LICENSE
├── .gitignore
├── find_free_model.py          # model-selection helper (finds an available free Gemini model)
├── list_models.py              # lists the models the API key can access
├── pilot_gemini.py             # early pilot / smoke test against the Gemini API
└── harness/
    ├── requirements.txt
    ├── auto_score.py           # applies automatic (Layer 1) scoring
    ├── compare.py              # cross-architecture comparison
    ├── check_progress.py       # tracks collection progress / API quota
    ├── retry_empties.py        # re-requests responses that came back empty
    ├── diagnose_fails.py       # inspects failed / malformed generations
    ├── fix_catastrophic.py     # utility for the flagged catastrophic-failure case
    ├── inspect_flagged.py      # reviews responses flagged during grading
    ├── llmeval/                # core evaluation package
    │   ├── architectures.py    # the three systems (direct, demo, pipeline)
    │   ├── judge.py            # the LLM judge (Layer 2 rubric grading)
    │   ├── rubric.py          # five rubric dimensions + error-code taxonomy
    │   ├── metrics.py         # chrF++ and other automatic metrics
    │   ├── prompts.py         # prompt templates
    │   ├── adapters.py        # Gemini / OpenAI-compatible API adapters
    │   └── tracking.py        # run/quota tracking
    ├── scripts/
    │   ├── run_experiment.py   # collects responses from the three architectures
    │   ├── analyze.py         # architecture/category statistics
    │   ├── build_scoring_sets.py  # builds the blind (de-identified) grading sets
    │   ├── validate_judge.py   # judge-vs-native-speaker agreement
    │   └── export_appendices.py  # exports example prompts/outputs
    ├── data/                   # the 120-item Uzbek prompt suite
    └── runs/
        └── main/
            ├── auto_scores.csv     # per-response automatic scores
            └── auto_summary.csv    # aggregated summary
```

## Setup

Requires Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r harness\requirements.txt
```

Set your API key as an environment variable — **never commit it**:

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

## Reproducing the results

Run from inside the `harness/` folder. Paths and flags may need to be adjusted to
your setup; the top of each script documents what it expects.

```powershell
# 1. Collect responses from the three architectures (free Gemini tier)
python scripts\run_experiment.py

# 2. Automatic (Layer 1) scoring
python auto_score.py

# 3. Statistics and cross-architecture comparison
python scripts\analyze.py
python compare.py

# 4. Judge vs. native-speaker validation
python scripts\validate_judge.py
```

**Rate limits.** Collection runs on the free Gemini tier, which enforces a daily
request quota. The harness detects quota exhaustion and stops cleanly so no quota
is wasted, and supports a configurable delay between calls, so a full run may span
more than one day. `check_progress.py` reports how far collection has gotten.

## Data

- **`harness/data/`** — the 120-item prompt suite (30 per category), each item
  pairing an Uzbek prompt with a reference answer and an English gloss.
- **`harness/runs/main/transcripts/`** — the raw model responses.
- **`harness/runs/main/auto_scores.csv`** — every response with its scores and the
  architecture that produced it.

## Citing this work

```bibtex
@article{musayeva2026uzbek,
  author = {Musayeva, Muhsina},
  title  = {Does the Single-Agent Advantage Survive in a Low-Resource Language?
            A Budget-Matched Comparison of Standalone and Multi-Agent LLM
            Architectures for Uzbek},
  year   = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The data under
`harness/data/` and `harness/runs/` is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
