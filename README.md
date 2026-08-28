# Single-Agent vs. Multi-Agent LLM Architectures for Uzbek

Code, prompt suite, and graded data for the paper **"Does the Single-Agent
Advantage Survive in a Low-Resource Language? A Budget-Matched Comparison of
Standalone and Multi-Agent LLM Architectures for Uzbek"** (Musayeva, 2026).

The study compares three configurations built on a single shared backbone under a
matched generation budget — a direct standalone, a demonstration-augmented
standalone, and a sequential generate–critique–aggregate pipeline — across four
Uzbek task categories (factual, translation, reasoning, cultural), scored on a
five-dimension rubric and a thirteen-code error taxonomy.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── run_harness.py        # collects responses from the 3 architectures via the Gemini API
│   ├── analysis.py           # architecture/category statistics, chrF++, agreement
│   ├── check_progress.py     # tracks collection progress / API quota
│   ├── fix_catastrophic.py   # utility for the flagged catastrophic-failure case
│   └── make_figures.py       # regenerates the five figures
├── data/
│   ├── prompts.csv           # the 120-item prompt suite (prompt, reference, gloss, category, difficulty)
│   ├── responses_graded.csv  # the 1,049 graded responses (scores d1-d5, error codes, architecture)
│   └── validation_native.csv # the 300-response native-speaker validation subset
└── figures/                  # generated figures (optional)
```

> Adjust the names above to match your actual files.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your API key as an environment variable — **never commit it**:

```bash
# macOS/Linux
export GEMINI_API_KEY="your-key-here"
# Windows PowerShell
$env:GEMINI_API_KEY="your-key-here"
```

## Reproducing the results

```bash
# 1. Collect responses (uses the free Gemini tier; see rate-limit note below)
python src/run_harness.py

# 2. Compute all statistics reported in the paper
python src/analysis.py

# 3. Regenerate the figures
python src/make_figures.py
```

**Rate limits.** Collection runs on the free Gemini tier, which enforces a daily
request quota. The harness detects quota exhaustion and stops cleanly so no quota
is wasted, and it supports a configurable delay between calls. A full collection
run may therefore span more than one day.

## Data

- **`prompts.csv`** — 120 items, 30 per category, each with an Uzbek prompt, a
  reference answer, an English gloss, a category label, and a difficulty rating.
- **`responses_graded.csv`** — every graded response with its five dimension
  scores (`d1`–`d5`), total `Q`, outcome label, any error codes, and the
  architecture that produced it.
- **`validation_native.csv`** — the stratified 300-response subset re-graded by a
  native Uzbek speaker.

## Citing this work

If you use this code or data, please cite:

```bibtex
@article{musayeva2026uzbek,
  author  = {Musayeva, Muhsina},
  title   = {Does the Single-Agent Advantage Survive in a Low-Resource Language?
             A Budget-Matched Comparison of Standalone and Multi-Agent LLM
             Architectures for Uzbek},
  year    = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The data in `data/` is
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
