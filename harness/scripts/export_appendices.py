#!/usr/bin/env python3
"""
Generate Appendix C (rubric) and Appendix D (codebook) from llmeval/rubric.py.

Run this whenever the rubric changes -- before annotation begins -- so the
paper and the code can never disagree about what was measured.

  python scripts/export_appendices.py --out ../paper/sections
Requires \\usepackage{tabularx} in your preamble.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmeval.rubric import to_latex_rubric, to_latex_codebook, RUBRIC_VERSION

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="appendices")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "appendix_rubric.tex").write_text(to_latex_rubric(), encoding="utf-8")
(out / "appendix_codebook.tex").write_text(to_latex_codebook(), encoding="utf-8")
print(f"rubric v{RUBRIC_VERSION} -> {out}/appendix_rubric.tex")
print(f"codebook            -> {out}/appendix_codebook.tex")
