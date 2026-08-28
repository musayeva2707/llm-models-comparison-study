#!/usr/bin/env python3
"""
Ask Google directly: which models can MY key use?

No more guessing at model names. Run this, and it prints the exact list of
models available to your key right now.

    python3 list_models.py

(Set your key first, same as always:
    Windows PowerShell:  $env:GEMINI_API_KEY="your-key"
)
"""

import os
import sys

key = os.environ.get("GEMINI_API_KEY")
if not key:
    print("No GEMINI_API_KEY set. Set it first:")
    print('  Windows PowerShell:  $env:GEMINI_API_KEY="your-key"')
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Run:  pip install openai")
    sys.exit(1)

client = OpenAI(
    api_key=key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

print("Models your key can use:\n")
names = []
try:
    for m in client.models.list():
        names.append(m.id)
        print("  ", m.id)
except Exception as e:
    print("Error listing models:", e)
    sys.exit(1)

print("\n" + "=" * 60)
# Point the user at the best free options if present.
flash_lite = [n for n in names if "flash-lite" in n.lower()]
flash = [n for n in names if "flash" in n.lower() and "lite" not in n.lower()]

print("PICK ONE OF THESE for the best free daily limit:")
if flash_lite:
    print("\n  BEST (most requests/day) — a Flash-Lite model:")
    for n in flash_lite:
        print("    ", n)
if flash:
    print("\n  ALSO GOOD — a Flash model:")
    for n in flash[:5]:
        print("    ", n)

if not flash_lite and not flash:
    print("  (No Flash models found — paste this whole list to Claude.)")

print("\nUse the name EXACTLY as shown. For the pilot, set it like this:")
print('  Windows PowerShell:  $env:GEMINI_MODEL="paste-the-name-here"')
print("  then run:  python3 pilot_gemini.py")
