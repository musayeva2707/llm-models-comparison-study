#!/usr/bin/env python3
"""
Find which models are ACTUALLY FREE for your account.

Some models exist but cost money for your account (they return "limit: 0").
This script sends ONE tiny question to each likely-free model and reports
which actually answer. The winners are your usable models.

    python3 find_free_model.py

(Set your key first:  $env:GEMINI_API_KEY="your-key" )

It sends about a dozen tiny requests total, spaced out, so it won't burn
much quota.
"""

import os
import sys
import time

key = os.environ.get("GEMINI_API_KEY")
if not key:
    print('Set your key first:  $env:GEMINI_API_KEY="your-key"')
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

# Candidates most likely to have a real free tier, cheapest/oldest first.
# Older 'stable' models are likelier to be free than shiny new ones.
CANDIDATES = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

TEST_PROMPT = "Reply with exactly one word: hello"

free_ones = []
paid_ones = []
missing = []

print("Testing which models actually answer for free...\n")

for name in CANDIDATES:
    try:
        resp = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_completion_tokens=10,
            temperature=0.0,
        )
        txt = (resp.choices[0].message.content or "").strip()
        print(f"  FREE   ✓  {name}   (replied: {txt[:20]!r})")
        free_ones.append(name)
    except Exception as e:
        msg = str(e)
        if "limit: 0" in msg or "'limit': 0" in msg:
            print(f"  PAID   ✗  {name}   (no free tier for your account)")
            paid_ones.append(name)
        elif "404" in msg or "not found" in msg.lower() or "not available" in msg.lower():
            print(f"  GONE   ✗  {name}   (not available via this endpoint)")
            missing.append(name)
        elif "429" in msg:
            # A normal rate-limit (not limit:0) actually means it IS free,
            # we just went too fast. Count it as free.
            print(f"  FREE   ✓  {name}   (free, but rate-limited right now)")
            free_ones.append(name)
        else:
            print(f"  ????   ?  {name}   ({msg[:60]})")
        time.sleep(3)  # gentle spacing between probes

print("\n" + "=" * 60)
if free_ones:
    best = free_ones[0]
    print(f"USE THIS MODEL:  {best}")
    print()
    print("Set it and run the pilot:")
    print(f'  $env:GEMINI_MODEL="{best}"')
    print("  python3 pilot_gemini.py")
    if len(free_ones) > 1:
        print(f"\nBackups if that one gets busy: {', '.join(free_ones[1:])}")
else:
    print("No free models answered. This usually means the free tier isn't")
    print("enabled on your Google project, OR today's quota is fully used.")
    print("Paste this whole output to Claude and we'll sort out next steps.")
