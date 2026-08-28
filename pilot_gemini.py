#!/usr/bin/env python3
"""
The pilot, using Google Gemini's FREE API. No credit card, no local model.

    python3 pilot_gemini.py

Before running, get a free key (2 minutes):
  1. Go to https://aistudio.google.com/apikey
  2. Sign in with a Google account.
  3. Click "Create API key". Copy it.
  4. Set it in your terminal:
        Mac/Linux:   export GEMINI_API_KEY="paste-your-key-here"
        Windows:     set GEMINI_API_KEY=paste-your-key-here
  5. Then run this script in that same terminal.

Same 10 questions as the local pilot, same job: is the Uzbek any good?
This runs in the cloud on Google's free tier, so your laptop's RAM does
not matter at all.
"""

import os
import sys
import time

# Reuse the exact pilot items from the local pilot so results are comparable.
PILOT_ITEMS = [
    ("fact_001", "fact", "easy",   "O'zbekistonning poytaxti qaysi shahar?", "Toshkent"),
    ("fact_018", "fact", "hard",   "Xorazm Ma'mun akademiyasida faoliyat yuritgan, algebra faniga asos solgan buyuk olim kim?", "Muhammad al-Xorazmiy"),
    ("trans_003","trans","easy",   "Quyidagi gapni o'zbek tiliga tarjima qiling: The weather is very cold today.", "Bugun havo juda sovuq."),
    ("trans_013","trans","hard",   "Quyidagi maqolni ingliz tiliga ma'nosini saqlagan holda tarjima qiling: Ishtaha tishning tagida", "Appetite comes with eating."),
    ("reas_001", "reas", "easy",   "Bir savatda 8 ta olma bor edi. Ali 3 tasini oldi, keyin Vali 2 tasini qo'shdi. Hozir savatda nechta olma bor?", "7"),
    ("reas_007", "reas", "hard",   "Uch aka-uka bor. Kattasi o'rtanchasidan 4 yosh katta. O'rtanchasi kichigidan 3 yosh katta. Ularning yoshlari yig'indisi 40. Har birining yoshini toping.", "10, 13, 17"),
    ("cult_001", "cult", "easy",   "O'zbek uylarida mehmon kelganda dasturxonga birinchi navbatda qanday noz-ne'mat qo'yiladi?", "Non"),
    ("cult_016", "cult", "hard",   "O'zbek tilida 'siz' va 'sen' olmoshlari qachon ishlatiladi? Farqini tushuntiring.", "Register/honorific explanation"),
    ("cult_030", "cult", "hard",   "O'zbek madaniyatida 'andisha' tushunchasi nimani anglatadi va nega u muhim qadriyat hisoblanadi?", "Tact/consideration value"),
    ("trans_028","trans","hard",   "Quyidagi ingliz iborasini o'zbek tilidagi mos ma'no bilan bering: To kill two birds with one stone.", "Bir o'q bilan ikki quyonni urmoq"),
]

SYSTEM = (
    "You are assisting with a task in Uzbek. Respond in Uzbek (Latin script) "
    "unless the task explicitly asks for another language. Do not mix in "
    "Turkish, Russian, or English words where an Uzbek equivalent exists."
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("No GEMINI_API_KEY found.")
        print("Get one free at https://aistudio.google.com/apikey, then:")
        print('  Mac/Linux:  export GEMINI_API_KEY="your-key"')
        print("  Windows:    set GEMINI_API_KEY=your-key")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("Missing the 'openai' package. Install it once with:")
        print("  pip install openai")
        sys.exit(1)

    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    print("=" * 70)
    print(f"GEMINI PILOT — model: {MODEL}  (free tier)")
    print("Read each answer. Checking ONE thing: is the Uzbek any good?")
    print("=" * 70)

    total = 0.0
    for iid, cat, diff, prompt, ref in PILOT_ITEMS:
        print(f"\n{'-' * 70}")
        print(f"[{iid}]  {cat} / {diff}")
        print(f"Q:  {prompt}")
        print(f"Expected (rough): {ref}")
        print("-" * 70)
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_completion_tokens=2048,
            )
            dt = time.perf_counter() - t0
            total += dt
            answer = (resp.choices[0].message.content or "").strip()
            print(f"MODEL ({dt:.1f}s):")
            print(answer or "  [empty response]")
        except Exception as e:
            print(f"  ERROR: {e}")
            emsg = str(e)
            # Two different 429s. PerDay means waiting won't help today.
            if "PerDay" in emsg or "RequestsPerDay" in emsg:
                print("\n" + "!" * 60)
                print("DAILY LIMIT REACHED for this model.")
                print("Waiting will NOT help — this resets once every 24 hours")
                print("(at midnight US Pacific time).")
                print("Come back tomorrow and run again. Answers so far are above.")
                print("!" * 60)
                break
            # PerMinute: a short wait fixes it.
            if "429" in emsg or "quota" in emsg.lower() or "rate" in emsg.lower():
                print("  (per-minute limit — waiting 60s for it to reset...)")
                time.sleep(60)
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_completion_tokens=2048,
                    )
                    answer = (resp.choices[0].message.content or "").strip()
                    print("MODEL (after wait):")
                    print(answer or "  [empty response]")
                except Exception as e2:
                    if "PerDay" in str(e2):
                        print("\n  DAILY LIMIT REACHED — come back tomorrow.")
                        break
                    print(f"  STILL FAILED: {e2}")

        # Flash-Lite allows 15 requests/minute. 5s spacing = 12/min, safely
        # under. Only 10 questions, so this finishes in about a minute.
        time.sleep(5)

    print(f"\n{'=' * 70}")
    print(f"Done. Total: {total:.0f}s.")
    print("=" * 70)
    print("""
NOW JUDGE, honestly:
  1. Easy fact right and in clean Uzbek? (Capital = Toshkent.)
  2. Translations stay in the correct language?
  3. Any drift into Turkish or Russian?
  4. Cultural answers sound genuinely Uzbek?
  5. Reasoning maths correct?

If most answers are good Uzbek -> you're ready for the full run on Gemini.
If not -> tell me what failed and we adjust.
""")


if __name__ == "__main__":
    main()
