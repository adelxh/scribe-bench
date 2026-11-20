import re

from .llm import EXTRACT_MODEL, chat

SYSTEM_PROMPT = """You extract key clinical facts from a clinical note written by a doctor. Treat everything in the note as fact.

Output exactly these six sections, in this order, with these exact headings, each followed by one fact per line starting with "- ". Use only what the note states; never add anything. If the note has nothing for a section, write "- none".

ATOMIC FACTS (most important rule): every line must contain exactly ONE checkable detail. If a sentence holds several details (a date, a body part, a side, a cause, an activity, a number, a drug and its dose, ...), split it into separate lines. Never join details with "and", commas or "while". Each line must make sense on its own, so repeat the subject where needed.
Example. Bad (one line, 4 facts):
- Right wrist injury sustained 05/12/2022 slipping on stairs while carrying laundry basket
Good (four lines):
- Injury is to the right wrist
- Injury happened on 05/12/2022
- Patient slipped on the stairs
- Patient was carrying a laundry basket when injured
Likewise split "Ultram 50 mg every 6 hours" into the drug, the dose (50 mg) and the frequency (every 6 hours); split "numbness and tingling" into numbness and tingling; split each vital sign into its own line.
Do not list the same fact twice, even if it fits two sections: put it in the best-fitting section only.

DIAGNOSES/CONDITIONS:
SYMPTOMS:
MEDICATION:
VITALS/LABS:
EXAM FINDING:
PLAN/FOLLOW-UP:

Notes on where things go:
- DIAGNOSES/CONDITIONS: current diagnoses, suspected diagnoses and relevant past conditions.
- SYMPTOMS: what the patient reports (onset, severity, aggravating/relieving factors, negatives).
- MEDICATION: drugs mentioned, with dose and frequency when given, and whether current, prescribed or changed.
- VITALS/LABS: vitals, lab values, imaging and test results.
- EXAM FINDING: what the doctor found on physical examination.
- PLAN/FOLLOW-UP: treatments, tests ordered, referrals, counseling, instructions, follow-up.
Output only the sections, no preamble and no markdown formatting."""


def parse_facts(text: str) -> dict[str, list[str]]:
    """Turn the model's "HEADING:\n- fact" text into {heading: [facts]}."""
    sections, current, seen = {}, None, set()
    for line in text.splitlines():
        if re.fullmatch(r"[A-Z/ -]+:", line.strip()):
            current = line.strip()[:-1]
            sections[current] = []
        elif line.startswith("- ") and current and line[2:].strip().lower() != "none":
            fact = line[2:].strip()
            if fact.lower() not in seen:  # drop exact duplicates across sections
                seen.add(fact.lower())
                sections[current].append(fact)
    return sections


async def extract_facts(reference_note: str) -> tuple[str, dict[str, list[str]]]:
    text = await chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Clinical note:\n\n{reference_note}\n\nExtract the key facts."},
        ],
        model=EXTRACT_MODEL,
    )
    return text, parse_facts(text)
