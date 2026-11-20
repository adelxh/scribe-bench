"""Check which reference facts (facts.txt) made it into the LLM-generated note.

Each fact is judged by a Perplexity Decisions model (OpenRouter /api/alpha/decisions) as:
  correct - the LLM note states it with the same meaning
  missed  - the LLM note does not mention it
  wrong   - the LLM note mentions it but with a different/contradicting value (e.g. 500mg vs 1000mg)

Usage (from server/):  .venv/bin/python -m scripts.judge_facts [generated_notes.json]
Reads facts.txt and the generated note for the encounter named in it.
Writes outputs/<encounter_id>_judgement.json
"""
import asyncio
import json
import re
import sys

from app.judge import JUDGE_MODEL, judge_all  # noqa: E402


def parse_facts(path="facts.txt"):
    text = open(path).read()
    encounter = re.search(r"Encounter:\s*(\S+)\s*\((\w+)\)", text)
    sections, current = {}, None
    for line in text.splitlines():
        if re.fullmatch(r"[A-Z/ -]+:", line.strip()):
            current = line.strip()[:-1]
            sections[current] = []
        elif line.startswith("- ") and current and line[2:].strip().lower() != "none":
            sections[current].append(line[2:].strip())
    return encounter.group(1), encounter.group(2), sections


async def main(notes_path):
    encounter_id, split, sections = parse_facts()
    notes = json.load(open(notes_path))
    generated = next(n["generated_note"] for n in notes if n["encounter_id"] == encounter_id)

    result = await judge_all(generated, sections)
    judged, counts, total = result["sections"], result["counts"], result["total"]
    out = {"encounter_id": encounter_id, **result}
    path = f"outputs/{encounter_id}_judgement.json"
    json.dump(out, open(path, "w"), indent=2)

    print(f"\n{encounter_id}: {total} facts -> " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    for verdict in ("correct", "missed", "wrong", "unjudged"):
        if not counts[verdict]:
            continue
        print(f"\n=== {verdict.upper()} ({counts[verdict]}) ===")
        for name, rows in judged.items():
            for r in rows:
                if r["verdict"] == verdict:
                    extra = f"  (conf {r['confidence']:.2f})" if r["confidence"] is not None else ""
                    print(f"[{name}] {r['fact']}{extra}")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "outputs/valid.json"))
