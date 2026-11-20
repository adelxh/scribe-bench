import asyncio
import os

import httpx
from dotenv import load_dotenv

from .llm import LLMError

load_dotenv()

JUDGE_MODEL = "perplexity/pplx-decider-v1.3:free"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

CRITERIA = {
    "correct": "The generated note states this fact with the same meaning (wording may differ).",
    "missed": "The generated note does not mention this fact at all.",
    "wrong": (
        "The generated note mentions the same thing but with a different or contradicting detail "
        "(e.g. the fact says 500 mg but the note says 1000 mg; right wrist vs left wrist; 9/10 vs 5/10)."
    ),
}


def question(fact: str) -> dict:
    return {
        "type": "choice",
        "instructions": (
            f'The doctor\'s reference note (ground truth) contains this fact: "{fact}". '
            "Compare it with the generated clinical note in state.generated_note. "
            "Judge only against the generated note's text; do not use outside knowledge to excuse a missing fact."
        ),
        "criteria": CRITERIA,
    }


async def judge_section(client, key, generated, facts):
    questions = {f"q{i}": question(f) for i, f in enumerate(facts, 1)}
    body = {"model": JUDGE_MODEL, "state": {"generated_note": generated}, "questions": questions}
    for attempt in range(3):
        r = await client.post(DECISIONS_URL, headers={"Authorization": f"Bearer {key}"}, json=body)
        if r.status_code == 200 or (r.status_code != 429 and r.status_code < 500):
            break
        await asyncio.sleep(2 * (attempt + 1))
    data = r.json()
    if r.status_code != 200:
        raise LLMError(data.get("error", {}).get("message", r.reason_phrase))

    answers = data.get("answers") or {}
    results = []
    for i, fact in enumerate(facts, 1):
        a = answers.get(f"q{i}", {})
        verdict = a.get("choice") if a.get("type") != "refusal" else None
        results.append(
            {
                "fact": fact,
                "verdict": verdict if verdict in CRITERIA else "unjudged",
                "confidence": a.get("confidence"),
                "probabilities": a.get("probabilities"),
            }
        )
    return results


async def judge_all(generated: str, sections: dict[str, list[str]]) -> dict:
    """Judge every fact against the generated note. Returns {counts, total, sections}."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set in server/.env")
    names = [n for n, facts in sections.items() if facts]
    async with httpx.AsyncClient(timeout=180) as client:
        results = await asyncio.gather(
            *(judge_section(client, key, generated, sections[n]) for n in names)
        )
    judged = dict(zip(names, results))
    rows = [r for rs in judged.values() for r in rs]
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "missed", "wrong", "unjudged")}
    return {"judge_model": JUDGE_MODEL, "counts": counts, "total": len(rows), "sections": judged}
