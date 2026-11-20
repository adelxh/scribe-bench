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


RECALL_STATE = "generated_note"


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


# Precision stage 1: is a fact from the LLM note in the doctor's reference note?
REF_CRITERIA = {
    "in_reference": "The reference note states this fact with the same meaning (wording may differ).",
    "not_in_reference": (
        "The reference note does not mention this fact, or mentions the same thing with a different "
        "or contradicting detail."
    ),
}

# Precision stage 2: a fact missing from the reference -- did the conversation actually contain it?
DIALOGUE_CRITERIA = {
    "in_dialogue": "The conversation contains this information, said by the doctor or the patient (wording may differ).",
    "not_in_dialogue": (
        "The conversation does not contain this information, or contradicts it. Inferences, assumptions "
        "and details nobody said count as not_in_dialogue."
    ),
}


def ref_question(fact: str) -> dict:
    return {
        "type": "choice",
        "instructions": (
            f'A generated clinical note contains this fact: "{fact}". '
            "Check whether the doctor's reference note in state.reference_note states it. "
            "Judge only against the reference note's text."
        ),
        "criteria": REF_CRITERIA,
    }


def dialogue_question(fact: str) -> dict:
    return {
        "type": "choice",
        "instructions": (
            f'A generated clinical note contains this fact: "{fact}". '
            "Check whether the doctor-patient conversation in state.dialogue contains this information. "
            "Judge only against the conversation's text; do not use outside knowledge."
        ),
        "criteria": DIALOGUE_CRITERIA,
    }


MAX_CONCURRENT = 3  # free models rate-limit (429) when many section calls fire at once
_sem: asyncio.Semaphore | None = None


def _limit() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    return _sem


async def _post(client, key, body):
    async with _limit():
        for attempt in range(4):
            r = await client.post(DECISIONS_URL, headers={"Authorization": f"Bearer {key}"}, json=body)
            if r.status_code == 200 or (r.status_code != 429 and r.status_code < 500):
                break
            await asyncio.sleep(2 * (attempt + 1))
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code != 200:
        raise LLMError(data.get("error", {}).get("message", r.reason_phrase))
    return data


async def judge_section(client, key, state, facts, question_fn=question, criteria=CRITERIA):
    questions = {f"q{i}": question_fn(f) for i, f in enumerate(facts, 1)}
    body = {"model": JUDGE_MODEL, "state": state, "questions": questions}
    try:
        data = await _post(client, key, body)
    except LLMError as e:
        # One failed section should not sink the whole evaluation: mark its facts unjudged.
        print(f"judge section failed ({len(facts)} facts): {e}", flush=True)
        return [
            {"fact": f, "verdict": "unjudged", "confidence": None, "probabilities": None}
            for f in facts
        ]

    answers = data.get("answers") or {}
    results = []
    for i, fact in enumerate(facts, 1):
        a = answers.get(f"q{i}", {})
        verdict = a.get("choice") if a.get("type") != "refusal" else None
        results.append(
            {
                "fact": fact,
                "verdict": verdict if verdict in criteria else "unjudged",
                "confidence": a.get("confidence"),
                "probabilities": a.get("probabilities"),
            }
        )
    return results


def _key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set in server/.env")
    return key


async def judge_all(generated: str, sections: dict[str, list[str]]) -> dict:
    """Recall: judge every reference fact against the generated note. Returns {counts, total, sections}."""
    key = _key()
    names = [n for n, facts in sections.items() if facts]
    state = {"generated_note": generated}
    async with httpx.AsyncClient(timeout=180) as client:
        results = await asyncio.gather(*(judge_section(client, key, state, sections[n]) for n in names))
    judged = dict(zip(names, results))
    rows = [r for rs in judged.values() for r in rs]
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "missed", "wrong", "unjudged")}
    return {"judge_model": JUDGE_MODEL, "counts": counts, "total": len(rows), "sections": judged}


async def judge_precision(
    gen_sections: dict[str, list[str]], reference: str, dialogue: str
) -> dict:
    """Precision: judge every fact of the generated note.

    1. fact in the reference note (any wording)      -> "in_reference"   (good)
    2. else, fact in the conversation                -> "in_dialogue"    (good: the doctor just left it out)
    3. else                                          -> "hallucinated"
    """
    key = _key()
    names = [n for n, facts in gen_sections.items() if facts]
    async with httpx.AsyncClient(timeout=180) as client:
        stage1 = await asyncio.gather(
            *(
                judge_section(client, key, {"reference_note": reference}, gen_sections[n], ref_question, REF_CRITERIA)
                for n in names
            )
        )
        sections = dict(zip(names, stage1))

        # Stage 2 only for facts the reference does not back up (or that could not be judged).
        pending = {n: [r for r in rows if r["verdict"] != "in_reference"] for n, rows in sections.items()}
        pending = {n: rows for n, rows in pending.items() if rows}
        stage2 = await asyncio.gather(
            *(
                judge_section(client, key, {"dialogue": dialogue}, [r["fact"] for r in rows], dialogue_question, DIALOGUE_CRITERIA)
                for rows in pending.values()
            )
        )

    for (n, rows), checked in zip(pending.items(), stage2):
        for row, c in zip(rows, checked):
            if c["verdict"] == "in_dialogue":
                row["verdict"] = "in_dialogue"
            elif c["verdict"] == "not_in_dialogue" and row["verdict"] == "not_in_reference":
                row["verdict"] = "hallucinated"
            else:
                row["verdict"] = "unjudged"
            row["confidence"] = c["confidence"]
            row["probabilities"] = c["probabilities"]

    rows = [r for rs in sections.values() for r in rs]
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("in_reference", "in_dialogue", "hallucinated", "unjudged")}
    return {"counts": counts, "total": len(rows), "sections": sections}
